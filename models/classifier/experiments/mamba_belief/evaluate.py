"""
Mamba Belief 모델 평가 모듈

compressive_memory/evaluate.py 와 동일한 평가 파이프라인을 사용한다:
  1. 기본 지표: Subset Accuracy, Micro F1, Macro F1, Hamming Loss
  2. 클래스별: F1, Precision, Recall, Support
  3. Threshold 최적화: 클래스별 최적 threshold 탐색 (dev → test 적용)
  4. 스트리밍 정량 지표: Convergence Rate, First Correct Step, Oscillation Count
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from sklearn.metrics import (
    accuracy_score, f1_score, hamming_loss,
    precision_score, recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "compressive_memory"))
from dataset import create_dataloaders

from config import (
    ABLATION_CONFIGS, CHECKPOINT_DIR, DATA_DIR, DEVICE,
    EVAL_CONFIG, GPU_CONFIG, LOG_DIR, MELD_CONFIG, TRAIN_CONFIG,
)
from model import build_mamba_model


# ── 체크포인트 경로 결정 ───────────────────────────────────

def resolve_checkpoint_path(experiment_name: str, run_id: str | None = None) -> tuple[Path, str]:
    if run_id is not None:
        return CHECKPOINT_DIR / f"{experiment_name}_{run_id}_best.pt", run_id

    latest_meta = CHECKPOINT_DIR / f"{experiment_name}_latest.json"
    if latest_meta.exists():
        with open(latest_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return Path(meta["checkpoint_path"]), meta.get("run_id", "latest")

    return CHECKPOINT_DIR / f"{experiment_name}_best.pt", "legacy"


def build_eval_log_path(experiment_name: str, split: str, model_run_id: str) -> Path:
    eval_run_id = time.strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"{experiment_name}_{model_run_id}_{split}_{eval_run_id}_eval.json"


# ── 기본 / 클래스별 지표 ───────────────────────────────────

def compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "subset_accuracy": accuracy_score(y_true, y_pred),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
    }


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict:
    return {
        name: {
            "f1": f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "precision": precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "recall": recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "support": int(y_true[:, i].sum()),
        }
        for i, name in enumerate(class_names)
    }


# ── Threshold 최적화 ───────────────────────────────────────

def find_optimal_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    search_range: tuple = (0.2, 0.8, 0.05),
) -> list[float]:
    start, end, step = search_range
    candidates = np.arange(start, end + step, step)
    best = []
    for i in range(y_true.shape[1]):
        best_f1, best_t = 0.0, 0.5
        for t in candidates:
            f1 = f1_score(y_true[:, i], (y_probs[:, i] > t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        best.append(round(best_t, 2))
    return best


# ── 스트리밍 정량 지표 ─────────────────────────────────────

def compute_streaming_metrics(
    all_step_probs: list[np.ndarray],
    labels: np.ndarray,
    num_segments: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    convergence_counts, first_correct_steps, oscillation_counts = [], [], []

    for n in range(labels.shape[0]):
        n_segs = int(num_segments[n])
        if n_segs <= 1:
            continue

        label = labels[n]
        step_preds, step_probs_n = [], []
        for t in range(n_segs):
            if t < len(all_step_probs) and n < all_step_probs[t].shape[0]:
                p = all_step_probs[t][n]
                step_preds.append((p > threshold).astype(int))
                step_probs_n.append(p)
            else:
                break

        if len(step_preds) < 2:
            continue

        positive_classes = np.where(label == 1)[0]
        if len(positive_classes) > 0:
            increases = total = 0
            for t in range(1, len(step_probs_n)):
                for c in positive_classes:
                    increases += step_probs_n[t][c] >= step_probs_n[t - 1][c]
                    total += 1
            if total > 0:
                convergence_counts.append(increases / total)

        found = False
        for t, pred in enumerate(step_preds):
            if np.array_equal(pred, label):
                first_correct_steps.append(t)
                found = True
                break
        if not found:
            first_correct_steps.append(n_segs)

        osc = sum(
            1 for t in range(1, len(step_preds))
            if np.array_equal(step_preds[t - 1], label) != np.array_equal(step_preds[t], label)
        )
        oscillation_counts.append(osc)

    return {
        "convergence_rate": float(np.mean(convergence_counts)) if convergence_counts else 0.0,
        "first_correct_step": float(np.mean(first_correct_steps)) if first_correct_steps else 0.0,
        "oscillation_count": float(np.mean(oscillation_counts)) if oscillation_counts else 0.0,
        "num_streaming_samples": len(oscillation_counts),
    }


# ── 예측 수집 ──────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader,
    compute_streaming: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    model.eval()
    amp_enabled = GPU_CONFIG["ENABLED"] and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if GPU_CONFIG["DTYPE"] == "bf16" else torch.float16

    all_labels, all_probs, all_num_segs = [], [], []
    all_step_probs: list[list[np.ndarray]] = []
    past_batch_sizes: list[int] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        segment_mask = batch["segment_mask"].to(DEVICE)
        labels = batch["labels"]
        num_segments = batch["num_segments"]

        with autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask,
                segment_mask=segment_mask, num_segments=num_segments.to(DEVICE),
            )

        probs = torch.sigmoid(outputs["logits"].float()).cpu().numpy()
        all_labels.append(labels.numpy())
        all_probs.append(probs)
        all_num_segs.append(num_segments.numpy())

        if compute_streaming and "all_logits" in outputs:
            batch_size = probs.shape[0]
            num_classes = probs.shape[1]
            step_probs_batch = [
                torch.sigmoid(lt.float()).cpu().numpy()
                for lt in outputs["all_logits"]
            ]
            curr_steps = len(step_probs_batch)
            prev_max = len(all_step_probs)

            if curr_steps > prev_max:
                for _ in range(prev_max, curr_steps):
                    all_step_probs.append(
                        [np.zeros((b, num_classes), dtype=np.float32) for b in past_batch_sizes]
                    )

            for t in range(len(all_step_probs)):
                all_step_probs[t].append(
                    step_probs_batch[t] if t < curr_steps
                    else np.zeros((batch_size, num_classes), dtype=np.float32)
                )

            past_batch_sizes.append(batch_size)

    y_true = np.concatenate(all_labels, axis=0)
    y_probs = np.concatenate(all_probs, axis=0)
    num_segs_arr = np.concatenate(all_num_segs, axis=0)
    step_probs = [np.concatenate(t, axis=0) for t in all_step_probs] if all_step_probs else []

    return y_true, y_probs, num_segs_arr, step_probs


def apply_thresholds(y_probs: np.ndarray, thresholds: float | list[float]) -> np.ndarray:
    if isinstance(thresholds, (int, float)):
        return (y_probs > float(thresholds)).astype(int)
    return (y_probs > np.asarray(thresholds, dtype=float).reshape(1, -1)).astype(int)


# ── 전체 평가 파이프라인 ───────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader,
    threshold: float = 0.5,
    threshold_search_loader=None,
    compute_streaming: bool = True,
) -> dict:
    y_true, y_probs, num_segs_arr, step_probs = collect_predictions(
        model, loader, compute_streaming=compute_streaming,
    )

    y_pred = apply_thresholds(y_probs, threshold)
    basic = compute_basic_metrics(y_true, y_pred)
    basic["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    class_names = MELD_CONFIG["EMOTION_LABELS"] + MELD_CONFIG["SENTIMENT_LABELS"]
    per_class = compute_per_class_metrics(y_true, y_pred, class_names)

    search_y_true, search_y_probs = y_true, y_probs
    threshold_source = "current_split"
    if threshold_search_loader is not None:
        search_y_true, search_y_probs, _, _ = collect_predictions(
            model, threshold_search_loader, compute_streaming=False,
        )
        threshold_source = "dev"

    optimal_thresholds = find_optimal_thresholds(
        search_y_true, search_y_probs, EVAL_CONFIG["THRESHOLD_SEARCH_RANGE"],
    )
    basic_optimal = compute_basic_metrics(y_true, apply_thresholds(y_probs, optimal_thresholds))

    streaming = {}
    if compute_streaming and step_probs:
        streaming = compute_streaming_metrics(step_probs, y_true, num_segs_arr, threshold)

    return {
        "basic_metrics": basic,
        "basic_metrics_optimal_threshold": basic_optimal,
        "optimal_thresholds": {n: t for n, t in zip(class_names, optimal_thresholds)},
        "per_class_metrics": per_class,
        "streaming_metrics": streaming,
        "threshold_used": threshold,
        "optimal_threshold_source": threshold_source,
    }


def print_results(results: dict):
    print("\n" + "=" * 60)
    print(f"  평가 결과 (threshold={results['threshold_used']})")
    print("=" * 60)
    b = results["basic_metrics"]
    print(f"  Subset Accuracy: {b['subset_accuracy']:.4f}")
    print(f"  Micro F1:        {b['micro_f1']:.4f}")
    print(f"  Macro F1:        {b['macro_f1']:.4f}")
    print(f"  Weighted F1:     {b.get('weighted_f1', 0.0):.4f}")
    print(f"  Hamming Loss:    {b['hamming_loss']:.4f}")

    bo = results["basic_metrics_optimal_threshold"]
    print(f"\n  [최적 threshold 적용 — {results.get('optimal_threshold_source')}]")
    print(f"  Subset Accuracy: {bo['subset_accuracy']:.4f}")
    print(f"  Micro F1:        {bo['micro_f1']:.4f}")

    print(f"\n{'─'*60}")
    print(f"  {'클래스':<15} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Support':>8} {'Opt_t':>6}")
    print(f"  {'─'*55}")
    opt_t = results["optimal_thresholds"]
    for name, m in results["per_class_metrics"].items():
        print(
            f"  {name:<15} {m['f1']:>6.3f} {m['precision']:>6.3f} "
            f"{m['recall']:>6.3f} {m['support']:>8} {opt_t.get(name, 0.5):>6.2f}"
        )

    s = results.get("streaming_metrics", {})
    if s:
        print(f"\n{'─'*60}")
        print(f"  스트리밍 정량 지표")
        print(f"  Convergence Rate:   {s['convergence_rate']:.4f}  (높을수록 좋음)")
        print(f"  First Correct Step: {s['first_correct_step']:.2f}  (낮을수록 좋음)")
        print(f"  Oscillation Count:  {s['oscillation_count']:.2f}  (낮을수록 좋음)")
        print(f"  분석 샘플 수:       {s['num_streaming_samples']}")

    print("=" * 60)


# ── 메인 실행 ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mamba Belief 모델 평가")
    parser.add_argument(
        "--experiment", type=str, default="mamba_frozen",
        choices=list(ABLATION_CONFIGS.keys()),
    )
    parser.add_argument("--split", type=str, default="test", choices=["dev", "test"])
    parser.add_argument("--threshold", type=float, default=EVAL_CONFIG["THRESHOLD"])
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    print(f"\n[평가] 실험: {args.experiment}  Split: {args.split}")

    loaders = create_dataloaders(
        DATA_DIR,
        batch_size=TRAIN_CONFIG["BATCH_SIZE"],
        num_workers=args.num_workers,
    )
    if args.split not in loaders:
        raise FileNotFoundError(f"{args.split}.json 없음")

    ablation_config = ABLATION_CONFIGS.get(args.experiment, {})
    model = build_mamba_model(ablation_config)

    ckpt_path, model_run_id = resolve_checkpoint_path(args.experiment, args.run_id)
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
        print(f"[모델] 체크포인트 로드: {ckpt_path}  (run: {model_run_id})")
    else:
        print(f"[경고] 체크포인트 없음: {ckpt_path}. 초기 가중치로 평가.")
        model_run_id = "init"

    model = model.to(DEVICE)

    threshold_search_loader = None
    if args.split == "test":
        if "dev" not in loaders:
            raise FileNotFoundError("test 평가에는 dev.json 필요")
        threshold_search_loader = loaders["dev"]

    results = evaluate_model(
        model, loaders[args.split],
        threshold=args.threshold,
        threshold_search_loader=threshold_search_loader,
    )
    print_results(results)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    result_path = build_eval_log_path(args.experiment, args.split, model_run_id)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[저장] {result_path}")

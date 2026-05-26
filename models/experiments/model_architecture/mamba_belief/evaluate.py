"""
Mamba Belief 모델 평가 모듈

SNS 한국어 대화 20-class single-label 분류 지표:
  1. 기본 지표: Accuracy, Micro F1, Macro F1, Weighted F1
  2. 클래스별: F1, Precision, Recall, Support
  3. 스트리밍 정량 지표:
     - Convergence Rate: 각 스텝에서 정답 클래스 확률 증가 비율
     - First Correct Step: 처음 argmax == 정답이 되는 스텝
     - Oscillation Count: 예측이 정답↔오답을 오가는 횟수
     - Mean Step Entropy: 스텝별 Shannon Entropy 평균
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.amp import autocast

from config import (
    ABLATION_CONFIGS, CHECKPOINT_DIR, DATA_DIR, DEVICE,
    EVAL_CONFIG, GPU_CONFIG, LOG_DIR, SNS_CONFIG, TRAIN_CONFIG,
)
from dataset import create_dataloaders
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


# ── 기본 지표 ──────────────────────────────────────────────

def compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict:
    labels_idx = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_idx, zero_division=0
    )
    return {
        name: {
            "f1": float(f1[i]),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(class_names)
    }


# ── 스트리밍 정량 지표 ─────────────────────────────────────

def compute_entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon Entropy: H(p) = -sum(p * log(p)), shape (N,)"""
    eps = 1e-12
    return -np.sum(probs * np.log(probs + eps), axis=-1)


def compute_streaming_metrics(
    step_probs: list[np.ndarray],  # list of (N, C) softmax probs, one per step
    labels: np.ndarray,            # (N,) integer true class
    num_segments: np.ndarray,      # (N,) integer actual segment count
) -> dict:
    """
    스트리밍 Belief 업데이트 정량 지표 계산.

    Returns:
        convergence_rate:    각 스텝에서 p[y*]가 전 스텝보다 높아진 비율 (높을수록 좋음)
        first_correct_step:  argmax == y* 가 되는 첫 스텝 (낮을수록 좋음)
        oscillation_count:   예측이 정답↔오답을 오가는 횟수 (낮을수록 좋음)
        mean_step_entropy:   스텝별 평균 Shannon Entropy (낮을수록 확신)
        early_stop_rate:     entropy < threshold 에서 조기 종료 가능 비율
        num_streaming_samples: 분석된 샘플 수 (n_segs > 1)
    """
    entropy_threshold = EVAL_CONFIG["ENTROPY_THRESHOLD"]
    convergence_list, first_correct_list, oscillation_list = [], [], []
    step_entropy_sums: list[float] = []
    step_entropy_counts: list[int] = []
    early_stop_count = 0
    total_multi = 0

    for n in range(labels.shape[0]):
        n_segs = int(num_segments[n])
        if n_segs <= 1:
            continue
        total_multi += 1

        true_label = int(labels[n])
        step_prob_seq = []
        step_pred_seq = []
        step_ent_seq = []

        for t in range(n_segs):
            if t >= len(step_probs) or n >= step_probs[t].shape[0]:
                break
            p = step_probs[t][n]                      # (C,)
            step_prob_seq.append(p)
            step_pred_seq.append(int(np.argmax(p)))
            step_ent_seq.append(float(compute_entropy(p[np.newaxis])[0]))

        if len(step_prob_seq) < 2:
            continue

        # 엔트로피 집계 (스텝별 전체 평균에 기여)
        for t, ent in enumerate(step_ent_seq):
            if t >= len(step_entropy_sums):
                step_entropy_sums.extend([0.0] * (t + 1 - len(step_entropy_sums)))
                step_entropy_counts.extend([0] * (t + 1 - len(step_entropy_counts)))
            step_entropy_sums[t] += ent
            step_entropy_counts[t] += 1

        # Convergence rate: p[y*] 증가 비율
        increases = sum(
            1 for t in range(1, len(step_prob_seq))
            if step_prob_seq[t][true_label] >= step_prob_seq[t - 1][true_label]
        )
        convergence_list.append(increases / (len(step_prob_seq) - 1))

        # First correct step
        try:
            first_correct = next(
                t for t, pred in enumerate(step_pred_seq) if pred == true_label
            )
        except StopIteration:
            first_correct = n_segs
        first_correct_list.append(first_correct)

        # Oscillation count
        is_correct = [p == true_label for p in step_pred_seq]
        osc = sum(
            1 for t in range(1, len(is_correct))
            if is_correct[t - 1] != is_correct[t]
        )
        oscillation_list.append(osc)

        # Early stopping 가능 여부 (정규화 엔트로피 기준)
        max_entropy = float(np.log(SNS_CONFIG["NUM_LABELS"]))
        if any(ent / max_entropy < entropy_threshold for ent in step_ent_seq):
            early_stop_count += 1

    mean_step_entropy = [
        s / c if c > 0 else 0.0
        for s, c in zip(step_entropy_sums, step_entropy_counts)
    ]

    return {
        "convergence_rate": float(np.mean(convergence_list)) if convergence_list else 0.0,
        "first_correct_step": float(np.mean(first_correct_list)) if first_correct_list else 0.0,
        "oscillation_count": float(np.mean(oscillation_list)) if oscillation_list else 0.0,
        "mean_step_entropy": [round(e, 4) for e in mean_step_entropy],
        "early_stop_rate": float(early_stop_count / total_multi) if total_multi > 0 else 0.0,
        "num_streaming_samples": total_multi,
    }


# ── 예측 수집 ──────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader,
    compute_streaming: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """
    Returns:
        y_true:     (N,) integer true labels
        y_pred:     (N,) integer predicted labels
        num_segs:   (N,) integer segment counts
        step_probs: list[(N, C)] softmax probs per step t
    """
    model.eval()
    amp_enabled = GPU_CONFIG["ENABLED"] and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if GPU_CONFIG["DTYPE"] == "bf16" else torch.float16
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    all_true, all_pred, all_num_segs = [], [], []
    all_step_probs: list[list[np.ndarray]] = []
    past_batch_sizes: list[int] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        segment_mask = batch["segment_mask"].to(DEVICE)
        labels = batch["labels"]
        num_segments = batch["num_segments"]

        with autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask,
                segment_mask=segment_mask, num_segments=num_segments.to(DEVICE),
            )

        preds = outputs["logits"].float().argmax(dim=-1).cpu().numpy()  # (B,)
        all_true.append(labels.numpy())
        all_pred.append(preds)
        all_num_segs.append(num_segments.numpy())

        if compute_streaming and "all_logits" in outputs:
            batch_size = preds.shape[0]
            num_classes = outputs["logits"].shape[1]
            step_probs_batch = [
                F.softmax(lt.float(), dim=-1).cpu().numpy()
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

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    num_segs_arr = np.concatenate(all_num_segs, axis=0)
    step_probs = [np.concatenate(t_list, axis=0) for t_list in all_step_probs]

    return y_true, y_pred, num_segs_arr, step_probs


# ── 전체 평가 파이프라인 ───────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader,
    compute_streaming: bool = True,
) -> dict:
    y_true, y_pred, num_segs_arr, step_probs = collect_predictions(
        model, loader, compute_streaming=compute_streaming,
    )

    basic = compute_basic_metrics(y_true, y_pred)
    class_names = SNS_CONFIG["LABELS"]
    per_class = compute_per_class_metrics(y_true, y_pred, class_names)

    labels_idx = list(range(SNS_CONFIG["NUM_LABELS"]))
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)

    streaming: dict = {}
    if compute_streaming and step_probs:
        streaming = compute_streaming_metrics(step_probs, y_true, num_segs_arr)

    return {
        "basic_metrics": basic,
        "per_class_metrics": per_class,
        "confusion_matrix": cm.tolist(),
        "streaming_metrics": streaming,
        "num_samples": int(len(y_true)),
        "num_classes": SNS_CONFIG["NUM_LABELS"],
        "labels": class_names,
    }


def print_results(results: dict) -> None:
    print("\n" + "=" * 60)
    print(f"  평가 결과 ({results['num_samples']}개 샘플)")
    print("=" * 60)
    b = results["basic_metrics"]
    print(f"  Accuracy:    {b['accuracy']:.4f}")
    print(f"  Micro F1:    {b['micro_f1']:.4f}")
    print(f"  Macro F1:    {b['macro_f1']:.4f}")
    print(f"  Weighted F1: {b['weighted_f1']:.4f}")

    print(f"\n{'-'*60}")
    print(f"  {'클래스':<22} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Support':>8}")
    print(f"  {'-'*60}")
    for name, m in results["per_class_metrics"].items():
        print(
            f"  {name:<22} {m['f1']:>6.3f} {m['precision']:>6.3f} "
            f"{m['recall']:>6.3f} {m['support']:>8}"
        )

    s = results.get("streaming_metrics", {})
    if s:
        print(f"\n{'-'*60}")
        print("  스트리밍 정량 지표")
        print(f"  Convergence Rate:  {s['convergence_rate']:.4f}  (높을수록 좋음)")
        print(f"  First Correct Step:{s['first_correct_step']:.2f}  (낮을수록 좋음)")
        print(f"  Oscillation Count: {s['oscillation_count']:.2f}  (낮을수록 좋음)")
        print(f"  Early Stop Rate:   {s['early_stop_rate']:.4f}  (높을수록 좋음)")
        print(f"  분석 샘플 수:      {s['num_streaming_samples']}")
        entropy_per_step = s.get("mean_step_entropy", [])
        if entropy_per_step:
            shown = entropy_per_step[:5]
            print(f"  Step Entropy[0..{len(shown)-1}]: {shown}")

    print("=" * 60)


# ── 메인 실행 ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mamba Belief 모델 평가")
    parser.add_argument(
        "--experiment", type=str, default="mamba_frozen",
        choices=list(ABLATION_CONFIGS.keys()),
    )
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    print(f"\n[평가] 실험: {args.experiment}  Split: {args.split}")

    loaders = create_dataloaders(
        DATA_DIR,
        batch_size=TRAIN_CONFIG["BATCH_SIZE"],
        num_workers=args.num_workers,
        splits=(args.split,),
    )
    if args.split not in loaders:
        raise FileNotFoundError(f"{args.split}.json 없음. data_preprocessing.py 먼저 실행하세요.")

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

    results = evaluate_model(model, loaders[args.split])
    print_results(results)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    result_path = build_eval_log_path(args.experiment, args.split, model_run_id)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[저장] {result_path}")

"""
Compressive Memory 모델 평가 모듈

평가 지표:
  1. 기본 지표: Subset Accuracy, Micro F1, Macro F1, Hamming Loss
  2. 클래스별 분석: F1, Precision, Recall (특히 희귀 클래스)
  3. Threshold 최적화: 클래스별 최적 threshold 탐색
  4. 스트리밍 정량 지표: Convergence Rate, First Correct Step, Oscillation Count
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)

from config import (
    DEVICE, DATA_DIR, CHECKPOINT_DIR, LOG_DIR,
    MELD_CONFIG, EVAL_CONFIG, ABLATION_CONFIGS, GPU_CONFIG,
)
from dataset import create_dataloaders
from model import build_compressive_memory_model


# ── 기본 평가 지표 ─────────────────────────────────────────

def resolve_checkpoint_path(experiment_name: str, run_id: str | None = None) -> tuple[Path, str]:
    """
    평가할 체크포인트 경로를 결정한다.

    run_id가 지정되면 해당 run의 체크포인트를 사용하고,
    없으면 최신 메타데이터 파일을 우선 사용한다.
    """
    if run_id is not None:
        return CHECKPOINT_DIR / f"{experiment_name}_{run_id}_best.pt", run_id

    latest_meta_path = CHECKPOINT_DIR / f"{experiment_name}_latest.json"
    if latest_meta_path.exists():
        with open(latest_meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return Path(metadata["checkpoint_path"]), metadata.get("run_id", "latest")

    legacy_path = CHECKPOINT_DIR / f"{experiment_name}_best.pt"
    return legacy_path, "legacy"


def build_eval_log_path(experiment_name: str, split: str, model_run_id: str) -> Path:
    """평가 결과를 run별로 새 파일에 저장한다."""
    eval_run_id = time.strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"{experiment_name}_{model_run_id}_{split}_{eval_run_id}_eval.json"

def compute_basic_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    기본 multi-label 평가 지표를 계산한다.

    Args:
        y_true: (N, C) 정답 multi-hot (0/1)
        y_pred: (N, C) 예측 multi-hot (0/1)

    Returns:
        dict: 각 지표의 값
    """
    metrics = {
        # Subset Accuracy: 10개 레이블 모두 정확히 맞춘 비율 (가장 엄격)
        "subset_accuracy": accuracy_score(y_true, y_pred),

        # Micro F1: 전체 TP/FP/FN 합산 (클래스 불균형에 강함)
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),

        # Macro F1: 클래스별 F1 단순 평균 (희귀 클래스 반영)
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),

        # Hamming Loss: 잘못 예측한 레이블 비율 (낮을수록 좋음)
        "hamming_loss": hamming_loss(y_true, y_pred),
    }

    return metrics


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict:
    """
    클래스별 F1, Precision, Recall을 계산한다.

    Args:
        y_true: (N, C) 정답 multi-hot
        y_pred: (N, C) 예측 multi-hot
        class_names: 클래스 이름 리스트

    Returns:
        dict: 클래스별 {"f1", "precision", "recall"}
    """
    per_class = {}

    for i, name in enumerate(class_names):
        per_class[name] = {
            "f1": f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "precision": precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "recall": recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "support": int(y_true[:, i].sum()),  # 양성 샘플 수
        }

    return per_class


# ── Threshold 최적화 ───────────────────────────────────────

def find_optimal_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    search_range: tuple = (0.2, 0.8, 0.05),
) -> list[float]:
    """
    클래스별 최적 threshold를 탐색한다.
    각 클래스에서 F1을 최대화하는 threshold를 찾는다.

    Args:
        y_true: (N, C) 정답 multi-hot
        y_probs: (N, C) 예측 확률 (Sigmoid 출력)
        search_range: (start, end, step) threshold 탐색 범위

    Returns:
        list[float]: 클래스별 최적 threshold
    """
    start, end, step = search_range
    thresholds_to_try = np.arange(start, end + step, step)
    num_classes = y_true.shape[1]

    best_thresholds = []

    for i in range(num_classes):
        best_f1 = 0.0
        best_t = 0.5  # 기본값

        for t in thresholds_to_try:
            pred = (y_probs[:, i] > t).astype(int)
            f1 = f1_score(y_true[:, i], pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

        best_thresholds.append(round(best_t, 2))

    return best_thresholds


# ── 스트리밍 정량 지표 ─────────────────────────────────────

def compute_streaming_metrics(
    all_step_probs: list[np.ndarray],
    labels: np.ndarray,
    num_segments: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    스트리밍 동작의 정량 지표를 계산한다.

    Args:
        all_step_probs: 각 타임스텝의 확률값 리스트 [(N, C), ...]
        labels: (N, C) 정답 multi-hot
        num_segments: (N,) 각 샘플의 세그먼트 수
        threshold: 이진화 임계값

    Returns:
        dict: {convergence_rate, first_correct_step, oscillation_count}
    """
    N = labels.shape[0]
    max_T = len(all_step_probs)

    convergence_counts = []    # 확신도가 정답 방향으로 단조 증가하는 비율
    first_correct_steps = []   # 최초 정답 예측 세그먼트 인덱스
    oscillation_counts = []    # 정답↔오답 전환 횟수

    for n in range(N):
        n_segs = int(num_segments[n])
        label = labels[n]  # (C,)

        if n_segs <= 1:
            continue  # 세그먼트 1개인 씬은 스트리밍 분석 불가

        # 각 타임스텝의 예측
        step_preds = []
        step_probs_sample = []
        for t in range(n_segs):
            if t < max_T and n < all_step_probs[t].shape[0]:
                probs_t = all_step_probs[t][n]  # (C,)
                pred_t = (probs_t > threshold).astype(int)
                step_preds.append(pred_t)
                step_probs_sample.append(probs_t)
            else:
                break

        if len(step_preds) < 2:
            continue

        # Convergence Rate: 정답 레이블에 대한 확률이 단조 증가하는 비율
        # 정답이 1인 클래스의 확률이 이전 스텝보다 증가하는 비율
        monotone_increases = 0
        total_comparisons = 0
        positive_classes = np.where(label == 1)[0]

        if len(positive_classes) > 0:
            for t in range(1, len(step_probs_sample)):
                for c in positive_classes:
                    if step_probs_sample[t][c] >= step_probs_sample[t - 1][c]:
                        monotone_increases += 1
                    total_comparisons += 1

            if total_comparisons > 0:
                convergence_counts.append(monotone_increases / total_comparisons)

        # First Correct Step: 최초로 정답 예측이 나오는 세그먼트 인덱스
        found_first = False
        for t, pred in enumerate(step_preds):
            if np.array_equal(pred, label):
                first_correct_steps.append(t)
                found_first = True
                break
        if not found_first:
            first_correct_steps.append(n_segs)  # 끝까지 정답 못 맞춤

        # Oscillation Count: 정답↔오답 전환 횟수
        oscillations = 0
        for t in range(1, len(step_preds)):
            prev_correct = np.array_equal(step_preds[t - 1], label)
            curr_correct = np.array_equal(step_preds[t], label)
            if prev_correct != curr_correct:
                oscillations += 1
        oscillation_counts.append(oscillations)

    metrics = {
        # 확신도가 정답 방향으로 단조 증가하는 비율 (높을수록 좋음)
        "convergence_rate": float(np.mean(convergence_counts)) if convergence_counts else 0.0,

        # 최초 정답 예측 세그먼트 인덱스 평균 (낮을수록 좋음 = 빠른 판단)
        "first_correct_step": float(np.mean(first_correct_steps)) if first_correct_steps else 0.0,

        # 정답↔오답 전환 횟수 평균 (낮을수록 좋음 = 안정적)
        "oscillation_count": float(np.mean(oscillation_counts)) if oscillation_counts else 0.0,

        # 분석 대상 샘플 수 (세그먼트 ≥ 2인 씬)
        "num_streaming_samples": len(oscillation_counts),
    }

    return metrics


# ── 전체 평가 파이프라인 ───────────────────────────────────

@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader,
    compute_streaming: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """
    모델 출력으로부터 평가에 필요한 정답/확률/스트리밍 로그를 수집한다.
    """
    model.eval()

    all_labels = []       # 정답 레이블
    all_probs = []        # 예측 확률 (Sigmoid 출력)
    all_num_segments = [] # 세그먼트 수
    all_step_probs = []   # 각 타임스텝별 [배치1, 배치2, ...] 확률 배열 리스트
    past_batch_sizes = []

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        segment_mask = batch["segment_mask"].to(DEVICE)
        labels = batch["labels"]
        num_segments = batch["num_segments"]

        # AMP autocast 적용 (추론 속도 향상)
        amp_enabled = GPU_CONFIG["ENABLED"] and torch.cuda.is_available()
        amp_dtype = torch.bfloat16 if GPU_CONFIG["DTYPE"] == "bf16" else torch.float16

        with autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                segment_mask=segment_mask,
                num_segments=num_segments.to(DEVICE),
            )

        # 최종 logits → 확률 변환 (float32로 변환 후 sigmoid)
        probs = torch.sigmoid(outputs["logits"].float()).cpu().numpy()
        all_labels.append(labels.numpy())
        all_probs.append(probs)
        all_num_segments.append(num_segments.numpy())

        # 스트리밍 분석용: 타임스텝별 확률을 배치 정렬을 유지한 채로 저장
        if compute_streaming and "all_logits" in outputs:
            batch_size = probs.shape[0]
            num_classes = probs.shape[1]
            step_probs_this_batch = [
                torch.sigmoid(logits_t.float()).cpu().numpy()
                for logits_t in outputs["all_logits"]
            ]

            prev_max_steps = len(all_step_probs)
            curr_steps = len(step_probs_this_batch)

            # 현재 배치가 더 길면, 이전 배치들 크기에 맞춰 새 타임스텝 슬롯을 backfill
            if curr_steps > prev_max_steps:
                for _ in range(prev_max_steps, curr_steps):
                    all_step_probs.append(
                        [np.zeros((b, num_classes), dtype=np.float32) for b in past_batch_sizes]
                    )

            # 모든 타임스텝에 대해 현재 배치 값을 append (없는 타임스텝은 0으로 패딩)
            for t in range(len(all_step_probs)):
                if t < curr_steps:
                    all_step_probs[t].append(step_probs_this_batch[t])
                else:
                    all_step_probs[t].append(np.zeros((batch_size, num_classes), dtype=np.float32))

            past_batch_sizes.append(batch_size)

    y_true = np.concatenate(all_labels, axis=0)
    y_probs = np.concatenate(all_probs, axis=0)
    num_segments_arr = np.concatenate(all_num_segments, axis=0)

    step_probs_concat = []
    if compute_streaming and all_step_probs:
        for t in range(len(all_step_probs)):
            step_probs_concat.append(np.concatenate(all_step_probs[t], axis=0))

    return y_true, y_probs, num_segments_arr, step_probs_concat


def apply_thresholds(y_probs: np.ndarray, thresholds: float | list[float]) -> np.ndarray:
    """단일 threshold 또는 클래스별 threshold로 multi-hot 예측을 생성한다."""
    if isinstance(thresholds, (int, float)):
        return (y_probs > float(thresholds)).astype(int)

    threshold_arr = np.asarray(thresholds, dtype=float).reshape(1, -1)
    return (y_probs > threshold_arr).astype(int)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader,
    threshold: float = 0.5,
    threshold_search_loader=None,
    compute_streaming: bool = True,
) -> dict:
    """
    모델을 평가하여 모든 지표를 계산한다.

    Args:
        model: 학습된 모델
        loader: 평가용 DataLoader
        threshold: multi-hot 이진화 기본 임계값
        threshold_search_loader: 클래스별 threshold를 탐색할 별도 loader
        compute_streaming: 스트리밍 지표 계산 여부

    Returns:
        dict: 모든 평가 지표
    """
    y_true, y_probs, num_segments_arr, step_probs_concat = collect_predictions(
        model,
        loader,
        compute_streaming=compute_streaming,
    )

    # 고정 threshold로 이진화
    y_pred = apply_thresholds(y_probs, threshold)

    # 기본 지표
    basic = compute_basic_metrics(y_true, y_pred)

    # weighted_f1 추가 (sample 수 비례)
    basic["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # 클래스별 지표
    class_names = MELD_CONFIG["EMOTION_LABELS"] + MELD_CONFIG["SENTIMENT_LABELS"]
    per_class = compute_per_class_metrics(y_true, y_pred, class_names)

    # Threshold 최적화는 dev에서 수행하고, test에는 dev에서 찾은 threshold를 적용한다.
    threshold_source = "current_split"
    search_y_true = y_true
    search_y_probs = y_probs
    if threshold_search_loader is not None:
        search_y_true, search_y_probs, _, _ = collect_predictions(
            model,
            threshold_search_loader,
            compute_streaming=False,
        )
        threshold_source = "dev"

    optimal_thresholds = find_optimal_thresholds(
        search_y_true,
        search_y_probs,
        search_range=EVAL_CONFIG["THRESHOLD_SEARCH_RANGE"],
    )
    y_pred_optimal = apply_thresholds(y_probs, optimal_thresholds)
    basic_optimal = compute_basic_metrics(y_true, y_pred_optimal)

    # 스트리밍 지표
    streaming = {}
    if compute_streaming and step_probs_concat:
        streaming = compute_streaming_metrics(
            step_probs_concat, y_true, num_segments_arr, threshold,
        )

    results = {
        "basic_metrics": basic,
        "basic_metrics_optimal_threshold": basic_optimal,
        "optimal_thresholds": {
            name: t for name, t in zip(class_names, optimal_thresholds)
        },
        "per_class_metrics": per_class,
        "streaming_metrics": streaming,
        "threshold_used": threshold,
        "optimal_threshold_source": threshold_source,
    }

    return results


def print_results(results: dict):
    """평가 결과를 보기 좋게 출력한다."""
    print("\n" + "=" * 60)
    print(f"  평가 결과 (threshold={results['threshold_used']})")
    print("=" * 60)

    basic = results["basic_metrics"]
    print(f"  Subset Accuracy:  {basic['subset_accuracy']:.4f}")
    print(f"  Micro F1:         {basic['micro_f1']:.4f}")
    print(f"  Macro F1:         {basic['macro_f1']:.4f}")
    print(f"  Weighted F1:      {basic.get('weighted_f1', 0.0):.4f}")
    print(f"  Hamming Loss:     {basic['hamming_loss']:.4f}")

    # 최적 threshold 결과
    basic_opt = results["basic_metrics_optimal_threshold"]
    print(f"\n  [최적 threshold 적용]")
    print(f"  threshold source:  {results.get('optimal_threshold_source', 'current_split')}")
    print(f"  Subset Accuracy:  {basic_opt['subset_accuracy']:.4f}")
    print(f"  Micro F1:         {basic_opt['micro_f1']:.4f}")

    # 클래스별 지표
    print(f"\n{'─'*60}")
    print(f"  {'클래스':<15} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Support':>8} {'Opt_t':>6}")
    print(f"  {'─'*55}")

    per_class = results["per_class_metrics"]
    opt_t = results["optimal_thresholds"]
    for name, m in per_class.items():
        t = opt_t.get(name, 0.5)
        print(f"  {name:<15} {m['f1']:>6.3f} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['support']:>8} {t:>6.2f}")

    # 스트리밍 지표
    streaming = results.get("streaming_metrics", {})
    if streaming:
        print(f"\n{'─'*60}")
        print(f"  스트리밍 정량 지표")
        print(f"  Convergence Rate:    {streaming['convergence_rate']:.4f}  (높을수록 좋음)")
        print(f"  First Correct Step:  {streaming['first_correct_step']:.2f}  (낮을수록 좋음)")
        print(f"  Oscillation Count:   {streaming['oscillation_count']:.2f}  (낮을수록 좋음)")
        print(f"  분석 대상 샘플 수:   {streaming['num_streaming_samples']}")

    print("=" * 60)


# ── 메인 실행 ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compressive Memory 모델 평가")
    parser.add_argument(
        "--experiment", type=str, default="full",
        choices=list(ABLATION_CONFIGS.keys()),
        help="평가할 실험명",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["dev", "test"],
        help="평가할 데이터 split",
    )
    parser.add_argument(
        "--threshold", type=float, default=EVAL_CONFIG["THRESHOLD"],
        help="multi-hot 이진화 임계값",
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="특정 학습 run_id 평가 (없으면 최신 run 사용)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="DataLoader worker 수 오버라이드 (문제 발생 시 0 권장)",
    )
    args = parser.parse_args()

    print(f"\n[평가] 실험: {args.experiment}, Split: {args.split}")

    # DataLoader 생성
    from config import TRAIN_CONFIG
    loaders = create_dataloaders(
        DATA_DIR,
        batch_size=TRAIN_CONFIG["BATCH_SIZE"],
        num_workers=args.num_workers,
    )
    if args.split not in loaders:
        raise FileNotFoundError(f"{args.split}.json이 없습니다.")

    # 모델 로드
    ablation_config = ABLATION_CONFIGS.get(args.experiment, {})
    model = build_compressive_memory_model(ablation_config)

    ckpt_path, model_run_id = resolve_checkpoint_path(args.experiment, args.run_id)
    if ckpt_path.exists():
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        print(f"[모델] 체크포인트 로드: {ckpt_path}")
        print(f"[모델] Run ID: {model_run_id}")
    else:
        print(f"[경고] 체크포인트 없음: {ckpt_path}. 초기 가중치로 평가합니다.")
        model_run_id = "init"

    model = model.to(DEVICE)

    threshold_search_loader = None
    if args.split == "test":
        if "dev" not in loaders:
            raise FileNotFoundError("test 평가에는 dev.json이 필요합니다. dev split을 먼저 준비하세요.")
        threshold_search_loader = loaders["dev"]

    # 평가 실행
    results = evaluate_model(
        model,
        loaders[args.split],
        threshold=args.threshold,
        threshold_search_loader=threshold_search_loader,
    )

    # 결과 출력
    print_results(results)

    # 결과 저장
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    result_path = build_eval_log_path(args.experiment, args.split, model_run_id)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[저장] {result_path}")

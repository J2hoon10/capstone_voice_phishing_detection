"""
Streaming Belief v5 평가 스크립트.
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
    ABLATION_CONFIGS,
    CHECKPOINT_DIR,
    DATA_CONFIG,
    DATA_DIR,
    DEVICE,
    EVAL_CONFIG,
    GPU_CONFIG,
    LOG_DIR,
    TRAIN_CONFIG,
)
from dataset import create_dataloaders
from model import build_streaming_belief_model


def resolve_checkpoint_path(experiment_name: str, run_id: str | None = None) -> tuple[Path, str]:
    if run_id is not None:
        return CHECKPOINT_DIR / f"{experiment_name}_{run_id}_best.pt", run_id
    latest_meta = CHECKPOINT_DIR / f"{experiment_name}_latest.json"
    if latest_meta.exists():
        with latest_meta.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        return Path(meta["checkpoint_path"]), meta.get("run_id", "latest")
    return CHECKPOINT_DIR / f"{experiment_name}_best.pt", "legacy"


def build_eval_log_path(experiment_name: str, split: str, model_run_id: str) -> Path:
    eval_run_id = time.strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"{experiment_name}_{model_run_id}_{split}_{eval_run_id}_eval.json"


def compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
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


def compute_entropy(probs: np.ndarray) -> np.ndarray:
    eps = 1e-12
    return -np.sum(probs * np.log(probs + eps), axis=-1)


def compute_streaming_metrics(
    step_probs: list[np.ndarray],
    labels: np.ndarray,
    num_segments: np.ndarray,
) -> dict:
    entropy_threshold = EVAL_CONFIG["ENTROPY_THRESHOLD"]
    convergence_list, first_correct_list, oscillation_list = [], [], []
    step_entropy_sums: list[float] = []
    step_entropy_counts: list[int] = []
    early_stop_count, total_multi = 0, 0

    for n in range(labels.shape[0]):
        n_segs = int(num_segments[n])
        if n_segs <= 1:
            continue
        total_multi += 1
        true_label = int(labels[n])
        prob_seq, pred_seq, ent_seq = [], [], []

        for t in range(n_segs):
            if t >= len(step_probs):
                break
            p = step_probs[t][n]
            prob_seq.append(p)
            pred_seq.append(int(np.argmax(p)))
            ent_seq.append(float(compute_entropy(p[np.newaxis])[0]))

        if len(prob_seq) < 2:
            continue

        for t, ent in enumerate(ent_seq):
            if t >= len(step_entropy_sums):
                step_entropy_sums.extend([0.0] * (t + 1 - len(step_entropy_sums)))
                step_entropy_counts.extend([0] * (t + 1 - len(step_entropy_counts)))
            step_entropy_sums[t] += ent
            step_entropy_counts[t] += 1

        increases = sum(
            1 for t in range(1, len(prob_seq))
            if prob_seq[t][true_label] >= prob_seq[t - 1][true_label]
        )
        convergence_list.append(increases / (len(prob_seq) - 1))

        try:
            first_correct = next(t for t, pred in enumerate(pred_seq) if pred == true_label)
        except StopIteration:
            first_correct = n_segs
        first_correct_list.append(first_correct)

        is_correct = [pred == true_label for pred in pred_seq]
        osc = sum(1 for t in range(1, len(is_correct)) if is_correct[t - 1] != is_correct[t])
        oscillation_list.append(osc)

        max_entropy = float(np.log(DATA_CONFIG["NUM_LABELS"]))
        if any(ent / max_entropy < entropy_threshold for ent in ent_seq):
            early_stop_count += 1

    mean_step_entropy = [s / c if c > 0 else 0.0 for s, c in zip(step_entropy_sums, step_entropy_counts)]
    return {
        "convergence_rate": float(np.mean(convergence_list)) if convergence_list else 0.0,
        "first_correct_step": float(np.mean(first_correct_list)) if first_correct_list else 0.0,
        "oscillation_count": float(np.mean(oscillation_list)) if oscillation_list else 0.0,
        "mean_step_entropy": [round(v, 4) for v in mean_step_entropy],
        "early_stop_rate": float(early_stop_count / total_multi) if total_multi > 0 else 0.0,
        "num_streaming_samples": int(total_multi),
    }


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader,
    compute_streaming: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    model.eval()
    amp_enabled = GPU_CONFIG["ENABLED"] and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if GPU_CONFIG["DTYPE"] == "bf16" else torch.float16
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    all_true, all_pred, all_num = [], [], []
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
                input_ids=input_ids,
                attention_mask=attention_mask,
                segment_mask=segment_mask,
                num_segments=num_segments.to(DEVICE),
            )

        preds = outputs["logits"].float().argmax(dim=-1).cpu().numpy()
        all_true.append(labels.numpy())
        all_pred.append(preds)
        all_num.append(num_segments.numpy())

        if compute_streaming and "all_logits" in outputs:
            batch_size = preds.shape[0]
            num_classes = outputs["logits"].shape[1]
            step_probs_batch = [F.softmax(lt.float(), dim=-1).cpu().numpy() for lt in outputs["all_logits"]]
            curr_steps = len(step_probs_batch)

            if curr_steps > len(all_step_probs):
                for _ in range(len(all_step_probs), curr_steps):
                    all_step_probs.append([np.zeros((b, num_classes), dtype=np.float32) for b in past_batch_sizes])

            for t in range(len(all_step_probs)):
                all_step_probs[t].append(
                    step_probs_batch[t] if t < curr_steps
                    else np.zeros((batch_size, num_classes), dtype=np.float32)
                )

            past_batch_sizes.append(batch_size)

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    num_arr = np.concatenate(all_num, axis=0)
    step_probs = [np.concatenate(x, axis=0) for x in all_step_probs]
    return y_true, y_pred, num_arr, step_probs


@torch.no_grad()
def evaluate_model(model: nn.Module, loader, compute_streaming: bool = True) -> dict:
    y_true, y_pred, num_arr, step_probs = collect_predictions(model, loader, compute_streaming)
    basic = compute_basic_metrics(y_true, y_pred)
    per_class = compute_per_class_metrics(y_true, y_pred, DATA_CONFIG["LABELS"])
    cm = confusion_matrix(y_true, y_pred, labels=list(range(DATA_CONFIG["NUM_LABELS"])))

    streaming = {}
    if compute_streaming and step_probs:
        streaming = compute_streaming_metrics(step_probs, y_true, num_arr)

    return {
        "basic_metrics": basic,
        "per_class_metrics": per_class,
        "confusion_matrix": cm.tolist(),
        "streaming_metrics": streaming,
        "num_samples": int(len(y_true)),
        "num_classes": DATA_CONFIG["NUM_LABELS"],
        "labels": DATA_CONFIG["LABELS"],
    }


def print_results(results: dict) -> None:
    print("\n" + "=" * 60)
    print(f"  Eval Results ({results['num_samples']} samples)")
    print("=" * 60)
    b = results["basic_metrics"]
    print(f"  Accuracy:    {b['accuracy']:.4f}")
    print(f"  Micro F1:    {b['micro_f1']:.4f}")
    print(f"  Macro F1:    {b['macro_f1']:.4f}")
    print(f"  Weighted F1: {b['weighted_f1']:.4f}")

    print(f"\n{'-' * 60}")
    print(f"  {'Class':<18} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Support':>8}")
    print(f"  {'-' * 60}")
    for name, m in results["per_class_metrics"].items():
        print(
            f"  {name:<18} {m['f1']:>6.3f} {m['precision']:>6.3f} "
            f"{m['recall']:>6.3f} {m['support']:>8}"
        )

    s = results.get("streaming_metrics", {})
    if s:
        print(f"\n{'-' * 60}")
        print("  Streaming Metrics")
        print(f"  Convergence Rate:   {s['convergence_rate']:.4f}")
        print(f"  First Correct Step: {s['first_correct_step']:.2f}")
        print(f"  Oscillation Count:  {s['oscillation_count']:.2f}")
        print(f"  Early Stop Rate:    {s['early_stop_rate']:.4f}")
        print(f"  Num Samples:        {s['num_streaming_samples']}")
        entropy_per_step = s.get("mean_step_entropy", [])
        if entropy_per_step:
            shown = entropy_per_step[:5]
            print(f"  Step Entropy[0..{len(shown)-1}]: {shown}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Streaming Belief v5")
    parser.add_argument(
        "--experiment",
        type=str,
        default="v5_default",
        choices=list(ABLATION_CONFIGS.keys()),
    )
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    print(f"\n[eval] experiment={args.experiment} split={args.split}")
    loaders = create_dataloaders(
        DATA_DIR,
        batch_size=TRAIN_CONFIG["BATCH_SIZE"],
        num_workers=args.num_workers,
        splits=(args.split,),
    )
    if args.split not in loaders:
        raise FileNotFoundError(f"{args.split}.json missing. run data_preprocessing.py first.")

    ablation_config = ABLATION_CONFIGS.get(args.experiment, {})
    model = build_streaming_belief_model(ablation_config)
    ckpt_path, model_run_id = resolve_checkpoint_path(args.experiment, args.run_id)

    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
        print(f"[model] loaded checkpoint: {ckpt_path} (run={model_run_id})")
    else:
        print(f"[warn] checkpoint missing: {ckpt_path}. evaluate with init weights.")
        model_run_id = "init"

    model = model.to(DEVICE)
    results = evaluate_model(model, loaders[args.split])
    print_results(results)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    result_path = build_eval_log_path(args.experiment, args.split, model_run_id)
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[save] {result_path}")

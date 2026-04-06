"""
Evaluation script for SNS dialogue-level Compressive Memory experiments.
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
    DATA_DIR,
    DEVICE,
    GPU_CONFIG,
    LOG_DIR,
    SNS_CONFIG,
    TRAIN_CONFIG,
)
from dataset import create_dataloaders
from model import build_compressive_memory_model


def resolve_checkpoint_path(experiment_name: str, run_id: str | None = None) -> tuple[Path, str]:
    if run_id is not None:
        return CHECKPOINT_DIR / f"{experiment_name}_{run_id}_best.pt", run_id

    latest_meta_path = CHECKPOINT_DIR / f"{experiment_name}_latest.json"
    if latest_meta_path.exists():
        with latest_meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        return Path(metadata["checkpoint_path"]), metadata.get("run_id", "latest")

    legacy_path = CHECKPOINT_DIR / f"{experiment_name}_best.pt"
    return legacy_path, "legacy"


def build_eval_log_path(experiment_name: str, split: str, model_run_id: str) -> Path:
    eval_run_id = time.strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"{experiment_name}_{model_run_id}_{split}_{eval_run_id}_eval.json"


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    labels_idx = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels_idx,
        zero_division=0,
    )

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "f1": float(f1[i]),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "support": int(support[i]),
        }
    return per_class


def bucket_name(num_turns: int) -> str:
    if num_turns <= 30:
        return "short(<=30)"
    if num_turns <= 45:
        return "mid(31-45)"
    return "long(>=46)"


@torch.no_grad()
def collect_predictions(model: nn.Module, dataloader):
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []
    length_bucket_stats: dict[str, dict[str, int]] = {}

    amp_enabled = GPU_CONFIG["ENABLED"] and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if GPU_CONFIG["DTYPE"] == "bf16" else torch.float16
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    for batch in dataloader:
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        turn_mask = batch["turn_mask"].to(DEVICE, non_blocking=True)
        num_turns = batch["num_turns"].to(DEVICE, non_blocking=True)
        labels = batch["dialogue_labels"].to(DEVICE, non_blocking=True)

        with autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                turn_mask=turn_mask,
                num_turns=num_turns,
            )

        probs = F.softmax(output["logits"].float(), dim=-1)
        preds = probs.argmax(dim=-1)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        y_probs.extend(probs.cpu().tolist())

        for i in range(labels.shape[0]):
            n_turn = int(num_turns[i].item())
            name = bucket_name(n_turn)
            if name not in length_bucket_stats:
                length_bucket_stats[name] = {"total": 0, "correct": 0}
            length_bucket_stats[name]["total"] += 1
            if int(preds[i].item()) == int(labels[i].item()):
                length_bucket_stats[name]["correct"] += 1

    return y_true, y_pred, y_probs, length_bucket_stats


@torch.no_grad()
def evaluate_model(model: nn.Module, dataloader) -> dict:
    y_true, y_pred, y_probs, bucket_stats = collect_predictions(model, dataloader)

    class_names = SNS_CONFIG["LABELS"]
    labels_idx = list(range(SNS_CONFIG["NUM_LABELS"]))

    if len(y_true) == 0:
        per_class = {
            name: {"f1": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}
            for name in class_names
        }
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "per_class": per_class,
            "confusion_matrix": np.zeros((SNS_CONFIG["NUM_LABELS"], SNS_CONFIG["NUM_LABELS"]), dtype=int).tolist(),
            "length_bucket_accuracy": {},
            "num_samples": 0,
            "num_classes": SNS_CONFIG["NUM_LABELS"],
            "labels": class_names,
            "y_probs": y_probs,
        }

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    accuracy = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
    per_class = compute_per_class_metrics(y_true_arr, y_pred_arr, class_names)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels_idx)

    length_bucket_accuracy = {}
    for name, stats in sorted(bucket_stats.items()):
        total = stats["total"]
        correct = stats["correct"]
        length_bucket_accuracy[name] = {
            "accuracy": float(correct / total) if total > 0 else 0.0,
            "count": int(total),
        }

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "length_bucket_accuracy": length_bucket_accuracy,
        "num_samples": int(len(y_true)),
        "num_classes": SNS_CONFIG["NUM_LABELS"],
        "labels": class_names,
        "y_probs": y_probs,
    }


def print_results(results: dict) -> None:
    print("\n" + "=" * 60)
    print("  Evaluation Result")
    print("=" * 60)
    print(f"  Accuracy:     {results['accuracy']:.4f}")
    print(f"  Macro F1:     {results['macro_f1']:.4f}")
    print(f"  Weighted F1:  {results['weighted_f1']:.4f}")
    print(f"  Num Samples:  {results['num_samples']}")

    print(f"\n{'-' * 60}")
    print(f"  {'Class':<22} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Support':>8}")
    print(f"  {'-' * 60}")
    for name, m in results["per_class"].items():
        print(
            f"  {name:<22} {m['f1']:>6.3f} {m['precision']:>6.3f} "
            f"{m['recall']:>6.3f} {m['support']:>8}"
        )

    print(f"\n{'-' * 60}")
    print("  Dialogue Length Bucket Accuracy")
    for bucket, metric in results["length_bucket_accuracy"].items():
        print(f"  {bucket:<12} acc={metric['accuracy']:.4f} (n={metric['count']})")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Compressive Memory KR model")
    parser.add_argument(
        "--experiment",
        type=str,
        default="full",
        choices=list(ABLATION_CONFIGS.keys()),
        help="Experiment name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Evaluation split",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Specific run_id to evaluate (default: latest)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader worker override",
    )
    args = parser.parse_args()

    print(f"\n[eval] experiment={args.experiment}, split={args.split}")

    loaders = create_dataloaders(
        DATA_DIR,
        batch_size=TRAIN_CONFIG["BATCH_SIZE"],
        num_workers=args.num_workers,
        splits=(args.split,),
    )
    if args.split not in loaders:
        raise FileNotFoundError(f"{args.split}.json is missing.")

    ablation_config = ABLATION_CONFIGS.get(args.experiment, {})
    model = build_compressive_memory_model(ablation_config)

    ckpt_path, model_run_id = resolve_checkpoint_path(args.experiment, args.run_id)
    if ckpt_path.exists():
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        print(f"[model] loaded checkpoint: {ckpt_path}")
        print(f"[model] run_id: {model_run_id}")
    else:
        print(f"[warn] checkpoint not found: {ckpt_path}. evaluate with random init.")
        model_run_id = "init"

    model = model.to(DEVICE)
    results = evaluate_model(model, loaders[args.split])
    print_results(results)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    result_path = build_eval_log_path(args.experiment, args.split, model_run_id)
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[save] {result_path}")

"""
roberta_mamba_freeze_init / evaluate.py
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from config import CHECKPOINT_DIR, DATA_DIR, DEVICE, LABELS, LOG_DIR, NUM_LABELS, TRAIN_CONFIG
from dataset import create_dataloader
from model import build_model

EXP_NAME = "bert_mamba_freeze_init_4class"


def resolve_checkpoint_path(run_id: str | None = None) -> tuple[Path, str]:
    if run_id is not None:
        return CHECKPOINT_DIR / f"{EXP_NAME}_{run_id}_best.pt", run_id
    latest_meta = CHECKPOINT_DIR / f"{EXP_NAME}_latest.json"
    if latest_meta.exists():
        with latest_meta.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        return Path(meta["checkpoint_path"]), meta.get("run_id", "latest")
    raise FileNotFoundError(f"checkpoint meta not found: {latest_meta}")


@torch.no_grad()
def collect_predictions(model, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_true, all_pred = [], []
    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(DEVICE)
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            segment_mask=batch["segment_mask"],
            num_segments=batch["num_segments"],
        )
        preds = out["logits"].argmax(dim=-1).cpu().numpy()
        all_true.append(batch["labels"].cpu().numpy())
        all_pred.append(preds)
    return np.concatenate(all_true), np.concatenate(all_pred)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels_idx = list(range(NUM_LABELS))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_idx, zero_division=0
    )
    per_class = {
        LABELS[i]: {
            "f1": float(f1[i]),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "support": int(support[i]),
        }
        for i in range(NUM_LABELS)
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "num_samples": int(len(y_true)),
        "labels": LABELS,
    }


def print_results(results: dict, split: str, run_id: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {EXP_NAME}  split={split}  run={run_id}  ({results['num_samples']} samples)")
    print("=" * 60)
    print(f"  Accuracy:    {results['accuracy']:.4f}")
    print(f"  Macro F1:    {results['macro_f1']:.4f}")
    print(f"  Weighted F1: {results['weighted_f1']:.4f}")
    print(f"  Micro F1:    {results['micro_f1']:.4f}")
    print(f"\n  {'Class':<22} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Support':>8}")
    print(f"  {'-' * 54}")
    for name, m in results["per_class"].items():
        print(f"  {name:<22} {m['f1']:>6.3f} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['support']:>8}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Evaluate {EXP_NAME} on test/val split")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--run-id", default=None, help="특정 run_id 지정 (없으면 latest 사용)")
    args = parser.parse_args()

    ckpt_path, run_id = resolve_checkpoint_path(args.run_id)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    print(f"[eval] checkpoint: {ckpt_path}  (run={run_id})")
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    print(f"[eval] loading split={args.split} from {DATA_DIR}")

    loader = create_dataloader(DATA_DIR / f"{args.split}.csv", TRAIN_CONFIG["BATCH_SIZE"], shuffle=False)
    y_true, y_pred = collect_predictions(model, loader)
    results = compute_metrics(y_true, y_pred)
    print_results(results, args.split, run_id)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    eval_run_id = time.strftime("%Y%m%d_%H%M%S")
    save_path = LOG_DIR / f"{EXP_NAME}_{run_id}_{args.split}_{eval_run_id}_eval.json"
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[save] {save_path}")


if __name__ == "__main__":
    main()

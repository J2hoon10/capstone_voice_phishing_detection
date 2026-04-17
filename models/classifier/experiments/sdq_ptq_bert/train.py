"""
Fine-tuning script for klue/bert-base on NSMC.
Produces the FP32 baseline checkpoint for PTQ experiments.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import (
    CHECKPOINT_DIR,
    DEVICE,
    EVAL_CONFIG,
    GPU_CONFIG,
    LOG_DIR,
    TRAIN_CONFIG,
)
from dataset import create_dataloaders
from model import build_model, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_gpu() -> None:
    if not torch.cuda.is_available():
        print("[gpu] CUDA unavailable, running on CPU")
        return

    if GPU_CONFIG["TF32_MATMUL"]:
        torch.backends.cuda.matmul.allow_tf32 = True
    if GPU_CONFIG["TF32_CUDNN"]:
        torch.backends.cudnn.allow_tf32 = True
    if GPU_CONFIG["CUDNN_BENCHMARK"]:
        torch.backends.cudnn.benchmark = True

    gpu_name = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[gpu] {gpu_name}")
    print(f"[gpu] VRAM {vram_total:.1f}GB")
    print(f"[gpu] mixed precision: {GPU_CONFIG['DTYPE']}")


def get_amp_dtype() -> torch.dtype:
    return torch.bfloat16 if GPU_CONFIG["DTYPE"] == "bf16" else torch.float16


def build_scheduler(optimizer: AdamW, total_steps: int) -> LambdaLR:
    warmup_steps = int(total_steps * TRAIN_CONFIG["WARMUP_RATIO"])
    min_ratio = 0.0

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        token_type_ids = batch["token_type_ids"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            loss = outputs.loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["MAX_GRAD_NORM"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["MAX_GRAD_NORM"])
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, loader, amp_enabled: bool, amp_dtype: torch.dtype) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    y_true, y_pred = [], []
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        token_type_ids = batch["token_type_ids"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)

        with autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )

        total_loss += outputs.loss.item()
        num_batches += 1

        preds = outputs.logits.argmax(dim=-1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    val_loss = total_loss / max(num_batches, 1)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    return val_loss, metrics


def train(batch_size: int | None = None, num_workers: int | None = None):
    set_seed(TRAIN_CONFIG["SEED"])
    init_gpu()

    amp_enabled = GPU_CONFIG["ENABLED"] and torch.cuda.is_available()
    amp_dtype = get_amp_dtype()
    scaler = GradScaler() if (amp_enabled and GPU_CONFIG["DTYPE"] == "fp16") else None
    _batch_size = batch_size or TRAIN_CONFIG["BATCH_SIZE"]

    run_id = time.strftime("%Y%m%d_%H%M%S")
    print(f"\n{'=' * 60}")
    print(f"[train] SDQ-PTQ BERT baseline fine-tuning")
    print(f"  run_id: {run_id}")
    print(f"  model: klue/bert-base")
    print(f"  dataset: NSMC")
    print(f"  device: {DEVICE}")
    print(f"  mixed precision: {GPU_CONFIG['DTYPE'] if amp_enabled else 'OFF'}")
    print(f"  batch_size: {_batch_size}")
    print(f"  lr: {TRAIN_CONFIG['LR']}")
    print(f"  epochs: {TRAIN_CONFIG['EPOCHS']}")
    print(f"{'=' * 60}\n")

    # Load data — use test split as validation for PTQ baseline
    loaders = create_dataloaders(
        batch_size=_batch_size,
        num_workers=num_workers,
        splits=("train", "test"),
    )

    model = build_model().to(DEVICE)
    params = count_parameters(model)
    print(f"[model] total params: {params['total']:,}")
    print(f"[model] trainable params: {params['trainable']:,}")

    optimizer = AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG["LR"],
        weight_decay=TRAIN_CONFIG["WEIGHT_DECAY"],
    )
    total_steps = TRAIN_CONFIG["EPOCHS"] * len(loaders["train"])
    scheduler = build_scheduler(optimizer, total_steps)

    best_acc = -1.0
    best_ckpt_path = CHECKPOINT_DIR / f"baseline_{run_id}_best.pt"
    log = {
        "run_id": run_id,
        "config": {
            "model": "klue/bert-base",
            "dataset": "nsmc",
            "batch_size": _batch_size,
            "lr": TRAIN_CONFIG["LR"],
            "epochs": TRAIN_CONFIG["EPOCHS"],
            "amp": GPU_CONFIG["DTYPE"] if amp_enabled else "off",
        },
        "epochs": [],
    }

    for epoch in range(1, TRAIN_CONFIG["EPOCHS"] + 1):
        start = time.time()
        train_loss = train_one_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        val_loss, val_metrics = validate(
            model=model,
            loader=loaders["test"],
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        elapsed = time.time() - start

        acc = val_metrics["accuracy"]
        macro_f1 = val_metrics["macro_f1"]
        print(
            f"[epoch {epoch}/{TRAIN_CONFIG['EPOCHS']}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"acc={acc:.4f} macro_f1={macro_f1:.4f} "
            f"time={elapsed:.1f}s"
        )

        log["epochs"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics,
            "time": elapsed,
        })

        if acc > best_acc:
            best_acc = acc
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  [checkpoint] saved (acc={acc:.4f})")

    # Save log
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"baseline_{run_id}_train_log.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    # Save latest metadata
    meta = {
        "run_id": run_id,
        "checkpoint_path": str(best_ckpt_path),
        "train_log_path": str(log_path),
        "best_accuracy": best_acc,
    }
    meta_path = CHECKPOINT_DIR / "baseline_latest.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[done] best accuracy: {best_acc:.4f}")
    print(f"[done] checkpoint: {best_ckpt_path}")
    print(f"[done] log: {log_path}")

    return model, log


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune klue/bert-base on NSMC")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    train(batch_size=args.batch_size, num_workers=args.num_workers)

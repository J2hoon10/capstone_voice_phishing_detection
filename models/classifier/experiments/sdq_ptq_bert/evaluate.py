"""
Evaluation script for SDQ-PTQ BERT experiments.
Measures accuracy, F1, model size, inference latency, and VRAM usage.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.amp import autocast

from config import (
    CHECKPOINT_DIR,
    DATA_CONFIG,
    DEVICE,
    EVAL_CONFIG,
    GPU_CONFIG,
    LOG_DIR,
    QUANT_CONFIG,
)
from dataset import create_dataloaders
from model import build_model, get_linear_modules, load_model_from_checkpoint, model_size_bytes


def get_amp_dtype() -> torch.dtype:
    return torch.bfloat16 if GPU_CONFIG["DTYPE"] == "bf16" else torch.float16


def compute_effective_model_size(
    model: nn.Module,
    bits: float | None = None,
    skip_patterns: list[str] | None = None,
) -> dict:
    """Compute theoretical model size under quantization."""
    skip = skip_patterns or QUANT_CONFIG["SKIP_PATTERNS"]

    quantized_params = 0
    non_quantized_params = 0

    for name, param in model.named_parameters():
        n = param.numel()
        if bits is not None and isinstance(param.data, torch.Tensor):
            # Check if this layer would be quantized
            module_name = name.rsplit(".", 1)[0] if "." in name else name
            if any(pat in module_name for pat in skip):
                non_quantized_params += n
            elif "weight" in name:
                quantized_params += n
            else:
                non_quantized_params += n
        else:
            non_quantized_params += n

    if bits is not None:
        size_bytes = (quantized_params * bits / 8) + (non_quantized_params * 4)
    else:
        size_bytes = (quantized_params + non_quantized_params) * 4

    fp32_size = (quantized_params + non_quantized_params) * 4

    return {
        "size_bytes": size_bytes,
        "size_mb": size_bytes / (1024**2),
        "fp32_size_mb": fp32_size / (1024**2),
        "compression_ratio": fp32_size / max(size_bytes, 1),
        "quantized_params": quantized_params,
        "non_quantized_params": non_quantized_params,
    }


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    loader,
    num_warmup: int = 10,
    max_batches: int | None = None,
) -> dict:
    """Measure per-sample inference latency."""
    model.eval()
    amp_enabled = GPU_CONFIG["ENABLED"] and torch.cuda.is_available()
    amp_dtype = get_amp_dtype()
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    # Warmup
    warmup_count = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        token_type_ids = batch["token_type_ids"].to(DEVICE, non_blocking=True)

        with autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
            model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        warmup_count += 1
        if warmup_count >= num_warmup:
            break

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed inference
    total_samples = 0
    total_time = 0.0
    batch_count = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        token_type_ids = batch["token_type_ids"].to(DEVICE, non_blocking=True)
        bs = input_ids.shape[0]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        with autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
            model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t1 = time.perf_counter()
        total_time += (t1 - t0)
        total_samples += bs
        batch_count += 1

        if max_batches is not None and batch_count >= max_batches:
            break

    ms_per_sample = (total_time / max(total_samples, 1)) * 1000

    return {
        "total_samples": total_samples,
        "total_time_s": total_time,
        "ms_per_sample": ms_per_sample,
        "samples_per_second": total_samples / max(total_time, 1e-9),
    }


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader) -> dict:
    """Evaluate accuracy and F1 metrics."""
    model.eval()
    y_true, y_pred = [], []
    amp_enabled = GPU_CONFIG["ENABLED"] and torch.cuda.is_available()
    amp_dtype = get_amp_dtype()
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
            )

        preds = outputs.logits.argmax(dim=-1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr, y_pred_arr,
        labels=[0, 1],
        zero_division=0,
    )

    per_class = {}
    for i, name in enumerate(DATA_CONFIG["LABELS"]):
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        "per_class": per_class,
        "num_samples": len(y_true),
    }


def get_vram_usage() -> dict:
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_mb": 0.0}
    return {
        "allocated_mb": torch.cuda.memory_allocated(0) / (1024**2),
        "reserved_mb": torch.cuda.memory_reserved(0) / (1024**2),
        "peak_mb": torch.cuda.max_memory_allocated(0) / (1024**2),
    }


def full_evaluation(
    model: nn.Module,
    loader,
    config_name: str = "fp32_baseline",
    bits: float | None = None,
) -> dict:
    """Run full evaluation: accuracy + latency + size."""
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    acc_metrics = evaluate_accuracy(model, loader)
    latency_metrics = measure_latency(
        model, loader,
        num_warmup=EVAL_CONFIG["NUM_WARMUP_BATCHES"],
    )
    size_metrics = compute_effective_model_size(model, bits=bits)
    vram = get_vram_usage()

    return {
        "config_name": config_name,
        "bits": bits,
        **acc_metrics,
        "latency": latency_metrics,
        "size": size_metrics,
        "vram": vram,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def print_results(results: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Evaluation: {results['config_name']}")
    print(f"{'=' * 60}")
    print(f"  Accuracy:     {results['accuracy']:.4f}")
    print(f"  Macro F1:     {results['macro_f1']:.4f}")
    print(f"  Weighted F1:  {results['weighted_f1']:.4f}")
    print(f"  Num Samples:  {results['num_samples']}")
    print(f"\n  Latency:      {results['latency']['ms_per_sample']:.3f} ms/sample")
    print(f"  Throughput:   {results['latency']['samples_per_second']:.1f} samples/s")
    print(f"\n  Model Size:   {results['size']['size_mb']:.1f} MB")
    print(f"  FP32 Size:    {results['size']['fp32_size_mb']:.1f} MB")
    print(f"  Compression:  {results['size']['compression_ratio']:.2f}x")
    print(f"\n  Peak VRAM:    {results['vram']['peak_mb']:.1f} MB")

    print(f"\n  {'Class':<12} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}")
    print(f"  {'-' * 40}")
    for name, m in results["per_class"].items():
        print(f"  {name:<12} {m['precision']:>6.4f} {m['recall']:>6.4f} {m['f1']:>6.4f} {m['support']:>8}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate SDQ-PTQ BERT model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    # Resolve checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        meta_path = CHECKPOINT_DIR / "baseline_latest.json"
        if meta_path.exists():
            with meta_path.open("r") as f:
                meta = json.load(f)
            ckpt_path = Path(meta["checkpoint_path"])
        else:
            raise FileNotFoundError("No checkpoint found. Run train.py first.")

    print(f"[eval] checkpoint: {ckpt_path}")

    model = load_model_from_checkpoint(str(ckpt_path)).to(DEVICE)
    loaders = create_dataloaders(
        batch_size=args.batch_size or EVAL_CONFIG["BATCH_SIZE"],
        num_workers=args.num_workers,
        splits=("test",),
    )

    results = full_evaluation(model, loaders["test"], config_name="fp32_baseline")
    print_results(results)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    result_path = LOG_DIR / f"eval_fp32_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[save] {result_path}")

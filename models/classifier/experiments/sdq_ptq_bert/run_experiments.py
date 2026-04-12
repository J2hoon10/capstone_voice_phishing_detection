"""
Experiment orchestrator for SDQ-PTQ BERT experiments.

Runs all quantization configurations and collects results:
  1. FP32 baseline
  2. BF16 baseline
  3. RTN at each bit-width
  4. SDQ-LLM at each (bit-width, OSR) combination
  5. SDQ-LLM adaptive per-layer OSR
  6. Layer sensitivity analysis
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from config import (
    CHECKPOINT_DIR,
    DEVICE,
    EVAL_CONFIG,
    LOG_DIR,
    QUANT_CONFIG,
)
from dataset import create_dataloaders
from evaluate import full_evaluation, print_results
from model import build_model, load_model_from_checkpoint
from quantize import (
    compute_layer_stats,
    quantize_model,
    quantize_single_layer,
)
from quantize_baselines import quantize_model_rtn


def resolve_checkpoint() -> Path:
    """Find the best baseline checkpoint."""
    meta_path = CHECKPOINT_DIR / "baseline_latest.json"
    if meta_path.exists():
        with meta_path.open("r") as f:
            meta = json.load(f)
        return Path(meta["checkpoint_path"])
    raise FileNotFoundError("No baseline checkpoint. Run train.py first.")


def save_result(result: dict, name: str) -> Path:
    """Save a single experiment result to JSON."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = LOG_DIR / f"{name}_{ts}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return path


def run_baseline_evaluations(ckpt_path: Path, loader) -> list[dict]:
    """Run FP32 and BF16 baseline evaluations."""
    results = []

    # FP32 baseline
    print("\n" + "=" * 60)
    print("  [1/2] FP32 Baseline")
    print("=" * 60)
    model = load_model_from_checkpoint(str(ckpt_path)).to(DEVICE)
    result = full_evaluation(model, loader, config_name="fp32_baseline", bits=32)
    print_results(result)
    save_result(result, "baseline_fp32")
    results.append(result)
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # BF16 baseline (cast weights to BF16 then back)
    print("\n" + "=" * 60)
    print("  [2/2] BF16 Baseline")
    print("=" * 60)
    model = load_model_from_checkpoint(str(ckpt_path))
    model = model.to(torch.bfloat16).to(DEVICE)
    # Cast back for evaluation (simulate BF16 precision loss)
    model = model.float()
    result = full_evaluation(model, loader, config_name="bf16_baseline", bits=16)
    print_results(result)
    save_result(result, "baseline_bf16")
    results.append(result)
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


def run_rtn_experiments(ckpt_path: Path, loader) -> list[dict]:
    """Run RTN quantization at each bit-width."""
    results = []

    for bits in QUANT_CONFIG["BIT_WIDTHS"]:
        print(f"\n{'=' * 60}")
        print(f"  RTN {bits}-bit")
        print(f"{'=' * 60}")

        model = load_model_from_checkpoint(str(ckpt_path))
        q_model, q_info = quantize_model_rtn(model, bits=bits)
        q_model = q_model.to(DEVICE)

        config_name = f"rtn_{bits}bit"
        result = full_evaluation(q_model, loader, config_name=config_name, bits=bits)
        result["quant_info"] = q_info
        print_results(result)
        save_result(result, config_name)
        results.append(result)

        del q_model, model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


def run_sdq_fixed_osr_experiments(ckpt_path: Path, loader) -> list[dict]:
    """Run SDQ-LLM at each (bit-width, OSR) combination."""
    results = []

    for bits in QUANT_CONFIG["BIT_WIDTHS"]:
        osr_list = QUANT_CONFIG["OSR_MAP"].get(bits, [4])

        for osr in osr_list:
            print(f"\n{'=' * 60}")
            print(f"  SDQ {bits}-bit OSR={osr}")
            print(f"{'=' * 60}")

            model = load_model_from_checkpoint(str(ckpt_path))
            q_model, q_info = quantize_model(model, bits=bits, osr=osr)
            q_model = q_model.to(DEVICE)

            config_name = f"sdq_{bits}bit_osr{osr}"
            result = full_evaluation(q_model, loader, config_name=config_name, bits=bits)
            result["quant_info"] = {
                "aggregate_mse": q_info["aggregate_mse"],
                "aggregate_snr_db": q_info.get("aggregate_snr_db", 0),
                "num_quantized_layers": q_info["num_quantized_layers"],
                "bits": bits,
                "osr": osr,
                "adaptive": False,
            }
            print_results(result)
            save_result(result, config_name)
            results.append(result)

            del q_model, model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


def run_sdq_adaptive_experiments(ckpt_path: Path, loader) -> list[dict]:
    """Run SDQ-LLM with adaptive per-layer OSR."""
    results = []

    for target_bits in QUANT_CONFIG["ADAPTIVE_TARGETS"]:
        # Use ternary levels for fractional targets, binary for 1.0
        bits = 1 if target_bits == 1.0 else 1.58

        print(f"\n{'=' * 60}")
        print(f"  SDQ Adaptive target={target_bits} bits (levels={bits}-bit)")
        print(f"{'=' * 60}")

        model = load_model_from_checkpoint(str(ckpt_path))
        q_model, q_info = quantize_model(
            model, bits=bits, adaptive=True, target_avg_bits=target_bits,
        )
        q_model = q_model.to(DEVICE)

        config_name = f"sdq_adaptive_{target_bits}bit"
        result = full_evaluation(q_model, loader, config_name=config_name, bits=target_bits)
        result["quant_info"] = {
            "aggregate_mse": q_info["aggregate_mse"],
            "aggregate_snr_db": q_info.get("aggregate_snr_db", 0),
            "num_quantized_layers": q_info["num_quantized_layers"],
            "bits": bits,
            "target_avg_bits": target_bits,
            "adaptive": True,
            "osr_distribution": {
                name: info["osr_assigned"]
                for name, info in q_info["layers"].items()
            },
        }
        print_results(result)
        save_result(result, config_name)
        results.append(result)

        del q_model, model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


def run_sensitivity_analysis(ckpt_path: Path, loader, bits: float = 2, osr: int = 4) -> dict:
    """
    Quantize one layer at a time and measure accuracy drop.
    Uses 2-bit OSR=4 as default to detect sensitivity without catastrophic failure.
    """
    print(f"\n{'=' * 60}")
    print(f"  Layer Sensitivity Analysis ({bits}-bit, OSR={osr})")
    print(f"{'=' * 60}")

    # Get FP32 baseline accuracy first
    model = load_model_from_checkpoint(str(ckpt_path)).to(DEVICE)
    base_result = full_evaluation(model, loader, config_name="fp32_for_sensitivity")
    base_acc = base_result["accuracy"]
    print(f"  FP32 baseline accuracy: {base_acc:.4f}")
    del model

    # Get all quantizable layer names
    temp_model = build_model()
    layer_stats = compute_layer_stats(temp_model)
    del temp_model

    sensitivity = {}
    total = len(layer_stats)

    for i, layer_name in enumerate(layer_stats, 1):
        model = load_model_from_checkpoint(str(ckpt_path))
        q_model, q_info = quantize_single_layer(model, layer_name, bits=bits, osr=osr)
        q_model = q_model.to(DEVICE)

        result = full_evaluation(q_model, loader, config_name=f"sens_{layer_name}")
        acc_drop = base_acc - result["accuracy"]

        sensitivity[layer_name] = {
            "accuracy": result["accuracy"],
            "accuracy_drop": acc_drop,
            "mse": q_info["mse"],
            "snr_db": q_info["snr_db"],
            "numel": layer_stats[layer_name]["numel"],
            "shape": layer_stats[layer_name]["shape"],
        }

        print(f"  [{i:2d}/{total}] {layer_name:<50s} acc_drop={acc_drop:+.4f}")

        del q_model, model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    result = {
        "config_name": f"sensitivity_{bits}bit_osr{osr}",
        "base_accuracy": base_acc,
        "bits": bits,
        "osr": osr,
        "layers": sensitivity,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    save_result(result, f"sensitivity_{bits}bit_osr{osr}")
    return result


def run_all(
    skip_sensitivity: bool = False,
    resume_phase4: bool = False,
    batch_size: int | None = None,
    num_workers: int | None = None,
):
    """Run all experiment phases."""
    ckpt_path = resolve_checkpoint()
    print(f"[run] checkpoint: {ckpt_path}")

    _batch_size = batch_size or EVAL_CONFIG["BATCH_SIZE"]
    loaders = create_dataloaders(
        batch_size=_batch_size,
        num_workers=num_workers,
        splits=("test",),
    )
    loader = loaders["test"]

    all_results = []

    if not resume_phase4:
        # Phase 1: Baselines
        print("\n" + "#" * 60)
        print("#  Phase 1: Baseline Evaluations")
        print("#" * 60)
        all_results.extend(run_baseline_evaluations(ckpt_path, loader))

        # Phase 2: RTN
        print("\n" + "#" * 60)
        print("#  Phase 2: RTN Quantization")
        print("#" * 60)
        all_results.extend(run_rtn_experiments(ckpt_path, loader))

        # Phase 3: SDQ-LLM fixed OSR
        print("\n" + "#" * 60)
        print("#  Phase 3: SDQ-LLM Fixed OSR")
        print("#" * 60)
        all_results.extend(run_sdq_fixed_osr_experiments(ckpt_path, loader))
    else:
        print("\n[info] Skipping Phase 1~3 (Baseline, RTN, SDQ Fixed OSR) as requested.")

    # Phase 4: SDQ-LLM adaptive
    print("\n" + "#" * 60)
    print("#  Phase 4: SDQ-LLM Adaptive OSR")
    print("#" * 60)
    all_results.extend(run_sdq_adaptive_experiments(ckpt_path, loader))

    # Phase 5: Sensitivity
    if not skip_sensitivity:
        print("\n" + "#" * 60)
        print("#  Phase 5: Layer Sensitivity Analysis")
        print("#" * 60)
        sensitivity = run_sensitivity_analysis(ckpt_path, loader)
    else:
        sensitivity = None

    # Save combined results
    summary = {
        "experiments": [
            {
                "config_name": r["config_name"],
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "bits": r.get("bits"),
                "size_mb": r["size"]["size_mb"],
                "compression_ratio": r["size"]["compression_ratio"],
                "ms_per_sample": r["latency"]["ms_per_sample"],
            }
            for r in all_results
        ],
        "sensitivity": sensitivity,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    summary_path = save_result(summary, "all_results_summary")
    print(f"\n[done] Summary saved: {summary_path}")

    # Print comparison table
    print(f"\n{'=' * 90}")
    print(f"  {'Config':<30s} {'Bits':>5} {'Acc(%)':>8} {'MacF1':>7} {'Size(MB)':>10} {'CR':>6} {'ms/s':>7}")
    print(f"  {'-' * 84}")
    for r in all_results:
        bits_str = f"{r.get('bits', 32)}" if r.get("bits") else "32"
        print(
            f"  {r['config_name']:<30s} "
            f"{bits_str:>5} "
            f"{r['accuracy']*100:>8.2f} "
            f"{r['macro_f1']:>7.4f} "
            f"{r['size']['size_mb']:>10.1f} "
            f"{r['size']['compression_ratio']:>6.2f} "
            f"{r['latency']['ms_per_sample']:>7.3f}"
        )
    print(f"{'=' * 90}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run all SDQ-PTQ experiments")
    parser.add_argument("--skip-sensitivity", action="store_true",
                        help="Skip layer sensitivity analysis (saves time)")
    parser.add_argument("--resume-phase4", action="store_true",
                        help="Skip Phases 1~3 and resume from Phase 4 directly")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    run_all(
        skip_sensitivity=args.skip_sensitivity,
        resume_phase4=args.resume_phase4,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

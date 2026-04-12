"""
SDQ-LLM (Sigma-Delta Quantization) core implementation.

Adapts the Sigma-Delta Quantization method from:
  "SDQ-LLM: Sigma-Delta Quantization for 1-bit LLMs of any size"
  (arXiv 2510.03275)

for post-training quantization of BERT-family encoder models.

Core algorithm:
  1. Flatten weight matrix to 1D signal
  2. Upsample by Over-Sampling Ratio (OSR)
  3. Run 1st-order Sigma-Delta modulator (noise shaping)
  4. Downsample back to original length
  5. Reshape to original weight matrix shape
"""

from __future__ import annotations

import copy
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from config import QUANT_CONFIG


# ---------------------------------------------------------------------------
# Quantization level sets
# ---------------------------------------------------------------------------

def get_quantization_levels(bits: float, scale: float) -> np.ndarray:
    """Return the quantization level set for a given bit-width and scale."""
    if bits == 1:
        # Binary: {-s, +s}
        return np.array([-scale, scale])
    elif bits == 1.58 or abs(bits - 1.58) < 0.1:
        # Ternary: {-s, 0, +s}
        return np.array([-scale, 0.0, scale])
    elif bits == 2:
        # 4-level: {-3s, -s, +s, +3s}
        return np.array([-3 * scale, -scale, scale, 3 * scale])
    else:
        raise ValueError(f"Unsupported bit-width: {bits}")


def nearest_level(value: float, levels: np.ndarray) -> float:
    """Quantize a scalar to the nearest level."""
    idx = np.argmin(np.abs(levels - value))
    return levels[idx]


# ---------------------------------------------------------------------------
# Signal resampling
# ---------------------------------------------------------------------------

def upsample_signal(signal: np.ndarray, osr: int) -> np.ndarray:
    """Upsample 1D signal by OSR using linear interpolation."""
    n = len(signal)
    x_orig = np.arange(n)
    x_up = np.linspace(0, n - 1, n * osr)
    return np.interp(x_up, x_orig, signal)


def downsample_signal(signal: np.ndarray, osr: int, original_length: int) -> np.ndarray:
    """Downsample by averaging groups of OSR samples (boxcar filter)."""
    # Truncate to exact multiple if needed
    usable = original_length * osr
    signal = signal[:usable]
    reshaped = signal.reshape(original_length, osr)
    return reshaped.mean(axis=1)


# ---------------------------------------------------------------------------
# Sigma-Delta Modulator (1st order)
# ---------------------------------------------------------------------------

def sigma_delta_modulate(
    signal: np.ndarray,
    levels: np.ndarray,
) -> np.ndarray:
    """
    First-order Sigma-Delta modulator.

    The integrator accumulates quantization error, and the quantizer
    operates on the sum of the input and accumulated error.
    This shapes the quantization noise toward higher frequencies,
    improving in-band SNR.
    """
    n = len(signal)
    output = np.empty(n)
    integrator = 0.0

    for i in range(n):
        # Add input to accumulated error
        x = signal[i] + integrator
        # Quantize
        q = nearest_level(x, levels)
        output[i] = q
        # Accumulate error
        integrator = x - q

    return output


# ---------------------------------------------------------------------------
# Weight tensor quantization
# ---------------------------------------------------------------------------

def quantize_weight_tensor(
    weight: torch.Tensor,
    bits: float,
    osr: int,
    scale: float | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    Apply Sigma-Delta quantization to a single weight tensor.

    Returns:
        quantized_weight: Dequantized weight in the original dtype
        info: Dict with quantization statistics
    """
    original_shape = weight.shape
    original_dtype = weight.dtype
    w_np = weight.detach().float().cpu().numpy().flatten()
    original_length = len(w_np)

    # Determine scale factor (per-tensor std)
    if scale is None:
        scale = float(np.std(w_np))
    if scale < 1e-10:
        scale = 1e-10

    levels = get_quantization_levels(bits, scale)

    # Step 1: Upsample
    upsampled = upsample_signal(w_np, osr)

    # Step 2: Sigma-Delta modulation
    quantized_up = sigma_delta_modulate(upsampled, levels)

    # Step 3: Downsample
    quantized = downsample_signal(quantized_up, osr, original_length)

    # Step 4: Reshape back
    quantized_tensor = torch.tensor(quantized, dtype=original_dtype).reshape(original_shape)

    # Compute statistics
    mse = float(np.mean((w_np - quantized) ** 2))
    snr = float(10 * np.log10(np.mean(w_np**2) / max(mse, 1e-20)))

    info = {
        "scale": scale,
        "mse": mse,
        "snr_db": snr,
        "bits": bits,
        "osr": osr,
        "num_elements": original_length,
    }

    return quantized_tensor, info


# ---------------------------------------------------------------------------
# Per-layer variance analysis
# ---------------------------------------------------------------------------

def compute_layer_stats(model: nn.Module, skip_patterns: list[str] | None = None) -> dict[str, dict]:
    """Compute weight statistics for each linear layer."""
    skip = skip_patterns or QUANT_CONFIG["SKIP_PATTERNS"]
    stats = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(pat in name for pat in skip):
            continue

        w = module.weight.detach().float().cpu()
        stats[name] = {
            "numel": w.numel(),
            "std": float(w.std()),
            "variance": float(w.var()),
            "mean": float(w.mean()),
            "shape": list(w.shape),
        }

    return stats


# ---------------------------------------------------------------------------
# Per-layer adaptive OSR allocation
# ---------------------------------------------------------------------------

def allocate_osr_per_layer(
    layer_stats: dict[str, dict],
    target_avg_bits: float,
    bits: float,
    min_osr: int = 2,
    max_osr: int = 32,
) -> dict[str, int]:
    """
    Allocate OSR per layer to minimize total quantization noise
    under a target average bit-width constraint.

    Higher-variance layers get higher OSR (more resources).
    The effective bits per weight for a layer = bits (since OSR affects
    reconstruction quality, not storage in the SDQ-LLM framework).

    Strategy: distribute OSR proportionally to weight variance × layer size.
    """
    # Compute importance scores
    scores = {}
    total_params = 0
    for name, stat in layer_stats.items():
        score = stat["variance"] * stat["numel"]
        scores[name] = score
        total_params += stat["numel"]

    total_score = sum(scores.values())
    if total_score < 1e-20:
        return {name: min_osr for name in layer_stats}

    # Normalize and scale OSR range
    osr_allocation = {}
    for name in layer_stats:
        normalized = scores[name] / total_score
        # Scale: higher importance -> higher OSR
        raw_osr = min_osr + normalized * len(layer_stats) * (max_osr - min_osr)
        osr = int(np.clip(round(raw_osr), min_osr, max_osr))
        osr_allocation[name] = osr

    return osr_allocation


# ---------------------------------------------------------------------------
# Full model quantization
# ---------------------------------------------------------------------------

def quantize_model(
    model: nn.Module,
    bits: float,
    osr: int | None = None,
    adaptive: bool = False,
    target_avg_bits: float | None = None,
    skip_patterns: list[str] | None = None,
    inplace: bool = False,
) -> tuple[nn.Module, dict]:
    """
    Apply SDQ-LLM quantization to all linear layers in the model.

    Args:
        model: The model to quantize
        bits: Target bit-width (1, 1.58, or 2)
        osr: Fixed OSR for all layers (ignored if adaptive=True)
        adaptive: Use per-layer adaptive OSR allocation
        target_avg_bits: Target average bits for adaptive mode
        skip_patterns: Layer name patterns to skip
        inplace: Modify model in place or create a copy

    Returns:
        quantized_model: Model with quantized weights
        quant_info: Dict with per-layer quantization statistics
    """
    skip = skip_patterns or QUANT_CONFIG["SKIP_PATTERNS"]

    if not inplace:
        model = copy.deepcopy(model)

    # Get layer stats for adaptive allocation
    layer_stats = compute_layer_stats(model, skip)

    if adaptive:
        _target = target_avg_bits or bits
        osr_map = allocate_osr_per_layer(layer_stats, _target, bits)
    else:
        if osr is None:
            raise ValueError("osr must be specified when adaptive=False")
        osr_map = {name: osr for name in layer_stats}

    quant_info = {"layers": {}, "config": {"bits": bits, "adaptive": adaptive}}
    total_mse = 0.0
    total_elements = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(pat in name for pat in skip):
            continue
        if name not in osr_map:
            continue

        layer_osr = osr_map[name]
        quantized_weight, info = quantize_weight_tensor(
            module.weight.data,
            bits=bits,
            osr=layer_osr,
        )
        module.weight.data = quantized_weight.to(module.weight.device)
        info["osr_assigned"] = layer_osr
        quant_info["layers"][name] = info

        total_mse += info["mse"] * info["num_elements"]
        total_elements += info["num_elements"]

    if total_elements > 0:
        quant_info["aggregate_mse"] = total_mse / total_elements
        quant_info["aggregate_snr_db"] = float(
            10 * np.log10(
                sum(s["variance"] for s in layer_stats.values())
                / max(quant_info["aggregate_mse"], 1e-20)
            )
        )
    else:
        quant_info["aggregate_mse"] = 0.0
        quant_info["aggregate_snr_db"] = float("inf")

    quant_info["num_quantized_layers"] = len(quant_info["layers"])
    quant_info["total_quantized_elements"] = total_elements

    return model, quant_info


# ---------------------------------------------------------------------------
# Single-layer quantization (for sensitivity analysis)
# ---------------------------------------------------------------------------

def quantize_single_layer(
    model: nn.Module,
    layer_name: str,
    bits: float,
    osr: int,
) -> tuple[nn.Module, dict]:
    """Quantize only a single named linear layer (for sensitivity analysis)."""
    model = copy.deepcopy(model)

    for name, module in model.named_modules():
        if name == layer_name and isinstance(module, nn.Linear):
            quantized_weight, info = quantize_weight_tensor(
                module.weight.data,
                bits=bits,
                osr=osr,
            )
            module.weight.data = quantized_weight.to(module.weight.device)
            return model, info

    raise ValueError(f"Layer not found: {layer_name}")


if __name__ == "__main__":
    from model import build_model

    print("[test] Loading klue/bert-base...")
    model = build_model()

    print("\n[test] Layer statistics:")
    stats = compute_layer_stats(model)
    for name, s in list(stats.items())[:6]:
        print(f"  {name}: shape={s['shape']}, std={s['std']:.6f}, numel={s['numel']:,}")

    print(f"\n[test] Total quantizable layers: {len(stats)}")
    print(f"[test] Total quantizable params: {sum(s['numel'] for s in stats.values()):,}")

    print("\n[test] Quantizing with SDQ 2-bit OSR=4...")
    q_model, q_info = quantize_model(model, bits=2, osr=4)
    print(f"  Quantized layers: {q_info['num_quantized_layers']}")
    print(f"  Aggregate MSE: {q_info['aggregate_mse']:.8f}")
    print(f"  Aggregate SNR: {q_info['aggregate_snr_db']:.2f} dB")

    print("\n[test] Adaptive quantization with target 1.5 bits...")
    q_model_a, q_info_a = quantize_model(model, bits=1.58, adaptive=True, target_avg_bits=1.5)
    print(f"  Quantized layers: {q_info_a['num_quantized_layers']}")
    print(f"  OSR distribution:")
    osrs = [info["osr_assigned"] for info in q_info_a["layers"].values()]
    print(f"    min={min(osrs)}, max={max(osrs)}, mean={np.mean(osrs):.1f}")

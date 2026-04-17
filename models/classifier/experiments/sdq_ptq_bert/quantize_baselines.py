"""
Baseline quantization methods for comparison with SDQ-LLM.

RTN (Round-to-Nearest): naive per-tensor uniform quantization.
This serves as the primary baseline to isolate SDQ-LLM's noise-shaping benefit.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from config import QUANT_CONFIG


def rtn_quantize_weight(
    weight: torch.Tensor,
    bits: float,
) -> tuple[torch.Tensor, dict]:
    """
    Round-to-Nearest (RTN) uniform quantization for a single weight tensor.

    For binary (1-bit): threshold at 0, map to {-s, +s}
    For ternary (1.58-bit): two thresholds, map to {-s, 0, +s}
    For 2-bit: uniform 4-level quantization
    """
    original_shape = weight.shape
    original_dtype = weight.dtype
    w = weight.detach().float()
    scale = float(w.std())
    if scale < 1e-10:
        scale = 1e-10

    w_np = w.cpu().numpy().flatten()

    if bits == 1:
        # Binary: sign-based
        quantized = np.where(w_np >= 0, scale, -scale)
    elif bits == 1.58 or abs(bits - 1.58) < 0.1:
        # Ternary: threshold at +/- 0.5 * scale
        threshold = 0.5 * scale
        quantized = np.zeros_like(w_np)
        quantized[w_np > threshold] = scale
        quantized[w_np < -threshold] = -scale
    elif bits == 2:
        # 4-level uniform: {-3s, -s, +s, +3s}
        levels = np.array([-3 * scale, -scale, scale, 3 * scale])
        quantized = np.empty_like(w_np)
        for i in range(len(w_np)):
            idx = np.argmin(np.abs(levels - w_np[i]))
            quantized[i] = levels[idx]
    else:
        raise ValueError(f"Unsupported bit-width: {bits}")

    quantized_tensor = torch.tensor(quantized, dtype=original_dtype).reshape(original_shape)

    mse = float(np.mean((w_np - quantized) ** 2))
    snr = float(10 * np.log10(np.mean(w_np**2) / max(mse, 1e-20)))

    info = {
        "scale": scale,
        "mse": mse,
        "snr_db": snr,
        "bits": bits,
        "method": "rtn",
        "num_elements": len(w_np),
    }

    return quantized_tensor, info


def quantize_model_rtn(
    model: nn.Module,
    bits: float,
    skip_patterns: list[str] | None = None,
    inplace: bool = False,
) -> tuple[nn.Module, dict]:
    """
    Apply RTN quantization to all linear layers.

    Args:
        model: The model to quantize
        bits: Target bit-width
        skip_patterns: Layer name patterns to skip
        inplace: Modify model in place or create a copy

    Returns:
        quantized_model: Model with RTN-quantized weights
        quant_info: Per-layer quantization statistics
    """
    skip = skip_patterns or QUANT_CONFIG["SKIP_PATTERNS"]

    if not inplace:
        model = copy.deepcopy(model)

    quant_info = {"layers": {}, "config": {"bits": bits, "method": "rtn"}}
    total_mse = 0.0
    total_elements = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(pat in name for pat in skip):
            continue

        quantized_weight, info = rtn_quantize_weight(module.weight.data, bits)
        module.weight.data = quantized_weight.to(module.weight.device)
        quant_info["layers"][name] = info

        total_mse += info["mse"] * info["num_elements"]
        total_elements += info["num_elements"]

    if total_elements > 0:
        quant_info["aggregate_mse"] = total_mse / total_elements
    else:
        quant_info["aggregate_mse"] = 0.0

    quant_info["num_quantized_layers"] = len(quant_info["layers"])
    quant_info["total_quantized_elements"] = total_elements

    return model, quant_info

"""
Post-hoc analysis and visualization for SDQ-PTQ experiments.

Generates:
  1. Comparison table (console + CSV)
  2. Pareto frontier plot (model size vs accuracy)
  3. OSR vs accuracy curves per bit-width
  4. Layer sensitivity heatmap
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import LOG_DIR


def load_all_results(log_dir: Path | None = None) -> list[dict]:
    """Load all experiment result JSONs from the log directory."""
    _dir = log_dir or LOG_DIR
    results = []
    for path in sorted(_dir.glob("*.json")):
        if "sensitivity" in path.name or "summary" in path.name:
            continue
        if "train_log" in path.name:
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "config_name" in data and "accuracy" in data:
            results.append(data)
    return results


def load_sensitivity(log_dir: Path | None = None) -> dict | None:
    """Load the sensitivity analysis result."""
    _dir = log_dir or LOG_DIR
    for path in sorted(_dir.glob("sensitivity_*.json"), reverse=True):
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


def print_comparison_table(results: list[dict]) -> None:
    """Print a formatted comparison table."""
    print(f"\n{'=' * 100}")
    print(f"  {'Config':<30s} {'Bits':>5} {'Acc(%)':>8} {'dAcc(%)':>9} "
          f"{'MacF1':>7} {'Size(MB)':>10} {'CR':>6} {'ms/s':>7}")
    print(f"  {'-' * 94}")

    # Find FP32 baseline accuracy
    base_acc = None
    for r in results:
        if r["config_name"] == "fp32_baseline":
            base_acc = r["accuracy"]
            break

    for r in results:
        bits = r.get("bits", 32) or 32
        acc = r["accuracy"]
        dacc = (base_acc - acc) * 100 if base_acc is not None else 0.0

        print(
            f"  {r['config_name']:<30s} "
            f"{bits:>5.2f} "
            f"{acc*100:>8.2f} "
            f"{dacc:>+9.2f} "
            f"{r['macro_f1']:>7.4f} "
            f"{r['size']['size_mb']:>10.1f} "
            f"{r['size']['compression_ratio']:>6.2f} "
            f"{r['latency']['ms_per_sample']:>7.3f}"
        )
    print(f"{'=' * 100}")


def plot_pareto_frontier(results: list[dict], save_path: Path | None = None) -> None:
    """Plot model size vs accuracy with Pareto frontier."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by method
    groups = {"fp32/bf16": [], "rtn": [], "sdq_fixed": [], "sdq_adaptive": []}
    for r in results:
        name = r["config_name"]
        point = (r["size"]["size_mb"], r["accuracy"] * 100)
        if "baseline" in name:
            groups["fp32/bf16"].append((*point, name))
        elif "rtn" in name:
            groups["rtn"].append((*point, name))
        elif "adaptive" in name:
            groups["sdq_adaptive"].append((*point, name))
        elif "sdq" in name:
            groups["sdq_fixed"].append((*point, name))

    markers = {"fp32/bf16": "s", "rtn": "^", "sdq_fixed": "o", "sdq_adaptive": "D"}
    colors = {"fp32/bf16": "#2196F3", "rtn": "#FF9800", "sdq_fixed": "#4CAF50", "sdq_adaptive": "#E91E63"}

    for group_name, points in groups.items():
        if not points:
            continue
        sizes, accs, labels = zip(*points)
        ax.scatter(sizes, accs, marker=markers[group_name], c=colors[group_name],
                   s=80, label=group_name, zorder=5, edgecolors="white", linewidth=0.5)
        for s, a, l in zip(sizes, accs, labels):
            ax.annotate(l, (s, a), fontsize=6, ha="left", va="bottom",
                        xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Model Size (MB)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("SDQ-LLM PTQ: Model Size vs Accuracy (klue/bert-base + NSMC)", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    _path = save_path or LOG_DIR / "pareto_frontier.png"
    fig.savefig(_path, dpi=150)
    print(f"[plot] Pareto frontier saved: {_path}")
    plt.close(fig)


def plot_osr_vs_accuracy(results: list[dict], save_path: Path | None = None) -> None:
    """Plot accuracy as a function of OSR for each bit-width."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect SDQ fixed-OSR results
    bit_data: dict[float, list[tuple[int, float]]] = {}
    rtn_data: dict[float, float] = {}

    for r in results:
        name = r["config_name"]
        if name.startswith("sdq_") and "adaptive" not in name:
            # Parse bits and OSR from config name
            qi = r.get("quant_info", {})
            bits = qi.get("bits", r.get("bits"))
            osr = qi.get("osr")
            if bits is not None and osr is not None:
                bit_data.setdefault(bits, []).append((osr, r["accuracy"] * 100))
        elif name.startswith("rtn_"):
            qi = r.get("quant_info", {})
            bits = qi.get("config", {}).get("bits", r.get("bits"))
            if bits is not None:
                rtn_data[bits] = r["accuracy"] * 100

    colors = {1: "#E91E63", 1.58: "#4CAF50", 2: "#2196F3"}

    for bits, points in sorted(bit_data.items()):
        points.sort(key=lambda x: x[0])
        osrs, accs = zip(*points)
        ax.plot(osrs, accs, "o-", color=colors.get(bits, "gray"),
                label=f"SDQ {bits}-bit", markersize=6)

        # RTN horizontal line for same bit-width
        if bits in rtn_data:
            ax.axhline(y=rtn_data[bits], color=colors.get(bits, "gray"),
                       linestyle="--", alpha=0.5, label=f"RTN {bits}-bit")

    ax.set_xlabel("Over-Sampling Ratio (OSR)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("SDQ-LLM: OSR vs Accuracy by Bit-Width", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    plt.tight_layout()

    _path = save_path or LOG_DIR / "osr_vs_accuracy.png"
    fig.savefig(_path, dpi=150)
    print(f"[plot] OSR vs accuracy saved: {_path}")
    plt.close(fig)


def plot_sensitivity_heatmap(sensitivity: dict, save_path: Path | None = None) -> None:
    """Plot layer sensitivity as a heatmap (12 layers x 6 sublayers)."""
    layers_data = sensitivity.get("layers", {})
    if not layers_data:
        print("[skip] No sensitivity data to plot")
        return

    # Organize into grid: 12 layers x sublayer types
    sublayer_types = ["attention.self.query", "attention.self.key", "attention.self.value",
                      "attention.output.dense", "intermediate.dense", "output.dense"]
    sublayer_short = ["Q", "K", "V", "Attn Out", "FFN Up", "FFN Down"]

    grid = np.full((12, 6), np.nan)
    for name, info in layers_data.items():
        for layer_idx in range(12):
            layer_prefix = f"bert.encoder.layer.{layer_idx}."
            if layer_prefix in name:
                for j, sub in enumerate(sublayer_types):
                    if sub in name:
                        grid[layer_idx, j] = info["accuracy_drop"] * 100
                        break
                break

    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(grid, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(6))
    ax.set_xticklabels(sublayer_short, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(12))
    ax.set_yticklabels([f"Layer {i}" for i in range(12)], fontsize=9)
    ax.set_title(f"Layer Sensitivity (Accuracy Drop %)\n{sensitivity.get('bits', 2)}-bit OSR={sensitivity.get('osr', 4)}", fontsize=13)

    # Add text annotations
    for i in range(12):
        for j in range(6):
            val = grid[i, j]
            if not np.isnan(val):
                color = "white" if val > np.nanmax(grid) * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="Accuracy Drop (%)")
    plt.tight_layout()

    _path = save_path or LOG_DIR / "sensitivity_heatmap.png"
    fig.savefig(_path, dpi=150)
    print(f"[plot] Sensitivity heatmap saved: {_path}")
    plt.close(fig)


def main():
    print("[analyze] Loading results from", LOG_DIR)

    results = load_all_results()
    if not results:
        print("[error] No result files found. Run experiments first.")
        return

    print(f"[analyze] Found {len(results)} experiment results")

    # Comparison table
    print_comparison_table(results)

    # Plots
    plot_pareto_frontier(results)
    plot_osr_vs_accuracy(results)

    sensitivity = load_sensitivity()
    if sensitivity:
        plot_sensitivity_heatmap(sensitivity)
    else:
        print("[skip] No sensitivity data found")

    print("\n[done] Analysis complete")


if __name__ == "__main__":
    main()

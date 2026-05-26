import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import hadamard

from config import CHECKPOINT_DIR, LOG_DIR
from model import load_model_from_checkpoint

def resolve_checkpoint() -> Path:
    meta_path = CHECKPOINT_DIR / "baseline_latest.json"
    if meta_path.exists():
        with meta_path.open("r") as f:
            meta = json.load(f)
        return Path(meta["checkpoint_path"])
    raise FileNotFoundError("No baseline checkpoint found.")

def get_hadamard_matrix(n_padded: int) -> np.ndarray:
    """Return a Hadamard matrix of size n_padded."""
    H = hadamard(n_padded) / np.sqrt(n_padded)
    return H

def plot_weight_outliers():
    print("[plot_outliers] Loading checkpoint...")
    ckpt_path = resolve_checkpoint()
    model = load_model_from_checkpoint(str(ckpt_path))
    
    # Let's pick a very sensitive layer identified from our sensitivity analysis
    # Layer 5 FFN Up (intermediate.dense) and Layer 5 Q-proj (attention.self.query)
    target_layers = {
        "Layer 5 FFN Up": "bert.encoder.layer.5.intermediate.dense.weight",
        "Layer 5 Query": "bert.encoder.layer.5.attention.self.query.weight",
    }
    
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    for name, param_name in target_layers.items():
        weight = model.state_dict()[param_name].cpu().numpy()
        
        # Take the first output channel (row 0) to visualize
        w_row = weight[0, :]
        n = len(w_row)
        
        # 1. Hadamard Transformation
        # Standard approach for these charts is to transform the features/weights to smooth outliers
        n_padded = 1 << (n - 1).bit_length()
        H = get_hadamard_matrix(n_padded)
        
        # Pad w_row to n_padded
        w_padded = np.zeros(n_padded)
        w_padded[:n] = w_row
        
        # Apply transformation and truncate back
        w_hadamard = (w_padded @ H)[:n]
        
        # 2. FFT Frequency Domain Map
        w_fft = np.abs(np.fft.fft(w_row))
        w_hadamard_fft = np.abs(np.fft.fft(w_hadamard))
        
        # Truncate FFT to half (symmetric)
        half_n = n // 2
        w_fft = w_fft[:half_n]
        w_hadamard_fft = w_hadamard_fft[:half_n]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Time/Spatial Domain
        ax1.plot(np.abs(w_row), label="weight", color="tab:blue", alpha=0.8, linewidth=1.5)
        ax1.plot(np.abs(w_hadamard), label="w_hadamard", color="tab:orange", alpha=0.8, linewidth=1.5)
        ax1.set_title(f"Spatial Domain: {name}\n({param_name})", fontsize=11)
        ax1.set_xlabel("Input Dimension Index")
        ax1.set_ylabel("Absolute Value")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Frequency Domain (FFT Magnitude)
        ax2.plot(w_fft, label=f"FFT Mag of weight[0]", color="tab:blue", alpha=0.8)
        ax2.plot(w_hadamard_fft, label=f"FFT Mag of w_hadamard[0]", color="tab:orange", alpha=0.8)
        ax2.set_title(f"Frequency Domain: {name}", fontsize=11)
        ax2.set_xlabel("Frequency Bin")
        ax2.set_ylabel("Magnitude")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = LOG_DIR / f"outlier_analysis_{name.replace(' ', '_')}.png"
        fig.savefig(save_path, dpi=150)
        print(f"[plot_outliers] Saved plot for {name} -> {save_path}")
        plt.close()

if __name__ == "__main__":
    plot_weight_outliers()

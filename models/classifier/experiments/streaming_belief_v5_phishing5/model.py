from pathlib import Path
import sys
import importlib.util

from config import ENCODER_CONFIG, HEAD_CONFIG, MAMBA_CONFIG, NUM_LABELS

SRC_DIR = Path(__file__).resolve().parents[1] / "streaming_belief_v5"
model_path = SRC_DIR / "model.py"
if not model_path.exists():
    raise FileNotFoundError(f"Cannot find streaming belief model file: {model_path}")

spec = importlib.util.spec_from_file_location("streaming_belief_v5_model", model_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load module from {model_path}")
streaming_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = streaming_module
spec.loader.exec_module(streaming_module)
StreamingBeliefClassifier = streaming_module.StreamingBeliefClassifier


def build_model() -> StreamingBeliefClassifier:
    return StreamingBeliefClassifier(
        encoder_name=ENCODER_CONFIG["MODEL_NAME"],
        d_model=MAMBA_CONFIG["D_MODEL"],
        d_state=MAMBA_CONFIG["D_STATE"],
        d_conv=MAMBA_CONFIG["D_CONV"],
        expand=MAMBA_CONFIG["EXPAND"],
        num_mamba_layers=MAMBA_CONFIG["NUM_LAYERS"],
        mamba_dropout=MAMBA_CONFIG["DROPOUT"],
        num_labels=NUM_LABELS,
        head_hidden_dim=HEAD_CONFIG["HIDDEN_DIM"],
        head_dropout=HEAD_CONFIG["DROPOUT"],
        freeze_embeddings=True,
        freeze_lower_layers=10,
        unfreeze_all=False,
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        lora_target_modules=["query", "key", "value"],
    )


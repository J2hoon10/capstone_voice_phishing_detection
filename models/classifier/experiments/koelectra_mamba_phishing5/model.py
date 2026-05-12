from pathlib import Path
import sys
import importlib.util
import torch
import torch.nn as nn

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
BaseStreamingBeliefClassifier = streaming_module.StreamingBeliefClassifier


class StreamingBeliefClassifier(BaseStreamingBeliefClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 보조 헤드 추가 (세그먼트별 위험도 3단계 분류)
        self.aux_head = nn.Linear(self.d_model, 3)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_mask: torch.Tensor,
        num_segments: torch.Tensor,
        **kwargs,
    ) -> dict:
        B, max_S, _ = input_ids.shape
        device = input_ids.device
        enc_dtype = next(self.encoder.parameters()).dtype

        segment_vectors = []
        for t in range(max_S):
            valid = segment_mask[:, t]
            if not valid.any():
                break

            x_t = torch.zeros(B, self.d_model, device=device, dtype=enc_dtype)
            H_valid = self._encode_segment(
                input_ids=input_ids[:, t, :][valid],
                attention_mask=attention_mask[:, t, :][valid],
            )
            x_valid = self.pooling(H_valid, attention_mask[:, t, :][valid])
            x_t[valid] = x_valid.to(enc_dtype)
            segment_vectors.append(x_t.float())

        T = len(segment_vectors)
        if T == 0:
            dummy = torch.zeros(B, self.num_labels, device=device)
            dummy_aux = torch.zeros(B, max_S, 3, device=device)
            return {"logits": dummy, "all_logits": [dummy], "aux_logits": dummy_aux}

        # (B, T, D) - Mamba 입력 전 세그먼트 벡터
        x_stacked = torch.stack(segment_vectors, dim=1)
        
        # 보조 태스크 출력 (위험도)
        aux_logits = self.aux_head(x_stacked)  # (B, T, 3)

        y = x_stacked
        for mamba, dropout in zip(self.mamba_layers, self.mamba_dropouts):
            y = dropout(mamba(y))

        # (B, T, C)
        logits_all = self.head(y)
        all_logits = [logits_all[:, t, :] for t in range(T)]
        final_logits = self._gather_last_logits(all_logits, num_segments)
        
        return {"logits": final_logits, "all_logits": all_logits, "aux_logits": aux_logits}


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


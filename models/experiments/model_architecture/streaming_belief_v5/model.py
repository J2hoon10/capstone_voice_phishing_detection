"""
Streaming Belief v5 모델.

구성:
1) KoELECTRA-base (청크 인코딩)
2) Attention-weighted Pooling ([CLS]/[SEP] 제외)
3) 2-layer Mamba SSM (청크 간 belief 추적)
4) MLP classifier
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout

import torch
import torch.nn as nn
from transformers import AutoModel

from mamba_ssm import Mamba


IGNORED_LOAD_REPORT_PREFIXES = (
    "lm_head.",
    "pooler.",
    "roberta.embeddings.position_ids",
    "electra.embeddings.position_ids",
    "discriminator_predictions.",
    "generator_predictions.",
    "generator_lm_head.",
)


def load_encoder_backbone(encoder_name: str):
    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        encoder = AutoModel.from_pretrained(encoder_name, trust_remote_code=True)

    noteworthy_rows = []
    for line in buffer.getvalue().splitlines():
        if "|" not in line:
            continue
        key = line.split("|", 1)[0].strip()
        if (
            not key
            or key == "Key"
            or key.startswith("-")
            or any(key.startswith(prefix) for prefix in IGNORED_LOAD_REPORT_PREFIXES)
        ):
            continue
        noteworthy_rows.append(line.rstrip())

    if noteworthy_rows:
        print(f"[model] {encoder_name} load report (backbone only)")
        for row in noteworthy_rows:
            print(row)
    else:
        print(f"[model] {encoder_name} loaded")

    return encoder


def freeze_encoder_layers(
    encoder: nn.Module,
    freeze_embeddings: bool,
    freeze_lower_layers: int,
) -> None:
    if freeze_embeddings and hasattr(encoder, "embeddings"):
        for p in encoder.embeddings.parameters():
            p.requires_grad = False

    layers = None
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        layers = encoder.encoder.layer
    elif hasattr(encoder, "electra") and hasattr(encoder.electra, "encoder"):
        if hasattr(encoder.electra.encoder, "layer"):
            layers = encoder.electra.encoder.layer

    if layers is None or freeze_lower_layers <= 0:
        return

    n = min(int(freeze_lower_layers), len(layers))
    for i in range(n):
        for p in layers[i].parameters():
            p.requires_grad = False


def maybe_apply_lora(
    encoder: nn.Module,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list[str],
) -> nn.Module:
    if not use_lora:
        return encoder

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError:
        print("[warn] peft not installed. continue without LoRA.")
        return encoder

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=target_modules,
    )
    encoder = get_peft_model(encoder, lora_cfg)
    print(
        f"[model] LoRA enabled (r={lora_r}, alpha={lora_alpha}, "
        f"dropout={lora_dropout}, targets={target_modules})"
    )
    return encoder


class AttentionWeightedPooling(nn.Module):
    """
    Attention-weighted Pooling.
    - [CLS], [SEP]를 제외한 실제 토큰만 softmax 가중합한다.
    """

    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.W_b = nn.Linear(hidden_dim, hidden_dim)
        self.W_a = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, L, D), attention_mask: (B, L)
        H_tokens = hidden_states[:, 1:-1, :]
        mask_tokens = attention_mask[:, 1:-1]

        e = self.W_a(torch.tanh(self.W_b(H_tokens)))  # (B, L-2, 1)
        e = e.masked_fill(mask_tokens.unsqueeze(-1) == 0, -1e4)
        alpha = torch.softmax(e, dim=1)

        # 유효 토큰이 없는 케이스에 대한 안전 처리
        alpha = alpha * mask_tokens.unsqueeze(-1).float()
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp(min=1e-8)
        x_t = (alpha * H_tokens).sum(dim=1)  # (B, D)
        return x_t


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class StreamingBeliefClassifier(nn.Module):
    """
    KoELECTRA + Attention-weighted Pooling + Mamba SSM 분류기.
    """

    def __init__(
        self,
        encoder_name: str,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        num_mamba_layers: int,
        mamba_dropout: float,
        num_labels: int,
        head_hidden_dim: int,
        head_dropout: float,
        freeze_embeddings: bool,
        freeze_lower_layers: int,
        unfreeze_all: bool,
        use_lora: bool,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_target_modules: list[str],
    ):
        super().__init__()
        self.d_model = d_model
        self.num_labels = num_labels

        self.encoder = load_encoder_backbone(encoder_name)

        if not unfreeze_all:
            freeze_encoder_layers(
                self.encoder,
                freeze_embeddings=freeze_embeddings,
                freeze_lower_layers=freeze_lower_layers,
            )

        self.encoder = maybe_apply_lora(
            encoder=self.encoder,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
        )

        self.pooling = AttentionWeightedPooling(hidden_dim=d_model)
        self.mamba_layers = nn.ModuleList(
            [Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(num_mamba_layers)]
        )
        self.mamba_dropouts = nn.ModuleList([nn.Dropout(mamba_dropout) for _ in range(num_mamba_layers)])
        self.head = ClassificationHead(
            input_dim=d_model,
            hidden_dim=head_hidden_dim,
            num_labels=num_labels,
            dropout=head_dropout,
        )

    def _encode_segment(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state

    def _gather_last_logits(self, all_logits: list[torch.Tensor], num_segments: torch.Tensor) -> torch.Tensor:
        B = num_segments.shape[0]
        final = all_logits[0].new_zeros(B, self.num_labels)
        for b in range(B):
            last_t = min(int(num_segments[b].item()) - 1, len(all_logits) - 1)
            final[b] = all_logits[last_t][b]
        return final

    def forward(
        self,
        input_ids: torch.Tensor,      # (B, max_S, L)
        attention_mask: torch.Tensor,  # (B, max_S, L)
        segment_mask: torch.Tensor,    # (B, max_S)
        num_segments: torch.Tensor,    # (B,)
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
            return {"logits": dummy, "all_logits": [dummy]}

        # (B, T, D)
        y = torch.stack(segment_vectors, dim=1)
        for mamba, dropout in zip(self.mamba_layers, self.mamba_dropouts):
            y = dropout(mamba(y))

        # (B, T, C)
        logits_all = self.head(y)
        all_logits = [logits_all[:, t, :] for t in range(T)]
        final_logits = self._gather_last_logits(all_logits, num_segments)
        return {"logits": final_logits, "all_logits": all_logits}


def build_streaming_belief_model(ablation_config: dict) -> StreamingBeliefClassifier:
    from config import DATA_CONFIG, ENCODER_CONFIG, HEAD_CONFIG, MAMBA_CONFIG

    return StreamingBeliefClassifier(
        encoder_name=ENCODER_CONFIG["MODEL_NAME"],
        d_model=MAMBA_CONFIG["D_MODEL"],
        d_state=ablation_config.get("D_STATE", MAMBA_CONFIG["D_STATE"]),
        d_conv=MAMBA_CONFIG["D_CONV"],
        expand=MAMBA_CONFIG["EXPAND"],
        num_mamba_layers=ablation_config.get("NUM_LAYERS", MAMBA_CONFIG["NUM_LAYERS"]),
        mamba_dropout=MAMBA_CONFIG["DROPOUT"],
        num_labels=DATA_CONFIG["NUM_LABELS"],
        head_hidden_dim=HEAD_CONFIG["HIDDEN_DIM"],
        head_dropout=HEAD_CONFIG["DROPOUT"],
        freeze_embeddings=ablation_config.get("FREEZE_EMBEDDINGS", ENCODER_CONFIG["FREEZE_EMBEDDINGS"]),
        freeze_lower_layers=ablation_config.get("FREEZE_LOWER_LAYERS", ENCODER_CONFIG["FREEZE_LOWER_LAYERS"]),
        unfreeze_all=ablation_config.get("UNFREEZE_ALL", ENCODER_CONFIG["UNFREEZE_ALL"]),
        use_lora=ablation_config.get("USE_LORA", ENCODER_CONFIG["USE_LORA"]),
        lora_r=ablation_config.get("LORA_R", ENCODER_CONFIG["LORA_R"]),
        lora_alpha=ablation_config.get("LORA_ALPHA", ENCODER_CONFIG["LORA_ALPHA"]),
        lora_dropout=ablation_config.get("LORA_DROPOUT", ENCODER_CONFIG["LORA_DROPOUT"]),
        lora_target_modules=ablation_config.get("LORA_TARGET_MODULES", ENCODER_CONFIG["LORA_TARGET_MODULES"]),
    )

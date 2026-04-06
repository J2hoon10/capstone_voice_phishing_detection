"""
Compressive Memory classifier for SNS dialogue-level subject prediction.

Flow:
turn -> encoder -> projection -> memory update (FM/CM) -> final FM pooling -> head
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout

import torch
import torch.nn as nn
from transformers import AutoModel


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
    """
    Load huggingface encoder and suppress noisy non-backbone warnings.
    """
    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        encoder = AutoModel.from_pretrained(encoder_name)

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


class LinearProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class CompressiveMemory(nn.Module):
    """
    Two-level memory:
    - Fine memory (FM): recent segments
    - Compressed memory (CM): compressed old segments
    """

    def __init__(
        self,
        slot_dim: int,
        fm_size: int,
        cm_size: int,
        conv_kernel_size: int = 2,
        use_learnable_init: bool = True,
        compress_fn: str = "conv",
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.fm_size = fm_size
        self.cm_size = cm_size
        self.compress_fn = compress_fn

        if compress_fn == "conv" and cm_size > 0:
            self.compressor = nn.Conv1d(
                in_channels=slot_dim,
                out_channels=slot_dim,
                kernel_size=conv_kernel_size,
            )
        else:
            self.compressor = None

        if use_learnable_init:
            if fm_size > 0:
                self.fm_init = nn.Parameter(torch.randn(fm_size, slot_dim) * 0.02)
            else:
                self.fm_init = None
            if cm_size > 0:
                self.cm_init = nn.Parameter(torch.randn(cm_size, slot_dim) * 0.02)
            else:
                self.cm_init = None
        else:
            self.fm_init = None
            self.cm_init = None

    def init_memory(self, batch_size: int, device: torch.device) -> dict:
        state = {"fm_count": torch.zeros(batch_size, dtype=torch.long, device=device)}

        if self.fm_size > 0:
            if self.fm_init is not None:
                fm = self.fm_init.unsqueeze(0).expand(batch_size, -1, -1).clone()
            else:
                fm = torch.zeros(batch_size, self.fm_size, self.slot_dim, device=device)
        else:
            fm = None

        if self.cm_size > 0:
            if self.cm_init is not None:
                cm = self.cm_init.unsqueeze(0).expand(batch_size, -1, -1).clone()
            else:
                cm = torch.zeros(batch_size, self.cm_size, self.slot_dim, device=device)
        else:
            cm = None

        state["fine_memory"] = fm
        state["compressed_memory"] = cm
        return state

    def _compress(self, cm_last: torch.Tensor, fm_oldest: torch.Tensor) -> torch.Tensor:
        if self.compress_fn == "mean" or self.compressor is None:
            return (cm_last + fm_oldest) / 2.0

        # Input shape for Conv1d: (B, C=slot_dim, L=2)
        stacked = torch.stack([cm_last, fm_oldest], dim=2)
        compressed = self.compressor(stacked)
        return compressed.squeeze(2)

    def update(self, state: dict, new_segment: torch.Tensor, valid_mask: torch.Tensor) -> dict:
        if self.fm_size == 0:
            return state

        fm = state["fine_memory"].clone()
        cm = state["compressed_memory"].clone() if state["compressed_memory"] is not None else None
        fm_count = state["fm_count"].clone()

        batch_size = new_segment.shape[0]
        for b in range(batch_size):
            if not bool(valid_mask[b]):
                continue

            cur_count = int(fm_count[b].item())
            if cur_count < self.fm_size:
                fm[b, cur_count] = new_segment[b]
                fm_count[b] = cur_count + 1
                continue

            fm_oldest = fm[b, 0]
            if cm is not None and self.cm_size > 0:
                cm_last = cm[b, -1]
                c_new = self._compress(cm_last.unsqueeze(0), fm_oldest.unsqueeze(0)).squeeze(0)
                cm[b] = torch.cat([cm[b, 1:], c_new.unsqueeze(0)], dim=0)

            fm[b] = torch.cat([fm[b, 1:], new_segment[b].unsqueeze(0)], dim=0)

        return {
            "fine_memory": fm,
            "compressed_memory": cm,
            "fm_count": fm_count,
        }

    def final_pool(self, state: dict) -> torch.Tensor:
        """
        Pool final memory state into one vector per dialogue.

        Priority:
        1) mean of valid FM slots
        2) mean of CM slots (if FM unavailable)
        3) zeros fallback
        """
        fm = state["fine_memory"]
        cm = state["compressed_memory"]
        fm_count = state["fm_count"]

        if fm is not None:
            batch_size = fm.shape[0]
            pooled = torch.zeros(batch_size, self.slot_dim, device=fm.device, dtype=fm.dtype)
            for b in range(batch_size):
                c = int(fm_count[b].item())
                if c > 0:
                    pooled[b] = fm[b, :c].mean(dim=0)
                elif cm is not None:
                    pooled[b] = cm[b].mean(dim=0)
            return pooled

        if cm is not None:
            return cm.mean(dim=1)

        raise RuntimeError("Invalid memory state: both FM and CM are None")


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


class CompressiveMemoryClassifier(nn.Module):
    """
    Dialogue-level classifier.

    - baseline: mean-pool turn representations
    - memory model: sequential memory update + final FM pooling
    """

    def __init__(
        self,
        encoder_name: str = "monologg/koelectra-base-v3-discriminator",
        encoder_hidden_size: int = 768,
        slot_dim: int = 128,
        fm_size: int = 3,
        cm_size: int = 4,
        num_labels: int = 20,
        head_hidden_dim: int = 64,
        dropout: float = 0.1,
        conv_kernel_size: int = 2,
        use_learnable_init: bool = True,
        freeze_encoder: bool = True,
        compress_fn: str = "conv",
    ):
        super().__init__()

        self.slot_dim = slot_dim
        self.fm_size = fm_size
        self.cm_size = cm_size
        self.num_labels = num_labels
        self.is_baseline = (fm_size == 0 and cm_size == 0)

        self.encoder = load_encoder_backbone(encoder_name)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.projection = LinearProjection(encoder_hidden_size, slot_dim)
        self.head = ClassificationHead(
            input_dim=slot_dim,
            hidden_dim=head_hidden_dim,
            num_labels=num_labels,
            dropout=dropout,
        )

        if self.is_baseline:
            self.memory = None
        else:
            self.memory = CompressiveMemory(
                slot_dim=slot_dim,
                fm_size=fm_size,
                cm_size=cm_size,
                conv_kernel_size=conv_kernel_size,
                use_learnable_init=use_learnable_init,
                compress_fn=compress_fn,
            )

    def encode_turn(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.projection(cls)

    def encode_dialogue_turns(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        turn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode all valid turns in one encoder forward pass.

        Args:
            input_ids: (B, T, L)
            attention_mask: (B, T, L)
            turn_mask: (B, T) True for real turns

        Returns:
            projected_turns: (B, T, slot_dim)
        """
        batch_size, max_turns, seq_len = input_ids.shape
        device = input_ids.device
        out_dtype = self.projection.projection.weight.dtype

        flat_ids = input_ids.reshape(batch_size * max_turns, seq_len)
        flat_attn = attention_mask.reshape(batch_size * max_turns, seq_len)
        flat_mask = turn_mask.reshape(batch_size * max_turns)

        flat_proj = torch.zeros(
            batch_size * max_turns,
            self.slot_dim,
            device=device,
            dtype=out_dtype,
        )
        if flat_mask.any():
            encoded = self.encode_turn(flat_ids[flat_mask], flat_attn[flat_mask]).to(out_dtype)
            flat_proj[flat_mask] = encoded

        return flat_proj.reshape(batch_size, max_turns, self.slot_dim)

    def _forward_baseline(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        turn_mask: torch.Tensor,
    ) -> dict:
        projected_turns = self.encode_dialogue_turns(input_ids, attention_mask, turn_mask)
        mask = turn_mask.unsqueeze(-1).to(projected_turns.dtype)
        turn_sum = (projected_turns * mask).sum(dim=1)
        turn_count = mask.sum(dim=1).clamp(min=1.0)
        pooled = turn_sum / turn_count
        logits = self.head(pooled)
        return {"logits": logits, "pooled_state": pooled}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        turn_mask: torch.Tensor,
        num_turns: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        if self.is_baseline:
            return self._forward_baseline(
                input_ids=input_ids,
                attention_mask=attention_mask,
                turn_mask=turn_mask,
            )

        batch_size, max_turns, _ = input_ids.shape
        device = input_ids.device
        mem_state = self.memory.init_memory(batch_size, device)
        projected_turns = self.encode_dialogue_turns(input_ids, attention_mask, turn_mask)

        for t in range(max_turns):
            valid = turn_mask[:, t]
            if not valid.any():
                break
            s_t = projected_turns[:, t, :]
            mem_state = self.memory.update(mem_state, s_t, valid)

        pooled = self.memory.final_pool(mem_state)
        logits = self.head(pooled)
        return {"logits": logits, "pooled_state": pooled}


def build_compressive_memory_model(config: dict) -> CompressiveMemoryClassifier:
    from config import ENCODER_CONFIG, HEAD_CONFIG, MEMORY_CONFIG, SNS_CONFIG

    fm_size = config.get("FM_SIZE", MEMORY_CONFIG["FM_SIZE"])
    cm_size = config.get("CM_SIZE", MEMORY_CONFIG["CM_SIZE"])
    compress_fn = config.get("COMPRESS_FN", "conv")

    return CompressiveMemoryClassifier(
        encoder_name=ENCODER_CONFIG["MODEL_NAME"],
        encoder_hidden_size=ENCODER_CONFIG["HIDDEN_SIZE"],
        slot_dim=MEMORY_CONFIG["SLOT_DIM"],
        fm_size=fm_size,
        cm_size=cm_size,
        num_labels=SNS_CONFIG["NUM_LABELS"],
        head_hidden_dim=HEAD_CONFIG["HIDDEN_DIM"],
        dropout=HEAD_CONFIG["DROPOUT"],
        conv_kernel_size=MEMORY_CONFIG["CONV_KERNEL_SIZE"],
        use_learnable_init=MEMORY_CONFIG["USE_LEARNABLE_INIT"],
        freeze_encoder=ENCODER_CONFIG["FREEZE_ENCODER"],
        compress_fn=compress_fn,
    )

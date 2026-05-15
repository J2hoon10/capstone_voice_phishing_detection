import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from audio_processor import AudioProcessor


LABELS = [
    "상품 가입 및 해지",
    "이체 출금 대출서비스",
    "잔고 및 거래내역",
    "수사기관 사칭형",
    "대출 사기형",
]
PHISHING_LABELS = {"수사기관 사칭형", "대출 사기형"}
IDX_TO_LABEL = {idx: label for idx, label in enumerate(LABELS)}


class MeanPooling(nn.Module):
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1e-9)
        return summed / count


class RobertaAvgPoolFiveLabelModel(nn.Module):
    def __init__(self, encoder_name: str, head_hidden_dim: int = 64, head_dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.pooling = MeanPooling()
        self.aux_head = nn.Linear(768, 3)
        self.head = nn.Sequential(
            nn.Linear(768, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, len(LABELS)),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_mask: torch.Tensor,
        num_segments: torch.Tensor,
    ) -> dict:
        batch_size, segment_count, _ = input_ids.shape
        enc_dtype = next(self.encoder.parameters()).dtype
        device = input_ids.device
        seg_repr = torch.zeros(batch_size, segment_count, 768, device=device, dtype=enc_dtype)

        for idx in range(segment_count):
            valid = segment_mask[:, idx]
            if not valid.any():
                break
            out = self.encoder(
                input_ids=input_ids[:, idx, :][valid],
                attention_mask=attention_mask[:, idx, :][valid],
            )
            seg_repr[valid, idx] = self.pooling(out.last_hidden_state, attention_mask[:, idx, :][valid])

        seg_repr = seg_repr.float()
        aux_logits = self.aux_head(seg_repr)
        seg_mask = segment_mask.unsqueeze(-1).float()
        pooled = (seg_repr * seg_mask).sum(dim=1) / seg_mask.sum(dim=1).clamp(min=1)
        return {"logits": self.head(pooled), "aux_logits": aux_logits}


def _resolve_checkpoint(exp_dir: Path) -> str:
    explicit = os.getenv("FIVE_LABEL_MODEL_PATH")
    if explicit:
        return explicit

    meta_path = Path(os.getenv("FIVE_LABEL_META_PATH", exp_dir / "checkpoints" / "roberta_avgpool_latest.json"))
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        ckpt = meta.get("checkpoint_path")
        if ckpt and Path(ckpt).exists():
            return ckpt
        if ckpt:
            local = exp_dir / "checkpoints" / Path(ckpt).name
            if local.exists():
                return str(local)

    candidates = sorted((exp_dir / "checkpoints").glob("*.pt"))
    if candidates:
        return str(candidates[-1])
    raise FileNotFoundError(
        "5-label checkpoint not found. Set FIVE_LABEL_MODEL_PATH to the RoBERTa AvgPool .pt checkpoint."
    )


def _build_segments(tokenizer, text: str, window_size: int, stride: int) -> list[dict]:
    token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if not token_ids:
        return []

    content_size = window_size - 2
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    if cls_id is None or sep_id is None:
        raise ValueError("Tokenizer missing cls/sep ids")

    if len(token_ids) <= content_size:
        starts = [0]
    else:
        starts = list(range(0, len(token_ids) - content_size + 1, stride))
        last_start = len(token_ids) - content_size
        if starts[-1] != last_start:
            starts.append(last_start)

    segments = []
    for start in starts:
        body = token_ids[start : start + content_size]
        ids = [cls_id] + body + [sep_id]
        attn = [1] * len(ids)
        if len(ids) < window_size:
            pad_len = window_size - len(ids)
            ids.extend([pad_id] * pad_len)
            attn.extend([0] * pad_len)
        segments.append({"input_ids": ids, "attention_mask": attn})
    return segments


class FiveLabelVoicePhishingDetector:
    def __init__(self, model_path: str | None = None, device: str | None = None) -> None:
        base_dir = Path(__file__).resolve().parent
        self.exp_dir = base_dir / "experiments" / "roberta_avgpool"
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder_name = os.getenv("FIVE_LABEL_ENCODER_NAME", "klue/roberta-base")
        self.window_size = int(os.getenv("FIVE_LABEL_WINDOW_SIZE", "128"))
        self.stride = int(os.getenv("FIVE_LABEL_STRIDE", "100"))
        self.checkpoint_path = model_path or _resolve_checkpoint(self.exp_dir)

        print(f"--- [5-label Inference] Initializing RoBERTa AvgPool on {self.device} ---")
        self.processor = AudioProcessor(whisper_model_size=os.getenv("WHISPER_MODEL_SIZE", "Systran/faster-whisper-base"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
        self.model = RobertaAvgPoolFiveLabelModel(self.encoder_name).to(self.device)
        self._load_weights(self.checkpoint_path)
        self.model.eval()
        print("[5-label Inference] System Ready.")

    def _load_weights(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict") or checkpoint.get("model_state") or checkpoint
        else:
            state_dict = checkpoint
        cleaned = {}
        for key, value in state_dict.items():
            new_key = key.removeprefix("module.")
            cleaned[new_key] = value
        keys = self.model.load_state_dict(cleaned, strict=False)
        if keys.missing_keys:
            print(f"⚠️ [5-label] Missing keys: {len(keys.missing_keys)}")
        if keys.unexpected_keys:
            print(f"⚠️ [5-label] Unexpected keys: {len(keys.unexpected_keys)}")

    def _predict_text(self, text: str) -> dict:
        segments = _build_segments(self.tokenizer, text.strip(), self.window_size, self.stride)
        if not segments:
            raise ValueError("empty text after tokenization")

        input_ids = torch.tensor([[segment["input_ids"] for segment in segments]], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(
            [[segment["attention_mask"] for segment in segments]], dtype=torch.long, device=self.device
        )
        segment_mask = torch.ones((1, len(segments)), dtype=torch.bool, device=self.device)
        num_segments = torch.tensor([len(segments)], dtype=torch.long, device=self.device)

        start = time.time()
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    segment_mask=segment_mask,
                    num_segments=num_segments,
                )
                probs = F.softmax(out["logits"].float(), dim=-1)[0]
        print(f"   [Profile] 5-label NLP Inference: {(time.time() - start):.4f}s")

        pred_id = int(torch.argmax(probs).item())
        pred_label = IDX_TO_LABEL[pred_id]
        class_probs = {IDX_TO_LABEL[i]: float(probs[i].item()) for i in range(len(LABELS))}
        phishing_score = sum(class_probs[label] for label in PHISHING_LABELS) * 100.0

        return {
            "pred_label_id": pred_id,
            "pred_label": pred_label,
            "confidence": float(probs[pred_id].item()),
            "class_probs": class_probs,
            "is_phishing": pred_label in PHISHING_LABELS,
            "max_risk_score": phishing_score,
            "num_segments": len(segments),
        }

    def predict(self, audio_file_path: str, threshold: float = 0.5) -> dict:
        cleaned_sentences = self.processor.process_file(audio_file_path)
        if not cleaned_sentences:
            return {"status": "fail", "message": "No text detected"}

        text = " ".join(cleaned_sentences)
        result = self._predict_text(text)
        result["status"] = "success"
        result["dangerous_segment"] = text
        result["is_phishing"] = result["is_phishing"] or result["max_risk_score"] >= (threshold * 100)
        result["checkpoint_path"] = self.checkpoint_path
        return result

"""
koelectra_mamba_phishing6_hce / inference.py

phishing5 와 동일한 인터페이스. 
checkpoint 이름만 koelectra_mamba_phishing6_hce_* 로 변경됩니다.

aux 관련 출력(segment 위험도 확률)을 선택적으로 반환합니다.
"""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from config import (
    CHECKPOINT_DIR,
    DEVICE,
    ENCODER_CONFIG,
    IDX_TO_LABEL,
    STRIDE,
    WINDOW_SIZE,
)
from model import build_model

EXP_NAME = "koelectra_mamba_hce_ordinal"


def build_segments(tokenizer, text: str):
    token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if not token_ids:
        return []

    content = WINDOW_SIZE - 2
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    if cls_id is None or sep_id is None:
        raise ValueError("Tokenizer missing cls/sep ids")

    if len(token_ids) <= content:
        starts = [0]
    else:
        starts = list(range(0, len(token_ids) - content + 1, STRIDE))
        last = len(token_ids) - content
        if starts[-1] != last:
            starts.append(last)

    segments = []
    for s in starts:
        body = token_ids[s : s + content]
        ids = [cls_id] + body + [sep_id]
        attn = [1] * len(ids)
        if len(ids) < WINDOW_SIZE:
            pad_len = WINDOW_SIZE - len(ids)
            ids.extend([pad_id] * pad_len)
            attn.extend([0] * pad_len)
        segments.append((ids, attn))
    return segments


class PhishingInfer:
    """
    koelectra_mamba_phishing6_hce 추론 클래스.

    predict_text() 반환 dict:
      pred_label_id    : int
      pred_label       : str (5-class)
      confidence       : float
      probs            : dict[str, float]  5-class 확률
      super_label      : str  ("일반" / "피싱")
      super_probs      : dict[str, float]
      segment_risks    : list[dict]  세그먼트별 위험도 (0~2)
      num_segments     : int
      checkpoint_path  : str
    """

    RISK_LEVELS = ["안전", "의심", "위험"]

    def __init__(self, checkpoint_path: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(ENCODER_CONFIG["MODEL_NAME"])
        self.model = build_model().to(DEVICE)
        ckpt = checkpoint_path or self._resolve_latest_checkpoint()
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.checkpoint_path = ckpt

    def _resolve_latest_checkpoint(self) -> str:
        meta_path = CHECKPOINT_DIR / f"{EXP_NAME}_latest.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"latest checkpoint meta not found: {meta_path}")
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        ckpt = meta.get("checkpoint_path")
        if not ckpt or not Path(ckpt).exists():
            raise FileNotFoundError(f"checkpoint_path not found in meta: {ckpt}")
        return ckpt

    @torch.no_grad()
    def predict_text(self, text: str) -> dict:
        segments = build_segments(self.tokenizer, text.strip())
        if not segments:
            raise ValueError("empty text after tokenization")

        input_ids = torch.tensor([[s[0] for s in segments]], dtype=torch.long, device=DEVICE)
        attention_mask = torch.tensor([[s[1] for s in segments]], dtype=torch.long, device=DEVICE)
        segment_mask = torch.ones((1, len(segments)), dtype=torch.bool, device=DEVICE)
        num_segments = torch.tensor([len(segments)], dtype=torch.long, device=DEVICE)

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            segment_mask=segment_mask,
            num_segments=num_segments,
        )

        # ── 5-class 최종 예측 ────────────────────────────────────────────
        probs_5 = torch.exp(out["logits"])[0]     # (5,)  log-prob → prob
        pred_5 = int(probs_5.argmax().item())

        # ── 상위 클래스 ─────────────────────────────────────────────────
        super_probs = torch.softmax(out["super_logits"][0], dim=-1)   # (2,)
        super_label = "피싱" if super_probs[1] > super_probs[0] else "일반"

        # ── 세그먼트별 위험도 (ordinal) ──────────────────────────────────
        # aux_logits: (1, T, 2)  → 이진 시그모이드로 누적 확률
        aux = torch.sigmoid(out["aux_logits"][0])                      # (T, 2)
        # P(y=0) = 1-σ(f0),  P(y=1) = σ(f0)-σ(f1),  P(y=2) = σ(f1)
        p0 = 1.0 - aux[:, 0]
        p1 = aux[:, 0] - aux[:, 1]
        p2 = aux[:, 1]
        risk_probs = torch.stack([p0, p1, p2], dim=-1).clamp(min=0)  # (T, 3)
        risk_probs = risk_probs / risk_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        segment_risks = []
        for t in range(len(segments)):
            rp = risk_probs[t]
            risk_idx = int(rp.argmax().item())
            segment_risks.append({
                "segment": t,
                "risk_label": self.RISK_LEVELS[risk_idx],
                "risk_id": risk_idx,
                "risk_probs": {self.RISK_LEVELS[i]: float(rp[i].item()) for i in range(3)},
            })

        return {
            "pred_label_id": pred_5,
            "pred_label": IDX_TO_LABEL[pred_5],
            "confidence": float(probs_5[pred_5].item()),
            "probs": {IDX_TO_LABEL[i]: float(probs_5[i].item()) for i in range(len(probs_5))},
            "super_label": super_label,
            "super_probs": {
                "일반": float(super_probs[0].item()),
                "피싱": float(super_probs[1].item()),
            },
            "segment_risks": segment_risks,
            "num_segments": len(segments),
            "checkpoint_path": str(self.checkpoint_path),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=f"{EXP_NAME} inference")
    parser.add_argument("--text", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    infer = PhishingInfer(checkpoint_path=args.checkpoint)
    result = infer.predict_text(args.text)
    print(json.dumps(result, ensure_ascii=False, indent=2))

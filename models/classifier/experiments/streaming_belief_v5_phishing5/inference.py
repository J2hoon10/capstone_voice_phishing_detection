import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from config import (
    CHECKPOINT_DIR,
    DEVICE,
    ENCODER_CONFIG,
    IDX_TO_LABEL,
    LABEL_TO_IDX,
    STRIDE,
    WINDOW_SIZE,
)
from model import build_model


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


class StreamingBeliefV5Phishing5Infer:
    def __init__(self, checkpoint_path: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(ENCODER_CONFIG["MODEL_NAME"])
        self.model = build_model().to(DEVICE)
        ckpt = checkpoint_path or self._resolve_latest_checkpoint()
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.checkpoint_path = ckpt

    def _resolve_latest_checkpoint(self) -> str:
        meta_path = CHECKPOINT_DIR / "streaming_belief_v5_phishing5_latest.json"
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
        probs = torch.softmax(out["logits"], dim=-1)[0]
        pred = int(torch.argmax(probs).item())
        return {
            "pred_label_id": pred,
            "pred_label": IDX_TO_LABEL[pred],
            "confidence": float(probs[pred].item()),
            "probs": {IDX_TO_LABEL[i]: float(probs[i].item()) for i in range(len(probs))},
            "num_segments": len(segments),
            "checkpoint_path": self.checkpoint_path,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Streaming Belief v5 (5-class) inference")
    parser.add_argument("--text", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    infer = StreamingBeliefV5Phishing5Infer(checkpoint_path=args.checkpoint)
    result = infer.predict_text(args.text)
    print(json.dumps(result, ensure_ascii=False, indent=2))


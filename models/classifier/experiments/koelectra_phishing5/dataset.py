import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from config import ENCODER_CONFIG, LABEL_TO_IDX, STRIDE, WINDOW_SIZE


def merge_phishing_category(category: str) -> str:
    if category == "바로 이 목소리":
        return "수사기관 사칭형"
    return category


def build_segments(tokenizer, text: str) -> list[dict]:
    token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if not token_ids:
        return []

    content_size = WINDOW_SIZE - 2
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    if cls_id is None or sep_id is None:
        raise ValueError("Tokenizer missing cls/sep ids")

    if len(token_ids) <= content_size:
        starts = [0]
    else:
        starts = list(range(0, len(token_ids) - content_size + 1, STRIDE))
        last_start = len(token_ids) - content_size
        if starts[-1] != last_start:
            starts.append(last_start)

    segments = []
    for start in starts:
        body = token_ids[start : start + content_size]
        ids = [cls_id] + body + [sep_id]
        attn = [1] * len(ids)
        if len(ids) < WINDOW_SIZE:
            pad_len = WINDOW_SIZE - len(ids)
            ids.extend([pad_id] * pad_len)
            attn.extend([0] * pad_len)
        segments.append({"input_ids": ids, "attention_mask": attn})
    return segments


class CsvStreamingDataset(Dataset):
    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(ENCODER_CONFIG["MODEL_NAME"])
        self.rows = []
        with self.csv_path.open("r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                category = merge_phishing_category((row.get("category") or "").strip())
                text = (row.get("text") or "").strip()
                if category not in LABEL_TO_IDX or not text:
                    continue
                segs = build_segments(self.tokenizer, text)
                if not segs:
                    continue
                self.rows.append(
                    {
                        "input_ids": torch.tensor([s["input_ids"] for s in segs], dtype=torch.long),
                        "attention_mask": torch.tensor([s["attention_mask"] for s in segs], dtype=torch.long),
                        "num_segments": len(segs),
                        "label": LABEL_TO_IDX[category],
                    }
                )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        return self.rows[idx]


def collate_fn(batch):
    bsz = len(batch)
    max_s = max(x["num_segments"] for x in batch)
    seq_len = batch[0]["input_ids"].shape[1]

    input_ids = torch.zeros((bsz, max_s, seq_len), dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_s, seq_len), dtype=torch.long)
    segment_mask = torch.zeros((bsz, max_s), dtype=torch.bool)
    num_segments = torch.tensor([x["num_segments"] for x in batch], dtype=torch.long)
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)

    for i, item in enumerate(batch):
        n = item["num_segments"]
        input_ids[i, :n] = item["input_ids"]
        attention_mask[i, :n] = item["attention_mask"]
        segment_mask[i, :n] = True

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "segment_mask": segment_mask,
        "num_segments": num_segments,
        "labels": labels,
    }


def create_dataloader(csv_path: str | Path, batch_size: int, shuffle: bool) -> DataLoader:
    ds = CsvStreamingDataset(csv_path)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


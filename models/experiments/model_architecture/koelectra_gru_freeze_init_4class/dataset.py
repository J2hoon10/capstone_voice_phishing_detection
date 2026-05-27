"""
roberta_gru_freeze_init_4class / dataset.py

roberta_mamba_freeze_init_4class 와 동일한 데이터셋 구성.
  - WINDOW_SIZE=64, STRIDE=32
  - segment_risks: 4class CSV의 segment_risks 컬럼에서 직접 로드
"""

import csv
import json
import logging
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from config import ENCODER_CONFIG, LABEL_TO_IDX, MAX_SEQ_LEN, STRIDE, WINDOW_SIZE


def merge_phishing_category(category: str) -> str:
    if category == "바로 이 목소리":
        return "수사기관 사칭형"
    return category


def _sentence_token_ranges(text: str, offsets: list) -> list[tuple[int, int]]:
    """문장 끝 부호(.?!\n)를 기준으로 토큰 인덱스 범위의 문장 목록을 반환."""
    n = len(offsets)
    if n == 0:
        return []
    boundary_ends = {m.end() for m in re.finditer(r'[.?!\n]', text)}
    sentences: list[tuple[int, int]] = []
    s_start = 0
    for i, (c0, c1) in enumerate(offsets):
        if any(c0 < b <= c1 for b in boundary_ends):
            sentences.append((s_start, i + 1))
            s_start = i + 1
    if s_start < n:
        sentences.append((s_start, n))
    return sentences


def _is_word_start(offsets: list, text: str, i: int) -> bool:
    """토큰 i가 새 단어의 시작이면 True (앞에 공백이 있거나 첫 토큰)."""
    if i == 0:
        return True
    c0 = offsets[i][0]
    return c0 == 0 or text[c0 - 1] in ' \t\n\r'


def build_segments(tokenizer, text: str) -> list[dict]:
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False,
    )
    token_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    n = len(token_ids)
    if not token_ids:
        return []

    content_size = WINDOW_SIZE - 2
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    if cls_id is None or sep_id is None:
        raise ValueError("Tokenizer missing cls/sep ids")

    sentences = _sentence_token_ranges(text, offsets)
    token_to_sent = [0] * n
    for si, (ss, se) in enumerate(sentences):
        for ti in range(ss, se):
            token_to_sent[ti] = si

    if n <= content_size:
        starts = [0]
    else:
        starts = list(range(0, n - content_size + 1, STRIDE))
        last_start = n - content_size
        if starts[-1] != last_start:
            starts.append(last_start)

    segments = []
    for start in starts:
        end = min(start + content_size, n)

        # ── 문장 경계 조정 ──────────────────────────────────────────────
        if end < n:
            si = token_to_sent[end - 1]
            ss, se = sentences[si]
            if se > end:
                cut_off = se - end
                sent_len = se - ss
                if cut_off * 2 > sent_len:
                    new_end = ss
                    if new_end > start:
                        end = new_end

        # ── 단어 경계 조정 ──────────────────────────────────────────────
        if end < n and not _is_word_start(offsets, text, end):
            while end < n and not _is_word_start(offsets, text, end):
                end += 1

        if end <= start:
            continue

        body = token_ids[start:end]
        ids = [cls_id] + body + [sep_id]

        if len(ids) > MAX_SEQ_LEN:
            ids = ids[:MAX_SEQ_LEN - 1] + [sep_id]
        attn = [1] * len(ids)
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

                segment_risks_str = row.get("segment_risks")
                if segment_risks_str:
                    risks_raw = json.loads(segment_risks_str)
                    raw = [r - 1 for r in risks_raw]
                    if len(raw) >= len(segs):
                        mapped_risks = raw[: len(segs)]
                    else:
                        mapped_risks = raw + [0] * (len(segs) - len(raw))
                else:
                    mapped_risks = [0] * len(segs)

                self.rows.append(
                    {
                        "input_ids": [torch.tensor(s["input_ids"], dtype=torch.long) for s in segs],
                        "attention_mask": [torch.tensor(s["attention_mask"], dtype=torch.long) for s in segs],
                        "num_segments": len(segs),
                        "label": LABEL_TO_IDX[category],
                        "segment_risks": torch.tensor(mapped_risks, dtype=torch.long),
                    }
                )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        return self.rows[idx]


def collate_fn(batch):
    bsz = len(batch)
    max_s = max(x["num_segments"] for x in batch)
    max_seq_len = max(seg.shape[0] for x in batch for seg in x["input_ids"])

    input_ids = torch.zeros((bsz, max_s, max_seq_len), dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_s, max_seq_len), dtype=torch.long)
    segment_mask = torch.zeros((bsz, max_s), dtype=torch.bool)
    num_segments = torch.tensor([x["num_segments"] for x in batch], dtype=torch.long)
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
    segment_risks = torch.full((bsz, max_s), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        n = item["num_segments"]
        for j in range(n):
            seg_len = item["input_ids"][j].shape[0]
            input_ids[i, j, :seg_len] = item["input_ids"][j]
            attention_mask[i, j, :seg_len] = item["attention_mask"][j]
        segment_mask[i, :n] = True
        segment_risks[i, :n] = item["segment_risks"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "segment_mask": segment_mask,
        "num_segments": num_segments,
        "labels": labels,
        "segment_risks": segment_risks,
    }


def create_dataloader(csv_path: str | Path, batch_size: int, shuffle: bool) -> DataLoader:
    ds = CsvStreamingDataset(csv_path)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

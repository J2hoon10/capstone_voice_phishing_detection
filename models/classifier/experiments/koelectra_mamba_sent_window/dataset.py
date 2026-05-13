"""
koelectra_mamba_sent_window / dataset.py

hce_ordinal 대비 핵심 변경:
  build_segments_from_sentences() — 새 분할 함수
    - phishing_labeled.csv 의 문장 단위 위험도 데이터를 직접 사용
    - regex 기반 text 분리 대신 사전 레이블링된 문장 목록 활용
    - 각 그룹의 segment_risk = max(문장 위험도) - 1  (0-indexed)
    - 피싱 키워드가 세그먼트 경계에서 잘리는 현상 방지
    - 위험도 레이블이 실제 문장 경계에 정렬됨
"""

import csv
import logging
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from config import ENCODER_CONFIG, LABEL_TO_IDX, PHISHING_LABELED_CSV, WINDOW_SIZE


def merge_phishing_category(category: str) -> str:
    if category == "바로 이 목소리":
        return "수사기관 사칭형"
    return category


def load_phishing_labeled(csv_path: Path) -> dict[str, list[tuple[str, int]]]:
    """
    phishing_labeled.csv → {conv_id: [(sentence_text, risk_1indexed), ...]}
    sentence_idx 기준으로 정렬하여 반환.
    파일이 없으면 빈 dict 반환.
    """
    if not csv_path.exists():
        return {}
    conv_data: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            conv_id = row["id"]
            idx = int(row["sentence_idx"])
            text = row["sentence"]
            risk = int(row["risk"])
            conv_data[conv_id].append((idx, text, risk))

    result: dict[str, list[tuple[str, int]]] = {}
    for conv_id, entries in conv_data.items():
        entries.sort(key=lambda x: x[0])
        result[conv_id] = [(text, risk) for _, text, risk in entries]
    return result


def build_segments_from_sentences(
    tokenizer,
    sentences: list[tuple[str, int]],  # (sentence_text, risk_1indexed)
) -> tuple[list[dict], list[int]]:
    """
    phishing_labeled.csv 의 사전 분리된 문장들을 WINDOW_SIZE 이하로 그룹화.
    각 그룹의 risk = max(문장 위험도) - 1  (0-indexed: 0=안전, 1=의심, 2=위험)
    Returns: (segments, segment_risks)
    """
    content_size = WINDOW_SIZE - 2
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    if cls_id is None or sep_id is None:
        raise ValueError("Tokenizer missing cls/sep ids")

    # 문장별 토큰화
    sent_data: list[tuple[list[int], int]] = []
    for sent_text, risk in sentences:
        ids = tokenizer(
            sent_text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            verbose=False,
        )["input_ids"]
        if ids:
            sent_data.append((ids, risk))

    if not sent_data:
        return [], []

    # WINDOW_SIZE 이하로 문장 그룹화
    groups: list[tuple[list[int], int]] = []  # (token_ids, max_risk_1indexed)
    current_tokens: list[int] = []
    current_risk = 1

    for tokens, risk in sent_data:
        if len(tokens) > content_size:
            # 단일 문장이 content_size 초과: 현재 그룹을 먼저 닫고 강제 분할
            if current_tokens:
                groups.append((current_tokens, current_risk))
                current_tokens = []
                current_risk = 1
            for i in range(0, len(tokens), content_size):
                groups.append((tokens[i : i + content_size], risk))
        elif len(current_tokens) + len(tokens) > content_size:
            if current_tokens:
                groups.append((current_tokens, current_risk))
            current_tokens = list(tokens)
            current_risk = risk
        else:
            current_tokens.extend(tokens)
            current_risk = max(current_risk, risk)

    if current_tokens:
        groups.append((current_tokens, current_risk))

    if not groups:
        return [], []

    # 그룹 → segment dict 변환
    segments: list[dict] = []
    segment_risks: list[int] = []
    for body, seg_risk in groups:
        body = body[:content_size]
        ids = [cls_id] + body + [sep_id]
        attn = [1] * len(ids)
        if len(ids) < WINDOW_SIZE:
            pad_len = WINDOW_SIZE - len(ids)
            ids.extend([pad_id] * pad_len)
            attn.extend([0] * pad_len)
        segments.append({"input_ids": ids, "attention_mask": attn})
        segment_risks.append(seg_risk - 1)  # 1-indexed → 0-indexed

    return segments, segment_risks


def build_segments_fallback(tokenizer, text: str) -> tuple[list[dict], list[int]]:
    """phishing_labeled.csv에 id가 없는 경우 regex 분할 fallback."""
    import re

    content_size = WINDOW_SIZE - 2
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    if cls_id is None or sep_id is None:
        raise ValueError("Tokenizer missing cls/sep ids")

    raw_sentences = re.split(r"(?<=[.?!\n])\s*", text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    if not sentences:
        return [], []

    sent_token_ids: list[list[int]] = []
    for sent in sentences:
        ids = tokenizer(
            sent,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            verbose=False,
        )["input_ids"]
        if ids:
            sent_token_ids.append(ids)

    if not sent_token_ids:
        return [], []

    groups: list[list[int]] = []
    current_tokens: list[int] = []
    for tokens in sent_token_ids:
        if len(tokens) > content_size:
            if current_tokens:
                groups.append(current_tokens)
                current_tokens = []
            for i in range(0, len(tokens), content_size):
                groups.append(tokens[i : i + content_size])
        elif len(current_tokens) + len(tokens) > content_size:
            if current_tokens:
                groups.append(current_tokens)
            current_tokens = list(tokens)
        else:
            current_tokens.extend(tokens)
    if current_tokens:
        groups.append(current_tokens)

    segments = []
    for body in groups:
        body = body[:content_size]
        ids = [cls_id] + body + [sep_id]
        attn = [1] * len(ids)
        if len(ids) < WINDOW_SIZE:
            pad_len = WINDOW_SIZE - len(ids)
            ids.extend([pad_id] * pad_len)
            attn.extend([0] * pad_len)
        segments.append({"input_ids": ids, "attention_mask": attn})

    return segments, [0] * len(segments)


class CsvStreamingDataset(Dataset):
    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(ENCODER_CONFIG["MODEL_NAME"])
        stn_lookup = load_phishing_labeled(PHISHING_LABELED_CSV)
        if not stn_lookup:
            print(
                f"[warn] {PHISHING_LABELED_CSV} 없음 → segment_risks 전부 0으로 대체됩니다."
            )
        self.rows = []

        with self.csv_path.open("r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                category = merge_phishing_category((row.get("category") or "").strip())
                text = (row.get("text") or "").strip()
                if category not in LABEL_TO_IDX or not text:
                    continue

                conv_id = (row.get("id") or "").strip()
                if conv_id in stn_lookup:
                    segs, mapped_risks = build_segments_from_sentences(
                        self.tokenizer, stn_lookup[conv_id]
                    )
                else:
                    segs, mapped_risks = build_segments_fallback(self.tokenizer, text)

                if not segs:
                    continue

                self.rows.append(
                    {
                        "input_ids": torch.tensor(
                            [s["input_ids"] for s in segs], dtype=torch.long
                        ),
                        "attention_mask": torch.tensor(
                            [s["attention_mask"] for s in segs], dtype=torch.long
                        ),
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
    seq_len = batch[0]["input_ids"].shape[1]

    input_ids = torch.zeros((bsz, max_s, seq_len), dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_s, seq_len), dtype=torch.long)
    segment_mask = torch.zeros((bsz, max_s), dtype=torch.bool)
    num_segments = torch.tensor([x["num_segments"] for x in batch], dtype=torch.long)
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
    segment_risks = torch.full((bsz, max_s), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        n = item["num_segments"]
        input_ids[i, :n] = item["input_ids"]
        attention_mask[i, :n] = item["attention_mask"]
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

"""
SNS 대화 Dataset 및 DataLoader (Mamba Belief 실험용)

mamba_belief model.py 의 forward() 인터페이스에 맞춰
segment_mask / num_segments / labels 키를 반환한다.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from config import ENCODER_CONFIG, GPU_CONFIG


class SNSDialogueDataset(Dataset):
    """
    대화 단위 데이터셋.

    각 샘플은 하나의 대화로, 가변 개수의 청크(발화)와 단일 클래스 레이블을 가진다.

    JSON 포맷 (data_preprocessing.py 출력):
        {
            "dialogue_label": int,   # 0~19 클래스 인덱스
            "num_turns": int,
            "turns": [{"input_ids": [...], "attention_mask": [...]}, ...]
        }
    """

    def __init__(self, data_path: str | Path, max_segments: int | None = None):
        data_path = Path(data_path)
        with data_path.open("r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.max_segments = max_segments

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        turns = item["turns"]
        if self.max_segments is not None:
            turns = turns[: self.max_segments]

        input_ids = torch.tensor([t["input_ids"] for t in turns], dtype=torch.long)
        attention_mask = torch.tensor(
            [t["attention_mask"] for t in turns], dtype=torch.long
        )
        num_segments = len(turns)

        return {
            "input_ids": input_ids,             # (num_segments, seq_len)
            "attention_mask": attention_mask,    # (num_segments, seq_len)
            "num_segments": num_segments,        # int
            "label": int(item["dialogue_label"]),  # 0~19
            "dialogue_id": item["dialogue_id"],
        }

    def get_num_segments_list(self) -> list[int]:
        if self.max_segments is None:
            return [int(d["num_turns"]) for d in self.data]
        return [min(int(d["num_turns"]), self.max_segments) for d in self.data]


class BucketBatchSampler(Sampler[list[int]]):
    """
    길이 인식 배치 샘플러.

    비슷한 청크 수의 샘플끼리 묶어 패딩 낭비를 최소화한다.
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        bucket_size_multiplier: int = 20,
        seed: int = 42,
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.bucket_size = max(batch_size, batch_size * bucket_size_multiplier)
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _ordered_indices(self) -> list[int]:
        n = len(self.lengths)
        indices = list(range(n))
        if not self.shuffle:
            return sorted(indices, key=lambda i: self.lengths[i])

        rng = random.Random(self.seed + self._epoch)
        rng.shuffle(indices)
        buckets = [
            indices[i : i + self.bucket_size]
            for i in range(0, n, self.bucket_size)
        ]
        for bucket in buckets:
            bucket.sort(key=lambda i: self.lengths[i])

        return [idx for bucket in buckets for idx in bucket]

    def __iter__(self):
        ordered = self._ordered_indices()
        batches = [
            ordered[i : i + self.batch_size]
            for i in range(0, len(ordered), self.batch_size)
        ]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]

        if self.shuffle:
            rng = random.Random(self.seed + 1000 + self._epoch)
            rng.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return math.ceil(len(self.lengths) / self.batch_size)


def collate_fn(batch: list[dict]) -> dict:
    """
    가변 청크 수를 가진 배치를 패딩하여 텐서로 통일한다.

    Returns:
        dict:
            - input_ids:     (B, max_S, L)
            - attention_mask:(B, max_S, L)
            - segment_mask:  (B, max_S)  실제 청크=True, 패딩=False
            - num_segments:  (B,)        각 대화의 실제 청크 수
            - labels:        (B,)        long, 클래스 인덱스 (0~19)
            - dialogue_ids:  list[str]
    """
    max_num_seg = max(item["num_segments"] for item in batch)
    seq_len = batch[0]["input_ids"].shape[1]
    batch_size = len(batch)
    pad_token_id = ENCODER_CONFIG["PAD_TOKEN_ID"]

    input_ids = torch.full(
        (batch_size, max_num_seg, seq_len),
        fill_value=pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros(batch_size, max_num_seg, seq_len, dtype=torch.long)
    segment_mask = torch.zeros(batch_size, max_num_seg, dtype=torch.bool)
    num_segments = torch.tensor(
        [item["num_segments"] for item in batch], dtype=torch.long
    )
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    dialogue_ids = [item["dialogue_id"] for item in batch]

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
        "dialogue_ids": dialogue_ids,
    }


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int = 8,
    max_segments: int | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, DataLoader]:
    """
    train/val/test DataLoader를 생성한다.

    Args:
        data_dir: 전처리된 JSON 캐시 파일 디렉토리
        batch_size: 배치 크기
        max_segments: 최대 청크 수 제한 (None이면 무제한)
        num_workers: DataLoader worker 수 (None이면 GPU_CONFIG 사용)
        pin_memory: pin_memory 여부 (None이면 GPU_CONFIG 사용)
        splits: 로드할 split 목록

    Returns:
        dict: {split: DataLoader}
    """
    if num_workers is None:
        num_workers = GPU_CONFIG["NUM_WORKERS"]
    if pin_memory is None:
        pin_memory = GPU_CONFIG["PIN_MEMORY"] and torch.cuda.is_available()

    data_dir = Path(data_dir)
    loaders: dict[str, DataLoader] = {}

    for split in splits:
        json_path = data_dir / f"{split}.json"
        if not json_path.exists():
            print(f"[warn] missing {json_path}, skip {split} loader")
            continue

        dataset = SNSDialogueDataset(json_path, max_segments=max_segments)
        lengths = dataset.get_num_segments_list()
        shuffle = split == "train"
        sampler = BucketBatchSampler(
            lengths=lengths,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )
        loaders[split] = loader
        print(
            f"[dataloader] {split}: {len(dataset)} dialogues, "
            f"batch={batch_size}, workers={num_workers}, bucketed=True"
        )

    return loaders

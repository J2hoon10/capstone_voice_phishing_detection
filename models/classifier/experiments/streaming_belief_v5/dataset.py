"""
Streaming Belief v5 Dataset / DataLoader.

model.py forward 인터페이스와 맞추어 아래 키를 반환한다.
- input_ids, attention_mask, segment_mask, num_segments, labels
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from config import ENCODER_CONFIG, GPU_CONFIG


class StreamingBeliefDataset(Dataset):
    """
    샘플 단위 데이터셋.

    Supports both schemas:
    1) streaming_v5 schema:
    {
      "sample_id": str,
      "label": int,
      "num_segments": int,
      "segments": [{"input_ids": [...], "attention_mask": [...]}, ...]
    }
    2) mamba_belief schema:
    {
      "dialogue_id": str,
      "dialogue_label": int,
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
        segments = item.get("segments")
        if segments is None:
            segments = item.get("turns", [])
        if self.max_segments is not None:
            segments = segments[: self.max_segments]

        input_ids = torch.tensor([s["input_ids"] for s in segments], dtype=torch.long)
        attention_mask = torch.tensor([s["attention_mask"] for s in segments], dtype=torch.long)
        num_segments = len(segments)

        label = item.get("label")
        if label is None:
            label = item.get("dialogue_label", -1)

        sample_id = item.get("sample_id")
        if sample_id is None:
            sample_id = item.get("dialogue_id", f"sample_{idx}")

        return {
            "input_ids": input_ids,  # (S, L)
            "attention_mask": attention_mask,  # (S, L)
            "num_segments": num_segments,
            "label": int(label),
            "sample_id": sample_id,
        }

    def get_num_segments_list(self) -> list[int]:
        lengths: list[int] = []
        for item in self.data:
            base_len = item.get("num_segments")
            if base_len is None:
                base_len = item.get("num_turns")
            if base_len is None:
                segs = item.get("segments")
                if segs is None:
                    segs = item.get("turns", [])
                base_len = len(segs)
            base_len = int(base_len)
            if self.max_segments is not None:
                base_len = min(base_len, self.max_segments)
            lengths.append(base_len)
        return lengths


class BucketBatchSampler(Sampler[list[int]]):
    """
    길이 기반 배치 샘플러.
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
        buckets = [indices[i : i + self.bucket_size] for i in range(0, n, self.bucket_size)]
        for bucket in buckets:
            bucket.sort(key=lambda i: self.lengths[i])
        return [idx for bucket in buckets for idx in bucket]

    def __iter__(self):
        ordered = self._ordered_indices()
        batches = [ordered[i : i + self.batch_size] for i in range(0, len(ordered), self.batch_size)]
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
    가변 길이 세그먼트 시퀀스를 (B, max_S, L) 텐서로 패딩한다.
    """
    batch_size = len(batch)
    max_num_seg = max(item["num_segments"] for item in batch)
    seq_len = batch[0]["input_ids"].shape[1]
    pad_token_id = ENCODER_CONFIG["PAD_TOKEN_ID"]

    input_ids = torch.full(
        (batch_size, max_num_seg, seq_len),
        fill_value=pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros(batch_size, max_num_seg, seq_len, dtype=torch.long)
    segment_mask = torch.zeros(batch_size, max_num_seg, dtype=torch.bool)
    num_segments = torch.tensor([item["num_segments"] for item in batch], dtype=torch.long)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    sample_ids = [item["sample_id"] for item in batch]

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
        "sample_ids": sample_ids,
    }


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int = 8,
    max_segments: int | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, DataLoader]:
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

        dataset = StreamingBeliefDataset(json_path, max_segments=max_segments)
        lengths = dataset.get_num_segments_list()
        sampler = BucketBatchSampler(
            lengths=lengths,
            batch_size=batch_size,
            shuffle=(split == "train"),
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
            f"[dataloader] {split}: {len(dataset)} samples, "
            f"batch={batch_size}, workers={num_workers}, bucketed=True"
        )

    return loaders

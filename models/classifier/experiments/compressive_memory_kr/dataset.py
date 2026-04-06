"""
Dataset and dataloader utilities for SNS dialogue-level classification.
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
    Dialogue-level dataset.

    Each sample is one dialogue:
    - turns: tokenized utterances
    - dialogue_label: single class index
    """

    def __init__(self, data_path: str | Path, max_turns: int | None = None):
        data_path = Path(data_path)
        with data_path.open("r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.max_turns = max_turns

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        turns = item["turns"]
        if self.max_turns is not None:
            turns = turns[: self.max_turns]

        input_ids = torch.tensor([t["input_ids"] for t in turns], dtype=torch.long)
        attention_mask = torch.tensor([t["attention_mask"] for t in turns], dtype=torch.long)
        num_turns = len(turns)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "num_turns": num_turns,
            "dialogue_label": int(item["dialogue_label"]),
            "dialogue_id": item["dialogue_id"],
        }

    def get_num_turns_list(self) -> list[int]:
        if self.max_turns is None:
            return [int(d["num_turns"]) for d in self.data]
        return [min(int(d["num_turns"]), self.max_turns) for d in self.data]


class BucketBatchSampler(Sampler[list[int]]):
    """
    Length-aware batch sampler.

    Samples are grouped by similar number of turns to reduce turn-level padding.
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

        ordered = [idx for bucket in buckets for idx in bucket]
        return ordered

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
    Collate variable-turn dialogues into padded batch tensors.
    """
    max_num_turns = max(item["num_turns"] for item in batch)
    seq_len = batch[0]["input_ids"].shape[1]
    batch_size = len(batch)
    pad_token_id = ENCODER_CONFIG["PAD_TOKEN_ID"]

    input_ids = torch.full(
        (batch_size, max_num_turns, seq_len),
        fill_value=pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros(batch_size, max_num_turns, seq_len, dtype=torch.long)
    turn_mask = torch.zeros(batch_size, max_num_turns, dtype=torch.bool)
    num_turns = torch.tensor([item["num_turns"] for item in batch], dtype=torch.long)
    dialogue_labels = torch.tensor([item["dialogue_label"] for item in batch], dtype=torch.long)
    dialogue_ids = [item["dialogue_id"] for item in batch]

    for i, item in enumerate(batch):
        n = item["num_turns"]
        input_ids[i, :n] = item["input_ids"]
        attention_mask[i, :n] = item["attention_mask"]
        turn_mask[i, :n] = True

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "turn_mask": turn_mask,
        "num_turns": num_turns,
        "dialogue_labels": dialogue_labels,
        "dialogue_ids": dialogue_ids,
    }


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int = 8,
    max_turns: int | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, DataLoader]:
    """
    Create train/val/test dataloaders from preprocessed JSON cache.
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

        dataset = SNSDialogueDataset(json_path, max_turns=max_turns)
        lengths = dataset.get_num_turns_list()
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

"""
NSMC (Naver Sentiment Movie Corpus) dataset loading.

datasets 3.x에서 레거시 dataset script가 제거됨에 따라
GitHub 원본 TSV 파일을 직접 다운로드하여 로드합니다.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from config import DATA_CONFIG, DATA_DIR, ENCODER_CONFIG, GPU_CONFIG

# NSMC raw files from official GitHub repo
_NSMC_URLS = {
    "train": "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt",
    "test":  "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt",
}
_NSMC_LOCAL = {
    "train": DATA_DIR / "ratings_train.txt",
    "test":  DATA_DIR / "ratings_test.txt",
}


def _ensure_nsmc_downloaded() -> None:
    """Download NSMC TSV files if not already cached locally."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split, url in _NSMC_URLS.items():
        local_path = _NSMC_LOCAL[split]
        if not local_path.exists():
            print(f"[dataset] Downloading NSMC {split} split from GitHub...")
            urllib.request.urlretrieve(url, local_path)
            print(f"[dataset] Saved to {local_path}")
        else:
            print(f"[dataset] Using cached {split} split: {local_path}")


def _load_nsmc_hf():
    """
    Load NSMC via HuggingFace datasets.

    datasets 3.x는 dataset script를 지원하지 않으므로
    GitHub 원본 TSV 파일을 직접 다운로드하여 CSV(탭 구분) 로더로 읽습니다.
    """
    _ensure_nsmc_downloaded()

    dataset = load_dataset(
        "csv",
        data_files={
            "train": str(_NSMC_LOCAL["train"]),
            "test":  str(_NSMC_LOCAL["test"]),
        },
        delimiter="\t",
    )
    return dataset


class NSMCDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Filter out samples with None/empty text
        self.data = [
            sample for sample in hf_dataset
            if sample[DATA_CONFIG["TEXT_COLUMN"]] is not None
            and str(sample[DATA_CONFIG["TEXT_COLUMN"]]).strip()
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]
        text = str(sample[DATA_CONFIG["TEXT_COLUMN"]])
        label = int(sample[DATA_CONFIG["LABEL_COLUMN"]])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get(
                "token_type_ids",
                torch.zeros(self.max_length, dtype=torch.long),
            ).squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_nsmc_datasets() -> tuple:
    """Load NSMC train and test splits."""
    dataset = _load_nsmc_hf()
    return dataset["train"], dataset["test"]


def create_dataloaders(
    batch_size: int | None = None,
    num_workers: int | None = None,
    splits: tuple[str, ...] = ("train", "test"),
) -> dict[str, DataLoader]:
    """Create DataLoaders for requested splits."""
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_CONFIG["MODEL_NAME"])
    dataset = _load_nsmc_hf()

    _batch_size = batch_size or DATA_CONFIG.get("BATCH_SIZE", 32)
    _num_workers = num_workers if num_workers is not None else GPU_CONFIG["NUM_WORKERS"]
    _pin_memory = GPU_CONFIG["PIN_MEMORY"] and torch.cuda.is_available()

    loaders = {}
    split_map = {"train": "train", "test": "test"}

    for split in splits:
        hf_split = split_map.get(split)
        if hf_split is None or hf_split not in dataset:
            continue

        ds = NSMCDataset(
            hf_dataset=dataset[hf_split],
            tokenizer=tokenizer,
            max_length=DATA_CONFIG["MAX_LENGTH"],
        )

        loaders[split] = DataLoader(
            ds,
            batch_size=_batch_size,
            shuffle=(split == "train"),
            num_workers=_num_workers,
            pin_memory=_pin_memory,
            drop_last=False,
        )

    return loaders

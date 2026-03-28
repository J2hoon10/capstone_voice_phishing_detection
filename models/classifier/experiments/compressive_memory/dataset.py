"""
MELD 데이터셋 및 DataLoader 모듈

씬(Dialogue) 단위로 가변 길이 세그먼트를 가진 데이터를 처리한다.
배치 내에서 세그먼트 수를 통일하기 위한 collate_fn을 제공한다.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


class MELDSegmentDataset(Dataset):
    """
    MELD 씬 단위 데이터셋.

    각 샘플은 하나의 씬으로, 가변 개수의 세그먼트와 1개의 multi-hot 레이블을 가진다.

    Args:
        data_path: 전처리된 JSON 파일 경로 (data_preprocessing.py의 출력)
        max_segments: 최대 세그먼트 수 제한 (None이면 제한 없음)
    """

    def __init__(self, data_path: str | Path, max_segments: int | None = None):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.max_segments = max_segments

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict:
                - input_ids: (num_segments, seq_len) 세그먼트별 토큰 ID
                - attention_mask: (num_segments, seq_len) 세그먼트별 어텐션 마스크
                - label: (NUM_LABELS,) multi-hot 레이블
                - num_segments: int 실제 세그먼트 수
                - dialogue_id: int 씬 ID
        """
        item = self.data[idx]
        segments = item["segments"]

        # 세그먼트 수 제한
        if self.max_segments is not None:
            segments = segments[:self.max_segments]

        input_ids = torch.tensor([seg["input_ids"] for seg in segments], dtype=torch.long)
        attention_mask = torch.tensor([seg["attention_mask"] for seg in segments], dtype=torch.long)
        label = torch.tensor(item["label"], dtype=torch.float)
        num_segments = len(segments)

        return {
            "input_ids": input_ids,             # (num_segments, seq_len)
            "attention_mask": attention_mask,    # (num_segments, seq_len)
            "label": label,                     # (NUM_LABELS,)
            "num_segments": num_segments,        # int
            "dialogue_id": item["dialogue_id"],  # int
        }


def collate_fn(batch: list[dict]) -> dict:
    """
    가변 세그먼트 수를 가진 배치를 패딩하여 텐서로 통일한다.

    패딩된 세그먼트는 segment_mask=0으로 표시하여
    모델에서 메모리 업데이트 시 skip할 수 있게 한다.

    Args:
        batch: Dataset.__getitem__의 출력 리스트

    Returns:
        dict:
            - input_ids: (batch, max_num_segments, seq_len)
            - attention_mask: (batch, max_num_segments, seq_len)
            - segment_mask: (batch, max_num_segments) 실제 세그먼트=1, 패딩=0
            - labels: (batch, NUM_LABELS)
            - num_segments: (batch,) 각 씬의 실제 세그먼트 수
            - dialogue_ids: list[int]
    """
    from config import ENCODER_CONFIG

    # 배치 내 최대 세그먼트 수
    max_num_seg = max(item["num_segments"] for item in batch)
    seq_len = batch[0]["input_ids"].shape[1]  # 모든 세그먼트의 토큰 길이는 동일 (패딩됨)

    batch_size = len(batch)
    pad_token_id = ENCODER_CONFIG["PAD_TOKEN_ID"]

    # 패딩된 텐서 초기화
    input_ids = torch.full(
        (batch_size, max_num_seg, seq_len),
        fill_value=pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros(batch_size, max_num_seg, seq_len, dtype=torch.long)
    segment_mask = torch.zeros(batch_size, max_num_seg, dtype=torch.bool)
    labels = torch.stack([item["label"] for item in batch])
    num_segments = torch.tensor([item["num_segments"] for item in batch], dtype=torch.long)
    dialogue_ids = [item["dialogue_id"] for item in batch]

    # 실제 세그먼트 데이터를 채워넣기
    for i, item in enumerate(batch):
        n = item["num_segments"]
        input_ids[i, :n] = item["input_ids"]
        attention_mask[i, :n] = item["attention_mask"]
        segment_mask[i, :n] = True  # 실제 세그먼트 위치만 True

    return {
        "input_ids": input_ids,             # (B, max_S, L)
        "attention_mask": attention_mask,    # (B, max_S, L)
        "segment_mask": segment_mask,       # (B, max_S)
        "labels": labels,                   # (B, NUM_LABELS)
        "num_segments": num_segments,        # (B,)
        "dialogue_ids": dialogue_ids,        # list[int]
    }


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int = 8,
    max_segments: int | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
) -> dict[str, DataLoader]:
    """
    train/dev/test DataLoader를 생성한다.
    GPU_CONFIG 설정을 자동 반영한다.

    Args:
        data_dir: 전처리된 JSON 파일들이 있는 디렉토리
        batch_size: 배치 크기
        max_segments: 최대 세그먼트 수 제한
        num_workers: DataLoader worker 수 (None이면 GPU_CONFIG 사용)
        pin_memory: pin_memory 사용 여부 (None이면 GPU_CONFIG 사용)

    Returns:
        dict: {"train": DataLoader, "dev": DataLoader, "test": DataLoader}
    """
    from config import GPU_CONFIG

    # GPU_CONFIG 기본값 적용
    if num_workers is None:
        num_workers = GPU_CONFIG["NUM_WORKERS"]
    if pin_memory is None:
        pin_memory = GPU_CONFIG["PIN_MEMORY"] and torch.cuda.is_available()

    data_dir = Path(data_dir)
    loaders = {}

    for split in ["train", "dev", "test"]:
        json_path = data_dir / f"{split}.json"
        if not json_path.exists():
            print(f"[경고] {json_path}가 없습니다. {split} DataLoader를 건너뜁니다.")
            continue

        dataset = MELDSegmentDataset(json_path, max_segments=max_segments)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),  # train만 셔플
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),  # worker 재생성 오버헤드 방지
            drop_last=False,
        )

        loaders[split] = loader
        print(f"[DataLoader] {split}: {len(dataset)}개 씬, 배치 {batch_size}, workers {num_workers}")

    return loaders

"""
Mamba Belief 모델 학습 모듈

compressive_memory/train.py 와 동일한 학습 인프라를 사용한다:
  - BF16 Mixed Precision
  - Gradient Accumulation
  - 차등 학습률 (인코더 2e-5 / Mamba+Head 5e-4)
  - Linear warmup + Cosine decay
  - L3 Loss (시간 비례 가중치)
  - Early Stopping (macro_f1 기준)

dataset / 전처리는 compressive_memory 모듈을 직접 임포트하여 재사용한다.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import f1_score

# compressive_memory의 dataset 모듈 재사용
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "compressive_memory"))
from dataset import create_dataloaders

from config import (
    ABLATION_CONFIGS, DATA_DIR, CHECKPOINT_DIR, DEVICE,
    ENCODER_CONFIG, GPU_CONFIG, HEAD_CONFIG, LOG_DIR,
    MAMBA_CONFIG, MELD_CONFIG, TRAIN_CONFIG,
)
from model import build_mamba_model


# ── Run 관리 유틸리티 ──────────────────────────────────────

def create_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def build_run_paths(experiment_name: str, run_id: str) -> dict[str, Path]:
    return {
        "checkpoint": CHECKPOINT_DIR / f"{experiment_name}_{run_id}_best.pt",
        "train_log": LOG_DIR / f"{experiment_name}_{run_id}_train_log.json",
        "latest_meta": CHECKPOINT_DIR / f"{experiment_name}_latest.json",
    }


def save_latest_run_metadata(
    experiment_name: str,
    run_id: str,
    loss_strategy: str,
    checkpoint_path: Path,
    train_log_path: Path,
):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment": experiment_name,
        "run_id": run_id,
        "loss_strategy": loss_strategy,
        "checkpoint_path": str(checkpoint_path),
        "train_log_path": str(train_log_path),
        "timestamp": run_id,
    }
    with open(CHECKPOINT_DIR / f"{experiment_name}_latest.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def build_config_snapshot(
    experiment_name: str,
    loss_strategy: str,
    ablation_config: dict,
    effective_batch: int,
    accum_steps: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    steps_per_epoch: int,
    total_steps: int,
) -> dict:
    resolved_mamba = dict(MAMBA_CONFIG)
    for key in ("D_STATE", "NUM_LAYERS"):
        if key in ablation_config:
            resolved_mamba[key] = ablation_config[key]

    return {
        "experiment_name": experiment_name,
        "ablation_override": dict(ablation_config),
        "resolved": {
            "encoder": dict(ENCODER_CONFIG),
            "mamba": resolved_mamba,
            "head": dict(HEAD_CONFIG),
            "train": dict(TRAIN_CONFIG),
            "gpu": dict(GPU_CONFIG),
            "meld": dict(MELD_CONFIG),
            "loss_strategy": loss_strategy,
            "effective_batch_size": effective_batch,
            "gradient_accumulation_steps": accum_steps,
            "amp_enabled": bool(amp_enabled),
            "amp_dtype": str(amp_dtype),
            "steps_per_epoch": int(steps_per_epoch),
            "total_steps": int(total_steps),
        },
    }


# ── GPU 초기화 ─────────────────────────────────────────────

def init_gpu():
    if not torch.cuda.is_available():
        print("[GPU] CUDA 사용 불가. CPU로 실행합니다.")
        return

    if GPU_CONFIG["TF32_MATMUL"]:
        torch.backends.cuda.matmul.allow_tf32 = True
    if GPU_CONFIG["TF32_CUDNN"]:
        torch.backends.cudnn.allow_tf32 = True
    if GPU_CONFIG["CUDNN_BENCHMARK"]:
        torch.backends.cudnn.benchmark = True

    gpu_name = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[GPU] {gpu_name}")
    print(f"[GPU] VRAM: {vram_total:.1f}GB (목표: {GPU_CONFIG['TARGET_VRAM_GB']:.1f}GB)")
    print(f"[GPU] BF16: {GPU_CONFIG['DTYPE']}, TF32: {GPU_CONFIG['TF32_MATMUL']}")
    print(f"[GPU] Gradient Accumulation: {GPU_CONFIG['GRADIENT_ACCUMULATION_STEPS']} steps")


def get_vram_usage() -> dict:
    if not torch.cuda.is_available():
        return {"allocated_gb": 0, "reserved_gb": 0, "free_gb": 0}
    allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return {
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "free_gb": round(total - reserved, 2),
    }


def get_amp_dtype() -> torch.dtype:
    return torch.bfloat16 if GPU_CONFIG["DTYPE"] == "bf16" else torch.float16


# ── 학습 유틸리티 ──────────────────────────────────────────

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module) -> AdamW:
    """인코더와 Mamba+Head에 차등 학습률을 적용한다."""
    encoder_params, upper_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (encoder_params if name.startswith("encoder.") else upper_params).append(param)

    return AdamW(
        [
            {"params": encoder_params, "lr": TRAIN_CONFIG["ENCODER_LR"]},
            {"params": upper_params, "lr": TRAIN_CONFIG["UPPER_LR"]},
        ],
        weight_decay=TRAIN_CONFIG["WEIGHT_DECAY"],
    )


def build_scheduler(optimizer: AdamW, total_steps: int) -> LambdaLR:
    warmup_steps = int(total_steps * TRAIN_CONFIG["WARMUP_RATIO"])

    def make_lr_lambda(base_lr: float):
        import math
        min_ratio = min(1.0, TRAIN_CONFIG["MIN_LR"] / base_lr)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return lr_lambda

    return LambdaLR(
        optimizer,
        [make_lr_lambda(TRAIN_CONFIG["ENCODER_LR"]), make_lr_lambda(TRAIN_CONFIG["UPPER_LR"])],
    )


def load_pos_weight() -> torch.Tensor:
    pw_path = DATA_DIR / "pos_weight.json"
    if pw_path.exists():
        with open(pw_path, "r") as f:
            data = json.load(f)
        pw = torch.tensor(data["pos_weight"], dtype=torch.float)
        pw = pw.clamp(max=TRAIN_CONFIG["POS_WEIGHT_CLIP"])
        print(f"[pos_weight] 로드 완료: {pw_path}")
    else:
        print("[pos_weight] 파일 없음. 균등 가중치 사용")
        pw = torch.ones(MELD_CONFIG["NUM_LABELS"])
    return pw.to(DEVICE)


def build_temporal_weights(num_segments: torch.Tensor, max_S: int) -> torch.Tensor:
    """L3: t번째 세그먼트 가중치 = t / T"""
    B = num_segments.shape[0]
    weights = torch.zeros(B, max_S, device=num_segments.device)
    for b in range(B):
        n = num_segments[b].item()
        if n > 0:
            weights[b, :n] = torch.arange(1, n + 1, dtype=torch.float, device=num_segments.device) / n
    return weights


def compute_loss(
    all_logits: list[torch.Tensor],
    labels: torch.Tensor,
    num_segments: torch.Tensor,
    segment_mask: torch.Tensor,
    criterion: nn.BCEWithLogitsLoss,
    loss_strategy: str = "L3",
) -> torch.Tensor:
    B = labels.shape[0]

    if loss_strategy == "L1":
        final_logits = torch.zeros_like(labels)
        for b in range(B):
            last_t = min(num_segments[b].item() - 1, len(all_logits) - 1)
            final_logits[b] = all_logits[last_t][b]
        return criterion(final_logits, labels)

    if loss_strategy == "L2":
        total, count = torch.tensor(0.0, device=labels.device), 0
        for t, logits_t in enumerate(all_logits):
            valid = segment_mask[:, t]
            if valid.any():
                total += criterion(logits_t[valid], labels[valid])
                count += 1
        return total / max(count, 1)

    # L3: 시간 비례 가중치
    max_S = len(all_logits)
    temporal_w = build_temporal_weights(num_segments, max_S)
    total = torch.tensor(0.0, device=labels.device)
    total_w = torch.tensor(0.0, device=labels.device)

    for t, logits_t in enumerate(all_logits):
        valid = segment_mask[:, t]
        if valid.any():
            loss_per = nn.functional.binary_cross_entropy_with_logits(
                logits_t, labels,
                pos_weight=criterion.pos_weight,
                reduction="none",
            ).mean(dim=1)
            w = temporal_w[:, t] * valid.float()
            total += (loss_per * w).sum()
            total_w += w.sum()

    return total / max(total_w, 1e-8)


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float, checkpoint_path: Path):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.best_f1 = -1.0
        self.counter = 0
        self.should_stop = False

    def step(self, macro_f1: float, model: nn.Module) -> bool:
        if macro_f1 > self.best_f1 + self.min_delta:
            self.best_f1 = macro_f1
            self.counter = 0
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.checkpoint_path)
            print(f"  [체크포인트] 저장 (macro_f1={macro_f1:.4f})")
            return False
        self.counter += 1
        print(f"  [Early Stopping] {self.counter}/{self.patience}")
        if self.counter >= self.patience:
            self.should_stop = True
            return True
        return False


# ── 학습 / 검증 루프 ──────────────────────────────────────

def train_one_epoch(
    model, loader, criterion, optimizer, scheduler,
    scaler, loss_strategy, accum_steps, amp_enabled, amp_dtype,
) -> float:
    model.train()
    total_loss, num_batches = 0.0, 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        segment_mask = batch["segment_mask"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)
        num_segments = batch["num_segments"].to(DEVICE, non_blocking=True)

        with autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask,
                segment_mask=segment_mask, num_segments=num_segments,
            )
            loss = compute_loss(
                outputs["all_logits"], labels, num_segments,
                segment_mask, criterion, loss_strategy,
            ) / accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["MAX_GRAD_NORM"])
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["MAX_GRAD_NORM"])
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        num_batches += 1

        interval = GPU_CONFIG["EMPTY_CACHE_INTERVAL"]
        if interval > 0 and (batch_idx + 1) % interval == 0:
            torch.cuda.empty_cache()

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, loss_strategy, amp_enabled, amp_dtype):
    model.eval()
    total_loss, num_batches = 0.0, 0
    all_probs, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        segment_mask = batch["segment_mask"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)
        num_segments = batch["num_segments"].to(DEVICE, non_blocking=True)

        with autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask,
                segment_mask=segment_mask, num_segments=num_segments,
            )
            loss = compute_loss(
                outputs["all_logits"], labels, num_segments,
                segment_mask, criterion, loss_strategy,
            )

        total_loss += loss.item()
        num_batches += 1
        all_probs.append(torch.sigmoid(outputs["logits"].float()).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    import numpy as np
    y_probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    y_pred = (y_probs > 0.5).astype(int)

    class_names = MELD_CONFIG["EMOTION_LABELS"] + MELD_CONFIG["SENTIMENT_LABELS"]
    per_class_f1 = {
        name: round(float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)), 4)
        for i, name in enumerate(class_names)
    }
    support = y_true.sum(axis=0)
    weighted_f1 = float(
        sum(per_class_f1[n] * support[i] for i, n in enumerate(class_names))
        / max(support.sum(), 1)
    )

    return total_loss / max(num_batches, 1), per_class_f1, round(weighted_f1, 4)


# ── 메인 학습 함수 ─────────────────────────────────────────

def train(experiment_name: str = "mamba_frozen", loss_strategy: str | None = None):
    set_seed(TRAIN_CONFIG["SEED"])
    init_gpu()

    amp_enabled = GPU_CONFIG["ENABLED"] and torch.cuda.is_available()
    amp_dtype = get_amp_dtype()
    accum_steps = GPU_CONFIG["GRADIENT_ACCUMULATION_STEPS"]
    scaler = GradScaler() if (amp_enabled and GPU_CONFIG["DTYPE"] == "fp16") else None

    ablation_config = ABLATION_CONFIGS.get(experiment_name, {})
    _loss_strategy = loss_strategy or TRAIN_CONFIG["LOSS_STRATEGY"]
    effective_batch = TRAIN_CONFIG["BATCH_SIZE"] * accum_steps
    run_id = create_run_id()
    run_paths = build_run_paths(experiment_name, run_id)

    print(f"\n{'='*60}")
    print(f"[학습 시작] 실험: {experiment_name}  Run: {run_id}")
    print(f"  설명: {ablation_config.get('description', '기본 설정')}")
    print(f"  Loss: {_loss_strategy}  Device: {DEVICE}")
    print(f"  Mixed Precision: {GPU_CONFIG['DTYPE'] if amp_enabled else 'OFF'}")
    print(f"  배치: {TRAIN_CONFIG['BATCH_SIZE']} × {accum_steps} = {effective_batch}")
    print(f"{'='*60}\n")

    loaders = create_dataloaders(DATA_DIR, batch_size=TRAIN_CONFIG["BATCH_SIZE"])
    if "train" not in loaders or "dev" not in loaders:
        raise FileNotFoundError("train.json / dev.json 없음. compressive_memory 전처리 먼저 실행")

    model = build_mamba_model(ablation_config).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[모델] 전체: {total_params:,}  학습 가능: {trainable_params:,}")
    vram = get_vram_usage()
    print(f"[GPU] 모델 로드 후 VRAM: {vram['allocated_gb']:.2f}GB\n")

    pos_weight = load_pos_weight()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = build_optimizer(model)
    steps_per_epoch = (len(loaders["train"]) + accum_steps - 1) // accum_steps
    total_steps = TRAIN_CONFIG["EPOCHS"] * steps_per_epoch
    scheduler = build_scheduler(optimizer, total_steps)

    early_stopping = EarlyStopping(
        TRAIN_CONFIG["PATIENCE"], TRAIN_CONFIG["MIN_DELTA"], run_paths["checkpoint"]
    )

    log = {
        "experiment": experiment_name,
        "run_id": run_id,
        "loss_strategy": _loss_strategy,
        "checkpoint_path": str(run_paths["checkpoint"]),
        "config_snapshot": build_config_snapshot(
            experiment_name, _loss_strategy, ablation_config,
            effective_batch, accum_steps, amp_enabled, amp_dtype,
            steps_per_epoch, total_steps,
        ),
        "epochs": [],
    }

    for epoch in range(1, TRAIN_CONFIG["EPOCHS"] + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scheduler,
            scaler, _loss_strategy, accum_steps, amp_enabled, amp_dtype,
        )
        val_loss, per_class_f1, weighted_f1 = validate(
            model, loaders["dev"], criterion, _loss_strategy, amp_enabled, amp_dtype,
        )

        elapsed = time.time() - t0
        macro_f1 = float(np.mean(list(per_class_f1.values())))
        vram = get_vram_usage()

        print(
            f"[Epoch {epoch:02d}/{TRAIN_CONFIG['EPOCHS']}] "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"macro_f1={macro_f1:.4f}  weighted_f1={weighted_f1:.4f}  "
            f"VRAM={vram['allocated_gb']:.1f}GB  time={elapsed:.1f}s"
        )
        print("  F1: " + "  ".join(f"{n[:4]}={v:.3f}" for n, v in per_class_f1.items()))

        log["epochs"].append({
            "epoch": epoch,
            "train_loss": train_loss, "val_loss": val_loss,
            "macro_f1": macro_f1, "weighted_f1": weighted_f1,
            "per_class_f1": per_class_f1,
            "encoder_lr": optimizer.param_groups[0]["lr"],
            "upper_lr": optimizer.param_groups[1]["lr"],
            "vram_gb": vram["allocated_gb"],
            "time": elapsed,
        })

        if early_stopping.step(macro_f1, model):
            print(f"\n[Early Stopping] {TRAIN_CONFIG['PATIENCE']}에폭 연속 미개선. 중단.")
            break

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(run_paths["train_log"], "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"\n[로그] {run_paths['train_log']}")
    save_latest_run_metadata(
        experiment_name, run_id, _loss_strategy,
        run_paths["checkpoint"], run_paths["train_log"],
    )
    print(f"[학습 완료] 최적 macro_f1: {early_stopping.best_f1:.4f}")
    return model, log


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mamba Belief 모델 학습")
    parser.add_argument(
        "--experiment", type=str, default="mamba_frozen",
        choices=list(ABLATION_CONFIGS.keys()),
    )
    parser.add_argument("--loss", type=str, default=None, choices=["L1", "L2", "L3"])
    args = parser.parse_args()

    train(experiment_name=args.experiment, loss_strategy=args.loss)

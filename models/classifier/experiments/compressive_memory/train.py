"""
Compressive Memory 모델 학습 모듈

주요 기능:
  - BF16 Mixed Precision (RTX 4070 Ti Super 최적화)
  - TF32 행렬 연산 가속
  - Gradient Accumulation (VRAM 절약)
  - VRAM 사용량 실시간 모니터링
  - 차등 학습률 (인코더 2e-5 / 상위 모듈 1e-3)
  - Linear warmup + Cosine decay 스케줄러
  - Early Stopping (patience=3)
  - L1/L2/L3 Loss 전략 지원
  - pos_weight 기반 클래스 불균형 처리
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score

from config import (
    DEVICE, DATA_DIR, CHECKPOINT_DIR, LOG_DIR,
    TRAIN_CONFIG, MELD_CONFIG, ABLATION_CONFIGS, GPU_CONFIG,
    ENCODER_CONFIG, MEMORY_CONFIG, ATTENTION_CONFIG, HEAD_CONFIG,
)
from dataset import create_dataloaders
from model import build_compressive_memory_model


def create_run_id() -> str:
    """실행 시각 기반 run_id를 생성한다."""
    return time.strftime("%Y%m%d_%H%M%S")


def build_run_paths(experiment_name: str, run_id: str) -> dict[str, Path]:
    """학습 run별 저장 경로를 구성한다."""
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
    """최신 run 정보를 평가 스크립트가 읽을 수 있도록 저장한다."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment": experiment_name,
        "run_id": run_id,
        "loss_strategy": loss_strategy,
        "checkpoint_path": str(checkpoint_path),
        "train_log_path": str(train_log_path),
        "timestamp": run_id,
    }
    latest_meta_path = CHECKPOINT_DIR / f"{experiment_name}_latest.json"
    with open(latest_meta_path, "w", encoding="utf-8") as f:
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
    """현재 run에서 실제 사용한 설정을 JSON 직렬화 가능한 형태로 정리한다."""
    resolved_memory = dict(MEMORY_CONFIG)
    if "FM_SIZE" in ablation_config:
        resolved_memory["FM_SIZE"] = ablation_config["FM_SIZE"]
    if "CM_SIZE" in ablation_config:
        resolved_memory["CM_SIZE"] = ablation_config["CM_SIZE"]
    if "COMPRESS_FN" in ablation_config:
        resolved_memory["COMPRESS_FN"] = ablation_config["COMPRESS_FN"]

    return {
        "experiment_name": experiment_name,
        "ablation_override": dict(ablation_config),
        "resolved": {
            "encoder": dict(ENCODER_CONFIG),
            "memory": resolved_memory,
            "attention": dict(ATTENTION_CONFIG),
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
    """
    GPU 최적화 설정을 초기화한다.
    TF32, cuDNN benchmark 등 RTX 4070 Ti Super에 맞는 설정을 적용한다.
    """
    if not torch.cuda.is_available():
        print("[GPU] CUDA 사용 불가. CPU로 실행합니다.")
        return

    # TF32 활성화: fp32 연산을 TF32로 가속 (정밀도 소폭 감소, 속도 대폭 향상)
    if GPU_CONFIG["TF32_MATMUL"]:
        torch.backends.cuda.matmul.allow_tf32 = True
    if GPU_CONFIG["TF32_CUDNN"]:
        torch.backends.cudnn.allow_tf32 = True

    # cuDNN benchmark: 입력 크기가 고정적이면 최적 알고리즘 자동 탐색
    if GPU_CONFIG["CUDNN_BENCHMARK"]:
        torch.backends.cudnn.benchmark = True

    # GPU 정보 출력
    gpu_name = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[GPU] {gpu_name}")
    print(f"[GPU] VRAM: {vram_total:.1f}GB (목표 사용량: {GPU_CONFIG['TARGET_VRAM_GB']:.1f}GB)")
    print(f"[GPU] BF16: {GPU_CONFIG['DTYPE']}, TF32: {GPU_CONFIG['TF32_MATMUL']}")
    print(f"[GPU] Gradient Accumulation: {GPU_CONFIG['GRADIENT_ACCUMULATION_STEPS']} steps")


def get_vram_usage() -> dict:
    """현재 GPU VRAM 사용량을 반환한다."""
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
    """GPU_CONFIG에 따른 AMP dtype을 반환한다."""
    if GPU_CONFIG["DTYPE"] == "bf16":
        return torch.bfloat16
    return torch.float16


# ── 기존 유틸리티 ──────────────────────────────────────────

def set_seed(seed: int):
    """재현성을 위한 시드 고정"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module) -> AdamW:
    """
    차등 학습률 옵티마이저를 생성한다.
    인코더(RoBERTa)와 상위 모듈(Projection, Memory, Attention, Head)에
    서로 다른 학습률을 적용한다.
    """
    encoder_params = []
    upper_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(param)
        else:
            upper_params.append(param)

    param_groups = [
        {"params": encoder_params, "lr": TRAIN_CONFIG["ENCODER_LR"]},
        {"params": upper_params, "lr": TRAIN_CONFIG["UPPER_LR"]},
    ]

    optimizer = AdamW(param_groups, weight_decay=TRAIN_CONFIG["WEIGHT_DECAY"])
    return optimizer


def build_scheduler(optimizer: AdamW, total_steps: int) -> LambdaLR:
    """Linear warmup + Cosine decay 스케줄러를 생성한다."""
    warmup_steps = int(total_steps * TRAIN_CONFIG["WARMUP_RATIO"])

    def make_lr_lambda(base_lr: float):
        min_ratio = min(1.0, TRAIN_CONFIG["MIN_LR"] / base_lr)

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            import math
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_ratio, cosine_decay)

        return lr_lambda

    return LambdaLR(
        optimizer,
        [
            make_lr_lambda(TRAIN_CONFIG["ENCODER_LR"]),
            make_lr_lambda(TRAIN_CONFIG["UPPER_LR"]),
        ],
    )


def load_pos_weight() -> torch.Tensor:
    """전처리 단계에서 계산된 pos_weight를 로드한다."""
    pw_path = DATA_DIR / "pos_weight.json"
    if pw_path.exists():
        with open(pw_path, "r") as f:
            data = json.load(f)
        pos_weight = torch.tensor(data["pos_weight"], dtype=torch.float)
        print(f"[pos_weight] 로드 완료: {pw_path}")
    else:
        print("[pos_weight] 파일 없음. 균등 가중치(1.0) 사용")
        pos_weight = torch.ones(MELD_CONFIG["NUM_LABELS"])

    return pos_weight.to(DEVICE)


def build_temporal_weights(num_segments: torch.Tensor, max_S: int) -> torch.Tensor:
    """L3 Loss 전략: 뒤쪽 세그먼트에 높은 가중치를 부여한다."""
    B = num_segments.shape[0]
    weights = torch.zeros(B, max_S, device=num_segments.device)

    for b in range(B):
        n = num_segments[b].item()
        if n > 0:
            w = torch.arange(1, n + 1, dtype=torch.float, device=num_segments.device) / n
            weights[b, :n] = w

    return weights


def compute_loss(
    all_logits: list[torch.Tensor],
    labels: torch.Tensor,
    num_segments: torch.Tensor,
    segment_mask: torch.Tensor,
    criterion: nn.BCEWithLogitsLoss,
    loss_strategy: str = "L1",
) -> torch.Tensor:
    """Loss 전략에 따라 손실을 계산한다."""
    B = labels.shape[0]

    if loss_strategy == "L1":
        final_logits = torch.zeros_like(labels)
        for b in range(B):
            last_t = num_segments[b].item() - 1
            last_t = min(last_t, len(all_logits) - 1)
            final_logits[b] = all_logits[last_t][b]
        return criterion(final_logits, labels)

    elif loss_strategy == "L2":
        total_loss = torch.tensor(0.0, device=labels.device)
        count = 0
        for t in range(len(all_logits)):
            valid = segment_mask[:, t]
            if valid.any():
                loss_t = criterion(all_logits[t][valid], labels[valid])
                total_loss += loss_t
                count += 1
        return total_loss / max(count, 1)

    elif loss_strategy == "L3":
        max_S = len(all_logits)
        temporal_w = build_temporal_weights(num_segments, max_S)

        total_loss = torch.tensor(0.0, device=labels.device)
        total_weight = torch.tensor(0.0, device=labels.device)

        for t in range(max_S):
            valid = segment_mask[:, t]
            if valid.any():
                loss_per_sample = nn.functional.binary_cross_entropy_with_logits(
                    all_logits[t], labels,
                    pos_weight=criterion.pos_weight,
                    reduction='none',
                ).mean(dim=1)

                w = temporal_w[:, t] * valid.float()
                total_loss += (loss_per_sample * w).sum()
                total_weight += w.sum()

        return total_loss / max(total_weight, 1e-8)

    else:
        raise ValueError(f"알 수 없는 Loss 전략: {loss_strategy}")


class EarlyStopping:
    """macro_f1이 개선되지 않으면 학습을 조기 종료하고, 개선 시 체크포인트를 저장한다."""

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
        else:
            self.counter += 1
            print(f"  [Early Stopping] {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
                return True
            return False


# ── 학습/검증 루프 (AMP + Gradient Accumulation) ──────────

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.BCEWithLogitsLoss,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: GradScaler | None,
    loss_strategy: str,
    accum_steps: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> float:
    """
    1에폭 학습 (AMP + Gradient Accumulation 적용).

    Args:
        scaler: GradScaler (fp16일 때만 사용, bf16은 None)
        accum_steps: Gradient Accumulation 스텝 수
        amp_enabled: AMP 사용 여부
        amp_dtype: autocast dtype (torch.bfloat16 또는 torch.float16)

    Returns:
        평균 학습 loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        # 배치 데이터를 디바이스로 이동 (non_blocking으로 비동기 전송)
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        segment_mask = batch["segment_mask"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)
        num_segments = batch["num_segments"].to(DEVICE, non_blocking=True)

        # Forward pass (AMP autocast 적용)
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                segment_mask=segment_mask,
                num_segments=num_segments,
            )

            loss = compute_loss(
                all_logits=outputs["all_logits"],
                labels=labels,
                num_segments=num_segments,
                segment_mask=segment_mask,
                criterion=criterion,
                loss_strategy=loss_strategy,
            )

            # Gradient Accumulation: loss를 누적 스텝 수로 나누기
            loss = loss / accum_steps

        # Backward pass
        if scaler is not None:
            # FP16: GradScaler 사용
            scaler.scale(loss).backward()
        else:
            # BF16 / FP32: GradScaler 불필요
            loss.backward()

        # Gradient Accumulation: accum_steps마다 업데이트
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

        total_loss += loss.item() * accum_steps  # 원래 scale로 복원
        num_batches += 1

        # VRAM 캐시 관리 (설정된 경우)
        cache_interval = GPU_CONFIG["EMPTY_CACHE_INTERVAL"]
        if cache_interval > 0 and (batch_idx + 1) % cache_interval == 0:
            torch.cuda.empty_cache()

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.BCEWithLogitsLoss,
    loss_strategy: str,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[float, dict]:
    """검증 데이터에 대한 loss와 per-class F1을 계산한다 (AMP 적용).

    Returns:
        (val_loss, per_class_f1): val_loss는 float, per_class_f1은 클래스별 F1 dict
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_probs = []
    all_labels = []

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        segment_mask = batch["segment_mask"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)
        num_segments = batch["num_segments"].to(DEVICE, non_blocking=True)

        with autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                segment_mask=segment_mask,
                num_segments=num_segments,
            )

            loss = compute_loss(
                all_logits=outputs["all_logits"],
                labels=labels,
                num_segments=num_segments,
                segment_mask=segment_mask,
                criterion=criterion,
                loss_strategy=loss_strategy,
            )

        total_loss += loss.item()
        num_batches += 1

        probs = torch.sigmoid(outputs["logits"].float()).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

    val_loss = total_loss / max(num_batches, 1)

    # Per-class F1 (threshold=0.5 고정)
    y_probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    y_pred = (y_probs > 0.5).astype(int)

    class_names = MELD_CONFIG["EMOTION_LABELS"] + MELD_CONFIG["SENTIMENT_LABELS"]
    per_class_f1 = {
        name: round(float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)), 4)
        for i, name in enumerate(class_names)
    }

    # sample counts per class (for weighted-F1)
    support = y_true.sum(axis=0)  # (num_labels,)
    total_support = support.sum()
    weighted_f1 = float(
        sum(per_class_f1[name] * support[i] for i, name in enumerate(class_names))
        / max(total_support, 1)
    )

    return val_loss, per_class_f1, round(weighted_f1, 4)


# ── 메인 학습 함수 ─────────────────────────────────────────

def train(
    experiment_name: str = "full",
    loss_strategy: str | None = None,
):
    """
    메인 학습 함수.

    Args:
        experiment_name: Ablation 실험명 (config.py의 ABLATION_CONFIGS 키)
        loss_strategy: Loss 전략 오버라이드 (None이면 TRAIN_CONFIG 기본값 사용)
    """
    set_seed(TRAIN_CONFIG["SEED"])

    # GPU 초기화 (TF32, cuDNN benchmark 등)
    init_gpu()

    # AMP 설정
    amp_enabled = GPU_CONFIG["ENABLED"] and torch.cuda.is_available()
    amp_dtype = get_amp_dtype()
    accum_steps = GPU_CONFIG["GRADIENT_ACCUMULATION_STEPS"]

    # GradScaler: fp16일 때만 사용 (bf16은 동적 범위가 넓어서 불필요)
    scaler = GradScaler() if (amp_enabled and GPU_CONFIG["DTYPE"] == "fp16") else None

    # 실험 설정
    ablation_config = ABLATION_CONFIGS.get(experiment_name, {})
    _loss_strategy = loss_strategy or TRAIN_CONFIG["LOSS_STRATEGY"]
    effective_batch = TRAIN_CONFIG["BATCH_SIZE"] * accum_steps
    run_id = create_run_id()
    run_paths = build_run_paths(experiment_name, run_id)

    print(f"\n{'='*60}")
    print(f"[학습 시작] 실험: {experiment_name}")
    print(f"  Run ID: {run_id}")
    print(f"  설명: {ablation_config.get('description', '기본 설정')}")
    print(f"  Loss 전략: {_loss_strategy}")
    print(f"  디바이스: {DEVICE}")
    print(f"  Mixed Precision: {GPU_CONFIG['DTYPE'] if amp_enabled else 'OFF'}")
    print(f"  배치 크기: {TRAIN_CONFIG['BATCH_SIZE']} × {accum_steps} (accum) = {effective_batch}")
    print(f"{'='*60}\n")

    # DataLoader 생성
    loaders = create_dataloaders(DATA_DIR, batch_size=TRAIN_CONFIG["BATCH_SIZE"])
    if "train" not in loaders or "dev" not in loaders:
        raise FileNotFoundError("train.json 또는 dev.json이 없습니다. 먼저 data_preprocessing.py를 실행하세요.")

    # 모델 생성
    model = build_compressive_memory_model(ablation_config)
    model = model.to(DEVICE)

    # 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[모델] 전체 파라미터: {total_params:,}")
    print(f"[모델] 학습 가능 파라미터: {trainable_params:,}")

    # 모델 로드 후 VRAM 확인
    vram = get_vram_usage()
    print(f"[GPU] 모델 로드 후 VRAM: {vram['allocated_gb']:.2f}GB 사용 / {vram['free_gb']:.2f}GB 여유\n")

    # pos_weight 로드 및 Loss 함수 생성
    pos_weight = load_pos_weight()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 옵티마이저 & 스케줄러
    optimizer = build_optimizer(model)
    # total_steps는 옵티마이저 업데이트 횟수 기준 (accum 고려)
    steps_per_epoch = len(loaders["train"]) // accum_steps + (1 if len(loaders["train"]) % accum_steps != 0 else 0)
    total_steps = TRAIN_CONFIG["EPOCHS"] * steps_per_epoch
    scheduler = build_scheduler(optimizer, total_steps)
    config_snapshot = build_config_snapshot(
        experiment_name=experiment_name,
        loss_strategy=_loss_strategy,
        ablation_config=ablation_config,
        effective_batch=effective_batch,
        accum_steps=accum_steps,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
    )

    # Early Stopping
    ckpt_path = run_paths["checkpoint"]
    early_stopping = EarlyStopping(
        patience=TRAIN_CONFIG["PATIENCE"],
        min_delta=TRAIN_CONFIG["MIN_DELTA"],
        checkpoint_path=ckpt_path,
    )

    # 학습 로그
    log = {
        "experiment": experiment_name,
        "run_id": run_id,
        "loss_strategy": _loss_strategy,
        "checkpoint_path": str(ckpt_path),
        "config_snapshot": config_snapshot,
        "gpu_config": {
            "dtype": GPU_CONFIG["DTYPE"] if amp_enabled else "fp32",
            "gradient_accumulation": accum_steps,
            "effective_batch_size": effective_batch,
        },
        "epochs": [],
    }

    # 학습 루프
    for epoch in range(1, TRAIN_CONFIG["EPOCHS"] + 1):
        start_time = time.time()

        # 학습
        train_loss = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scheduler,
            scaler, _loss_strategy, accum_steps, amp_enabled, amp_dtype,
        )

        # 검증
        val_loss, per_class_f1, weighted_f1 = validate(
            model, loaders["dev"], criterion, _loss_strategy,
            amp_enabled, amp_dtype,
        )

        elapsed = time.time() - start_time
        encoder_lr = optimizer.param_groups[0]["lr"]
        upper_lr = optimizer.param_groups[1]["lr"]
        vram = get_vram_usage()

        micro_f1 = float(np.mean(list(per_class_f1.values())))
        print(
            f"[Epoch {epoch:02d}/{TRAIN_CONFIG['EPOCHS']}] "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"macro_f1={micro_f1:.4f}  weighted_f1={weighted_f1:.4f}  "
            f"enc_lr={encoder_lr:.2e}  upper_lr={upper_lr:.2e}  "
            f"VRAM={vram['allocated_gb']:.1f}GB  "
            f"time={elapsed:.1f}s"
        )
        # 클래스별 F1 출력 (Fear/Disgust 등 소수 클래스 모니터링)
        f1_line = "  F1: " + "  ".join(
            f"{name[:4]}={f1:.3f}" for name, f1 in per_class_f1.items()
        )
        print(f1_line)

        # 로그 기록
        log["epochs"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "macro_f1": micro_f1,
            "weighted_f1": weighted_f1,
            "per_class_f1": per_class_f1,
            "encoder_lr": encoder_lr,
            "upper_lr": upper_lr,
            "vram_gb": vram["allocated_gb"],
            "time": elapsed,
        })

        # Early Stopping 체크 (macro_f1 기준)
        if early_stopping.step(micro_f1, model):
            print(f"\n[Early Stopping] {TRAIN_CONFIG['PATIENCE']}에폭 연속 개선 없음. 학습 중단.")
            break

    # 학습 로그 저장
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = run_paths["train_log"]
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"\n[로그] {log_path}")
    save_latest_run_metadata(
        experiment_name=experiment_name,
        run_id=run_id,
        loss_strategy=_loss_strategy,
        checkpoint_path=ckpt_path,
        train_log_path=log_path,
    )
    print(f"[메타데이터] {run_paths['latest_meta']}")

    # 최종 VRAM 상태
    vram = get_vram_usage()
    print(f"[GPU] 최종 VRAM: {vram['allocated_gb']:.2f}GB 사용 / {vram['free_gb']:.2f}GB 여유")
    print(f"\n[학습 완료] 최적 모델: {ckpt_path}")
    print(f"[학습 완료] 최적 macro_f1: {early_stopping.best_f1:.4f}")

    return model, log


# ── 메인 실행 ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compressive Memory 모델 학습")
    parser.add_argument(
        "--experiment", type=str, default="full",
        choices=list(ABLATION_CONFIGS.keys()),
        help="실험명 (Ablation 설정)",
    )
    parser.add_argument(
        "--loss", type=str, default=None,
        choices=["L1", "L2", "L3"],
        help="Loss 전략 오버라이드",
    )
    args = parser.parse_args()

    train(experiment_name=args.experiment, loss_strategy=args.loss)

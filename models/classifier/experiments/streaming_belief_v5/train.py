"""
Streaming Belief v5 학습 스크립트.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import (
    ABLATION_CONFIGS,
    CHECKPOINT_DIR,
    DATA_CONFIG,
    DATA_DIR,
    DEVICE,
    ENCODER_CONFIG,
    GPU_CONFIG,
    HEAD_CONFIG,
    LOG_DIR,
    MAMBA_CONFIG,
    TRAIN_CONFIG,
)
from dataset import create_dataloaders
from model import build_streaming_belief_model


def create_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def build_run_paths(experiment_name: str, run_id: str) -> dict[str, Path]:
    return {
        "checkpoint": CHECKPOINT_DIR / f"{experiment_name}_{run_id}_best.pt",
        "resume_checkpoint": CHECKPOINT_DIR / f"{experiment_name}_{run_id}_resume.pt",
        "train_log": LOG_DIR / f"{experiment_name}_{run_id}_train_log.json",
        "latest_meta": CHECKPOINT_DIR / f"{experiment_name}_latest.json",
    }


def save_latest_run_metadata(
    experiment_name: str,
    run_id: str,
    best_checkpoint_path: Path,
    resume_checkpoint_path: Path,
    train_log_path: Path,
    lambda_alpha: float,
    last_completed_epoch: int | None = None,
    best_macro_f1: float | None = None,
) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": experiment_name,
        "run_id": run_id,
        "checkpoint_path": str(best_checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "resume_checkpoint_path": str(resume_checkpoint_path),
        "train_log_path": str(train_log_path),
        "lambda_alpha": float(lambda_alpha),
        "last_completed_epoch": (
            int(last_completed_epoch) if last_completed_epoch is not None else None
        ),
        "best_macro_f1": float(best_macro_f1) if best_macro_f1 is not None else None,
        "timestamp": run_id,
    }
    with (CHECKPOINT_DIR / f"{experiment_name}_latest.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_resume_checkpoint(
    resume_path: Path,
    *,
    experiment_name: str,
    run_id: str,
    epoch: int,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: GradScaler | None,
    early_stopping: "EarlyStopping",
    lambda_alpha: float,
    focal_gamma: float,
    log: dict,
) -> None:
    resume_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": experiment_name,
        "run_id": run_id,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": (scaler.state_dict() if scaler is not None else None),
        "best_f1": float(early_stopping.best_f1),
        "early_counter": int(early_stopping.counter),
        "lambda_alpha": float(lambda_alpha),
        "focal_gamma": float(focal_gamma),
        "log": log,
    }
    torch.save(payload, resume_path)


def resolve_resume_checkpoint_path(
    experiment_name: str,
    resume_path: str | None = None,
    resume_run_id: str | None = None,
) -> Path | None:
    if resume_path is not None:
        p = Path(resume_path)
        return p if p.exists() else None

    if resume_run_id is not None:
        p = CHECKPOINT_DIR / f"{experiment_name}_{resume_run_id}_resume.pt"
        return p if p.exists() else None

    latest_meta = CHECKPOINT_DIR / f"{experiment_name}_latest.json"
    if latest_meta.exists():
        with latest_meta.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        resume_ckpt = meta.get("resume_checkpoint_path")
        if resume_ckpt:
            p = Path(resume_ckpt)
            if p.exists():
                return p
    return None


def init_gpu() -> None:
    if not torch.cuda.is_available():
        print("[gpu] CUDA unavailable, run on CPU")
        return
    if GPU_CONFIG["TF32_MATMUL"]:
        torch.backends.cuda.matmul.allow_tf32 = True
    if GPU_CONFIG["TF32_CUDNN"]:
        torch.backends.cudnn.allow_tf32 = True
    if GPU_CONFIG["CUDNN_BENCHMARK"]:
        torch.backends.cudnn.benchmark = True
    gpu_name = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[gpu] {gpu_name}")
    print(f"[gpu] VRAM: {vram_total:.1f}GB")


def get_vram_usage() -> dict:
    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "free_gb": 0.0}
    allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return {
        "allocated_gb": round(float(allocated), 2),
        "reserved_gb": round(float(reserved), 2),
        "free_gb": round(float(total - reserved), 2),
    }


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_amp_dtype() -> torch.dtype:
    return torch.bfloat16 if GPU_CONFIG["DTYPE"] == "bf16" else torch.float16


def load_focal_alpha() -> torch.Tensor:
    path = DATA_DIR / "class_weight.json"
    if not path.exists():
        return torch.ones(DATA_CONFIG["NUM_LABELS"], dtype=torch.float, device=DEVICE)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    alpha = payload.get("focal_alpha", [1.0] * DATA_CONFIG["NUM_LABELS"])
    if len(alpha) != DATA_CONFIG["NUM_LABELS"]:
        alpha = [1.0] * DATA_CONFIG["NUM_LABELS"]
    return torch.tensor(alpha, dtype=torch.float, device=DEVICE)


def build_optimizer(model: nn.Module) -> AdamW:
    encoder_params, upper_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(p)
        else:
            upper_params.append(p)
    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": TRAIN_CONFIG["ENCODER_LR"]})
    if upper_params:
        groups.append({"params": upper_params, "lr": TRAIN_CONFIG["UPPER_LR"]})
    return AdamW(groups, weight_decay=TRAIN_CONFIG["WEIGHT_DECAY"])


def build_scheduler(optimizer: AdamW, total_steps: int) -> LambdaLR:
    warmup_steps = int(total_steps * TRAIN_CONFIG["WARMUP_RATIO"])

    def make_lr_lambda(base_lr: float):
        import math
        min_ratio = min(1.0, TRAIN_CONFIG["MIN_LR"] / max(base_lr, 1e-12))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return lr_lambda

    return LambdaLR(optimizer, [make_lr_lambda(g["lr"]) for g in optimizer.param_groups])


def focal_loss_per_sample(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    log_pt = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()
    return -alpha[labels] * ((1.0 - pt) ** gamma) * log_pt


def compute_temporal_focal_loss(
    all_logits: list[torch.Tensor],
    labels: torch.Tensor,
    segment_mask: torch.Tensor,
    focal_alpha: torch.Tensor,
    gamma: float,
    lambda_alpha: float,
) -> torch.Tensor:
    if not all_logits:
        return torch.tensor(0.0, device=labels.device)
    t_index = torch.arange(len(all_logits), device=labels.device, dtype=torch.float)
    lambda_t = torch.exp(-lambda_alpha * t_index)
    total = torch.tensor(0.0, device=labels.device)
    total_w = torch.tensor(0.0, device=labels.device)
    for t, logits_t in enumerate(all_logits):
        valid = segment_mask[:, t]
        if not valid.any():
            continue
        loss_per = focal_loss_per_sample(logits_t, labels, focal_alpha, gamma)
        w = lambda_t[t] * valid.float()
        total += (loss_per * w).sum()
        total_w += w.sum()
    return total / total_w.clamp(min=1e-8)


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float, checkpoint_path: Path):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.best_f1 = -1.0
        self.counter = 0

    def step(self, macro_f1: float, model: nn.Module) -> bool:
        if macro_f1 > self.best_f1 + self.min_delta:
            self.best_f1 = macro_f1
            self.counter = 0
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.checkpoint_path)
            print(f"  [checkpoint] saved (macro_f1={macro_f1:.4f})")
            return False
        self.counter += 1
        print(f"  [early stopping] {self.counter}/{self.patience}")
        return self.counter >= self.patience


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: GradScaler | None,
    focal_alpha: torch.Tensor,
    gamma: float,
    lambda_alpha: float,
    accum_steps: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> float:
    model.train()
    total_loss, num_batches = 0.0, 0
    optimizer.zero_grad()
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        segment_mask = batch["segment_mask"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)
        num_segments = batch["num_segments"].to(DEVICE, non_blocking=True)

        with autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                segment_mask=segment_mask,
                num_segments=num_segments,
            )
            loss = compute_temporal_focal_loss(
                all_logits=outputs["all_logits"],
                labels=labels,
                segment_mask=segment_mask,
                focal_alpha=focal_alpha,
                gamma=gamma,
                lambda_alpha=lambda_alpha,
            )
            loss = loss / accum_steps

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
        if interval > 0 and torch.cuda.is_available() and (batch_idx + 1) % interval == 0:
            torch.cuda.empty_cache()

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    focal_alpha: torch.Tensor,
    gamma: float,
    lambda_alpha: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[float, dict[str, float], float, float]:
    model.eval()
    total_loss, num_batches = 0.0, 0
    y_true, y_pred = [], []
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        segment_mask = batch["segment_mask"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)
        num_segments = batch["num_segments"].to(DEVICE, non_blocking=True)

        with autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                segment_mask=segment_mask,
                num_segments=num_segments,
            )
            loss = compute_temporal_focal_loss(
                all_logits=outputs["all_logits"],
                labels=labels,
                segment_mask=segment_mask,
                focal_alpha=focal_alpha,
                gamma=gamma,
                lambda_alpha=lambda_alpha,
            )

        total_loss += loss.item()
        num_batches += 1
        preds = outputs["logits"].argmax(dim=-1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    labels_idx = list(range(DATA_CONFIG["NUM_LABELS"]))
    per_class_arr = f1_score(y_true, y_pred, labels=labels_idx, average=None, zero_division=0)
    per_class_f1 = {
        name: round(float(per_class_arr[i]), 4)
        for i, name in enumerate(DATA_CONFIG["LABELS"])
    }
    macro_f1 = float(np.mean(list(per_class_f1.values())))
    accuracy = float(accuracy_score(y_true, y_pred))
    return total_loss / max(num_batches, 1), per_class_f1, macro_f1, accuracy


def build_config_snapshot(
    experiment_name: str,
    ablation_config: dict,
    effective_batch: int,
    accum_steps: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    steps_per_epoch: int,
    total_steps: int,
    lambda_alpha: float,
) -> dict:
    resolved_mamba = dict(MAMBA_CONFIG)
    for key in ("D_STATE", "NUM_LAYERS"):
        if key in ablation_config:
            resolved_mamba[key] = ablation_config[key]

    resolved_encoder = dict(ENCODER_CONFIG)
    for key in (
        "FREEZE_EMBEDDINGS",
        "FREEZE_LOWER_LAYERS",
        "UNFREEZE_ALL",
        "USE_LORA",
        "LORA_R",
        "LORA_ALPHA",
        "LORA_DROPOUT",
    ):
        if key in ablation_config:
            resolved_encoder[key] = ablation_config[key]

    return {
        "experiment_name": experiment_name,
        "ablation_override": dict(ablation_config),
        "resolved": {
            "encoder": resolved_encoder,
            "mamba": resolved_mamba,
            "head": dict(HEAD_CONFIG),
            "train": dict(TRAIN_CONFIG),
            "gpu": dict(GPU_CONFIG),
            "data": dict(DATA_CONFIG),
            "lambda_alpha": float(lambda_alpha),
            "effective_batch_size": int(effective_batch),
            "gradient_accumulation_steps": int(accum_steps),
            "amp_enabled": bool(amp_enabled),
            "amp_dtype": str(amp_dtype),
            "steps_per_epoch": int(steps_per_epoch),
            "total_steps": int(total_steps),
        },
    }


def resolve_lambda_alpha(ablation_config: dict, override_value: float | None) -> float:
    if override_value is not None:
        return float(override_value)
    if "LAMBDA_ALPHA" in ablation_config:
        return float(ablation_config["LAMBDA_ALPHA"])
    return float(TRAIN_CONFIG["LAMBDA_ALPHA"])


def train(
    experiment_name: str = "v5_default",
    lambda_alpha_override: float | None = None,
    resume: bool = False,
    resume_path: str | None = None,
    resume_run_id: str | None = None,
):
    set_seed(TRAIN_CONFIG["SEED"])
    init_gpu()

    amp_enabled = GPU_CONFIG["ENABLED"] and torch.cuda.is_available()
    amp_dtype = get_amp_dtype()
    accum_steps = int(GPU_CONFIG["GRADIENT_ACCUMULATION_STEPS"])
    scaler = GradScaler() if (amp_enabled and GPU_CONFIG["DTYPE"] == "fp16") else None

    ablation_config = ABLATION_CONFIGS.get(experiment_name, {})
    lambda_alpha = resolve_lambda_alpha(ablation_config, lambda_alpha_override)
    focal_gamma = float(TRAIN_CONFIG["FOCAL_GAMMA"])

    resume_state = None
    if resume or resume_path is not None or resume_run_id is not None:
        resolved_resume_path = resolve_resume_checkpoint_path(
            experiment_name=experiment_name,
            resume_path=resume_path,
            resume_run_id=resume_run_id,
        )
        if resolved_resume_path is None:
            raise FileNotFoundError(
                "resume checkpoint not found. use --resume-path or check latest metadata."
            )
        resume_state = torch.load(resolved_resume_path, map_location="cpu", weights_only=False)
        print(f"[resume] loaded: {resolved_resume_path}")

    run_id = (
        str(resume_state.get("run_id"))
        if resume_state is not None and resume_state.get("run_id")
        else create_run_id()
    )
    run_paths = build_run_paths(experiment_name, run_id)
    effective_batch = int(TRAIN_CONFIG["BATCH_SIZE"] * accum_steps)

    if resume_state is not None:
        lambda_alpha = float(resume_state.get("lambda_alpha", lambda_alpha))
        focal_gamma = float(resume_state.get("focal_gamma", focal_gamma))

    print(f"\n{'=' * 72}")
    print(f"[train] experiment={experiment_name} run_id={run_id}")
    print(f"  description: {ablation_config.get('description', 'default')}")
    print(f"  device={DEVICE} mixed_precision={GPU_CONFIG['DTYPE'] if amp_enabled else 'OFF'}")
    print(f"  batch={TRAIN_CONFIG['BATCH_SIZE']} x {accum_steps} = {effective_batch}")
    print(f"  focal_gamma={focal_gamma} lambda_alpha={lambda_alpha}")
    print(f"{'=' * 72}\n")

    loaders = create_dataloaders(DATA_DIR, batch_size=TRAIN_CONFIG["BATCH_SIZE"], splits=("train", "val"))
    if "train" not in loaders or "val" not in loaders:
        raise FileNotFoundError("train.json/val.json missing. run data_preprocessing.py first.")

    model = build_streaming_belief_model(ablation_config).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] total params: {total_params:,}")
    print(f"[model] trainable params: {trainable_params:,}")
    vram = get_vram_usage()
    print(f"[gpu] after model load: {vram['allocated_gb']:.2f}GB\n")

    focal_alpha = load_focal_alpha()
    optimizer = build_optimizer(model)
    steps_per_epoch = (len(loaders["train"]) + accum_steps - 1) // accum_steps
    total_steps = int(TRAIN_CONFIG["EPOCHS"] * steps_per_epoch)
    scheduler = build_scheduler(optimizer, total_steps)
    early_stopping = EarlyStopping(
        patience=TRAIN_CONFIG["PATIENCE"],
        min_delta=TRAIN_CONFIG["MIN_DELTA"],
        checkpoint_path=run_paths["checkpoint"],
    )

    log = {
        "experiment": experiment_name,
        "run_id": run_id,
        "checkpoint_path": str(run_paths["checkpoint"]),
        "resume_checkpoint_path": str(run_paths["resume_checkpoint"]),
        "config_snapshot": build_config_snapshot(
            experiment_name=experiment_name,
            ablation_config=ablation_config,
            effective_batch=effective_batch,
            accum_steps=accum_steps,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            steps_per_epoch=steps_per_epoch,
            total_steps=total_steps,
            lambda_alpha=lambda_alpha,
        ),
        "epochs": [],
    }

    start_epoch = 1
    if resume_state is not None:
        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        scaler_state = resume_state.get("scaler_state_dict")
        if scaler is not None and scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        early_stopping.best_f1 = float(resume_state.get("best_f1", early_stopping.best_f1))
        early_stopping.counter = int(resume_state.get("early_counter", early_stopping.counter))
        loaded_log = resume_state.get("log")
        if isinstance(loaded_log, dict) and "epochs" in loaded_log:
            log = loaded_log
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        print(
            f"[resume] run_id={run_id} next_epoch={start_epoch} "
            f"best_f1={early_stopping.best_f1:.4f}"
        )

    last_completed_epoch = max(0, start_epoch - 1)
    interrupted = False
    try:
        for epoch in range(start_epoch, TRAIN_CONFIG["EPOCHS"] + 1):
            if hasattr(loaders["train"], "batch_sampler") and hasattr(loaders["train"].batch_sampler, "set_epoch"):
                loaders["train"].batch_sampler.set_epoch(epoch)

            t0 = time.time()
            train_loss = train_one_epoch(
                model=model,
                loader=loaders["train"],
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                focal_alpha=focal_alpha,
                gamma=focal_gamma,
                lambda_alpha=lambda_alpha,
                accum_steps=accum_steps,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            val_loss, per_class_f1, macro_f1, accuracy = validate(
                model=model,
                loader=loaders["val"],
                focal_alpha=focal_alpha,
                gamma=focal_gamma,
                lambda_alpha=lambda_alpha,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )

            elapsed = time.time() - t0
            vram = get_vram_usage()
            print(
                f"[epoch {epoch:02d}/{TRAIN_CONFIG['EPOCHS']}] "
                f"train={train_loss:.4f} val={val_loss:.4f} "
                f"acc={accuracy:.4f} macro_f1={macro_f1:.4f} "
                f"vram={vram['allocated_gb']:.1f}GB time={elapsed:.1f}s"
            )
            print("  F1: " + "  ".join(f"{k}={v:.3f}" for k, v in per_class_f1.items()))

            log["epochs"].append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "per_class_f1": per_class_f1,
                    "encoder_lr": optimizer.param_groups[0]["lr"],
                    "upper_lr": optimizer.param_groups[-1]["lr"],
                    "vram_gb": vram["allocated_gb"],
                    "time": elapsed,
                }
            )

            last_completed_epoch = epoch
            stop_now = early_stopping.step(macro_f1, model)
            save_resume_checkpoint(
                run_paths["resume_checkpoint"],
                experiment_name=experiment_name,
                run_id=run_id,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                early_stopping=early_stopping,
                lambda_alpha=lambda_alpha,
                focal_gamma=focal_gamma,
                log=log,
            )
            save_latest_run_metadata(
                experiment_name=experiment_name,
                run_id=run_id,
                best_checkpoint_path=run_paths["checkpoint"],
                resume_checkpoint_path=run_paths["resume_checkpoint"],
                train_log_path=run_paths["train_log"],
                lambda_alpha=lambda_alpha,
                last_completed_epoch=last_completed_epoch,
                best_macro_f1=early_stopping.best_f1,
            )
            if stop_now:
                print(f"\n[early stopping] no improvement for {TRAIN_CONFIG['PATIENCE']} epochs.")
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\n[interrupt] training interrupted. saving resumable checkpoint...")
        save_resume_checkpoint(
            run_paths["resume_checkpoint"],
            experiment_name=experiment_name,
            run_id=run_id,
            epoch=last_completed_epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            early_stopping=early_stopping,
            lambda_alpha=lambda_alpha,
            focal_gamma=focal_gamma,
            log=log,
        )
        save_latest_run_metadata(
            experiment_name=experiment_name,
            run_id=run_id,
            best_checkpoint_path=run_paths["checkpoint"],
            resume_checkpoint_path=run_paths["resume_checkpoint"],
            train_log_path=run_paths["train_log"],
            lambda_alpha=lambda_alpha,
            last_completed_epoch=last_completed_epoch,
            best_macro_f1=early_stopping.best_f1,
        )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with run_paths["train_log"].open("w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"\n[log] {run_paths['train_log']}")

    save_latest_run_metadata(
        experiment_name=experiment_name,
        run_id=run_id,
        best_checkpoint_path=run_paths["checkpoint"],
        resume_checkpoint_path=run_paths["resume_checkpoint"],
        train_log_path=run_paths["train_log"],
        lambda_alpha=lambda_alpha,
        last_completed_epoch=last_completed_epoch,
        best_macro_f1=early_stopping.best_f1,
    )
    if interrupted:
        print(f"[done] interrupted; resume from epoch {last_completed_epoch + 1}")
    else:
        print(f"[done] best macro_f1={early_stopping.best_f1:.4f}")
    return model, log


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Streaming Belief v5")
    parser.add_argument(
        "--experiment",
        type=str,
        default="v5_default",
        choices=list(ABLATION_CONFIGS.keys()),
    )
    parser.add_argument("--lambda-alpha", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-run-id", type=str, default=None)
    args = parser.parse_args()

    train(
        experiment_name=args.experiment,
        lambda_alpha_override=args.lambda_alpha,
        resume=args.resume,
        resume_path=args.resume_path,
        resume_run_id=args.resume_run_id,
    )

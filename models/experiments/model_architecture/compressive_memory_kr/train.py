"""
Training script for SNS dialogue-level Compressive Memory experiments.
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
    DATA_DIR,
    DEVICE,
    ENCODER_CONFIG,
    GPU_CONFIG,
    HEAD_CONFIG,
    LOG_DIR,
    MEMORY_CONFIG,
    SNS_CONFIG,
    TRAIN_CONFIG,
)
from dataset import create_dataloaders
from model import build_compressive_memory_model


def create_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def build_run_paths(experiment_name: str, run_id: str) -> dict[str, Path]:
    return {
        "checkpoint": CHECKPOINT_DIR / f"{experiment_name}_{run_id}_best.pt",
        "resume_state": CHECKPOINT_DIR / f"{experiment_name}_{run_id}_last_state.pt",
        "latest_state": CHECKPOINT_DIR / f"{experiment_name}_latest_state.pt",
        "train_log": LOG_DIR / f"{experiment_name}_{run_id}_train_log.json",
        "latest_meta": CHECKPOINT_DIR / f"{experiment_name}_latest.json",
    }


def save_latest_run_metadata(
    experiment_name: str,
    run_id: str,
    loss_fn: str,
    checkpoint_path: Path,
    train_log_path: Path,
    resume_state_path: Path | None = None,
) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": experiment_name,
        "run_id": run_id,
        "loss_fn": loss_fn,
        "checkpoint_path": str(checkpoint_path),
        "train_log_path": str(train_log_path),
        "resume_state_path": str(resume_state_path) if resume_state_path is not None else None,
        "timestamp": run_id,
    }
    with (CHECKPOINT_DIR / f"{experiment_name}_latest.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def resolve_resume_state_path(experiment_name: str, resume_path: str | None = None) -> Path | None:
    if resume_path:
        path = Path(resume_path)
        return path if path.exists() else None

    latest_meta_path = CHECKPOINT_DIR / f"{experiment_name}_latest.json"
    if latest_meta_path.exists():
        try:
            with latest_meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            state_path = meta.get("resume_state_path")
            if state_path:
                p = Path(state_path)
                if p.exists():
                    return p
        except Exception:
            pass

    fallback = CHECKPOINT_DIR / f"{experiment_name}_latest_state.pt"
    if fallback.exists():
        return fallback
    return None


def save_training_state(
    run_paths: dict[str, Path],
    experiment_name: str,
    run_id: str,
    epoch: int,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: GradScaler | None,
    best_f1: float,
    early_stop_counter: int,
    loss_fn: str,
    log: dict,
) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "experiment": experiment_name,
        "run_id": run_id,
        "epoch": int(epoch),
        "loss_fn": loss_fn,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_f1": float(best_f1),
        "early_stop_counter": int(early_stop_counter),
        "log": log,
    }
    torch.save(state, run_paths["resume_state"])
    torch.save(state, run_paths["latest_state"])


def load_training_state(
    state_path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: GradScaler | None,
) -> dict:
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])

    scaler_state = state.get("scaler_state_dict")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    return state


def build_config_snapshot(
    experiment_name: str,
    loss_fn: str,
    ablation_config: dict,
    effective_batch: int,
    accum_steps: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    steps_per_epoch: int,
    total_steps: int,
) -> dict:
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
            "head": dict(HEAD_CONFIG),
            "train": dict(TRAIN_CONFIG),
            "gpu": dict(GPU_CONFIG),
            "sns": dict(SNS_CONFIG),
            "loss_fn": loss_fn,
            "effective_batch_size": effective_batch,
            "gradient_accumulation_steps": accum_steps,
            "amp_enabled": bool(amp_enabled),
            "amp_dtype": str(amp_dtype),
            "steps_per_epoch": int(steps_per_epoch),
            "total_steps": int(total_steps),
        },
    }


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
    print(f"[gpu] VRAM {vram_total:.1f}GB (target {GPU_CONFIG['TARGET_VRAM_GB']:.1f}GB)")
    print(f"[gpu] mixed precision: {GPU_CONFIG['DTYPE']}")


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


def get_amp_dtype() -> torch.dtype:
    return torch.bfloat16 if GPU_CONFIG["DTYPE"] == "bf16" else torch.float16


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module) -> AdamW:
    encoder_params = []
    upper_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(param)
        else:
            upper_params.append(param)

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


def load_class_weights() -> torch.Tensor:
    path = DATA_DIR / "class_weight.json"
    if not path.exists():
        print("[weight] class_weight.json missing, fallback to ones")
        return torch.ones(SNS_CONFIG["NUM_LABELS"], dtype=torch.float, device=DEVICE)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    weights = data.get("class_weight", [1.0] * SNS_CONFIG["NUM_LABELS"])
    return torch.tensor(weights, dtype=torch.float, device=DEVICE)


def focal_loss_multiclass(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: torch.Tensor | None = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()
    ce = -log_pt

    if alpha is not None:
        ce = ce * alpha[targets]

    return ((1.0 - pt) ** gamma * ce).mean()


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: str,
    class_weight: torch.Tensor,
    focal_gamma: float,
) -> torch.Tensor:
    if loss_fn == "cross_entropy":
        return F.cross_entropy(logits, labels)
    if loss_fn == "weighted_ce":
        return F.cross_entropy(logits, labels, weight=class_weight)
    if loss_fn == "focal":
        return focal_loss_multiclass(logits, labels, alpha=class_weight, gamma=focal_gamma)
    raise ValueError(f"Unknown loss function: {loss_fn}")


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
    loss_fn: str,
    class_weight: torch.Tensor,
    focal_gamma: float,
    accum_steps: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        turn_mask = batch["turn_mask"].to(DEVICE, non_blocking=True)
        num_turns = batch["num_turns"].to(DEVICE, non_blocking=True)
        labels = batch["dialogue_labels"].to(DEVICE, non_blocking=True)

        with autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                turn_mask=turn_mask,
                num_turns=num_turns,
            )
            loss = compute_loss(
                logits=outputs["logits"],
                labels=labels,
                loss_fn=loss_fn,
                class_weight=class_weight,
                focal_gamma=focal_gamma,
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

        cache_interval = GPU_CONFIG["EMPTY_CACHE_INTERVAL"]
        if cache_interval > 0 and torch.cuda.is_available() and (batch_idx + 1) % cache_interval == 0:
            torch.cuda.empty_cache()

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    loss_fn: str,
    class_weight: torch.Tensor,
    focal_gamma: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    y_true = []
    y_pred = []
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        turn_mask = batch["turn_mask"].to(DEVICE, non_blocking=True)
        num_turns = batch["num_turns"].to(DEVICE, non_blocking=True)
        labels = batch["dialogue_labels"].to(DEVICE, non_blocking=True)

        with autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                turn_mask=turn_mask,
                num_turns=num_turns,
            )
            loss = compute_loss(
                logits=outputs["logits"],
                labels=labels,
                loss_fn=loss_fn,
                class_weight=class_weight,
                focal_gamma=focal_gamma,
            )

        total_loss += loss.item()
        num_batches += 1

        preds = outputs["logits"].argmax(dim=-1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    val_loss = total_loss / max(num_batches, 1)

    if not y_true:
        zero_per_class = {name: 0.0 for name in SNS_CONFIG["LABELS"]}
        return val_loss, {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "per_class_f1": zero_per_class,
            "per_class_accuracy": zero_per_class,
        }

    labels_idx = list(range(SNS_CONFIG["NUM_LABELS"]))
    per_class_arr = f1_score(y_true, y_pred, labels=labels_idx, average=None, zero_division=0)
    per_class_f1 = {
        name: round(float(per_class_arr[i]), 4)
        for i, name in enumerate(SNS_CONFIG["LABELS"])
    }
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    per_class_accuracy = {}
    for i, name in enumerate(SNS_CONFIG["LABELS"]):
        class_mask = y_true_arr == i
        support = int(class_mask.sum())
        if support == 0:
            per_class_accuracy[name] = 0.0
            continue
        correct = int((y_pred_arr[class_mask] == i).sum())
        per_class_accuracy[name] = round(float(correct / support), 4)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_f1": per_class_f1,
        "per_class_accuracy": per_class_accuracy,
    }
    return val_loss, metrics


def train(
    experiment_name: str = "full",
    loss_fn: str | None = None,
    num_workers: int | None = None,
    batch_size: int | None = None,
    accum_steps: int | None = None,
    resume: bool = False,
    resume_path: str | None = None,
):
    set_seed(TRAIN_CONFIG["SEED"])
    init_gpu()

    amp_enabled = GPU_CONFIG["ENABLED"] and torch.cuda.is_available()
    amp_dtype = get_amp_dtype()
    train_batch_size = int(batch_size or TRAIN_CONFIG["BATCH_SIZE"])
    accum_steps = int(accum_steps or GPU_CONFIG["GRADIENT_ACCUMULATION_STEPS"])
    scaler = GradScaler() if (amp_enabled and GPU_CONFIG["DTYPE"] == "fp16") else None

    ablation_config = ABLATION_CONFIGS.get(experiment_name, {})
    _loss_fn = loss_fn or TRAIN_CONFIG["LOSS_FN"]
    focal_gamma = float(TRAIN_CONFIG["FOCAL_GAMMA"])
    run_id = create_run_id()
    run_paths = build_run_paths(experiment_name, run_id)

    print(f"\n{'=' * 60}")
    print(f"[train] experiment: {experiment_name}")
    print(f"  run_id: {run_id}")
    print(f"  description: {ablation_config.get('description', 'default')}")
    print(f"  loss_fn: {_loss_fn} (gamma={focal_gamma})")
    print(f"  device: {DEVICE}")
    print(f"  mixed precision: {GPU_CONFIG['DTYPE'] if amp_enabled else 'OFF'}")
    print(f"  batch: {train_batch_size} x {accum_steps} = {train_batch_size * accum_steps}")
    print(f"  resume: {resume} ({resume_path if resume_path else 'auto'})")
    print(f"{'=' * 60}\n")

    loaders = create_dataloaders(
        DATA_DIR,
        batch_size=train_batch_size,
        num_workers=num_workers,
        splits=("train", "val"),
    )
    if "train" not in loaders or "val" not in loaders:
        raise FileNotFoundError("train.json or val.json is missing. Run data_preprocessing.py first.")

    model = build_compressive_memory_model(ablation_config).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] total params: {total_params:,}")
    print(f"[model] trainable params: {trainable_params:,}")

    vram = get_vram_usage()
    print(f"[gpu] after model load: {vram['allocated_gb']:.2f}GB used / {vram['free_gb']:.2f}GB free\n")

    class_weight = load_class_weights()
    optimizer = build_optimizer(model)
    steps_per_epoch = len(loaders["train"]) // accum_steps + (1 if len(loaders["train"]) % accum_steps else 0)
    total_steps = TRAIN_CONFIG["EPOCHS"] * steps_per_epoch
    scheduler = build_scheduler(optimizer, total_steps)
    effective_batch = train_batch_size * accum_steps

    config_snapshot = build_config_snapshot(
        experiment_name=experiment_name,
        loss_fn=_loss_fn,
        ablation_config=ablation_config,
        effective_batch=effective_batch,
        accum_steps=accum_steps,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
    )

    early_stopping = EarlyStopping(
        patience=TRAIN_CONFIG["PATIENCE"],
        min_delta=TRAIN_CONFIG["MIN_DELTA"],
        checkpoint_path=run_paths["checkpoint"],
    )

    log = {
        "experiment": experiment_name,
        "run_id": run_id,
        "loss_fn": _loss_fn,
        "checkpoint_path": str(run_paths["checkpoint"]),
        "config_snapshot": config_snapshot,
        "epochs": [],
    }
    start_epoch = 1

    if resume:
        state_path = resolve_resume_state_path(experiment_name, resume_path)
        if state_path is None:
            print("[resume] no state file found. start fresh training.")
        else:
            print(f"[resume] loading state: {state_path}")
            state = load_training_state(
                state_path=state_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )
            resumed_run_id = state.get("run_id", run_id)
            if resumed_run_id != run_id:
                run_id = resumed_run_id
                run_paths = build_run_paths(experiment_name, run_id)
                print(f"[resume] continue run_id: {run_id}")
            start_epoch = int(state.get("epoch", 0)) + 1
            early_stopping.best_f1 = float(state.get("best_f1", -1.0))
            early_stopping.counter = int(state.get("early_stop_counter", 0))
            if isinstance(state.get("log"), dict):
                log = state["log"]
            if "config_snapshot" not in log:
                log["config_snapshot"] = config_snapshot
            print(
                f"[resume] start epoch={start_epoch}, "
                f"best_macro_f1={early_stopping.best_f1:.4f}, "
                f"early_counter={early_stopping.counter}"
            )

    try:
        for epoch in range(start_epoch, TRAIN_CONFIG["EPOCHS"] + 1):
            if hasattr(loaders["train"], "batch_sampler") and hasattr(loaders["train"].batch_sampler, "set_epoch"):
                loaders["train"].batch_sampler.set_epoch(epoch)

            start = time.time()
            train_loss = train_one_epoch(
                model=model,
                loader=loaders["train"],
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                loss_fn=_loss_fn,
                class_weight=class_weight,
                focal_gamma=focal_gamma,
                accum_steps=accum_steps,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            val_loss, val_metrics = validate(
                model=model,
                loader=loaders["val"],
                loss_fn=_loss_fn,
                class_weight=class_weight,
                focal_gamma=focal_gamma,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )

            elapsed = time.time() - start
            encoder_lr = optimizer.param_groups[0]["lr"]
            upper_lr = optimizer.param_groups[1]["lr"]
            vram = get_vram_usage()

            macro_f1 = val_metrics["macro_f1"]
            weighted_f1 = val_metrics["weighted_f1"]
            accuracy = val_metrics["accuracy"]

            print(
                f"[epoch {epoch:02d}/{TRAIN_CONFIG['EPOCHS']}] "
                f"train={train_loss:.4f} val={val_loss:.4f} "
                f"acc={accuracy:.4f} macro_f1={macro_f1:.4f} weighted_f1={weighted_f1:.4f} "
                f"enc_lr={encoder_lr:.2e} upper_lr={upper_lr:.2e} "
                f"vram={vram['allocated_gb']:.1f}GB time={elapsed:.1f}s"
            )
            cls_acc_line = "  per_class_acc: " + "  ".join(
                f"{name}={score:.3f}" for name, score in val_metrics["per_class_accuracy"].items()
            )
            print(cls_acc_line)

            log["epochs"].append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "weighted_f1": weighted_f1,
                    "per_class_f1": val_metrics["per_class_f1"],
                    "per_class_accuracy": val_metrics["per_class_accuracy"],
                    "encoder_lr": encoder_lr,
                    "upper_lr": upper_lr,
                    "vram_gb": vram["allocated_gb"],
                    "time": elapsed,
                }
            )

            should_stop = early_stopping.step(macro_f1, model)
            save_training_state(
                run_paths=run_paths,
                experiment_name=experiment_name,
                run_id=run_id,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_f1=early_stopping.best_f1,
                early_stop_counter=early_stopping.counter,
                loss_fn=_loss_fn,
                log=log,
            )

            if should_stop:
                print(f"\n[early stopping] no improvement for {TRAIN_CONFIG['PATIENCE']} epochs.")
                break
    except KeyboardInterrupt:
        print("\n[interrupt] keyboard interrupt detected. latest saved state is kept for resume.")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with run_paths["train_log"].open("w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"\n[log] {run_paths['train_log']}")

    save_latest_run_metadata(
        experiment_name=experiment_name,
        run_id=run_id,
        loss_fn=_loss_fn,
        checkpoint_path=run_paths["checkpoint"],
        train_log_path=run_paths["train_log"],
        resume_state_path=run_paths["latest_state"],
    )
    print(f"[meta] {run_paths['latest_meta']}")

    vram = get_vram_usage()
    print(f"[gpu] final: {vram['allocated_gb']:.2f}GB used / {vram['free_gb']:.2f}GB free")
    print(f"[done] best checkpoint: {run_paths['checkpoint']}")
    print(f"[done] best macro_f1: {early_stopping.best_f1:.4f}")

    return model, log


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Compressive Memory KR model")
    parser.add_argument(
        "--experiment",
        type=str,
        default="full",
        choices=list(ABLATION_CONFIGS.keys()),
        help="Experiment name",
    )
    parser.add_argument(
        "--loss-fn",
        type=str,
        default=None,
        choices=["cross_entropy", "focal", "weighted_ce"],
        help="Loss function override",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader worker override",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size override (default from config)",
    )
    parser.add_argument(
        "--accum-steps",
        type=int,
        default=None,
        help="Gradient accumulation steps override (default from config)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest saved training state for this experiment",
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default=None,
        help="Resume from a specific training state path (*.pt)",
    )
    args = parser.parse_args()

    train(
        experiment_name=args.experiment,
        loss_fn=args.loss_fn,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        resume=args.resume,
        resume_path=args.resume_path,
    )

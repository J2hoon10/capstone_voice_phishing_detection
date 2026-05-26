"""
roberta_mamba_freeze_init / train.py

freeze_init 전략:
  epoch 1 ~ FREEZE_INIT_EPOCHS(=3) : encoder 전체 freeze → Mamba + head 만 학습
  epoch FREEZE_INIT_EPOCHS+1 ~     : encoder layer[10~11] unfreeze → fine-tune
  embeddings + layer[0~9] 는 끝까지 freeze 유지

스케줄러:
  - encoder group : 1e-5 → cosine → 1e-6  (ENCODER_MIN_LR)
  - Mamba+Head group : 5e-4 → cosine → 1e-4 (UPPER_MIN_LR)
  두 그룹에 독립적인 min_lr 를 적용한 per-group LambdaLR

비등방성(Anisotropy) 지표:
  모델 forward() 에서 "anisotropy_metric" 으로 반환된 평균 코사인 유사도를
  학습/검증 루프에서 수집하여 JSON 로그에 기록.

  [WandB / TensorBoard 연동 방법]
  학습 스텝 로그 레코드에 "anisotropy" 키가 포함되어 있습니다.
  아래 코드를 `run_epoch` 내 LOG_STEP_INTERVAL 블록 또는
  `train()` 의 epoch 루프 끝에 삽입하면 됩니다:

    # WandB
    import wandb
    wandb.log({"train/anisotropy": avg_anisotropy, "step": global_step})
    wandb.log({"epoch/train_anisotropy": tr_anisotropy, "epoch/val_anisotropy": val["anisotropy"],
                "epoch": epoch})

    # TensorBoard
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=str(LOG_DIR))
    writer.add_scalar("train/anisotropy", avg_anisotropy, global_step)
    writer.add_scalar("epoch/train_anisotropy", tr_anisotropy, epoch)
    writer.add_scalar("epoch/val_anisotropy", val["anisotropy"], epoch)
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import (
    CHECKPOINT_DIR,
    DATA_DIR,
    DEVICE,
    LABELS,
    LOG_DIR,
    LOSS_CONFIG,
    TRAIN_CONFIG,
    WINDOW_SIZE,
    STRIDE,
)
from dataset import create_dataloader
from losses import HierarchicalCrossEntropyLoss, OrdinalRegressionLoss
from model import build_model

LOG_STEP_INTERVAL = 200
RESUME_CKPT = CHECKPOINT_DIR / "resume_latest.pt"
EXP_NAME = "roberta_mamba_freeze_init"
FREEZE_INIT_EPOCHS = TRAIN_CONFIG["FREEZE_INIT_EPOCHS"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_encoder_all(model: nn.Module) -> None:
    for p in model.encoder.parameters():
        p.requires_grad = False


def unfreeze_encoder_upper(model: nn.Module, freeze_lower: int = 10) -> None:
    encoder = model.encoder
    layers = None
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        layers = encoder.encoder.layer
    elif hasattr(encoder, "electra") and hasattr(encoder.electra, "encoder"):
        if hasattr(encoder.electra.encoder, "layer"):
            layers = encoder.electra.encoder.layer
    if layers is None:
        return
    n = min(freeze_lower, len(layers))
    for i in range(n, len(layers)):
        for p in layers[i].parameters():
            p.requires_grad = True


def build_optimizer(model: nn.Module) -> AdamW:
    enc, upper = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder."):
            enc.append(p)
        else:
            upper.append(p)
    groups = []
    if enc:
        groups.append({"params": enc, "lr": TRAIN_CONFIG["ENCODER_LR"]})
    if upper:
        groups.append({"params": upper, "lr": TRAIN_CONFIG["UPPER_LR"]})
    return AdamW(groups, weight_decay=TRAIN_CONFIG["WEIGHT_DECAY"])


def build_scheduler(optimizer: AdamW, total_steps: int) -> LambdaLR:
    """
    per-group 독립 LambdaLR.
      - encoder group  : ENCODER_LR(1e-5) → cosine → ENCODER_MIN_LR(1e-6)
      - upper group    : UPPER_LR(5e-4)   → cosine → UPPER_MIN_LR(1e-4)

    param_groups 순서는 build_optimizer() 기준 [encoder, upper].
    """
    warmup_steps = int(total_steps * TRAIN_CONFIG["WARMUP_RATIO"])
    # build_optimizer 가 생성하는 순서: 0=encoder, 1=upper
    group_min_lrs = [
        TRAIN_CONFIG["ENCODER_MIN_LR"],
        TRAIN_CONFIG["UPPER_MIN_LR"],
    ]

    def make(base_lr: float, min_lr: float):
        min_ratio = min(1.0, min_lr / max(base_lr, 1e-12))

        def lr_lambda(step: int):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return lr_lambda

    lambdas = [
        make(g["lr"], group_min_lrs[i] if i < len(group_min_lrs) else group_min_lrs[-1])
        for i, g in enumerate(optimizer.param_groups)
    ]
    return LambdaLR(optimizer, lambdas)


def curriculum_beta(epoch: int, total_epochs: int) -> float:
    beta_s = LOSS_CONFIG["AUX_BETA_START"]
    beta_e = LOSS_CONFIG["AUX_BETA_END"]
    ratio = min(1.0, (epoch - 1) / max(total_epochs - 1, 1))
    return beta_s + (beta_e - beta_s) * ratio


class Logger:
    def __init__(self, log_dir: Path, run_id: str) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self.text_path = log_dir / f"{run_id}_train.log"
        self.json_path = log_dir / f"{run_id}_train.json"
        self.records: list[dict] = []

    def write(self, record: dict) -> None:
        self.records.append(record)
        with self.text_path.open("a", encoding="utf-8") as f:
            t = record["type"]
            if t == "step":
                f.write(
                    f"[step {record['global_step']:6d}] "
                    f"epoch={record['epoch']} "
                    f"loss={record['loss']:.4f} "
                    f"aniso={record.get('anisotropy', 0.0):.4f} "
                    f"lr={record['lr']:.2e} "
                    f"beta={record['beta']:.4f} "
                    f"elapsed={record['elapsed']:.0f}s\n"
                )
            elif t == "epoch":
                suffix = " <- best" if record.get("is_best") else ""
                f.write(
                    f"[epoch {record['epoch']:3d}] "
                    f"train={record['train_loss']:.4f} "
                    f"val={record['val_loss']:.4f} "
                    f"acc={record['val_acc']:.4f} "
                    f"macro_f1={record['val_macro_f1']:.4f} "
                    f"weighted_f1={record['val_weighted_f1']:.4f} "
                    f"tr_aniso={record.get('train_anisotropy', 0.0):.4f} "
                    f"val_aniso={record.get('val_anisotropy', 0.0):.4f}"
                    f"{suffix}\n"
                )
                for label, score in record["per_class_f1"].items():
                    f.write(f"          {label}: {score:.4f}\n")
            elif t == "freeze_event":
                f.write(f"[freeze] epoch={record['epoch']} event={record['event']}\n")
            elif t == "resume":
                f.write(
                    f"[resume] epoch {record['from_epoch']}까지 완료, "
                    f"global_step={record['global_step']}, "
                    f"best_f1={record['best_f1']:.4f}\n"
                )
            elif t == "start":
                f.write(f"[start] run_id={record['run_id']} {record['timestamp']}\n")

    def flush_json(self) -> None:
        with self.json_path.open("w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)


def run_epoch(
    model: nn.Module,
    loader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    criterion_main: HierarchicalCrossEntropyLoss,
    criterion_aux: OrdinalRegressionLoss,
    epoch: int,
    global_step: int,
    logger: Logger,
    t0: float,
    beta: float,
) -> tuple[float, float, int]:
    """학습 1 epoch 실행. (train_loss, epoch_anisotropy, global_step) 반환."""
    model.train()
    total_loss = 0.0
    step_loss_buf: list[float] = []
    step_aniso_buf: list[float] = []   # step-level anisotropy 버퍼
    all_aniso: list[float] = []        # epoch 평균용

    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(DEVICE)

        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            segment_mask=batch["segment_mask"],
            num_segments=batch["num_segments"],
        )

        loss_main = criterion_main(
            super_logits=out["super_logits"],
            normal_logits=out["normal_logits"],
            phishing_logits=out["phishing_logits"],
            targets=batch["labels"],
        )

        aux_logits = out["aux_logits"]
        T = aux_logits.shape[1]
        segment_risks = batch["segment_risks"][:, :T]
        loss_aux = criterion_aux(aux_logits, segment_risks)

        loss = loss_main + beta * loss_aux

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["MAX_GRAD_NORM"])
        optimizer.step()
        scheduler.step()

        loss_val = float(loss.item())
        aniso_val = float(out.get("anisotropy_metric", torch.tensor(0.0)).item())
        total_loss += loss_val
        step_loss_buf.append(loss_val)
        step_aniso_buf.append(aniso_val)
        all_aniso.append(aniso_val)
        global_step += 1

        if global_step % LOG_STEP_INTERVAL == 0:
            avg_loss = sum(step_loss_buf) / len(step_loss_buf)
            avg_aniso = sum(step_aniso_buf) / len(step_aniso_buf)
            step_loss_buf.clear()
            step_aniso_buf.clear()
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            record = {
                "type": "step",
                "global_step": global_step,
                "epoch": epoch,
                "loss": avg_loss,
                "anisotropy": avg_aniso,
                "lr": lr,
                "beta": beta,
                "elapsed": elapsed,
            }
            logger.write(record)
            print(
                f"  [step {global_step:6d}] epoch={epoch} "
                f"loss={avg_loss:.4f} aniso={avg_aniso:.4f} "
                f"lr={lr:.2e} beta={beta:.3f} elapsed={elapsed:.0f}s"
            )

    epoch_anisotropy = sum(all_aniso) / max(len(all_aniso), 1)
    return total_loss / max(len(loader), 1), epoch_anisotropy, global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion_main: HierarchicalCrossEntropyLoss,
    criterion_aux: OrdinalRegressionLoss,
    beta: float,
) -> dict:
    model.eval()
    total = 0.0
    y_true, y_pred = [], []
    aniso_vals: list[float] = []

    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(DEVICE)

        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            segment_mask=batch["segment_mask"],
            num_segments=batch["num_segments"],
        )

        loss_main = criterion_main(
            super_logits=out["super_logits"],
            normal_logits=out["normal_logits"],
            phishing_logits=out["phishing_logits"],
            targets=batch["labels"],
        )
        aux_logits = out["aux_logits"]
        T = aux_logits.shape[1]
        segment_risks = batch["segment_risks"][:, :T]
        loss_aux = criterion_aux(aux_logits, segment_risks)
        loss = loss_main + beta * loss_aux

        total += float(loss.item())
        pred = out["logits"].argmax(dim=-1)
        y_true.extend(batch["labels"].cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
        aniso_vals.append(float(out.get("anisotropy_metric", torch.tensor(0.0)).item()))

    per_class_f1 = f1_score(
        y_true, y_pred, average=None, zero_division=0, labels=list(range(len(LABELS)))
    )
    return {
        "loss": total / max(len(loader), 1),
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_f1": {LABELS[i]: round(float(v), 4) for i, v in enumerate(per_class_f1)},
        "anisotropy": sum(aniso_vals) / max(len(aniso_vals), 1),
    }


def save_resume_ckpt(
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    epoch: int,
    global_step: int,
    best_f1: float,
    best_path: Path,
    run_id: str,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_f1": best_f1,
            "best_path": str(best_path),
            "run_id": run_id,
        },
        RESUME_CKPT,
    )


def train(resume: bool = False) -> None:
    set_seed(TRAIN_CONFIG["SEED"])
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    train_loader = create_dataloader(DATA_DIR / "train.csv", TRAIN_CONFIG["BATCH_SIZE"], True)
    val_loader = create_dataloader(DATA_DIR / "val.csv", TRAIN_CONFIG["BATCH_SIZE"], False)

    # ── 모델: unfreeze_all=True (모든 파라미터 trainable) ──────────────────────
    model = build_model().to(DEVICE)

    # ── Optimizer: encoder 파라미터도 처음부터 포함 ────────────────────────────
    # freeze 후에도 optimizer 내 등록은 유지 → unfreeze 시 add_param_group 불필요
    optimizer = build_optimizer(model)
    total_steps = TRAIN_CONFIG["EPOCHS"] * max(len(train_loader), 1)
    scheduler = build_scheduler(optimizer, total_steps)
    criterion_main = HierarchicalCrossEntropyLoss()
    criterion_aux = OrdinalRegressionLoss(ignore_index=-100)

    start_epoch = 1
    global_step = 0
    best_f1 = -1.0

    if resume and RESUME_CKPT.exists():
        ckpt = torch.load(RESUME_CKPT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_f1 = ckpt["best_f1"]
        best_path = Path(ckpt["best_path"])
        run_id = ckpt["run_id"]
        print(
            f"[resume] epoch {start_epoch}부터 재개 "
            f"(global_step={global_step}, best_f1={best_f1:.4f})"
        )
    elif resume:
        print(f"[warn] --resume 지정했지만 {RESUME_CKPT} 없음 → 처음부터 시작")
        run_id = time.strftime("%Y%m%d_%H%M%S")
        best_path = CHECKPOINT_DIR / f"{EXP_NAME}_{run_id}_best.pt"
    else:
        run_id = time.strftime("%Y%m%d_%H%M%S")
        best_path = CHECKPOINT_DIR / f"{EXP_NAME}_{run_id}_best.pt"

    # ── freeze_init 상태 복원 ─────────────────────────────────────────────────
    if start_epoch <= FREEZE_INIT_EPOCHS:
        freeze_encoder_all(model)
        print(
            f"[freeze_init] encoder 완전 freeze "
            f"(epoch {start_epoch}~{FREEZE_INIT_EPOCHS})"
        )
    else:
        unfreeze_encoder_upper(model, freeze_lower=10)
        print(f"[freeze_init] encoder layer[10~11] 이미 unfreeze 상태로 시작")

    logger = Logger(LOG_DIR, run_id)
    logger.write({"type": "start", "run_id": run_id, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})

    if resume and RESUME_CKPT.exists():
        logger.write({
            "type": "resume",
            "from_epoch": start_epoch - 1,
            "global_step": global_step,
            "best_f1": best_f1,
        })

    t0 = time.time()
    total_epochs = TRAIN_CONFIG["EPOCHS"]
    print(f"[train] {EXP_NAME}  run_id={run_id}  log={logger.text_path}")
    print(
        f"[train] WINDOW_SIZE={WINDOW_SIZE} STRIDE={STRIDE} "
        f"BATCH_SIZE={TRAIN_CONFIG['BATCH_SIZE']} "
        f"MAMBA_LAYERS=1 freeze_init_epochs={FREEZE_INIT_EPOCHS}"
    )
    print(
        f"[train] encoder_lr={TRAIN_CONFIG['ENCODER_LR']:.0e}→{TRAIN_CONFIG['ENCODER_MIN_LR']:.0e} "
        f"upper_lr={TRAIN_CONFIG['UPPER_LR']:.0e}→{TRAIN_CONFIG['UPPER_MIN_LR']:.0e} "
        f"(cosine)"
    )
    print(
        f"[train] epochs={start_epoch}~{total_epochs}  "
        f"steps_per_epoch≈{len(train_loader)}  device={DEVICE}"
    )

    for epoch in range(start_epoch, total_epochs + 1):
        # ── freeze_init: unfreeze 시점 ──────────────────────────────────────
        if epoch == FREEZE_INIT_EPOCHS + 1:
            unfreeze_encoder_upper(model, freeze_lower=10)
            print(f"[freeze_init] epoch {epoch}: encoder layer[10~11] unfreeze → fine-tune 시작")
            logger.write({"type": "freeze_event", "epoch": epoch, "event": "encoder_upper_unfreeze"})

        beta = curriculum_beta(epoch, total_epochs)

        tr_loss, tr_anisotropy, global_step = run_epoch(
            model, train_loader, optimizer, scheduler,
            criterion_main, criterion_aux,
            epoch, global_step, logger, t0, beta,
        )
        val = evaluate(model, val_loader, criterion_main, criterion_aux, beta)
        is_best = val["macro_f1"] > best_f1

        epoch_record = {
            "type": "epoch",
            "epoch": epoch,
            "beta": beta,
            "encoder_frozen": epoch <= FREEZE_INIT_EPOCHS,
            "train_loss": tr_loss,
            "train_anisotropy": tr_anisotropy,
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_macro_f1": val["macro_f1"],
            "val_weighted_f1": val["weighted_f1"],
            "val_anisotropy": val["anisotropy"],
            "per_class_f1": val["per_class_f1"],
            "is_best": is_best,
            "global_step": global_step,
        }
        logger.write(epoch_record)
        logger.flush_json()

        frozen_tag = " [enc frozen]" if epoch <= FREEZE_INIT_EPOCHS else ""
        suffix = " <- best" if is_best else ""
        print(
            f"[epoch {epoch}]{frozen_tag} β={beta:.3f} train={tr_loss:.4f} val={val['loss']:.4f} "
            f"acc={val['acc']:.4f} macro_f1={val['macro_f1']:.4f} "
            f"weighted_f1={val['weighted_f1']:.4f} "
            f"tr_aniso={tr_anisotropy:.4f} val_aniso={val['anisotropy']:.4f}"
            f"{suffix}"
        )
        for label, score in val["per_class_f1"].items():
            print(f"  {label}: {score:.4f}")

        if is_best:
            best_f1 = val["macro_f1"]
            torch.save(model.state_dict(), best_path)
            print(f"  [checkpoint] saved: {best_path.name}")

        save_resume_ckpt(model, optimizer, scheduler, epoch, global_step, best_f1, best_path, run_id)

    latest_meta = CHECKPOINT_DIR / f"{EXP_NAME}_latest.json"
    with latest_meta.open("w", encoding="utf-8") as f:
        json.dump(
            {"checkpoint_path": str(best_path), "best_macro_f1": best_f1, "run_id": run_id},
            f, ensure_ascii=False, indent=2,
        )

    print(f"[done] best_f1={best_f1:.4f}  log={logger.text_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", action="store_true",
        help="checkpoints/resume_latest.pt 에서 이어서 학습",
    )
    args = parser.parse_args()
    train(resume=args.resume)

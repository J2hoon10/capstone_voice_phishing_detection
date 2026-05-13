"""
koelectra_mamba_short_window / train.py

hce_ordinal 대비 전처리 변경 사항:
  - WINDOW_SIZE: 128 → 64
  - STRIDE:      100 → 32
  - BATCH_SIZE:  8   → 4  (세그먼트 수 증가에 따른 메모리 대응)
학습 전략(HCE + Ordinal + Curriculum β)은 동일.
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
)
from dataset import create_dataloader
from losses import HierarchicalCrossEntropyLoss, OrdinalRegressionLoss
from model import build_model

LOG_STEP_INTERVAL = 200
RESUME_CKPT = CHECKPOINT_DIR / "resume_latest.pt"
EXP_NAME = "koelectra_mamba_short_window"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    warmup_steps = int(total_steps * TRAIN_CONFIG["WARMUP_RATIO"])

    def make(base_lr: float):
        min_ratio = min(1.0, TRAIN_CONFIG["MIN_LR"] / max(base_lr, 1e-12))

        def lr_lambda(step: int):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return lr_lambda

    return LambdaLR(optimizer, [make(g["lr"]) for g in optimizer.param_groups])


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
                    f"weighted_f1={record['val_weighted_f1']:.4f}"
                    f"{suffix}\n"
                )
                for label, score in record["per_class_f1"].items():
                    f.write(f"          {label}: {score:.4f}\n")
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
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    step_loss_buf: list[float] = []

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
        total_loss += loss_val
        step_loss_buf.append(loss_val)
        global_step += 1

        if global_step % LOG_STEP_INTERVAL == 0:
            avg_loss = sum(step_loss_buf) / len(step_loss_buf)
            step_loss_buf.clear()
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            record = {
                "type": "step",
                "global_step": global_step,
                "epoch": epoch,
                "loss": avg_loss,
                "lr": lr,
                "beta": beta,
                "elapsed": elapsed,
            }
            logger.write(record)
            print(
                f"  [step {global_step:6d}] epoch={epoch} "
                f"loss={avg_loss:.4f} lr={lr:.2e} beta={beta:.3f} elapsed={elapsed:.0f}s"
            )

    return total_loss / max(len(loader), 1), global_step


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

    per_class_f1 = f1_score(
        y_true, y_pred, average=None, zero_division=0, labels=list(range(len(LABELS)))
    )
    return {
        "loss": total / max(len(loader), 1),
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_f1": {LABELS[i]: round(float(v), 4) for i, v in enumerate(per_class_f1)},
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

    model = build_model().to(DEVICE)
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
    print(f"[train] run_id={run_id}  log={logger.text_path}")
    print(
        f"[train] WINDOW_SIZE={64} STRIDE={32} BATCH_SIZE={TRAIN_CONFIG['BATCH_SIZE']}"
    )
    print(
        f"[train] epochs={start_epoch}~{total_epochs}  "
        f"steps_per_epoch≈{len(train_loader)}  device={DEVICE}"
    )
    print(
        f"[train] loss=HCE (lambda={LOSS_CONFIG['HCE_LAMBDA']}, "
        f"focal_gamma={LOSS_CONFIG['FOCAL_GAMMA']}) + "
        f"OrdinalRegression (beta: {LOSS_CONFIG['AUX_BETA_START']}→{LOSS_CONFIG['AUX_BETA_END']})"
    )

    for epoch in range(start_epoch, total_epochs + 1):
        beta = curriculum_beta(epoch, total_epochs)

        tr_loss, global_step = run_epoch(
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
            "train_loss": tr_loss,
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_macro_f1": val["macro_f1"],
            "val_weighted_f1": val["weighted_f1"],
            "per_class_f1": val["per_class_f1"],
            "is_best": is_best,
            "global_step": global_step,
        }
        logger.write(epoch_record)
        logger.flush_json()

        suffix = " <- best" if is_best else ""
        print(
            f"[epoch {epoch}] β={beta:.3f} train={tr_loss:.4f} val={val['loss']:.4f} "
            f"acc={val['acc']:.4f} macro_f1={val['macro_f1']:.4f} "
            f"weighted_f1={val['weighted_f1']:.4f}{suffix}"
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

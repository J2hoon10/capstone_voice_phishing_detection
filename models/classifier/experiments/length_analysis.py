"""
length_analysis.py

대화 길이(세그먼트 수)에 따른 모델별 성능 비교.
4개 모델(avgpool, mamba, gru, lstm)을 동일한 test.csv로 추론하고
버킷별 Accuracy / Macro F1 을 출력한다.

버킷 정의:
  Short     : 1 ~ 4  세그먼트
  Medium    : 5 ~ 8  세그먼트
  Long      : 9 ~ 15 세그먼트
  Very Long : 16+    세그먼트

사용법:
  python length_analysis.py
  python length_analysis.py --split val
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

# ── 실험 디렉토리 정의 ────────────────────────────────────────────────────────
EXPERIMENTS_DIR = Path(__file__).resolve().parent

EXPERIMENTS = [
    {
        "name": "roberta_avgpool_4class",
        "dir": EXPERIMENTS_DIR / "roberta_avgpool_4class",
    },
    {
        "name": "roberta_mamba_freeze_init_4class",
        "dir": EXPERIMENTS_DIR / "roberta_mamba_freeze_init_4class",
    },
    {
        "name": "roberta_gru_freeze_init_4class",
        "dir": EXPERIMENTS_DIR / "roberta_gru_freeze_init_4class",
    },
    {
        "name": "roberta_lstm_freeze_init_4class",
        "dir": EXPERIMENTS_DIR / "roberta_lstm_freeze_init_4class",
    },
]

# 버킷 정의: (레이블, min_segments, max_segments)
BUCKETS = [
    ("Short (1-4)",     1,  4),
    ("Medium (5-8)",    5,  8),
    ("Long (9-15)",     9, 15),
    ("Very Long (16+)", 16, 9999),
]


# ── 동적 모듈 로더 ────────────────────────────────────────────────────────────

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_experiment(exp: dict):
    """실험 디렉토리에서 config / dataset / model 모듈을 동적으로 로드."""
    exp_dir = exp["dir"]
    prefix = exp["name"]

    # 이전 실험의 모듈 캐시 제거 (config 충돌 방지)
    for key in ["config", "dataset", "model", "losses"]:
        sys.modules.pop(key, None)

    # sys.path에 실험 디렉토리 추가 (내부 상대 import 지원)
    sys.path.insert(0, str(exp_dir))
    try:
        cfg = _load_module(f"{prefix}_config", exp_dir / "config.py")
        # config 모듈을 단순 이름으로도 등록해 내부 from config import ... 지원
        sys.modules["config"] = cfg
        ds  = _load_module(f"{prefix}_dataset", exp_dir / "dataset.py")
        mdl = _load_module(f"{prefix}_model",   exp_dir / "model.py")
    finally:
        sys.path.pop(0)

    return cfg, ds, mdl


def resolve_checkpoint(exp_name: str, ckpt_dir: Path) -> Path:
    latest_meta = ckpt_dir / f"{exp_name}_latest.json"
    if not latest_meta.exists():
        raise FileNotFoundError(f"latest.json 없음: {latest_meta}")
    with latest_meta.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return Path(meta["checkpoint_path"])


# ── 추론 ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(y_true, y_pred, num_segments) 배열 반환."""
    model.eval()
    all_true, all_pred, all_nseg = [], [], []
    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(device)
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            segment_mask=batch["segment_mask"],
            num_segments=batch["num_segments"],
        )
        all_pred.append(out["logits"].argmax(dim=-1).cpu().numpy())
        all_true.append(batch["labels"].cpu().numpy())
        all_nseg.append(batch["num_segments"].cpu().numpy())
    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_nseg),
    )


# ── 메트릭 계산 ───────────────────────────────────────────────────────────────

def bucket_metrics(y_true, y_pred, num_segs) -> dict:
    results = {}
    for label, lo, hi in BUCKETS:
        mask = (num_segs >= lo) & (num_segs <= hi)
        n = mask.sum()
        if n == 0:
            results[label] = {"n": 0, "accuracy": None, "macro_f1": None}
            continue
        yt, yp = y_true[mask], y_pred[mask]
        results[label] = {
            "n": int(n),
            "accuracy": float(accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        }
    return results


# ── 출력 ──────────────────────────────────────────────────────────────────────

def print_table(all_results: dict) -> None:
    exp_names = list(all_results.keys())
    short_names = {
        "roberta_avgpool_4class":            "avgpool",
        "roberta_mamba_freeze_init_4class":  "mamba",
        "roberta_gru_freeze_init_4class":    "gru",
        "roberta_lstm_freeze_init_4class":   "lstm",
    }

    print("\n" + "=" * 80)
    print("  대화 길이별 성능 비교  (Macro F1)")
    print("=" * 80)
    header = f"  {'Bucket':<20}" + "".join(f"{short_names.get(n,n):>12}" for n in exp_names)
    print(header)
    print("  " + "-" * 78)
    for label, _, _ in BUCKETS:
        row = f"  {label:<20}"
        for name in exp_names:
            m = all_results[name].get(label, {})
            if m.get("macro_f1") is None:
                row += f"{'N/A':>12}"
            else:
                row += f"{m['macro_f1']:>11.4f} "
        print(row)

    print("\n" + "=" * 80)
    print("  대화 길이별 성능 비교  (Accuracy)")
    print("=" * 80)
    print(header)
    print("  " + "-" * 78)
    for label, _, _ in BUCKETS:
        row = f"  {label:<20}"
        for name in exp_names:
            m = all_results[name].get(label, {})
            if m.get("accuracy") is None:
                row += f"{'N/A':>12}"
            else:
                row += f"{m['accuracy']:>11.4f} "
        print(row)

    print("\n" + "=" * 80)
    print("  버킷별 샘플 수")
    print("=" * 80)
    first = next(iter(all_results.values()))
    for label, _, _ in BUCKETS:
        n = first.get(label, {}).get("n", 0)
        print(f"  {label:<20}  n={n}")
    print("=" * 80)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="실행할 모델 이름 일부 지정 (예: --models mamba gru lstm). 미지정 시 전체 실행."
    )
    args = parser.parse_args()

    all_results = {}

    exps = EXPERIMENTS
    if args.models:
        exps = [e for e in EXPERIMENTS if any(m in e["name"] for m in args.models)]

    for exp in exps:
        name = exp["name"]
        print(f"\n[{name}] 로드 중...")

        cfg, ds, mdl = load_experiment(exp)
        device = cfg.DEVICE

        ckpt_path = resolve_checkpoint(name, cfg.CHECKPOINT_DIR)
        if not ckpt_path.exists():
            print(f"  체크포인트 없음: {ckpt_path}, 건너뜀")
            continue

        model = mdl.build_model().to(device)
        model.load_state_dict(
            torch.load(ckpt_path, map_location="cpu", weights_only=False)
        )
        print(f"  체크포인트 로드: {ckpt_path.name}")

        loader = ds.create_dataloader(
            cfg.DATA_DIR / f"{args.split}.csv",
            cfg.TRAIN_CONFIG["BATCH_SIZE"],
            shuffle=False,
        )
        y_true, y_pred, num_segs = collect(model, loader, device)
        all_results[name] = bucket_metrics(y_true, y_pred, num_segs)
        print(f"  추론 완료  (샘플={len(y_true)})")

        # 메모리 해제
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_table(all_results)

    # JSON 저장
    out_dir = Path(__file__).resolve().parent
    save_path = out_dir / f"length_analysis_{args.split}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[save] {save_path}")


if __name__ == "__main__":
    main()

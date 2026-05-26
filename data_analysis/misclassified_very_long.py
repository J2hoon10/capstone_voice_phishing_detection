"""
misclassified_very_long.py

Very Long (16+ 세그먼트) 구간에서 오분류된 샘플을 모델별로 출력한다.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

EXPERIMENTS_DIR = Path(__file__).resolve().parent

EXPERIMENTS = [
    {"name": "roberta_mamba_freeze_init_4class", "dir": EXPERIMENTS_DIR / "roberta_mamba_freeze_init_4class"},
    {"name": "roberta_gru_freeze_init_4class",   "dir": EXPERIMENTS_DIR / "roberta_gru_freeze_init_4class"},
    {"name": "roberta_lstm_freeze_init_4class",  "dir": EXPERIMENTS_DIR / "roberta_lstm_freeze_init_4class"},
]

VERY_LONG_MIN = 16
SPLIT = "test"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_experiment(exp):
    exp_dir = exp["dir"]
    prefix = exp["name"]
    for key in ["config", "dataset", "model", "losses"]:
        sys.modules.pop(key, None)
    sys.path.insert(0, str(exp_dir))
    try:
        cfg = _load_module(f"{prefix}_config", exp_dir / "config.py")
        sys.modules["config"] = cfg
        ds  = _load_module(f"{prefix}_dataset", exp_dir / "dataset.py")
        mdl = _load_module(f"{prefix}_model",   exp_dir / "model.py")
    finally:
        sys.path.pop(0)
    return cfg, ds, mdl


def resolve_checkpoint(exp_name, ckpt_dir):
    latest_meta = ckpt_dir / f"{exp_name}_latest.json"
    with latest_meta.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return Path(meta["checkpoint_path"])


@torch.no_grad()
def collect(model, loader, device):
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
    return np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_nseg)


def load_test_texts(data_dir):
    """test.csv에서 (text, category) 리스트를 순서대로 반환 (유효 행만)."""
    import csv
    rows = []
    label_set = {"상담 대화", "일상 대화", "대출 사기형", "수사기관 사칭형", "바로 이 목소리"}
    with (data_dir / f"{SPLIT}.csv").open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cat = (row.get("category") or "").strip()
            if cat == "바로 이 목소리":
                cat = "수사기관 사칭형"
            text = (row.get("text") or "").strip()
            if cat not in {"상담 대화", "일상 대화", "대출 사기형", "수사기관 사칭형"} or not text:
                continue
            rows.append({"text": text, "category": cat, "id": row.get("id", "")})
    return rows


def main():
    # 첫 번째 모델로 텍스트/정답 인덱스 수집
    first_exp = EXPERIMENTS[0]
    cfg0, ds0, _ = load_experiment(first_exp)
    texts = load_test_texts(cfg0.DATA_DIR)
    labels = cfg0.LABELS

    # 모델별 예측 수집
    model_preds = {}
    model_nsegs = {}

    for exp in EXPERIMENTS:
        name = exp["name"]
        short = name.replace("roberta_", "").replace("_freeze_init_4class", "").replace("_4class", "")
        print(f"[{short}] 추론 중...")

        cfg, ds, mdl = load_experiment(exp)
        device = cfg.DEVICE
        ckpt = resolve_checkpoint(name, cfg.CHECKPOINT_DIR)
        model = mdl.build_model().to(device)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))

        loader = ds.create_dataloader(cfg.DATA_DIR / f"{SPLIT}.csv", cfg.TRAIN_CONFIG["BATCH_SIZE"], shuffle=False)
        y_true, y_pred, num_segs = collect(model, loader, device)

        model_preds[short] = y_pred
        model_nsegs[short] = num_segs

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Very Long 오분류 샘플 추출
    # num_segs는 모든 모델에서 동일 (데이터 기반) → 첫 번째 모델 기준
    ref_nseg = list(model_nsegs.values())[0]
    ref_pred = list(model_preds.values())[0]
    y_true_arr = y_true  # 마지막 수집된 y_true (모두 동일)

    very_long_mask = ref_nseg >= VERY_LONG_MIN
    indices = np.where(very_long_mask)[0]

    print(f"\n{'='*80}")
    print(f"  Very Long (16+ 세그먼트) 오분류 샘플  (총 {very_long_mask.sum()}개 중)")
    print(f"{'='*80}")

    model_keys = list(model_preds.keys())
    any_error = False

    for idx in indices:
        true_label = labels[y_true_arr[idx]]
        preds_per_model = {k: labels[model_preds[k][idx]] for k in model_keys}
        nseg = ref_nseg[idx]

        # 하나라도 틀린 경우만 출력
        if all(p == true_label for p in preds_per_model.values()):
            continue

        any_error = True
        sample = texts[idx] if idx < len(texts) else {}
        text_preview = sample.get("text", "")[:120].replace("\n", " ")
        sid = sample.get("id", f"idx={idx}")

        print(f"\n  ID        : {sid}")
        print(f"  세그먼트수 : {nseg}")
        print(f"  정답       : {true_label}")
        for k, p in preds_per_model.items():
            mark = "✓" if p == true_label else "✗"
            print(f"  {k:<8}  : {p}  {mark}")
        print(f"  텍스트 앞부분: {text_preview}...")

    if not any_error:
        print("\n  모든 Very Long 샘플을 정확히 분류함.")

    print(f"\n{'='*80}")

    # 정답/오답 클래스 분포
    print("\n  [Very Long 구간 정답 클래스 분포]")
    for i, lbl in enumerate(labels):
        cnt = ((y_true_arr[very_long_mask]) == i).sum()
        print(f"  {lbl:<20} : {cnt}개")

    print(f"\n  [Very Long 구간 모델별 오분류 수]")
    for k in model_keys:
        wrong = (model_preds[k][very_long_mask] != y_true_arr[very_long_mask]).sum()
        print(f"  {k:<12} : {wrong}개 오분류")


if __name__ == "__main__":
    main()

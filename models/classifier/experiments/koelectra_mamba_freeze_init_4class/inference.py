"""
roberta_mamba_freeze_init_4class / inference.py

학습 시와 동일한 슬라이딩 윈도우(WINDOW_SIZE=64, STRIDE=32)로
텍스트를 세그먼트로 분할하여 4-class 분류를 수행한다.

사용 예시
---------
# 단일 텍스트
python inference.py --text "안녕하세요 대출 관련해서 연락드립니다..."

# CSV 파일 (text 열 필수, label 열 선택)
python inference.py --csv data.csv --output result.csv

# 최신 체크포인트 자동 선택 (기본값)
python inference.py --text "..."

# 특정 체크포인트 지정
python inference.py --text "..." --checkpoint path/to/model.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer

# ── 경로 설정: config·dataset·model import를 위해 현재 디렉터리 추가 ──────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import (
    CHECKPOINT_DIR,
    DEVICE,
    IDX_TO_LABEL,
    MAMBA_CONFIG,
    MAX_SEQ_LEN,
    STRIDE,
    WINDOW_SIZE,
)
from dataset import build_segments
from model import build_model


# ── 최신 체크포인트 자동 탐색 ────────────────────────────────────────────────
def _resolve_checkpoint(checkpoint_arg: str | None) -> Path:
    if checkpoint_arg:
        p = Path(checkpoint_arg)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p

    latest_json = CHECKPOINT_DIR / "roberta_mamba_freeze_init_4class_latest.json"
    if latest_json.exists():
        meta = json.loads(latest_json.read_text(encoding="utf-8"))
        p = Path(meta["checkpoint_path"])
        if p.exists():
            print(f"[INFO] Loaded checkpoint: {p.name}  (macro F1={meta.get('best_macro_f1', '?')})")
            return p

    # fallback: 가장 최근에 수정된 *_best.pt
    candidates = sorted(CHECKPOINT_DIR.glob("*_best.pt"), key=lambda x: x.stat().st_mtime)
    if candidates:
        p = candidates[-1]
        print(f"[INFO] Loaded checkpoint (fallback): {p.name}")
        return p

    raise FileNotFoundError(
        f"No checkpoint found in {CHECKPOINT_DIR}. "
        "Specify --checkpoint or run training first."
    )


# ── 모델 로드 ─────────────────────────────────────────────────────────────────
def load_model(checkpoint_path: Path) -> torch.nn.Module:
    model = build_model().to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    # 체크포인트가 dict로 저장된 경우 (train.py 저장 형식 대응)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


# ── 텍스트 → 배치 텐서 변환 ──────────────────────────────────────────────────
def text_to_batch(tokenizer: AutoTokenizer, text: str) -> dict:
    """학습 시와 동일한 슬라이딩 윈도우로 텍스트를 세그먼트 텐서로 변환."""
    segs = build_segments(tokenizer, text)
    if not segs:
        raise ValueError("입력 텍스트를 세그먼트로 분할할 수 없습니다.")

    max_seq = max(len(s["input_ids"]) for s in segs)
    max_seq = min(max_seq, MAX_SEQ_LEN)

    n = len(segs)
    input_ids = torch.zeros((1, n, max_seq), dtype=torch.long)
    attention_mask = torch.zeros((1, n, max_seq), dtype=torch.long)
    segment_mask = torch.zeros((1, n), dtype=torch.bool)

    for j, s in enumerate(segs):
        l = min(len(s["input_ids"]), max_seq)
        input_ids[0, j, :l] = torch.tensor(s["input_ids"][:l], dtype=torch.long)
        attention_mask[0, j, :l] = torch.tensor(s["attention_mask"][:l], dtype=torch.long)
        segment_mask[0, j] = True

    return {
        "input_ids": input_ids.to(DEVICE),
        "attention_mask": attention_mask.to(DEVICE),
        "segment_mask": segment_mask.to(DEVICE),
        "num_segments": torch.tensor([n], dtype=torch.long, device=DEVICE),
    }


# ── 단일 추론 ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model: torch.nn.Module, tokenizer: AutoTokenizer, text: str) -> dict:
    """
    반환값:
        label       : 예측 클래스 이름
        label_idx   : 예측 클래스 인덱스 (0~3)
        probs       : 4-class 확률 분포 dict
        super_label : 'general' | 'phishing'
        super_prob  : superclass 확률 dict
        num_segments: 사용된 세그먼트 수
    """
    batch = text_to_batch(tokenizer, text)
    out = model(**batch)

    logits = out["logits"]                    # (1, 4) log-probs
    probs = torch.exp(logits).squeeze(0)      # (4,)
    pred_idx = probs.argmax().item()
    pred_label = IDX_TO_LABEL[pred_idx]

    super_logits = out["super_logits"].squeeze(0)   # (2,)
    super_probs = F.softmax(super_logits, dim=-1)

    return {
        "label": pred_label,
        "label_idx": pred_idx,
        "probs": {IDX_TO_LABEL[i]: round(probs[i].item(), 4) for i in range(4)},
        "super_label": "phishing" if super_probs[1] > 0.5 else "general",
        "super_prob": {
            "general": round(super_probs[0].item(), 4),
            "phishing": round(super_probs[1].item(), 4),
        },
        "num_segments": batch["num_segments"].item(),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="roberta_mamba_freeze_init_4class inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="추론할 텍스트 (단일 입력)")
    group.add_argument("--csv", type=str, help="추론할 CSV 파일 경로 (text 열 필수)")
    parser.add_argument("--checkpoint", type=str, default=None, help="체크포인트 경로 (미지정 시 최신 자동 선택)")
    parser.add_argument("--output", type=str, default=None, help="결과 CSV 저장 경로 (--csv 사용 시)")
    args = parser.parse_args()

    ckpt_path = _resolve_checkpoint(args.checkpoint)
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Window size: {WINDOW_SIZE}, stride: {STRIDE}")

    print("[INFO] Loading model...", flush=True)
    model = load_model(ckpt_path)
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

    # ── 단일 텍스트 모드 ────────────────────────────────────────────────────
    if args.text:
        result = predict(model, tokenizer, args.text)
        print("\n── 추론 결과 ──────────────────────────────────────")
        print(f"  예측 클래스  : {result['label']}  (idx={result['label_idx']})")
        print(f"  Superclass   : {result['super_label']}  {result['super_prob']}")
        print(f"  세그먼트 수  : {result['num_segments']}")
        print(f"  클래스 확률  :")
        for name, p in result["probs"].items():
            bar = "█" * int(p * 30)
            print(f"    {name:<12}: {p:.4f}  {bar}")
        return

    # ── CSV 모드 ────────────────────────────────────────────────────────────
    df = pd.read_csv(args.csv)
    if "text" not in df.columns:
        raise ValueError("CSV 파일에 'text' 열이 필요합니다.")

    results = []
    for i, row in df.iterrows():
        text = str(row["text"]).strip()
        if not text:
            results.append({"pred_label": None, "pred_idx": None, **{f"prob_{IDX_TO_LABEL[j]}": None for j in range(4)}})
            continue
        try:
            r = predict(model, tokenizer, text)
            results.append({
                "pred_label": r["label"],
                "pred_idx": r["label_idx"],
                "super_label": r["super_label"],
                "prob_general": r["super_prob"]["general"],
                "prob_phishing": r["super_prob"]["phishing"],
                **{f"prob_{IDX_TO_LABEL[j]}": r["probs"][IDX_TO_LABEL[j]] for j in range(4)},
                "num_segments": r["num_segments"],
            })
        except Exception as e:
            print(f"[WARN] row {i} 처리 실패: {e}")
            results.append({"pred_label": "ERROR", "pred_idx": -1})

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(df)} 완료...", flush=True)

    out_df = pd.concat([df, pd.DataFrame(results)], axis=1)
    out_path = args.output or args.csv.replace(".csv", "_pred.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] 결과 저장: {out_path}")

    # 간단 통계 출력
    if "label" in out_df.columns:
        correct = (out_df["label"] == out_df["pred_label"]).sum()
        print(f"[INFO] 정확도: {correct}/{len(out_df)} = {correct/len(out_df):.4f}")


if __name__ == "__main__":
    main()

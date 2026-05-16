"""
streaming_inference.py

대화를 윈도우 단위로 순차 입력할 때 모델 예측이 어떻게 변하는지 시각화.

사용법:
  # test.csv에서 특정 ID의 대화를 스트리밍
  python streaming_inference.py --id "phishing_대출 사기형_1"

  # 직접 텍스트 입력
  python streaming_inference.py --text "대화 내용 전체..."

  # test.csv에서 각 클래스별 랜덤 샘플 1개씩
  python streaming_inference.py --sample-each
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# 실험 디렉토리를 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CHECKPOINT_DIR, DATA_DIR, DEVICE, IDX_TO_LABEL, LABELS, NUM_LABELS
from dataset import build_segments
from model import build_model
from transformers import AutoTokenizer
from config import ENCODER_CONFIG, MAX_SEQ_LEN


EXP_NAME = "roberta_avgpool_4class"
LABEL_COLORS = {
    "상담 대화":      "\033[94m",   # 파랑
    "일상 대화":      "\033[92m",   # 초록
    "대출 사기형":    "\033[93m",   # 노랑
    "수사기관 사칭형": "\033[91m",   # 빨강
}
RESET = "\033[0m"
BOLD  = "\033[1m"


def load_checkpoint(run_id: str | None = None) -> Path:
    if run_id:
        return CHECKPOINT_DIR / f"{EXP_NAME}_{run_id}_best.pt"
    meta = CHECKPOINT_DIR / f"{EXP_NAME}_latest.json"
    if meta.exists():
        return Path(json.loads(meta.read_text(encoding="utf-8"))["checkpoint_path"])
    raise FileNotFoundError(f"체크포인트 없음: {CHECKPOINT_DIR}")


def build_batch(segments: list[dict], k: int, device: torch.device):
    """첫 k개 윈도우만 사용하는 단일 배치 생성."""
    segs = segments[:k]
    max_len = max(len(s["input_ids"]) for s in segs)
    max_len = min(max_len, MAX_SEQ_LEN)

    S = k
    input_ids    = torch.zeros(1, S, max_len, dtype=torch.long)
    attention_mask = torch.zeros(1, S, max_len, dtype=torch.long)
    segment_mask = torch.ones(1, S, dtype=torch.bool)

    for j, seg in enumerate(segs):
        ids  = seg["input_ids"][:max_len]
        attn = seg["attention_mask"][:max_len]
        L = len(ids)
        input_ids[0, j, :L]     = torch.tensor(ids,  dtype=torch.long)
        attention_mask[0, j, :L] = torch.tensor(attn, dtype=torch.long)

    return {
        "input_ids":      input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "segment_mask":   segment_mask.to(device),
        "num_segments":   torch.tensor([S], dtype=torch.long).to(device),
    }


def bar(prob: float, width: int = 20) -> str:
    filled = int(prob * width)
    return "█" * filled + "░" * (width - filled)


@torch.no_grad()
def run_streaming(model, tokenizer, text: str, true_label: str | None = None):
    segments = build_segments(tokenizer, text)
    if not segments:
        print("[오류] 세그먼트 생성 실패 (텍스트가 너무 짧거나 비어 있음)")
        return

    N = len(segments)
    print(f"\n{'='*65}")
    if true_label:
        print(f"  실제 레이블: {BOLD}{true_label}{RESET}   |   총 윈도우: {N}개")
    else:
        print(f"  총 윈도우: {N}개")
    print(f"  텍스트 앞부분: {text[:80]}...")
    print(f"{'='*65}")
    print(f"  {'윈도우':>4}  {'예측 레이블':<14}  {'확신도':>6}  확률 분포")
    print(f"  {'-'*61}")

    prev_pred = None
    for k in range(1, N + 1):
        batch  = build_batch(segments, k, DEVICE)
        out    = model(**batch)
        probs  = F.softmax(out["logits"].float(), dim=-1)[0].cpu()
        pred_idx = probs.argmax().item()
        pred_label = IDX_TO_LABEL[pred_idx]
        conf = probs[pred_idx].item()

        changed = "◀ 변경!" if (prev_pred is not None and pred_label != prev_pred) else ""
        color = LABEL_COLORS.get(pred_label, "")

        print(f"  {k:>4}  {color}{pred_label:<14}{RESET}  {conf:>5.1%}  ", end="")
        for i, label in enumerate(LABELS):
            p = probs[i].item()
            print(f"{label[:4]}:{bar(p,8)}{p:.2f}  ", end="")
        if changed:
            print(f"  {BOLD}{changed}{RESET}", end="")
        print()

        prev_pred = pred_label

    final_label = IDX_TO_LABEL[probs.argmax().item()]
    print(f"{'='*65}")
    print(f"  최종 예측: {BOLD}{LABEL_COLORS.get(final_label,'')}{final_label}{RESET}")
    if true_label:
        match = "✓ 정답" if final_label == true_label else "✗ 오답"
        print(f"  결과:      {BOLD}{match}{RESET}")
    print(f"{'='*65}\n")


def load_rows_by_split(split: str = "test") -> list[dict]:
    path = DATA_DIR / f"{split}.csv"
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="순차 윈도우 스트리밍 추론")
    parser.add_argument("--run-id",      default=None, help="체크포인트 run_id (없으면 latest)")
    parser.add_argument("--split",       default="test", choices=["train", "val", "test"])
    parser.add_argument("--id",          default=None, help="데이터셋 내 특정 id")
    parser.add_argument("--text",        default=None, help="직접 텍스트 입력")
    parser.add_argument("--sample-each", action="store_true", help="클래스별 랜덤 1개씩 시연")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    # 모델 로드
    ckpt_path = load_checkpoint(args.run_id)
    print(f"[로드] {ckpt_path}")
    model = build_model().to(DEVICE)
    model.load_state_dict(
        torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_CONFIG["MODEL_NAME"])

    # ── 직접 텍스트 입력 ───────────────────────────────────────────────
    if args.text:
        run_streaming(model, tokenizer, args.text)
        return

    rows = load_rows_by_split(args.split)

    # ── 특정 id ───────────────────────────────────────────────────────
    if args.id:
        row = next((r for r in rows if r["id"] == args.id), None)
        if row is None:
            print(f"[오류] id '{args.id}' 를 {args.split}.csv 에서 찾을 수 없습니다.")
            sys.exit(1)
        run_streaming(model, tokenizer, row["text"], true_label=row["category"])
        return

    # ── 클래스별 랜덤 1개씩 ────────────────────────────────────────────
    if args.sample_each:
        rng = random.Random(args.seed)
        by_label: dict[str, list] = {}
        for r in rows:
            by_label.setdefault(r["category"], []).append(r)
        for label in LABELS:
            pool = by_label.get(label, [])
            if not pool:
                print(f"[경고] '{label}' 샘플 없음")
                continue
            row = rng.choice(pool)
            run_streaming(model, tokenizer, row["text"], true_label=label)
        return

    parser.print_help()


if __name__ == "__main__":
    main()

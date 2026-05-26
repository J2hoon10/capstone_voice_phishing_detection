"""
streaming_inference.py  (roberta_mamba_freeze_init_4class)

대화를 윈도우 단위로 순차 누적할 때 모델 예측이 어떻게 변하는지 시각화.

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CHECKPOINT_DIR, DATA_DIR, DEVICE, IDX_TO_LABEL, LABELS, NUM_LABELS, MAX_SEQ_LEN, WINDOW_SIZE
from dataset import build_segments
from model import build_model
from transformers import AutoTokenizer
from config import ENCODER_CONFIG


EXP_NAME = "koelectra_mamba_freeze_init_4class"
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
    input_ids     = torch.zeros(1, S, max_len, dtype=torch.long)
    attention_mask = torch.zeros(1, S, max_len, dtype=torch.long)
    segment_mask  = torch.ones(1, S, dtype=torch.bool)

    for j, seg in enumerate(segs):
        ids  = seg["input_ids"][:max_len]
        attn = seg["attention_mask"][:max_len]
        L = len(ids)
        input_ids[0, j, :L]      = torch.tensor(ids,  dtype=torch.long)
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
def encode_window(model, seg: dict, device: torch.device) -> torch.Tensor:
    """단일 세그먼트를 RoBERTa + AttentionWeightedPooling으로 인코딩 → (1, D)"""
    ids  = torch.tensor([seg["input_ids"]],       dtype=torch.long, device=device)
    attn = torch.tensor([seg["attention_mask"]], dtype=torch.long, device=device)
    enc_dtype = next(model.encoder.parameters()).dtype
    H   = model.encoder(input_ids=ids, attention_mask=attn).last_hidden_state
    vec = model.pooling(H, attn).to(enc_dtype).float()  # (1, D)
    return vec


def predict_from_mamba_out(model, mamba_out_last: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    """Mamba 마지막 출력 (1, H) → 4-class 확률 벡터 (CPU)"""
    head_out       = model.head(mamba_out_last)
    super_probs    = torch.softmax(head_out["super_logits"],    dim=-1)
    normal_probs   = torch.softmax(head_out["normal_logits"],   dim=-1)
    phishing_probs = torch.softmax(head_out["phishing_logits"], dim=-1)
    logits_4 = torch.cat([
        super_probs[:, 0:1] * normal_probs,
        super_probs[:, 1:2] * phishing_probs,
    ], dim=-1)   # (1, 4)
    log_logits = torch.log(logits_4 + 1e-8)
    return F.softmax(log_logits / temp, dim=-1)[0].cpu()


def get_temperature(k: int, warmup: int = 8, max_temp: float = 4.0) -> float:
    """초반 윈도우일수록 높은 temperature → 확률 분포 평탄화.
    k < warmup 구간에서 max_temp → 1.0 으로 선형 감소.
    """
    if k >= warmup:
        return 1.0
    return max_temp - (max_temp - 1.0) * (k - 1) / (warmup - 1)


@torch.no_grad()
def run_streaming(model, tokenizer, text: str, true_label: str | None = None,
                  temp_warmup: int = 8, temp_max: float = 4.0):
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
    print(f"  온도 스케일링: warmup={temp_warmup}  max_temp={temp_max:.1f}")
    print(f"{'='*65}")
    print(f"  {'윈도우':>4}  {'예측 레이블':<14}  {'확신도':>6}  {'T':>4}  확률 분포")
    print(f"  {'-'*65}")

    prev_pred = None
    for k in range(1, N + 1):
        batch    = build_batch(segments, k, DEVICE)
        out      = model(**batch)
        temp     = get_temperature(k, warmup=temp_warmup, max_temp=temp_max)
        probs    = F.softmax(out["logits"].float() / temp, dim=-1)[0].cpu()
        pred_idx = probs.argmax().item()
        pred_label = IDX_TO_LABEL[pred_idx]
        conf = probs[pred_idx].item()

        changed = "◀ 변경!" if (prev_pred is not None and pred_label != prev_pred) else ""
        color = LABEL_COLORS.get(pred_label, "")

        print(f"  {k:>4}  {color}{pred_label:<14}{RESET}  {conf:>5.1%}  {temp:>3.1f}  ", end="")
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


@torch.no_grad()
def run_interactive(model, tokenizer, temp_warmup: int = 8, temp_max: float = 4.0):
    """
    문장을 한 줄씩 입력받아 누적하고, 새로 생긴 윈도우만 incremental하게 처리.
    RoBERTa 인코딩 벡터를 캐싱하여 재인코딩을 방지.
    Mamba는 병렬 스캔 구조상 step caching 대신 캐시된 벡터 전체를 재실행.

    'q' 또는 빈 줄로 종료, 'r' 입력 시 상태 초기화.
    """
    print(f"\n{'='*65}")
    print("  [인터랙티브 incremental 모드]")
    print("  문장을 한 줄씩 입력하세요. 새 윈도우의 RoBERTa만 실행합니다 (벡터 캐싱).")
    print(f"  온도 스케일링: warmup={temp_warmup}  max_temp={temp_max:.1f}")
    print("  r  → 상태 초기화")
    print("  q  → 종료")
    print(f"{'='*65}\n")

    accumulated:       list[str]          = []
    encoded_cache:     list[torch.Tensor] = []   # (1, D) 인코딩 벡터 캐시
    processed_seg_count: int              = 0
    last_probs:          torch.Tensor | None = None

    while True:
        try:
            line = input(f"[{len(accumulated)+1}번째 문장] >> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[종료]")
            break

        if line.lower() == "q" or line == "":
            print("[종료]")
            break

        if line.lower() == "r":
            accumulated.clear()
            encoded_cache.clear()
            processed_seg_count = 0
            last_probs          = None
            print("  → 상태를 초기화했습니다.\n")
            continue

        accumulated.append(line)
        full_text = " ".join(accumulated)

        # 전체 누적 토큰 수가 content_size 미만이면 아직 대기
        token_ids    = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        content_size = WINDOW_SIZE - 2
        if len(token_ids) < content_size:
            print(f"  (누적 {len(token_ids)}/{content_size} 토큰 — 윈도우 크기만큼 쌓이면 시작됩니다)\n")
            continue

        segments  = build_segments(tokenizer, full_text)

        new_segs = segments[processed_seg_count:]
        if not new_segs:
            print(f"  (새 윈도우 없음 — 텍스트가 아직 충분하지 않습니다)\n")
            continue

        # 새 세그먼트만 RoBERTa로 인코딩하여 캐시에 추가
        for seg in new_segs:
            encoded_cache.append(encode_window(model, seg, DEVICE))  # (1, D)
            processed_seg_count += 1

        print(f"\n  누적 문장 수: {len(accumulated)}개  |  "
              f"전체 윈도우: {len(segments)}개  |  새 윈도우: {len(new_segs)}개")
        print(f"  {'윈도우':>4}  {'예측 레이블':<14}  {'확신도':>6}  확률 분포")
        print(f"  {'-'*61}")

        # 캐시된 벡터 전체로 Mamba 실행 (새 윈도우 추가될 때마다)
        x_stacked = torch.stack(encoded_cache, dim=1)  # (1, T, D)
        y = x_stacked
        for mamba, dropout in zip(model.mamba_layers, model.mamba_dropouts):
            y = dropout(mamba(y))

        # 새 윈도우 각각의 위치에서 예측 출력
        prev_label = IDX_TO_LABEL[last_probs.argmax().item()] if last_probs is not None else None
        new_start  = processed_seg_count - len(new_segs)

        for i in range(len(new_segs)):
            seg_idx    = new_start + i
            temp       = get_temperature(seg_idx + 1, temp_warmup, temp_max)
            last_probs = predict_from_mamba_out(model, y[:, seg_idx, :], temp)

            pred_idx   = last_probs.argmax().item()
            pred_label = IDX_TO_LABEL[pred_idx]
            conf       = last_probs[pred_idx].item()
            changed    = "◀ 변경!" if (prev_label is not None and pred_label != prev_label) else ""
            color      = LABEL_COLORS.get(pred_label, "")

            print(f"  {seg_idx+1:>4}  {color}{pred_label:<14}{RESET}  {conf:>5.1%}  ", end="")
            for j, label in enumerate(LABELS):
                p = last_probs[j].item()
                print(f"{label[:4]}:{bar(p,8)}{p:.2f}  ", end="")
            if changed:
                print(f"  {BOLD}{changed}{RESET}", end="")
            print()

            prev_label = pred_label

        final_label = IDX_TO_LABEL[last_probs.argmax().item()]
        print(f"  → 현재 판단: {BOLD}{LABEL_COLORS.get(final_label,'')}{final_label}{RESET}\n")


def load_rows_by_split(split: str = "test") -> list[dict]:
    path = DATA_DIR / f"{split}.csv"
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="순차 윈도우 스트리밍 추론 (Mamba)")
    parser.add_argument("--run-id",      default=None, help="체크포인트 run_id (없으면 latest)")
    parser.add_argument("--split",       default="test", choices=["train", "val", "test"])
    parser.add_argument("--id",          default=None, help="데이터셋 내 특정 id")
    parser.add_argument("--text",        default=None, help="직접 텍스트 입력")
    parser.add_argument("--interactive", action="store_true", help="문장 누적 인터랙티브 모드")
    parser.add_argument("--sample-each", action="store_true", help="클래스별 랜덤 1개씩 시연")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--temp-warmup", type=int,   default=8,   help="온도 스케일링 warmup 윈도우 수 (기본: 8)")
    parser.add_argument("--temp-max",    type=float, default=4.0, help="초반 최대 temperature (기본: 4.0)")
    args = parser.parse_args()

    ckpt_path = load_checkpoint(args.run_id)
    print(f"[로드] {ckpt_path}")
    model = build_model().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_CONFIG["MODEL_NAME"])

    if args.interactive:
        run_interactive(model, tokenizer,
                        temp_warmup=args.temp_warmup, temp_max=args.temp_max)
        return

    if args.text:
        run_streaming(model, tokenizer, args.text,
                      temp_warmup=args.temp_warmup, temp_max=args.temp_max)
        return

    rows = load_rows_by_split(args.split)

    if args.id:
        row = next((r for r in rows if r["id"] == args.id), None)
        if row is None:
            print(f"[오류] id '{args.id}' 를 {args.split}.csv 에서 찾을 수 없습니다.")
            sys.exit(1)
        run_streaming(model, tokenizer, row["text"], true_label=row["category"],
                      temp_warmup=args.temp_warmup, temp_max=args.temp_max)
        return

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
            run_streaming(model, tokenizer, row["text"], true_label=label,
                          temp_warmup=args.temp_warmup, temp_max=args.temp_max)
        return

    parser.print_help()


if __name__ == "__main__":
    main()

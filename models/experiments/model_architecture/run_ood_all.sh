#!/bin/bash
# OOD 평가 전체 실행 스크립트
# 실험 보고서의 모든 모델에 대해 ood.csv 평가 수행

set -e
EXPERIMENTS_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_OUT="$EXPERIMENTS_DIR/ood_results_summary.txt"

run_eval() {
    local exp_dir="$1"
    local label="$2"
    shift 2
    local extra_args="$@"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [$label]"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    cd "$EXPERIMENTS_DIR/$exp_dir"
    python evaluate.py --split ood $extra_args 2>&1 | grep -E "split=ood|Accuracy|Macro F1|상담 대화|일상 대화|대출 사기형|수사기관" | tee -a "$LOG_OUT"
}

echo "OOD 평가 결과 - $(date)" > "$LOG_OUT"
echo "========================================" >> "$LOG_OUT"

# ── 실험 1. Mamba Layer ──────────────────────────────────
run_eval "roberta_mamba_l1_freeze_init_4class" "실험1 L1 Mamba"
run_eval "roberta_mamba_freeze_init_4class"    "실험1 L2 Mamba (기본)"

# ── 실험 2. 인코더 ───────────────────────────────────────
# RoBERTa: 위에서 이미 실행 (L2 기본)
run_eval "bert_mamba_freeze_init_4class"       "실험2 BERT"
run_eval "kobert_mamba_freeze_init_4class"     "실험2 KoBERT"
run_eval "koelectra_mamba_freeze_init_4class"  "실험2 KoELECTRA"

# ── 실험 3. 디코더 구조 ──────────────────────────────────
# Mamba: 위에서 이미 실행 (L2 기본)
run_eval "roberta_gru_freeze_init_4class"      "실험3 GRU"
run_eval "roberta_lstm_freeze_init_4class"     "실험3 LSTM"

# ── 실험 4. d_state (L1 기반) ────────────────────────────
# d_state=16: roberta_mamba_l1_freeze_init_4class (위에서 실행)
run_eval "roberta_mamba_l1_dstate_4class"      "실험4 d_state=32" --d-state 32
run_eval "roberta_mamba_l1_dstate_4class"      "실험4 d_state=64" --d-state 64

# ── 실험 5. 윈도우 크기 ──────────────────────────────────
# w64: roberta_mamba_freeze_init_4class (위에서 실행)
run_eval "roberta_mamba_w32_freeze_init_4class" "실험5 w32"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  완료. 결과 저장: $LOG_OUT"

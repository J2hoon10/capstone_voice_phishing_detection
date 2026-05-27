"""
models/main/model_architecture/inference.py

학습 시와 동일한 슬라이딩 윈도우로 전사 텍스트를 세그먼트로 분할하여 4-class 분류를 수행한다.
FastAPI 서비스(app.py)용 VoicePhishingDetector와 CLI 테스트 기능을 통합 제공한다.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# ── 경로 설정: config·dataset·model import를 위해 현재 디렉터리 추가 ──────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from audio_processor import AudioProcessor
from config import (
    CHECKPOINT_DIR,
    CONFIG,
    DEVICE,
    IDX_TO_LABEL,
    MAX_SEQ_LEN,
    STRIDE,
    WINDOW_SIZE,
)
from dataset import build_segments
from model import build_model


class PhishingRiskScorer:
    def __init__(self):
        self.current_score = 0.0
        self.max_score = 100.0
        self.min_score = 0.0
        self.threshold_level1 = 30
        self.threshold_level2 = 60

    def update_score(self, prob: float):
        score_change = 0
        if prob > 0.8:
            score_change = 20
        elif 0.5 < prob <= 0.8:
            score_change = 10
        else:
            score_change = -10
        self.current_score += score_change
        self.current_score = max(self.min_score, min(self.current_score, self.max_score))
        return self.current_score, self._get_warning_level()

    def _get_warning_level(self):
        if self.current_score >= self.threshold_level2:
            return "LEVEL_2_WARNING"
        elif self.current_score >= self.threshold_level1:
            return "LEVEL_1_CAUTION"
        else:
            return "NORMAL"


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

    # config 에 적혀있는 DEFAULT_MODEL_PATH 로 fallback
    p = Path(CONFIG["MODEL_PATH"])
    if p.exists():
        return p

    raise FileNotFoundError(
        f"No checkpoint found in {CHECKPOINT_DIR} or {p.parent}. "
        "Specify --checkpoint or run training first."
    )


class VoicePhishingDetector:
    def __init__(self, model_path=None, device=None):
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device(CONFIG["DEVICE"])

        print(f"--- [Inference] Initializing Detector on {self.device} ---")

        self.base_model_name = CONFIG["BASE_MODEL_NAME"]
        self.max_length = CONFIG["MAX_LENGTH"]
        self.window_size = CONFIG["WINDOW_SIZE"]
        self.stride = CONFIG["STRIDE"]

        print(f"[Init] Loading Audio Processor ({CONFIG.get('WHISPER_MODEL_SIZE', 'base')})...")
        self.processor = AudioProcessor(whisper_model_size=CONFIG.get("WHISPER_MODEL_SIZE", "base"))

        print(f"[Init] Loading Tokenizer from {self.base_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

        print("[Init] Building Student Architecture (RoBERTa-Mamba)...")
        self.model = build_model().to(self.device)

        ckpt_path = _resolve_checkpoint(model_path)
        self._load_weights(ckpt_path)

        self.model.eval()

        # 실전 VAD-disabled STT & NLP 모델 웜업
        self._warmup()

        print("[Init] System Ready.")

    def _load_weights(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
        try:
            print(f" -> Loading weights from {os.path.basename(path)}...")
            checkpoint = torch.load(path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif isinstance(checkpoint, dict) and "model_state" in checkpoint:
                state_dict = checkpoint["model_state"]
            else:
                state_dict = checkpoint

            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k
                if new_key.startswith("student."):
                    new_key = new_key.replace("student.", "")
                if new_key.startswith("module."):
                    new_key = new_key.replace("module.", "")
                new_state_dict[new_key] = v

            keys = self.model.load_state_dict(new_state_dict, strict=False)
            if len(keys.missing_keys) > 0:
                print(f"⚠️ [Warning] Missing keys: {len(keys.missing_keys)}")
            print("✅ Weights loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"가중치 로드 중 치명적 오류: {e}")

    def _warmup(self):
        print("🔥 [Warm-up] Running full-path simulation (Forcing Decode)...")
        warmup_filename = "warmup_temp.wav"
        try:
            # 1. 15초 길이의 랜덤 노이즈 파일 생성
            sr = 16000
            duration = 15  # 15초
            audio_data = np.random.uniform(-0.5, 0.5, int(sr * duration)).astype(np.float32)
            sf.write(warmup_filename, audio_data, sr)

            # 2. VAD Filter를 끈 채로 Whisper 웜업
            if hasattr(self.processor, "whisper"):
                self.processor.whisper.transcribe(
                    warmup_filename,
                    language="ko",
                    beam_size=1,
                    temperature=0.0,
                    vad_filter=False,
                )

            # 3. NLP 모델 웜업
            long_dummy_text = "보이스피싱 탐지 시스템 웜업을 위한 긴 문장입니다. " * 5
            batch = text_to_batch(self.tokenizer, long_dummy_text)
            with torch.no_grad():
                self.model(**batch)

            print("✅ [Warm-up] Completed. Latency is optimized.")
        except Exception as e:
            print(f"⚠️ [Warm-up] Failed: {e}")
        finally:
            if os.path.exists(warmup_filename):
                try:
                    os.remove(warmup_filename)
                except:
                    pass

    @torch.no_grad()
    def predict(self, audio_file_path, threshold=0.5):
        """오디오 음성 파일 추론 수행."""
        # [Step 1] 전처리 & STT
        cleaned_sentences = self.processor.process_file(audio_file_path)
        if not cleaned_sentences:
            return {"status": "fail", "message": "No text detected"}

        full_text = " ".join(cleaned_sentences)

        # [Step 2] 슬라이딩 윈도우 생성
        segs = build_segments(self.tokenizer, full_text)
        if not segs:
            return {"status": "fail", "message": "Window creation failed"}

        # [Step 3] 텐서 배치화
        n = len(segs)
        max_seq = max(len(s["input_ids"]) for s in segs)
        max_seq = min(max_seq, MAX_SEQ_LEN)

        input_ids = torch.zeros((1, n, max_seq), dtype=torch.long)
        attention_mask = torch.zeros((1, n, max_seq), dtype=torch.long)
        segment_mask = torch.zeros((1, n), dtype=torch.bool)

        for j, s in enumerate(segs):
            l = min(len(s["input_ids"]), max_seq)
            input_ids[0, j, :l] = torch.tensor(s["input_ids"][:l], dtype=torch.long)
            attention_mask[0, j, :l] = torch.tensor(s["attention_mask"][:l], dtype=torch.long)
            segment_mask[0, j] = True

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        segment_mask = segment_mask.to(self.device)
        num_segments = torch.tensor([n], dtype=torch.long, device=self.device)

        # [Step 4] 모델 추론
        t_nlp_start = time.time()
        with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                segment_mask=segment_mask,
                num_segments=num_segments,
            )
            logits = out["logits"]                          # (1, 4)
            probs = torch.exp(logits).squeeze(0)            # (4,)
            super_logits = out["super_logits"].squeeze(0)   # (2,)
            super_probs = F.softmax(super_logits, dim=-1)   # (2,)

            aux_logits = out.get("aux_logits")
            if aux_logits is not None:
                aux_probs = F.softmax(aux_logits.squeeze(0), dim=-1)
                segment_phishing_probs = aux_probs[:, 1].cpu().numpy()
            else:
                segment_phishing_probs = np.zeros(n)
        t_nlp_end = time.time()
        print(f"   [Profile] NLP Inference: {(t_nlp_end - t_nlp_start):.4f}s")

        # [Step 5] 결과 집계
        risk_score = float(super_probs[1].item()) * 100
        max_idx = np.argmax(segment_phishing_probs)
        dangerous_token_ids = segs[max_idx]["input_ids"][1:-1]
        dangerous_segment = self.tokenizer.decode(dangerous_token_ids, skip_special_tokens=True)
        is_phishing = risk_score >= (threshold * 100)

        return {
            "status": "success",
            "is_phishing": is_phishing,
            "max_risk_score": risk_score,
            "dangerous_segment": dangerous_segment,
            "probs": {
                "상담 대화": round(probs[0].item(), 4),
                "일상 대화": round(probs[1].item(), 4),
                "대출 사기형": round(probs[2].item(), 4),
                "수사기관 사칭형": round(probs[3].item(), 4),
            },
        }


# ── CLI ───────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_text(model: torch.nn.Module, tokenizer: AutoTokenizer, text: str) -> dict:
    batch = text_to_batch(tokenizer, text)
    out = model(**batch)

    logits = out["logits"]
    probs = torch.exp(logits).squeeze(0)
    pred_idx = probs.argmax().item()
    pred_label = IDX_TO_LABEL[pred_idx]

    super_logits = out["super_logits"].squeeze(0)
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


def main():
    import pandas as pd

    parser = argparse.ArgumentParser(description="RoBERTa-Mamba 4class baseline inference")
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
    model = build_model().to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["BASE_MODEL_NAME"])

    # ── 단일 텍스트 모드 ────────────────────────────────────────────────────
    if args.text:
        result = predict_text(model, tokenizer, args.text)
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
            r = predict_text(model, tokenizer, text)
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


if __name__ == "__main__":
    main()

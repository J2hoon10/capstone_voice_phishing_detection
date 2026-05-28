import os
import sys
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audio_processor import AudioProcessor
from config import CONFIG


DEFAULT_EXP_DIR = Path(__file__).resolve().parent / "experiments" / "roberta_mamba_freeze_init_4class"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "roberta_mamba_freeze_init_4class_4class_20260517_174922_best.pt"
PHISHING_LABELS = {"대출 사기형", "수사기관 사칭형"}


def _get_temperature(k: int, warmup: int = 8, max_temp: float = 4.0) -> float:
    """초반 세그먼트일수록 높은 temperature로 확률 분포를 평탄화.
    k < warmup 구간에서 max_temp → 1.0 으로 선형 감소.
    """
    if k >= warmup:
        return 1.0
    return max_temp - (max_temp - 1.0) * (k - 1) / max(warmup - 1, 1)


def _patch_transformers_generation_for_mamba() -> None:
    try:
        import transformers.generation as generation
        import transformers.generation.utils as generation_utils
    except Exception:
        return

    aliases = {
        "GreedySearchDecoderOnlyOutput": "GenerateDecoderOnlyOutput",
        "SampleDecoderOnlyOutput": "GenerateDecoderOnlyOutput",
        "BeamSearchDecoderOnlyOutput": "GenerateBeamDecoderOnlyOutput",
        "BeamSampleDecoderOnlyOutput": "GenerateBeamDecoderOnlyOutput",
    }
    for old_name, new_name in aliases.items():
        if not hasattr(generation, old_name) and hasattr(generation_utils, new_name):
            setattr(generation, old_name, getattr(generation_utils, new_name))



@contextmanager
def _experiment_import_context(exp_dir: Path):
    names = ["config", "dataset", "model"]
    saved_modules = {name: sys.modules.get(name) for name in names}
    for name in names:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(exp_dir))
    try:
        yield
    finally:
        try:
            sys.path.remove(str(exp_dir))
        except ValueError:
            pass
        for name in names:
            sys.modules.pop(name, None)
            if saved_modules[name] is not None:
                sys.modules[name] = saved_modules[name]


class RobertaMamba4ClassTextDetector:
    def __init__(self, model_path: str | os.PathLike | None = None, device: str | None = None):
        self.exp_dir = Path(os.getenv("ROBERTA_MAMBA_EXP_DIR", str(DEFAULT_EXP_DIR))).resolve()
        self.model_path = Path(model_path or os.getenv("ROBERTA_MAMBA_MODEL_PATH", str(DEFAULT_CHECKPOINT))).resolve()
        self.device = torch.device(device or CONFIG["DEVICE"])
        self._predict_lock = Lock()

        if not self.model_path.exists():
            raise FileNotFoundError(f"RoBERTa-Mamba checkpoint not found: {self.model_path}")

        _patch_transformers_generation_for_mamba()
        previous_force_fallback = os.environ.get("FORCE_MAMBA_PYTORCH")
        if self.device.type == "cpu":
            os.environ["FORCE_MAMBA_PYTORCH"] = "1"
        try:
            with _experiment_import_context(self.exp_dir):
                import config as exp_config  # type: ignore
                import dataset as exp_dataset  # type: ignore
                import model as exp_model  # type: ignore
        finally:
            if previous_force_fallback is None:
                os.environ.pop("FORCE_MAMBA_PYTORCH", None)
            else:
                os.environ["FORCE_MAMBA_PYTORCH"] = previous_force_fallback

        self.labels = list(exp_config.LABELS)
        self.idx_to_label = dict(exp_config.IDX_TO_LABEL)
        self.max_seq_len = int(exp_config.MAX_SEQ_LEN)
        self.build_segments = exp_dataset.build_segments
        self.model = exp_model.build_model().to(self.device)

        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(exp_config.ENCODER_CONFIG["MODEL_NAME"])

    def _batch_from_segments(self, segments: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = min(max(len(seg["input_ids"]) for seg in segments), self.max_seq_len)
        num_segments = len(segments)
        input_ids = torch.zeros(1, num_segments, max_len, dtype=torch.long)
        attention_mask = torch.zeros(1, num_segments, max_len, dtype=torch.long)
        segment_mask = torch.ones(1, num_segments, dtype=torch.bool)

        for index, seg in enumerate(segments):
            ids = seg["input_ids"][:max_len]
            mask = seg["attention_mask"][:max_len]
            length = len(ids)
            input_ids[0, index, :length] = torch.tensor(ids, dtype=torch.long)
            attention_mask[0, index, :length] = torch.tensor(mask, dtype=torch.long)

        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "segment_mask": segment_mask.to(self.device),
            "num_segments": torch.tensor([num_segments], dtype=torch.long, device=self.device),
        }

    def _dangerous_excerpt(self, text: str, limit: int = 360) -> str:
        compact = " ".join(line.strip() for line in text.splitlines() if line.strip())
        if len(compact) <= limit:
            return compact
        return compact[-limit:].strip()

    @torch.no_grad()
    def _encode_window(self, seg: dict) -> torch.Tensor:
        """단일 세그먼트를 RoBERTa + Pooling으로 인코딩 → (1, D) float 텐서.
        반드시 _predict_lock 내부에서 호출해야 함.
        """
        ids  = torch.tensor([seg["input_ids"]],      dtype=torch.long, device=self.device)
        attn = torch.tensor([seg["attention_mask"]], dtype=torch.long, device=self.device)
        enc_dtype = next(self.model.encoder.parameters()).dtype
        H   = self.model.encoder(input_ids=ids, attention_mask=attn).last_hidden_state
        vec = self.model.pooling(H, attn).to(enc_dtype).float()  # (1, D)
        return vec

    def create_session(self, temp_warmup: int = 8, temp_max: float = 4.0) -> "StreamingSession":
        """통화 세션 단위의 증분 추론 컨텍스트를 생성해 반환."""
        return StreamingSession(self, temp_warmup=temp_warmup, temp_max=temp_max)

    @torch.no_grad()
    def predict_text(self, text: str, threshold: float = 0.5) -> dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {"status": "fail", "message": "empty transcript"}

        segments = self.build_segments(self.tokenizer, text)
        if not segments:
            return {"status": "fail", "message": "no token segments"}

        with self._predict_lock:
            batch = self._batch_from_segments(segments)
            with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                outputs = self.model(**batch)
                probs = F.softmax(outputs["logits"].float(), dim=-1)[0].detach().cpu()

        pred_idx = int(probs.argmax().item())
        pred_label = self.idx_to_label[pred_idx]
        class_probs = {label: float(probs[index].item()) for index, label in enumerate(self.labels)}
        phishing_prob = max(class_probs.get(label, 0.0) for label in PHISHING_LABELS)
        risk_score = phishing_prob * 100.0
        is_phishing = pred_label in PHISHING_LABELS or phishing_prob >= threshold

        return {
            "status": "success",
            "is_phishing": bool(is_phishing),
            "max_risk_score": risk_score,
            "dangerous_segment": self._dangerous_excerpt(text),
            "transcript": text,
            "pred_label_id": pred_idx,
            "pred_label": pred_label,
            "confidence": float(probs[pred_idx].item()),
            "class_probs": class_probs,
            "num_segments": len(segments),
            "model_kind": "roberta_mamba_4class",
            "model_path": str(self.model_path),
        }

    def predict_file(self, script_path: str | os.PathLike, threshold: float = 0.5) -> dict[str, Any]:
        path = Path(script_path)
        if not path.exists():
            return {"status": "fail", "message": f"script not found: {path}"}
        return self.predict_text(path.read_text(encoding="utf-8"), threshold=threshold)


class RobertaMamba4ClassAudioDetector:
    def __init__(self, model_path: str | os.PathLike | None = None, device: str | None = None):
        self.text_detector = RobertaMamba4ClassTextDetector(model_path=model_path, device=device)
        self.processor = AudioProcessor()

    def predict(self, audio_file_path: str | os.PathLike, threshold: float = 0.5) -> dict[str, Any]:
        sentences = self.processor.process_file(str(audio_file_path))
        if not sentences:
            return {"status": "fail", "message": "No text detected"}
        transcript = " ".join(sentences)
        result = self.text_detector.predict_text(transcript, threshold=threshold)
        result["sentences"] = sentences
        return result


class StreamingSession:
    """통화 세션 단위의 증분(incremental) 추론 컨텍스트.

    streaming_inference.py 의 run_interactive 방식을 서비스 환경에 적용:
      - 새로 생긴 세그먼트만 RoBERTa 로 인코딩하고 벡터를 캐싱
      - Mamba 는 매번 캐시 전체를 재실행 (병렬 스캔 구조상 step caching 불가)
      - 초반 세그먼트의 불안정한 예측을 temperature warmup 으로 완화

    세션이 끝나거나 reset 요청이 오면 reset() 을 호출해 캐시를 비워야 함.
    """

    def __init__(
        self,
        detector: RobertaMamba4ClassTextDetector,
        temp_warmup: int = 8,
        temp_max: float = 4.0,
    ) -> None:
        self._detector = detector
        self._temp_warmup = temp_warmup
        self._temp_max = temp_max
        self._encoded_cache: list[torch.Tensor] = []  # 세그먼트별 (1, D) 인코딩 벡터
        self._processed_seg_count: int = 0

    def reset(self) -> None:
        self._encoded_cache.clear()
        self._processed_seg_count = 0

    @torch.no_grad()
    def predict(self, full_text: str, threshold: float = 0.5) -> dict[str, Any]:
        text = (full_text or "").strip()
        if not text:
            return {"status": "fail", "message": "empty transcript"}

        segments = self._detector.build_segments(self._detector.tokenizer, text)
        if not segments:
            return {"status": "fail", "message": "no token segments"}

        new_segs = segments[self._processed_seg_count:]

        with self._detector._predict_lock:
            # 새 세그먼트만 RoBERTa 인코딩 후 캐시에 추가
            for seg in new_segs:
                self._encoded_cache.append(self._detector._encode_window(seg))
                self._processed_seg_count += 1

            if not self._encoded_cache:
                return {"status": "fail", "message": "no encoded segments"}

            # 캐시 전체로 Mamba 실행
            x_stacked = torch.stack(self._encoded_cache, dim=1)  # (1, T, D)
            y = x_stacked
            model = self._detector.model
            with torch.amp.autocast("cuda", enabled=self._detector.device.type == "cuda"):
                for mamba, dropout in zip(model.mamba_layers, model.mamba_dropouts):
                    y = dropout(mamba(y))
                # head도 동일한 autocast 컨텍스트 안에서 호출해야 dtype 일치
                T = y.shape[1]
                head_out = model.head(y[:, -1, :])

            # 마지막 위치에서 temperature 적용 후 확률 계산
            temp = _get_temperature(T, self._temp_warmup, self._temp_max)

            super_probs    = torch.softmax(head_out["super_logits"],    dim=-1)
            normal_probs   = torch.softmax(head_out["normal_logits"],   dim=-1)
            phishing_probs = torch.softmax(head_out["phishing_logits"], dim=-1)
            logits_4 = torch.cat([
                super_probs[:, 0:1] * normal_probs,
                super_probs[:, 1:2] * phishing_probs,
            ], dim=-1)
            log_logits = torch.log(logits_4 + 1e-8)
            probs = torch.softmax(log_logits / temp, dim=-1)[0].cpu()

        pred_idx   = int(probs.argmax().item())
        pred_label = self._detector.idx_to_label[pred_idx]
        labels     = self._detector.labels
        class_probs   = {label: float(probs[i].item()) for i, label in enumerate(labels)}
        phishing_prob = max(class_probs.get(lbl, 0.0) for lbl in PHISHING_LABELS)
        risk_score    = phishing_prob * 100.0
        is_phishing   = pred_label in PHISHING_LABELS or phishing_prob >= threshold

        return {
            "status": "success",
            "is_phishing": bool(is_phishing),
            "max_risk_score": risk_score,
            "dangerous_segment": self._detector._dangerous_excerpt(text),
            "transcript": text,
            "pred_label_id": pred_idx,
            "pred_label": pred_label,
            "confidence": float(probs[pred_idx].item()),
            "class_probs": class_probs,
            "num_segments": self._processed_seg_count,
            "model_kind": "roberta_mamba_4class",
            "model_path": str(self._detector.model_path),
        }

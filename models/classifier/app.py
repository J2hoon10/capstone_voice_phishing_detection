import os
import tempfile
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from config import CONFIG
from five_label_inference import FiveLabelVoicePhishingDetector
from inference import VoicePhishingDetector


app = FastAPI(title="Classifier Service", version="1.0.0")

_detector = None
_detector_lock = Lock()
_detector_error = None


def _get_detector() -> VoicePhishingDetector:
    global _detector, _detector_error
    if _detector is not None:
        return _detector

    with _detector_lock:
        if _detector is not None:
            return _detector
        model_kind = CONFIG.get("MODEL_KIND", "legacy")
        model_path = os.getenv("FIVE_LABEL_MODEL_PATH") if model_kind == "roberta_avgpool_5" else CONFIG["MODEL_PATH"]
        if model_kind == "roberta_avgpool_5" and not model_path:
            model_path = None
        elif model_path and not Path(model_path).exists():
            raise RuntimeError(f"model file not found: {model_path}")
        try:
            if model_kind == "roberta_avgpool_5":
                _detector = FiveLabelVoicePhishingDetector(model_path=model_path, device=CONFIG["DEVICE"])
            else:
                _detector = VoicePhishingDetector(model_path=model_path, device=CONFIG["DEVICE"])
            return _detector
        except Exception as exc:
            _detector_error = str(exc)
            raise


@app.on_event("startup")
def _startup() -> None:
    try:
        _get_detector()
    except Exception:
        # Keep service alive for health diagnostics even when the model is unavailable.
        pass


@app.get("/health")
def health() -> dict:
    model_loaded = _detector is not None
    model_kind = CONFIG.get("MODEL_KIND", "legacy")
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "device": CONFIG["DEVICE"],
        "model_kind": model_kind,
        "model_path": os.getenv("FIVE_LABEL_MODEL_PATH") if model_kind == "roberta_avgpool_5" else CONFIG["MODEL_PATH"],
        "error": _detector_error,
    }


@app.post("/predict")
async def predict(
    audio: UploadFile = File(...),
    threshold: float = Form(CONFIG["DEFAULT_THRESHOLD"]),
) -> dict:
    if threshold < 0.0 or threshold > 1.0:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1")

    detector = _get_detector()
    suffix = Path(audio.filename or "").suffix or ".wav"

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name
            tmp.write(await audio.read())

        result = detector.predict(temp_path, threshold=threshold)
        result["filename"] = audio.filename
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"prediction failed: {exc}") from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

import os
import tempfile
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket

from config import CONFIG
from realtime_streaming import RealtimeVoicePhishingSession
from roberta_mamba_4class_inference import (
    DEFAULT_CHECKPOINT,
    RobertaMamba4ClassAudioDetector,
    RobertaMamba4ClassTextDetector,
)


app = FastAPI(title="Classifier Service", version="1.0.0")

_detector: RobertaMamba4ClassAudioDetector | None = None
_detector_lock = Lock()
_detector_error: str | None = None


def _get_detector() -> RobertaMamba4ClassAudioDetector:
    global _detector, _detector_error
    if _detector is not None:
        return _detector

    with _detector_lock:
        if _detector is not None:
            return _detector
        model_path = os.getenv("ROBERTA_MAMBA_MODEL_PATH", str(DEFAULT_CHECKPOINT))
        if not Path(model_path).exists():
            raise RuntimeError(f"model file not found: {model_path}")
        try:
            _detector = RobertaMamba4ClassAudioDetector(model_path=model_path, device=CONFIG["DEVICE"])
            return _detector
        except Exception as exc:
            _detector_error = str(exc)
            raise


def _get_text_detector() -> RobertaMamba4ClassTextDetector:
    return _get_detector().text_detector


def _get_realtime_audio_processor():
    return _get_detector().processor


@app.on_event("startup")
def _startup() -> None:
    try:
        _get_detector()
    except Exception:
        pass


@app.get("/health")
def health() -> dict:
    model_loaded = _detector is not None
    model_path = os.getenv("ROBERTA_MAMBA_MODEL_PATH", str(DEFAULT_CHECKPOINT))
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "device": CONFIG["DEVICE"],
        "model_kind": "roberta_mamba_4class",
        "model_path": model_path,
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


@app.post("/predict_text")
async def predict_text(payload: dict) -> dict:
    text = str(payload.get("text") or "")
    threshold = float(payload.get("threshold", CONFIG["DEFAULT_THRESHOLD"]))
    if threshold < 0.0 or threshold > 1.0:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1")
    try:
        return _get_text_detector().predict_text(text, threshold=threshold)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"text prediction failed: {exc}") from exc


@app.websocket("/ws/realtime")
async def realtime_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        detector = _get_text_detector()
        audio_processor = _get_realtime_audio_processor()
    except Exception as exc:
        await websocket.send_json({"event": "error", "detail": f"realtime init failed: {exc}"})
        await websocket.close()
        return

    session = RealtimeVoicePhishingSession(
        websocket=websocket,
        detector=detector,
        audio_processor=audio_processor,
    )
    await session.run()

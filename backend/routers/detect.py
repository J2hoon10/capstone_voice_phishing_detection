from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from schemas.detect import DetectResponse
from services.classifier_client import classifier_client
from services.guidance_client import guidance_client


router = APIRouter(prefix="/api", tags=["detect"])


def _warning_level(score: float) -> str:
    if score >= 60:
        return "WARNING"
    if score >= 30:
        return "CAUTION"
    return "NORMAL"


@router.post("/detect", response_model=DetectResponse)
async def detect_audio(
    audio: UploadFile = File(...),
    threshold: float = Form(0.5),
) -> DetectResponse:
    if threshold < 0.0 or threshold > 1.0:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty audio payload")

    try:
        prediction = await classifier_client.predict_bytes(
            audio_bytes=audio_bytes,
            filename=audio.filename or "upload.wav",
            threshold=threshold,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"classifier call failed: {exc}") from exc

    if prediction.get("status") != "success":
        return DetectResponse(
            status=prediction.get("status", "fail"),
            is_phishing=False,
            max_risk_score=0.0,
            dangerous_segment="",
            warning_level="NORMAL",
            guidance=None,
            raw=prediction,
        )

    max_risk_score = float(prediction.get("max_risk_score", 0.0))
    warning_level = _warning_level(max_risk_score)
    dangerous_segment = prediction.get("dangerous_segment", "")

    try:
        guidance_response = await guidance_client.get_guidance(
            risk_score=max_risk_score,
            warning_level=warning_level,
            text=dangerous_segment,
        )
    except Exception:
        guidance_response = None

    return DetectResponse(
        status="success",
        is_phishing=bool(prediction.get("is_phishing", False)),
        max_risk_score=max_risk_score,
        dangerous_segment=dangerous_segment,
        warning_level=warning_level,
        guidance=guidance_response,
        raw=prediction,
    )


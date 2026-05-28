from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from schemas.detect import DetectResponse
from services.classifier_client import classifier_client
from services.guidance_client import guidance_client


router = APIRouter(prefix="/api", tags=["detect"])


def _warning_level(class_probs: dict) -> str:
    max_prob = max(
        class_probs.get("대출 사기형", 0.0),
        class_probs.get("수사기관 사칭형", 0.0),
    )
    if max_prob > 0.9:
        return "WARNING"
    if max_prob > 0.8:
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
    warning_level = _warning_level(prediction.get("class_probs") or {})
    dangerous_segment = prediction.get("dangerous_segment", "")
    pred_label = prediction.get("pred_label")

    try:
        guidance_response = await guidance_client.get_guidance(
            risk_score=max_risk_score,
            warning_level=warning_level,
            text=f"{pred_label or ''}\n{dangerous_segment}".strip(),
        )
    except Exception:
        guidance_response = None

    return DetectResponse(
        status="success",
        is_phishing=bool(prediction.get("is_phishing", False)),
        max_risk_score=max_risk_score,
        dangerous_segment=dangerous_segment,
        pred_label_id=prediction.get("pred_label_id"),
        pred_label=pred_label,
        confidence=prediction.get("confidence"),
        class_probs=prediction.get("class_probs"),
        warning_level=warning_level,
        guidance=guidance_response,
        raw=prediction,
    )

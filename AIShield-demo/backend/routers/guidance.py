from fastapi import APIRouter, HTTPException

from schemas.guidance import GuidanceRequest, GuidanceResponse
from services.guidance_client import guidance_client


router = APIRouter(prefix="/api", tags=["guidance"])


@router.post("/guidance", response_model=GuidanceResponse)
async def get_guidance(payload: GuidanceRequest) -> GuidanceResponse:
    try:
        response = await guidance_client.get_guidance(
            risk_score=payload.risk_score,
            warning_level=payload.warning_level,
            text=payload.text,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"guidance call failed: {exc}") from exc

    return GuidanceResponse.model_validate(response)


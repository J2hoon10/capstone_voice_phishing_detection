from fastapi import FastAPI
from pydantic import BaseModel, Field

from guidance_engine import GuidanceEngine


class GuidanceRequest(BaseModel):
    risk_score: float = Field(..., ge=0, le=100)
    warning_level: str = Field(default="NORMAL")
    text: str = Field(default="")


engine = GuidanceEngine()
app = FastAPI(title="Guidance Service", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/guidance")
def get_guidance(payload: GuidanceRequest) -> dict:
    guidance = engine.build_guidance(
        risk_score=payload.risk_score,
        warning_level=payload.warning_level,
        text=payload.text,
    )
    return {
        "risk_score": payload.risk_score,
        "warning_level": payload.warning_level,
        "text": payload.text,
        "guidance": guidance,
    }


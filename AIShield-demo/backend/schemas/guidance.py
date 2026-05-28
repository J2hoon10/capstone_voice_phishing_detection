from typing import Any

from pydantic import BaseModel, Field


class GuidanceRequest(BaseModel):
    risk_score: float = Field(..., ge=0, le=100)
    warning_level: str = Field(default="NORMAL")
    text: str = Field(default="")


class GuidanceResponse(BaseModel):
    risk_score: float
    warning_level: str
    text: str
    guidance: dict[str, Any]


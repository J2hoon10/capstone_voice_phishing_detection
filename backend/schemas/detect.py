from typing import Any

from pydantic import BaseModel, Field


class DetectResponse(BaseModel):
    status: str = Field(default="success")
    is_phishing: bool = Field(default=False)
    max_risk_score: float = Field(default=0.0, ge=0, le=100)
    dangerous_segment: str = Field(default="")
    warning_level: str = Field(default="NORMAL")
    guidance: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None


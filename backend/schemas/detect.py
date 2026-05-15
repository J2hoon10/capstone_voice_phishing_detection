from typing import Any

from pydantic import BaseModel, Field


class DetectResponse(BaseModel):
    status: str = Field(default="success")
    is_phishing: bool = Field(default=False)
    max_risk_score: float = Field(default=0.0, ge=0, le=100)
    dangerous_segment: str = Field(default="")
    pred_label_id: int | None = Field(default=None)
    pred_label: str | None = Field(default=None)
    confidence: float | None = Field(default=None)
    class_probs: dict[str, float] | None = Field(default=None)
    warning_level: str = Field(default="NORMAL")
    guidance: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None

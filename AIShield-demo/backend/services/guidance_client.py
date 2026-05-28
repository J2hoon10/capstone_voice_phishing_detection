import os

import httpx


class GuidanceClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("GUIDANCE_URL", "http://guidance:8002")
        self.timeout = float(os.getenv("GUIDANCE_TIMEOUT", "30"))

    async def get_guidance(self, risk_score: float, warning_level: str, text: str) -> dict:
        payload = {
            "risk_score": risk_score,
            "warning_level": warning_level,
            "text": text,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/guidance", json=payload)
            response.raise_for_status()
            return response.json()


guidance_client = GuidanceClient()


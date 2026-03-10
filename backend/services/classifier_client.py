import os

import httpx


class ClassifierClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("CLASSIFIER_URL", "http://classifier:8001")
        self.timeout = float(os.getenv("CLASSIFIER_TIMEOUT", "180"))

    async def predict_bytes(self, audio_bytes: bytes, filename: str, threshold: float = 0.5) -> dict:
        files = {"audio": (filename, audio_bytes, "application/octet-stream")}
        data = {"threshold": str(threshold)}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/predict", files=files, data=data)
            response.raise_for_status()
            return response.json()


classifier_client = ClassifierClient()


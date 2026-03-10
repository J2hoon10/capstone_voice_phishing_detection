import json
from dataclasses import dataclass
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from services.classifier_client import classifier_client
from services.guidance_client import guidance_client


router = APIRouter(tags=["stream"])


@dataclass
class SessionRiskScorer:
    current_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 100.0

    def update(self, risk_score: float) -> tuple[float, str]:
        if risk_score > 80:
            self.current_score += 20
        elif risk_score > 50:
            self.current_score += 10
        else:
            self.current_score -= 10

        self.current_score = max(self.min_score, min(self.current_score, self.max_score))
        if self.current_score >= 60:
            level = "WARNING"
        elif self.current_score >= 30:
            level = "CAUTION"
        else:
            level = "NORMAL"
        return self.current_score, level


@router.websocket("/ws/stream")
async def stream_detection(websocket: WebSocket) -> None:
    await websocket.accept()
    scorer = SessionRiskScorer()

    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"] is not None:
                audio_bytes = message["bytes"]
                filename = f"chunk-{uuid4().hex}.webm"
                prediction = await classifier_client.predict_bytes(audio_bytes, filename, threshold=0.5)

                if prediction.get("status") != "success":
                    await websocket.send_json(
                        {
                            "event": "prediction",
                            "status": "fail",
                            "score": scorer.current_score,
                            "warning_level": "NORMAL",
                            "transcript": "",
                            "guidance": None,
                        }
                    )
                    continue

                risk_score = float(prediction.get("max_risk_score", 0.0))
                transcript = prediction.get("dangerous_segment", "")
                accumulated, warning_level = scorer.update(risk_score)
                guidance_response = await guidance_client.get_guidance(
                    risk_score=accumulated,
                    warning_level=warning_level,
                    text=transcript,
                )

                await websocket.send_json(
                    {
                        "event": "prediction",
                        "status": "success",
                        "risk_score": risk_score,
                        "score": accumulated,
                        "warning_level": warning_level,
                        "transcript": transcript,
                        "guidance": guidance_response.get("guidance"),
                    }
                )
            elif "text" in message and message["text"] is not None:
                text = message["text"]
                if text == "ping":
                    await websocket.send_text("pong")
                    continue

                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    await websocket.send_json({"event": "error", "detail": "invalid json message"})
                    continue

                if payload.get("event") == "reset":
                    scorer = SessionRiskScorer()
                    await websocket.send_json({"event": "reset", "status": "ok"})
                else:
                    await websocket.send_json({"event": "ack", "status": "ignored"})
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_json({"event": "error", "detail": str(exc)})
        await websocket.close()


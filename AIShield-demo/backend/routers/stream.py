import asyncio
import json
import os
from urllib.parse import urlparse
from dataclasses import dataclass
from uuid import uuid4

import websockets
import websockets.exceptions
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


def _classifier_realtime_ws_url() -> str:
    base_url = os.getenv("CLASSIFIER_URL", "http://classifier:8001").rstrip("/")
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    netloc = parsed.netloc or parsed.path
    path = parsed.path if parsed.netloc else ""
    return f"{scheme}://{netloc}{path}/ws/realtime"


@router.websocket("/ws/realtime")
async def realtime_detection_proxy(websocket: WebSocket) -> None:
    await websocket.accept()
    upstream_url = _classifier_realtime_ws_url()
    try:
        async with websockets.connect(upstream_url, max_size=None) as upstream:
            async def client_to_classifier() -> None:
                while True:
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        await upstream.close()
                        return
                    if message.get("bytes") is not None:
                        await upstream.send(message["bytes"])
                    elif message.get("text") is not None:
                        await upstream.send(message["text"])

            async def classifier_to_client() -> None:
                async for message in upstream:
                    if isinstance(message, bytes):
                        await websocket.send_bytes(message)
                    else:
                        await websocket.send_text(message)

            tasks = [
                asyncio.create_task(client_to_classifier()),
                asyncio.create_task(classifier_to_client()),
            ]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                task.result()
    except (WebSocketDisconnect, websockets.exceptions.ConnectionClosedOK):
        return
    except Exception as exc:
        await websocket.send_json({"event": "error", "detail": f"classifier realtime stream failed: {exc}"})
        await websocket.close()


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


import asyncio
import os
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from audio_processor import AudioProcessor
from roberta_mamba_4class_inference import RobertaMamba4ClassTextDetector, StreamingSession


class RealtimeVoicePhishingSession:
    def __init__(
        self,
        websocket: WebSocket,
        detector: RobertaMamba4ClassTextDetector,
        audio_processor: AudioProcessor,
    ) -> None:
        self.websocket = websocket
        self.detector = detector
        self.audio_processor = audio_processor
        self.sample_rate = int(os.getenv("REALTIME_SAMPLE_RATE", "16000"))
        self.window_seconds = float(os.getenv("REALTIME_STT_WINDOW_SECONDS", "20"))
        self.stride_seconds = float(os.getenv("REALTIME_STT_STRIDE_SECONDS", "10"))
        self.model_poll_seconds = float(os.getenv("REALTIME_MODEL_POLL_SECONDS", "0.5"))
        self.threshold = float(os.getenv("REALTIME_THRESHOLD", "0.5"))
        self.script_dir = Path(os.getenv("REALTIME_SCRIPT_DIR", "runtime_scripts"))
        self.session_id = uuid4().hex
        self.script_path = self.script_dir / f"session_{self.session_id}.txt"

        self._audio = np.zeros(0, dtype=np.float32)
        self._audio_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._running = True
        self._next_stt_start = 0.0
        self._last_transcript = ""
        self._last_prediction_source = ""
        self._streaming_session: StreamingSession = detector.create_session(
            temp_warmup=int(os.getenv("REALTIME_TEMP_WARMUP", "8")),
            temp_max=float(os.getenv("REALTIME_TEMP_MAX", "4.0")),
        )

    async def run(self) -> None:
        self.script_dir.mkdir(parents=True, exist_ok=True)
        self.script_path.write_text("", encoding="utf-8")
        await self._send_json(
            {
                "event": "ready",
                "session_id": self.session_id,
                "sample_rate": self.sample_rate,
                "window_seconds": self.window_seconds,
                "stride_seconds": self.stride_seconds,
                "model_poll_seconds": self.model_poll_seconds,
                "script_path": str(self.script_path),
            }
        )

        tasks = [
            asyncio.create_task(self._stt_loop()),
            asyncio.create_task(self._prediction_loop()),
        ]
        try:
            await self._receive_loop()
        finally:
            self._running = False
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._send_json({"event": "closed", "session_id": self.session_id})

    async def _receive_loop(self) -> None:
        try:
            while self._running:
                message = await self.websocket.receive()
                if message.get("bytes") is not None:
                    await self._append_pcm16(message["bytes"])
                    continue

                text = message.get("text")
                if text is None:
                    continue
                if text == "ping":
                    await self._send_json({"event": "pong"})
                elif '"event":"stop"' in text or '"event": "stop"' in text:
                    self._running = False
                    return
                elif '"event":"reset"' in text or '"event": "reset"' in text:
                    await self._reset()
                else:
                    await self._send_json({"event": "ack", "status": "ignored"})
        except WebSocketDisconnect:
            self._running = False

    async def _append_pcm16(self, payload: bytes) -> None:
        if not payload:
            return
        chunk = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
        if chunk.size == 0:
            return
        async with self._audio_lock:
            self._audio = np.concatenate([self._audio, chunk])
            duration = self._audio.size / self.sample_rate
        await self._send_json({"event": "audio", "duration": duration})

    async def _reset(self) -> None:
        async with self._audio_lock:
            self._audio = np.zeros(0, dtype=np.float32)
        self._next_stt_start = 0.0
        self._last_transcript = ""
        self._last_prediction_source = ""
        self._streaming_session.reset()
        self.script_path.write_text("", encoding="utf-8")
        await self._send_json({"event": "reset", "status": "ok"})

    async def _stt_loop(self) -> None:
        window_samples = int(self.window_seconds * self.sample_rate)
        stride = self.stride_seconds
        while self._running:
            await asyncio.sleep(0.5)
            start = int(self._next_stt_start * self.sample_rate)
            end = start + window_samples
            async with self._audio_lock:
                if self._audio.size < end:
                    continue
                audio_window = self._audio[start:end].copy()

            window_start = self._next_stt_start
            window_end = window_start + self.window_seconds
            self._next_stt_start += stride

            started_at = time.time()
            transcript = await asyncio.to_thread(self._transcribe, audio_window)
            elapsed = time.time() - started_at
            if not transcript or transcript == self._last_transcript:
                continue

            self._last_transcript = transcript
            line = f"[{window_start:06.1f}-{window_end:06.1f}] {transcript}"
            with self.script_path.open("a", encoding="utf-8") as fp:
                fp.write(line + "\n")

            await self._send_json(
                {
                    "event": "transcript",
                    "session_id": self.session_id,
                    "window_start": window_start,
                    "window_end": window_end,
                    "text": transcript,
                    "script_path": str(self.script_path),
                    "stt_seconds": elapsed,
                }
            )

    def _transcribe(self, audio_window: np.ndarray) -> str:
        sentences = self.audio_processor.process_file(audio_window)
        return " ".join(sentence.strip() for sentence in sentences if sentence.strip()).strip()

    async def _prediction_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.model_poll_seconds)
            try:
                source = self.script_path.read_text(encoding="utf-8")
            except FileNotFoundError:
                continue
            if not source.strip() or source == self._last_prediction_source:
                continue
            self._last_prediction_source = source

            started_at = time.time()
            result = await asyncio.to_thread(self._streaming_session.predict, source, self.threshold)
            elapsed = time.time() - started_at
            if result.get("status") != "success":
                await self._send_json({"event": "prediction", "status": "fail", "raw": result})
                continue

            risk_score = float(result.get("max_risk_score", 0.0))
            warning_level = "WARNING" if risk_score >= 60 else "CAUTION" if risk_score >= 30 else "NORMAL"
            await self._send_json(
                {
                    "event": "prediction",
                    "status": "success",
                    "session_id": self.session_id,
                    "risk_score": risk_score,
                    "score": risk_score,
                    "warning_level": warning_level,
                    "is_phishing": bool(result.get("is_phishing", False)),
                    "transcript": result.get("transcript", ""),
                    "dangerous_segment": result.get("dangerous_segment", ""),
                    "pred_label": result.get("pred_label"),
                    "pred_label_id": result.get("pred_label_id"),
                    "confidence": result.get("confidence"),
                    "class_probs": result.get("class_probs"),
                    "num_segments": result.get("num_segments"),
                    "script_path": str(self.script_path),
                    "inference_seconds": elapsed,
                    "raw": result,
                }
            )

    async def _send_json(self, payload: dict) -> None:
        async with self._send_lock:
            try:
                await self.websocket.send_json(payload)
            except Exception:
                self._running = False

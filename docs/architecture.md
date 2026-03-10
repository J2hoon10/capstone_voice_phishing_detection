# Architecture

## Service Graph

```text
[Browser] <-> [frontend (nginx:80)]
                 |
                 +-> /api/* -> [backend:8000] -> [classifier:8001]
                 |                           \-> [guidance:8002]
                 |
                 +-> /ws/stream -> [backend:8000 websocket]
```

## Responsibilities

- `frontend`
  - File upload detection request
  - Live microphone chunk streaming via WebSocket
  - Risk gauge, warning banner, transcript, and guidance rendering
- `backend`
  - API gateway and response orchestration
  - Classifier + guidance service integration
  - Per-websocket-session risk score state management
- `classifier`
  - Audio enhancement + Whisper transcription
  - ModernBERT phishing binary classification
  - Returns `is_phishing`, `max_risk_score`, `dangerous_segment`
- `guidance`
  - Keyword-based phishing type matching
  - Returns 대응 행동 지침 + 긴급 연락처 템플릿

## Runtime Notes

- Model weights are mounted from host to `models/classifier/weights`.
- Hugging Face cache is persisted via named volume `hf-cache`.
- In non-GPU environments set `CLASSIFIER_DEVICE=cpu` in `.env`.

# API Spec

## 1) Classifier Service

### `GET /health`
- Response:
```json
{
  "status": "ok|degraded",
  "model_loaded": true,
  "device": "cuda",
  "model_path": "/app/weights/student_best.pt",
  "error": null
}
```

### `POST /predict`
- Content-Type: `multipart/form-data`
- Fields:
  - `audio`: audio file
  - `threshold`: float (0~1, optional)
- Success response:
```json
{
  "status": "success",
  "is_phishing": true,
  "max_risk_score": 87.3,
  "dangerous_segment": "...",
  "filename": "sample.wav"
}
```

## 2) Guidance Service

### `GET /health`
- Response: `{ "status": "ok" }`

### `POST /guidance`
- Body:
```json
{
  "risk_score": 75,
  "warning_level": "WARNING",
  "text": "검찰에서 계좌 동결..."
}
```
- Response:
```json
{
  "risk_score": 75,
  "warning_level": "WARNING",
  "text": "...",
  "guidance": {
    "matched_type": "impersonation_investigation",
    "matched_label": "수사기관 사칭형",
    "summary": "...",
    "actions": ["..."],
    "emergency_contacts": [{"name":"경찰청","phone":"112","description":"긴급 신고"}],
    "banks_notice": "..."
  }
}
```

## 3) Backend Gateway

### `GET /health`
- Response: `{ "status": "ok" }`

### `POST /api/detect`
- Content-Type: `multipart/form-data`
- Fields:
  - `audio`: audio file
  - `threshold`: float (optional)
- Response:
```json
{
  "status": "success",
  "is_phishing": true,
  "max_risk_score": 83.4,
  "dangerous_segment": "...",
  "warning_level": "WARNING",
  "guidance": { "...": "..." },
  "raw": { "...": "classifier raw response" }
}
```

### `POST /api/guidance`
- Body: guidance service request format
- Response: guidance service response format

### `WS /ws/stream`
- Client -> server:
  - binary audio chunk (`audio/webm`)
  - optional text JSON `{ "event": "reset" }`
- Server -> client (`event=prediction`):
```json
{
  "event": "prediction",
  "status": "success",
  "risk_score": 64.0,
  "score": 70.0,
  "warning_level": "WARNING",
  "transcript": "...",
  "guidance": { "...": "..." }
}
```

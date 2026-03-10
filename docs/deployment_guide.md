# Deployment Guide

## Prerequisites

- Docker + Docker Compose plugin
- Model weight file: `student_best.pt`
- Optional GPU runtime:
  - NVIDIA driver on host
  - NVIDIA Container Toolkit installed

## 1) Prepare

1. Copy environment file:
   - PowerShell: `Copy-Item .env.example .env`
2. Place model file:
   - `models/classifier/weights/student_best.pt`
3. For CPU-only run, set:
   - `CLASSIFIER_DEVICE=cpu` in `.env`

## 2) Build and Run

```bash
docker compose up --build
```

Open `http://localhost` (or configured `FRONTEND_PORT`).

## 3) Smoke Tests

### Classifier only
```bash
curl -X POST http://localhost:8001/predict -F "audio=@test.wav"
```

### Guidance only
```bash
curl -X POST http://localhost:8002/guidance \
  -H "Content-Type: application/json" \
  -d '{"risk_score":75,"warning_level":"WARNING","text":"검찰 계좌 동결"}'
```

### Gateway
```bash
curl -X POST http://localhost/api/detect -F "audio=@test.wav"
```

## 4) Operational Notes

- Whisper model files are cached in Docker volume `hf-cache`.
- `.pt` model files are excluded from git by `.gitignore`.
- If frontend WebSocket fails through proxy, verify `/ws/` upgrade headers in `frontend/nginx.conf`.

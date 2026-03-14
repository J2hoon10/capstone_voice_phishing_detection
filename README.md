# capstone_project

Seoultech capstone
Voice phishing real-time detection demo stack.

## Stack

- `models/classifier`: FastAPI wrapper for Whisper + ModernBERT inference
- `models/guidance`: Rule-based phishing 대응 가이드 서비스
- `backend`: API gateway (`/api/detect`, `/api/guidance`, `/ws/stream`)
- `frontend`: React + Vite UI served by Nginx

## Quick Start

1. Copy env file:
   - `cp .env.example .env` (PowerShell: `Copy-Item .env.example .env`)
2. Put model weights:
   - `models/classifier/weights/student_best.pt`
3. Run services:
   - `docker compose up --build`
4. Open:
   - `http://localhost`

## API Summary

- `POST /api/detect`: multipart audio upload and integrated detect + guidance response
- `POST /api/guidance`: direct guidance query
- `WS /ws/stream`: realtime chunk streaming

Detailed docs are in `docs/`:
- `docs/architecture.md`
- `docs/api_spec.md`
- `docs/deployment_guide.md`

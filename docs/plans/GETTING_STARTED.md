# 프로젝트 시작 가이드

## 필수 환경

| 도구 | 버전 | 비고 |
|------|------|------|
| Git | 2.30+ | |
| Python | 3.10+ | classifier, guidance, backend |
| Node.js | 18+ | frontend |
| Docker | 24+ | 전체 서비스 실행 시 |
| Docker Compose | v2+ | |

## 1. 저장소 클론

```bash
git clone https://github.com/<org>/capstone_voice_phishing_detection.git
cd capstone_voice_phishing_detection
```

## 2. 환경 파일 설정

```bash
cp .env.example .env
```

`.env` 파일에서 필요한 값을 수정합니다. 주요 설정:

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `CLASSIFIER_DEVICE` | `cuda` | GPU 없으면 `cpu`로 변경 |
| `MODEL_PATH` | `/app/weights/student_best.pt` | 모델 가중치 경로 |
| `WHISPER_MODEL_SIZE` | `Systran/faster-whisper-base` | Whisper 모델 크기 |
| `FRONTEND_PORT` | `80` | 프론트엔드 포트 |

## 3. 모델 가중치 배치

```bash
# student_best.pt 파일을 아래 경로에 배치
models/classifier/weights/student_best.pt
```

> 가중치 파일은 `.gitignore` 대상이므로 별도로 공유받아야 합니다.

## 4. 전체 서비스 실행 (Docker)

```bash
# GPU 환경
docker compose up --build

# CPU 전용 환경
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build
```

실행 후 `http://localhost`에서 확인할 수 있습니다.

## 5. 개별 서비스 실행 (로컬 개발)

각 서비스를 Docker 없이 직접 실행하려면:

### Classifier (포트 8001)

```bash
cd models/classifier
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

### Guidance (포트 8002)

```bash
cd models/guidance
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8002 --reload
```

### Backend (포트 8000)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend (포트 5173)

```bash
cd frontend
npm install
npm run dev
```

## 6. 동작 확인

각 서비스의 헬스체크 엔드포인트로 정상 동작을 확인합니다:

```bash
curl http://localhost:8001/health   # classifier
curl http://localhost:8002/health   # guidance
curl http://localhost:8000/health   # backend
```

상세한 API 테스트 방법은 [api_spec.md](api_spec.md)와 [deployment_guide.md](deployment_guide.md)를 참고하세요.

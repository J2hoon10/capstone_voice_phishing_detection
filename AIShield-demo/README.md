# AIShield 데모

AIShield 보이스피싱 탐지 데모 패키지입니다.

```text
AIShield-demo/
  frontend/          React + Vite 모바일 데모 UI
  backend/           FastAPI API 게이트웨이
  models/
    classifier/      분류기 서비스 소스
    guidance/        대응 가이던스 서비스 소스
  docker-compose.yml
  .env.example
```

## UI 단독 실행

백엔드 없이 UI 화면과 폴백 흐름만 확인할 때 사용합니다.

```bash
cd frontend
npm install
npm run dev
```

`http://localhost:5174` 에서 확인합니다.

UI는 업로드 음성 분석에 `/api/detect`, 실시간 마이크 분석에 `/ws/realtime` 을 호출합니다. 백엔드·모델 서비스가 실행 중이 아닌 경우 UI가 자동으로 발표용 폴백 결과로 전환되어 데모 화면은 그대로 동작합니다.

## 백엔드 단독 실행

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

헬스체크:

```bash
curl http://localhost:8000/health
```

실제 예측을 위해 백엔드는 아래 환경변수를 사용합니다:

- `CLASSIFIER_URL` 기본값: `http://classifier:8001`
- `GUIDANCE_URL` 기본값: `http://guidance:8002`

Docker 없이 로컬에서 테스트할 경우:

```bash
export CLASSIFIER_URL=http://localhost:8001
export GUIDANCE_URL=http://localhost:8002
```

## Docker 전체 실행

frontend / backend / classifier / guidance 4개 서비스를 한 번에 실행합니다.

```bash
cp .env.example .env
docker compose up --build
```

`http://localhost` 에서 확인합니다.

기본 설정은 `CLASSIFIER_DEVICE=cpu` 로 되어 있어 GPU가 없는 환경에서도 실행 가능합니다. CUDA GPU가 있는 경우 `.env` 를 수정하세요:

```bash
CLASSIFIER_DEVICE=cuda
```

## 모델 가중치

`.pt` 체크포인트 파일은 용량 문제로 저장소에 포함되지 않습니다.

Docker 기본 설정은 아래 경로에서 체크포인트를 찾습니다:

```text
models/classifier/checkpoints/roberta_mamba_freeze_init_4class_4class_20260517_174922_best.pt
```

다른 파일을 사용할 경우 `.env` 에서 경로를 지정하세요:

```bash
CLASSIFIER_MODEL_KIND=roberta_mamba_4class
ROBERTA_MAMBA_MODEL_PATH=/app/checkpoints/YOUR_MODEL.pt
```

체크포인트 파일이 없으면 classifier 헬스 엔드포인트가 `degraded` 상태로 표시되고 실제 예측이 동작하지 않습니다. 프론트엔드 폴백은 체크포인트 없이도 동작합니다.

## 서비스 URL 목록

| 서비스 | URL |
|--------|-----|
| 프론트엔드 (dev) | `http://localhost:5174` |
| 프론트엔드 (Docker) | `http://localhost` |
| 백엔드 헬스체크 | `http://localhost:8000/health` |
| 분류기 헬스체크 | `http://localhost:8001/health` |
| 가이던스 헬스체크 | `http://localhost:8002/health` |

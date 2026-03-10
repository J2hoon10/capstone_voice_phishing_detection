# 배포 가이드

## 사전 준비

- Docker + Docker Compose 플러그인
- 모델 가중치 파일: `student_best.pt`
- (선택) GPU 런타임: NVIDIA 드라이버 + NVIDIA Container Toolkit

## 1단계: 환경 설정

```bash
# .env 파일 생성
cp .env.example .env

# 모델 가중치 배치
cp /path/to/student_best.pt models/classifier/weights/student_best.pt

# CPU 전용 환경이라면 .env에서 수정
# CLASSIFIER_DEVICE=cpu
```

## 2단계: 빌드 및 실행

```bash
# GPU 환경
docker compose up --build

# CPU 전용 환경
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build
```

브라우저에서 `http://localhost` 접속 (또는 `.env`의 `FRONTEND_PORT` 포트).

## 3단계: 동작 확인 (Smoke Test)

### Classifier 단독 테스트
```bash
curl -X POST http://localhost:8001/predict -F "audio=@test.wav"
```

### Guidance 단독 테스트
```bash
curl -X POST http://localhost:8002/guidance \
  -H "Content-Type: application/json" \
  -d '{"risk_score":75,"warning_level":"WARNING","text":"검찰 계좌 동결"}'
```

### 게이트웨이 통합 테스트
```bash
curl -X POST http://localhost/api/detect -F "audio=@test.wav"
```

## 운영 참고사항

- Whisper 모델 파일은 `hf-cache` Docker 볼륨에 캐싱됨
- `.pt` 모델 파일은 `.gitignore`에 의해 git에서 제외됨
- WebSocket 연결 실패 시 `frontend/nginx.conf`의 `/ws/` 업그레이드 헤더 확인

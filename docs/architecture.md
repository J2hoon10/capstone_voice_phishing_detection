# 서비스 아키텍처

## 서비스 구성도

```text
[Browser] <-> [frontend (nginx:80)]
                 |
                 +-> /api/* -> [backend:8000] -> [classifier:8001]  ← 담당자A
                 |                           \-> [guidance:8002]    ← 담당자B, C
                 |
                 +-> /ws/stream -> [backend:8000 websocket]
```

## 서비스별 역할

| 서비스 | 포트 | 담당 | 설명 |
|--------|------|------|------|
| `frontend` | 80 | - | React + Vite 웹 UI, 녹음/업로드/결과 표시 |
| `backend` | 8000 | - | FastAPI 게이트웨이, 서비스 간 요청 조율 |
| `classifier` | 8001 | **담당자A** | 음성 → 텍스트 변환(Whisper) + 보이스피싱 분류(ModernBERT) |
| `guidance` | 8002 | **담당자B, C** | 키워드 기반 피싱 유형 매칭 + 대응 지침 제공 |

## 담당 디렉터리 매핑

```text
models/
├── classifier/    ← 담당자A
│   ├── app.py              # FastAPI 앱 (엔드포인트 정의)
│   ├── inference.py         # 추론 파이프라인 (Whisper + ModernBERT)
│   ├── audio_processor.py   # 오디오 전처리 (필터링, 노이즈 제거, VAD)
│   ├── audio_enhancer.py    # 오디오 품질 향상
│   ├── architecture.py      # 모델 아키텍처 정의
│   ├── config.py            # 설정 관리
│   └── weights/             # 모델 가중치 (.gitignore 대상)
│
└── guidance/      ← 담당자B, C
    ├── app.py               # FastAPI 앱 (엔드포인트 정의)
    ├── guidance_engine.py    # 대응 지침 생성 엔진
    └── knowledge_base/      # 피싱 유형 및 긴급 연락처 데이터
        ├── phishing_types.json
        └── emergency_contacts.json
```

## 런타임 참고사항

- 모델 가중치는 호스트의 `models/classifier/weights`를 컨테이너에 마운트
- Hugging Face 캐시는 `hf-cache` Docker 볼륨으로 유지
- GPU가 없는 환경에서는 `.env`에서 `CLASSIFIER_DEVICE=cpu` 설정

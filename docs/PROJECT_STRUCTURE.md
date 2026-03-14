# 프로젝트 파일 구조

> 최종 업데이트: 2026-03-13

## 전체 구조 요약

```
capstone_voice_phishing_detection/
│
├── models/                    ★ 핵심 서비스 (모델 + API)
│   ├── classifier/            ← 담당자A
│   └── guidance/              ← 담당자B, C
│
├── backend/                   API 게이트웨이
├── frontend/                  웹 UI (React + Vite)
├── docs/                      문서
├── Project/                   (아카이브) 연구/학습 코드
│
├── docker-compose.yml         서비스 오케스트레이션
├── docker-compose.cpu.yml     CPU 전용 오버라이드
├── .env.example               환경변수 템플릿
├── .gitignore                 Git 제외 규칙
└── README.md                  프로젝트 소개
```

---

## models/ — 핵심 서비스

### models/classifier/ (담당자A)

보이스피싱 탐지의 핵심. 음성 입력 → 텍스트 변환(STT) → 피싱 분류.

```
models/classifier/
├── app.py                  FastAPI 앱 (/health, /predict 엔드포인트)
├── inference.py            추론 파이프라인 (VoicePhishingDetector, PhishingRiskScorer)
├── audio_processor.py      Whisper STT + 텍스트 정제
├── audio_enhancer.py       오디오 전처리 (Bandpass → Noise Reduction → VAD → Normalize)
├── architecture.py         ModernBERT Student 모델 아키텍처 정의
├── config.py               설정 관리 (디바이스, 모델 경로, 임계값 등)
├── requirements.txt        Python 의존성
├── Dockerfile              컨테이너 빌드 정의
│
├── data/                   음성 데이터 (git 제외, 로컬 전용)
│   ├── phishing/           피싱 음성 파일 (515개)
│   │   ├── 대출 사기형/         185개
│   │   ├── 바로 이 목소리/       94개
│   │   └── 수사기관 사칭형/      236개
│   └── normal/             일반 음성 파일 (2,001개)
│       ├── 상품 가입 및 해지/    762개
│       ├── 이체 출금 대출서비스/  619개
│       └── 잔고 및 거래내역/     620개
│
├── STT_test/               STT 테스트 결과
│   ├── run_stt_test.py         테스트 스크립트
│   ├── stt_test_results.csv    결과 CSV
│   └── stt_test_report.txt     결과 리포트
│
└── weights/                모델 가중치 (git 제외)
    └── student_best.pt         학습된 Student 모델
```

### models/guidance/ (담당자B, C)

피싱 유형 매칭 및 대응 지침 제공.

```
models/guidance/
├── app.py                  FastAPI 앱 (/health, /guidance 엔드포인트)
├── guidance_engine.py      대응 지침 생성 엔진 (키워드 매칭, 행동 지침 구성)
├── knowledge_base/         피싱 지식 베이스
│   ├── phishing_types.json     피싱 유형별 키워드, 요약, 권장 행동
│   └── emergency_contacts.json 긴급 연락처 (경찰, 금융감독원 등)
├── requirements.txt
└── Dockerfile
```

---

## backend/ — API 게이트웨이

클라이언트 요청을 classifier/guidance 서비스로 라우팅.

```
backend/
├── main.py                 FastAPI 앱 초기화, CORS 설정
├── routers/
│   ├── detect.py           POST /api/detect (음성 파일 업로드 → 탐지)
│   ├── guidance.py         POST /api/guidance (대응 지침 요청)
│   └── stream.py           WS /ws/stream (실시간 마이크 스트리밍)
├── schemas/
│   ├── detect.py           탐지 요청/응답 Pydantic 모델
│   └── guidance.py         지침 요청/응답 Pydantic 모델
├── services/
│   ├── classifier_client.py    classifier 서비스 HTTP 클라이언트
│   └── guidance_client.py      guidance 서비스 HTTP 클라이언트
├── requirements.txt
└── Dockerfile
```

---

## frontend/ — 웹 UI

React + Vite 기반. 파일 업로드 및 실시간 마이크 녹음 지원.

```
frontend/
├── src/
│   ├── App.jsx                 메인 앱 컴포넌트
│   ├── main.jsx                엔트리 포인트
│   ├── styles/main.css         스타일시트
│   ├── components/
│   │   ├── AudioRecorder.jsx   실시간 마이크 녹음
│   │   ├── FileUpload.jsx      음성 파일 업로드
│   │   ├── RiskGauge.jsx       위험도 게이지 시각화
│   │   ├── WarningBanner.jsx   경고 배너
│   │   ├── TranscriptView.jsx  전사 텍스트 표시
│   │   └── GuidancePanel.jsx   대응 지침 패널
│   └── hooks/
│       └── useWebSocket.js     WebSocket 연결 훅
├── index.html
├── package.json
├── vite.config.js
├── nginx.conf                  Nginx 프록시 설정
└── Dockerfile
```

---

## docs/ — 문서

```
docs/
├── README.md               문서 인덱스 (목록)
├── GETTING_STARTED.md       환경 설정 및 실행 가이드
├── TEAM_WORKFLOW.md         브랜치 전략, 커밋 컨벤션, PR 규칙
├── ROLE_CLASSIFIER.md       Classifier 담당자(A) 가이드
├── ROLE_GUIDANCE.md         Guidance 담당자(B, C) 가이드
├── architecture.md          서비스 아키텍처 및 담당 영역
├── api_spec.md              API 엔드포인트 명세
├── deployment_guide.md      Docker 배포 가이드
├── PROJECT_STRUCTURE.md     이 파일 (프로젝트 구조 설명)
│
├── plans/                   담당자별 작업 계획서
│   ├── classifier/              담당자A 계획서
│   ├── guidance-B/              담당자B 계획서
│   └── guidance-C/              담당자C 계획서
│
└── archive/                 과거 시나리오 문서
    ├── SINARIO_v1.md            초기 모델 파이프라인 설계
    ├── SINARIO_v2.md            ModernBERT TA 기반 증류 계획
    └── SINARIO_v3.md            최종 증류 계획
```

---

## 서비스 간 통신 흐름

```
[Browser] ←→ [frontend:80 (nginx)]
                  |
                  ├─ /api/*     → [backend:8000] → [classifier:8001]  ← 담당자A
                  |                              └→ [guidance:8002]   ← 담당자B, C
                  |
                  └─ /ws/stream → [backend:8000 WebSocket]
```

## Git에서 제외되는 파일

| 패턴 | 대상 |
|------|------|
| `*.pt`, `*.pth` | 모델 가중치 |
| `*.wav`, `*.mp3`, `*.flac`, `*.m4a`, `*.ogg` | 음성 파일 |
| `*.zip` | 압축 파일 |
| `__pycache__/`, `*.pyc` | Python 캐시 |
| `frontend/node_modules/`, `frontend/dist/` | Node.js 의존성/빌드 |
| `.env` | 환경 변수 (비밀 정보) |

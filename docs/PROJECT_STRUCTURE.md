# 프로젝트 파일 구조

> 최종 업데이트: 2026-05-27

## 전체 구조 요약

```
capstone_voice_phishing_detection/
│
├── models/                    ★ 핵심 서비스 (모델 + API)
│   ├── main/                  ← Baseline 서비스 및 최신 데이터 증강
│   │   ├── model_architecture/← roberta+mamba 4class 모델 및 API
│   │   └── data_augmentation/ ← 노말 & 피싱 데이터 증강 파이프라인
│   │
│   ├── experiments/           ← 기타 모델 구조 및 증강 기법 실험
│   │   ├── model_architecture/← 기타 모델 구조 변경 실험 폴더 및 코드
│   │   └── data_augmentation/ ← 그 외 데이터 증강 방법 실험 코드
│   │
│   └── remain/                ← 분류가 애매하거나 보관이 필요한 파일/폴더
│
├── demo/                      데모 앱 골격 (작성 예정)
│   ├── frontend/              ← 프론트엔드 (React + Vite)
│   │   └── src/
│   │       ├── components/
│   │       ├── hooks/
│   │       └── styles/
│   └── backend/               ← 백엔드 API 게이트웨이
│       ├── routers/
│       ├── schemas/
│       └── services/
│
├── pipeline/                  실시간 스트리밍 파이프라인 속도 평가
├── normal_data_reference/     일반 대화 참조 데이터
├── docs/                      문서 정리
│   ├── plans/                 작업 계획서 / 설계서
│   ├── reports/               실험 결과 보고서 및 분석 리포트
│   └── archive/               아카이브 문서
│
├── presentation/              발표용 자료 (작성 예정)
│
├── docker-compose.yml         서비스 오케스트레이션
├── docker-compose.cpu.yml     CPU 전용 오버라이드
├── .env.example               환경변수 템플릿
├── .gitignore                 Git 제외 규칙
└── README.md                  프로젝트 소개
```

---

## models/ — 핵심 서비스

### models/main/ (Baseline 모델 및 최신 증강 기법)

보이스피싱 탐지 분류기(Classifier)의 핵심 baseline 환경입니다.

#### models/main/model_architecture/
- `app.py`: FastAPI 앱 구동 (/health, /predict 엔드포인트)
- `model.py`: RoBERTa-Mamba 4클래스 분류 모델 정의 (Self-contained)
- `train.py`: Baseline 모델 학습 루프 (freeze_init 전략 관리)
- `dataset.py`: 슬라이딩 윈도우 기반 데이터셋 빌더 및 로더
- `inference.py`: 추론기 서비스 엔진 (`VoicePhishingDetector`, `PhishingRiskScorer`) 및 CLI 테스트 도구
- `evaluate.py`: 검증 및 평가 루프
- `losses.py`: Hierarchical Cross Entropy Loss 및 Ordinal Regression Loss
- `config.py`: RoBERTa-Mamba 아키텍처 파라미터 및 API 서버 통합 설정
- `audio_processor.py`: Whisper STT + 텍스트 정제
- `audio_enhancer.py`: 오디오 전처리 (Bandpass → Noise Reduction → VAD → Normalize)
- `Dockerfile` & `requirements.txt`: FastAPI 배포 설정
- `weights/` (git 제외): Baseline 학습 완료 모델 가중치 (`student_best.pt` 또는 최신 pt 파일)
- `data/` (git 제외): 음성 및 학습/검증용 데이터

#### models/main/data_augmentation/
- `build_4class_dataset.py`: 4클래스 데이터셋 최종 조합 스크립트
- `build_final_dataset.py`: 최종 데이터셋 빌드 스크립트
- `batch_transcribe.py`: Whisper STT 일괄 전사 전처리 스크립트
- `normal_augmentation/`: 일반 데이터 LLM 증강 및 ASR 노이즈 주입 스크립트/프롬프트
- `phishing_augmentation/`: 피싱 데이터 LLM 증강 및 ASR 노이즈 주입 스크립트
- `error_analysis/`: Whisper 실측 에러 분석 결과 (`error_summary.json` 포함)

---

### models/experiments/ (기타 실험 모델 및 증강 기법)

- **`model_architecture/`**:
  - `bert_avgpool/`, `koelectra_mamba_freeze_init_4class/` 등 30여 개의 모델 구조 변경 실험 폴더 및 코드 보관.
  - `five_label_inference.py`, `whisper_error_analysis.py` 등 실험용 추론 및 분석 스크립트.
- **`data_augmentation/`**:
  - `augment_asr_noise.py`, `augment_llm_fewshot.py` 등 레거시/실험용 데이터 증강 스크립트.

---

### models/remain/ (기타 보관 파일)

분류가 모호하거나 레거시 코드 중 보관이 필요한 파일들을 임시 격리 보관하는 디렉토리입니다.
- `conversations.json`: 대화 텍스트 데이터 json 파일
- `architecture.py`: ModernBERT Student 오리지널 구조 정의 파일 (Baseline 교체로 보관)
- `legacy_v1/`, `stn_labeling/`, `stt_tools/`, `phishing_analysis/`: 레거시 전처리/라벨링 유틸리티 폴더

---

## demo/ — 데모 앱 (작성 예정)

데모 서비스의 프론트엔드 및 백엔드 코드를 위한 골격 디렉토리입니다.

### demo/frontend/
- `src/components/`: UI 컴포넌트
- `src/hooks/`: 커스텀 훅
- `src/styles/`: 스타일시트

### demo/backend/
- `routers/`: API 라우터
- `schemas/`: 요청/응답 스키마
- `services/`: 외부 모델 서비스 클라이언트

---

## pipeline/ — 스트리밍 파이프라인 평가

실시간 스트리밍 파이프라인의 처리 속도 및 성능 측정 실험 모음입니다.
- `run_pipeline.py`, `run_pipeline_watch.py`: 파이프라인 실행 스크립트
- `compare_streaming.py`: 스트리밍 방식 비교 분석
- `visualize_pipeline.py`, `visualize_comparison.py`: 결과 시각화
- `figure/`: 시각화 결과 이미지
- `results.json`: 실험 결과 데이터
- `PIPELINE_REPORT.md`: 실험 보고서

---

## docs/ — 문서 정리

- **`plans/`**: 작업 계획서 및 환경 설정 가이드 문서 (`experiment_plan.md`, `GETTING_STARTED.md`, `TEAM_WORKFLOW.md`, `ROLE_CLASSIFIER.md`, `api_spec.md`, `deployment_guide.md` 등)
- **`reports/`**: 실험 결과 보고서, 분석 리포트 및 스펙 정의 문서 (`2026-05-18_experiment_report.md`, `voice_phishing_model_spec.md`, `architecture.md` 등)
- **`archive/`**: 과거 아카이브용 시나리오 문서 (`SINARIO_v1.md`, `SINARIO_v2.md`, `SINARIO_v3.md` 등)

---

## presentation/ — 발표용 자료 (작성 예정)

발표 자료를 위한 골격 디렉토리입니다.

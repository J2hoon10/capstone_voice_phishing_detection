# 프로젝트 파일 구조

> 최종 업데이트: 2026-05-27

## 전체 구조 요약

```
capstone_voice_phishing_detection/
│
├── models/                    ★ 핵심 서비스 (모델 + API)
│   ├── main/                  ← Baseline 서비스 및 데이터 증강 파이프라인
│   │   ├── model_architecture/← RoBERTa-Mamba 4class 모델 및 API 서버
│   │   └── data_augmentation/ ← 노말 & 피싱 데이터 증강 파이프라인
│   │
│   ├── experiments/           ← 어블레이션 스터디 실험 모델 및 증강 실험
│   │   ├── model_architecture/← 인코더·분류기 구조 변경 실험 폴더
│   │   └── data_augmentation/ ← 실험용 데이터 증강 스크립트
│   │
│   └── remain/                ← 레거시·보관 파일
│
├── data_analysis/             데이터셋 분석 스크립트 및 결과
│   └── output/                ← 분석 결과 저장 (그래프, 리포트 등)
│
├── models_analysis/           모델 추론 결과 분석 스크립트 및 결과
│   └── output/                ← 분석 결과 저장 (JSON 등)
│
├── streaming_test/            실시간 스트리밍 파이프라인 속도 평가
│   └── figure/                ← 시각화 결과 이미지
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
├── normal_data_reference/     일반 대화 참조 데이터 (Training / Validation)
├── docs/                      문서 정리
│   ├── plans/                 작업 계획서 / 설계서
│   ├── reports/               실험 결과 보고서 및 분석 리포트
│   └── archive/               아카이브 문서
│
├── presentation/              발표용 자료 (작성 예정)
├── .env.example               환경변수 템플릿
├── .gitignore
└── README.md                  프로젝트 소개
```

---

## models/ — 핵심 서비스

### models/main/model_architecture/ — Baseline 모델 및 API

보이스피싱 탐지 분류기(Classifier)의 핵심 baseline 환경입니다.

| 파일 | 설명 |
|------|------|
| `app.py` | FastAPI 앱 구동 (`/health`, `/predict` 엔드포인트) |
| `model.py` | RoBERTa-Mamba 4클래스 분류 모델 정의 |
| `train.py` | Baseline 모델 학습 루프 (freeze_init 전략 관리) |
| `dataset.py` | 슬라이딩 윈도우 기반 데이터셋 빌더 및 로더 |
| `inference.py` | 추론 서비스 엔진 (`VoicePhishingDetector`, `PhishingRiskScorer`) |
| `streaming_inference.py` | 스트리밍 방식 추론 엔진 |
| `evaluate.py` | 검증 및 평가 루프 |
| `losses.py` | Hierarchical Cross Entropy Loss 및 Ordinal Regression Loss |
| `config.py` | RoBERTa-Mamba 아키텍처 파라미터 및 API 서버 통합 설정 |
| `audio_processor.py` | Whisper STT + 텍스트 정제 |
| `audio_enhancer.py` | 오디오 전처리 (Bandpass → Noise Reduction → VAD → Normalize) |
| `Dockerfile` & `requirements.txt` | FastAPI 배포 설정 |
| `weights/` *(git 제외)* | 학습 완료 모델 가중치 |
| `data/` *(git 제외)* | 음성 및 학습/검증용 데이터 |

---

### models/main/data_augmentation/ — 데이터 증강 파이프라인

| 파일 | 설명 |
|------|------|
| `pipeline_config.py` | 전처리 파이프라인 경로 및 Whisper 변형 설정 |
| `batch_transcribe.py` | Whisper STT 일괄 전사 전처리 (resume 지원) |
| `build_4class_dataset.py` | 4클래스 데이터셋 최종 조합 스크립트 |
| `build_final_dataset.py` | 최종 데이터셋 빌드 스크립트 |
| `build_ood_normal_dataset.py` | OOD 일반 데이터셋 구성 스크립트 |
| `preprocess_augmented_data.py` | 증강 데이터 전처리 스크립트 |
| `create_augmented_train.py` | 증강 학습 데이터 생성 |
| `generate_long_normal.py` | 긴 일반 대화 생성 |
| `generate_short_phishing.py` | 짧은 피싱 대화 생성 |
| `normal_augmentation/` | 일반 데이터 LLM 증강 및 ASR 노이즈 주입 스크립트/프롬프트 |
| `phishing_augmentation/` | 피싱 데이터 LLM 증강 및 ASR 노이즈 주입 스크립트 |
| `error_analysis/` | Whisper 실측 에러 분석 결과 (`error_summary.json` 포함) |
| `transcriptions/` *(git 제외)* | STT 처리 결과 CSV |
| `output/` *(git 제외)* | 4class 분할 데이터셋 (train/val/test CSV) |

---

### models/experiments/model_architecture/ — 어블레이션 스터디 실험

어블레이션 스터디에 포함된 인코더·분류기 구조 변경 실험 폴더입니다.

**GRU/LSTM + 비RoBERTa 인코더 조합**

| 폴더 | 설명 |
|------|------|
| `bert_gru_freeze_init_4class/` | BERT + GRU, freeze_init, 4class |
| `bert_mamba_freeze_init_4class/` | BERT + Mamba, freeze_init, 4class |
| `kobert_gru_freeze_init_4class/` | KoBERT + GRU, freeze_init, 4class |
| `kobert_mamba_freeze_init_4class/` | KoBERT + Mamba, freeze_init, 4class |
| `koelectra_gru_freeze_init_4class/` | KoELECTRA + GRU, freeze_init, 4class |
| `koelectra_mamba_freeze_init_4class/` | KoELECTRA + Mamba, freeze_init, 4class |

**RoBERTa + 분류기 조합 (기본)**

| 폴더 | 설명 |
|------|------|
| `roberta_gru_freeze_init_4class/` | RoBERTa + GRU, freeze_init, 4class |
| `roberta_lstm_freeze_init_4class/` | RoBERTa + LSTM, freeze_init, 4class |
| `roberta_mamba_freeze_init_4class/` | RoBERTa + Mamba, freeze_init, 4class (**Baseline**) |
| `roberta_mamba_l1_freeze_init_4class/` | Mamba Layer=1 변형 |
| `roberta_mamba_l1_dstate_4class/` | Mamba Layer=1 + d_state 변형 |

**Short Window 변형**

| 폴더 | 설명 |
|------|------|
| `roberta_gru_short_window_freeze_init/` | RoBERTa + GRU, Short Window |
| `roberta_lstm_short_window_freeze_init/` | RoBERTa + LSTM, Short Window |
| `roberta_mamba_short_window_freeze_init/` | RoBERTa + Mamba, Short Window |
| `roberta_mamba_w32_freeze_init_4class/` | RoBERTa + Mamba, Window=32 변형 |

**공용 스크립트**

| 파일 | 설명 |
|------|------|
| `five_label_inference.py` | 5레이블 추론 테스트 스크립트 |
| `plot_decoder_f1_heatmap.py` | 디코더 F1 히트맵 시각화 |
| `plot_experiment_f1_heatmaps.py` | 실험 F1 히트맵 시각화 |
| `run_ood_all.sh` | OOD 평가 일괄 실행 스크립트 |
| `figures/` | 시각화 결과 이미지 |

---

### models/experiments/data_augmentation/ — 실험용 데이터 증강

| 파일 | 설명 |
|------|------|
| `pipeline_config.py` | 실험용 증강 파이프라인 경로 설정 |
| `augment_asr_noise.py` | ASR 노이즈 주입 증강 스크립트 |
| `augment_llm_fewshot.py` | LLM Few-shot 증강 스크립트 |

---

### models/remain/ — 레거시·보관 파일

| 항목 | 설명 |
|------|------|
| `architecture.py` | ModernBERT Student 오리지널 구조 정의 (Baseline 교체로 보관) |
| `legacy_v1/` | 레거시 전처리 유틸리티 |
| `stn_labeling/` | STN 레이블링 유틸리티 |
| `stt_tools/` | STT 관련 레거시 도구 |

---

## data_analysis/ — 데이터셋 분석

데이터셋(CSV) 기반 분석 스크립트입니다. 모델 추론 없이 독립 실행 가능합니다.

| 파일 | 설명 |
|------|------|
| `analyze_baro.py` | 전체 데이터셋 통계 분석 (레이블/소스 분포, 길이 요약) |
| `outlier_phishing_length.py` | 피싱 데이터 이상 길이 샘플 탐지 |
| `plot_length_bias.py` | 클래스별 스크립트 길이 편향 시각화 (5종 그래프 저장) |
| `whisper_error_analysis.py` | Whisper STT 결과 vs 사람 정답 CER 분석 |
| `output/` | 분석 결과 저장 (PNG 그래프, error_report.csv, error_summary.json) |

> 입력: `models/main/data_augmentation/output/4class/{train,val,test}.csv`

---

## models_analysis/ — 모델 추론 결과 분석

모델 추론 결과를 기반으로 한 분석 스크립트입니다.

| 파일 | 설명 |
|------|------|
| `length_analysis.py` | 스크립트 길이에 따른 모델 성능 구간별 분석 |
| `misclassified_very_long.py` | 긴 스크립트 오분류 샘플 추출 및 분석 |
| `output/` | 분석 결과 저장 (length_analysis JSON 등) |

> 입력: `models/experiments/model_architecture/` 내 각 실험의 로그 파일

---

## streaming_test/ — 스트리밍 파이프라인 평가

실시간 스트리밍 파이프라인의 처리 속도 및 FPS/RTF 측정 실험 모음입니다.

| 파일 | 설명 |
|------|------|
| `run_pipeline.py` | 음성 → Whisper STT → 분류기 스트리밍 파이프라인 실행 (FPS/RTF 측정) |
| `run_pipeline_watch.py` | 파일 감시 기반 파이프라인 실행 |
| `compare_streaming.py` | 여러 실험 모델 streaming inference 결과 나란히 비교 출력 |
| `visualize_pipeline.py` | 파이프라인 속도 결과 시각화 |
| `visualize_comparison.py` | 모델 간 비교 결과 시각화 |
| `results.json` | 실험 결과 데이터 |
| `PIPELINE_REPORT.md` | 실험 보고서 |
| `figure/` | 시각화 결과 이미지 |

---

## demo/ — 데모 앱 (작성 예정)

데모 서비스의 프론트엔드 및 백엔드 코드를 위한 골격 디렉토리입니다.

- `frontend/src/`: components, hooks, styles
- `backend/`: routers, schemas, services

---

## docs/ — 문서

| 폴더 | 설명 |
|------|------|
| `plans/` | 작업 계획서, 설계서, 팀 가이드 (`GETTING_STARTED.md`, `TEAM_WORKFLOW.md`, `api_spec.md` 등) |
| `reports/` | 실험 결과 보고서 및 분석 리포트 (`2026-05-18_experiment_report.md`, `architecture.md` 등) |
| `archive/` | 과거 시나리오 문서 아카이브 |
| `PROJECT_FILE_STRUCTURE.md` | 이 문서 |
| `roberta_mamba_architecture.tex` | 모델 아키텍처 LaTeX 다이어그램 소스 |
| `sinario_poster.md` | 포스터용 시나리오 문서 |

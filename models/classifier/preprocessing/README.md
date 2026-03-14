# preprocessing/ — 학습 데이터 전처리 파이프라인

음성 데이터를 STT 처리하고 학습용 텍스트 데이터셋을 만드는 스크립트 모음.
실행 순서는 P1 → P2 순서를 따른다.

---

## 파일 설명

### 설정

| 파일 | 설명 |
|------|------|
| `pipeline_config.py` | 파이프라인 전체에서 공유하는 설정값 (경로, Whisper 모델, 증강 파라미터, 분할 비율 등) |

---

### P1 — 전사 + 오류 측정

| 파일 | 설명 |
|------|------|
| `batch_transcribe.py` | 음성 파일 전체(피싱 515개 + 일반 2,001개)를 Whisper로 일괄 STT 처리. 이력 기반 resume 지원 (`--resume`) |
| `sample_ground_truth.py` | 전사 결과에서 카테고리별 10개씩 샘플 추출 → 사람 어노테이션용 템플릿 CSV 생성 |
| `whisper_error_analysis.py` | 사람이 채운 정답과 STT 결과를 CER(문자 단위 오류율)로 비교 → 오류 분포(치환/삭제/삽입/띄어쓰기) 산출 |

---

### P2 — 데이터 증강 + 최종 데이터셋

| 파일 | 설명 |
|------|------|
| `augment_llm_fewshot.py` | 피싱 전사 결과에서 LLM API로 구어체 변형 생성. API 없으면 mock 모드 동작 |
| `augment_asr_noise.py` | `error_summary.json`의 오류 확률 기반으로 텍스트에 ASR 노이즈 주입 (한국어 음운 유사 치환 포함) |
| `build_training_dataset.py` | 원본 + LLM 증강 + ASR 노이즈를 합산하고 train/val/test 분할 → 최종 학습 데이터셋 생성 |

---

## 출력 디렉터리 구조

```
preprocessing/
├── transcriptions/
│   ├── gpu_small/              Whisper small (GPU, float16)
│   │   ├── phishing.csv
│   │   ├── normal.csv
│   │   └── all.csv             합본
│   └── cpu_base/               Whisper base (CPU, int8)
│       ├── phishing.csv
│       ├── normal.csv
│       └── all.csv
├── error_analysis/
│   ├── ground_truth_template.csv   사람 어노테이션용 템플릿 (STT 결과 자동 채움)
│   ├── ground_truth_annotated.csv  사람이 human_transcription 컬럼을 채운 파일
│   ├── error_report.csv            샘플별 CER 및 오류 유형
│   └── error_summary.json          집계 오류율 (augment_asr_noise.py 입력으로 사용)
├── augmented/
│   ├── llm_fewshot.csv         LLM이 생성한 피싱 구어체 변형
│   └── asr_noised.csv          ASR 노이즈가 주입된 텍스트
└── final/
    ├── train.csv               학습 데이터 (80%)
    ├── val.csv                 검증 데이터 (10%)
    ├── test.csv                테스트 데이터 (10%)
    └── dataset_stats.json      라벨 분포, 증강 비율 통계
```

**CSV 공통 컬럼**: `id, text, label, category, source, filename`

| 컬럼 | 값 |
|------|---|
| `label` | 1 (피싱) / 0 (일반) |
| `source` | `original` / `llm_fewshot` / `asr_noise` |
| `category` | 대출 사기형, 수사기관 사칭형, 바로 이 목소리, 등 |

---

## 실행 방법

> 모든 스크립트는 `models/classifier/` 디렉터리에서 실행

```bash
# P1 — Step 1: 전체 STT 처리 (GPU 권장)
python preprocessing/batch_transcribe.py --variant gpu_small

# P1 — Step 2: 오류 측정용 샘플 추출
python preprocessing/sample_ground_truth.py --variant gpu_small --label phishing

# P1 — Step 3: [수동] ground_truth_annotated.csv의 human_transcription 컬럼 채움

# P1 — Step 4: CER 오류 분석
python preprocessing/whisper_error_analysis.py --plot

# P2 — Step 5: LLM 구어체 증강 (API 없으면 mock 동작)
python preprocessing/augment_llm_fewshot.py --variant gpu_small

# P2 — Step 6: ASR 노이즈 주입
python preprocessing/augment_asr_noise.py

# P2 — Step 7: 최종 데이터셋 생성
python preprocessing/build_training_dataset.py
```

---

## Whisper 모델 선택

| 변형 | 모델 | 디바이스 | 정밀도 |
|------|------|----------|--------|
| `gpu_small` | `Systran/faster-whisper-small` | CUDA | float16 |
| `cpu_base` | `Systran/faster-whisper-base` | CPU | int8 |

학습 데이터와 추론 서비스에서 반드시 같은 변형을 사용해야 도메인 불일치가 없다.

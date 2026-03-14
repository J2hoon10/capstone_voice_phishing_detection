# P1/P2 데이터 파이프라인 구현 계획

## Context
`voice_intent_classification_plan.md`의 P1(전사+오류측정)과 P2(증강)를 `models/classifier/` 내부에서 구현한다. 현재 보유 데이터는 피싱 515개 + 일반 2,001개(총 2,516개 음성). 기존 `audio_processor.py`를 재사용하여 학습 데이터와 추론 데이터가 동일한 파이프라인을 거치도록 보장한다.

## 생성할 파일 (7개)
| 파일 | 역할 | 단계 |
| --- | --- | --- |
| `pipeline_config.py` | 파이프라인 전용 설정 (Whisper 버전, 경로, 증강 파라미터) | 공통 |
| `batch_transcribe.py` | 전체 음성 2,516개 일괄 STT. 이력 기반 resume 지원 | P1 |
| `sample_ground_truth.py` | 오류 측정용 60개 샘플 추출 → 사람 어노테이션 템플릿 생성 | P1 |
| `whisper_error_analysis.py` | 사람 정답 vs STT 결과 비교 → 오류 분포(치환/삭제/삽입/띄어쓰기) 산출 | P1 |
| `augment_llm_fewshot.py` | 피싱 seed 문장에서 LLM으로 구어체 변형 생성 | P2 |
| `augment_asr_noise.py` | P1 측정 오류 확률로 텍스트에 ASR 노이즈 주입 | P2 |
| `build_training_dataset.py` | 원본 + 증강 합치고 train/val/test 분할 → 최종 학습 데이터셋 | P2 |

## 기존 파일 수정 (1개)
| 파일 | 변경 내용 |
| --- | --- |
| `audio_processor.py` | `__init__`에 `device`, `compute_type` 선택 파라미터 추가 (기본값 None=자동감지, 하위호환) |

## 삭제할 파일/폴더 (1개)
| 대상 | 사유 |
| --- | --- |
| `STT_test/` | 역할이 `batch_transcribe.py` + `preprocessing/transcriptions/`로 대체됨 |

## 출력 디렉터리 구조
```
models/classifier/preprocessing/
├── transcriptions/
│   ├── cpu_base/                      # CPU 버전 (base int8)
│   │   ├── phishing.csv
│   │   ├── normal.csv
│   │   └── all.csv                    # 합본
│   └── gpu_small/                     # GPU 버전 (small float16)
│       ├── phishing.csv
│       ├── normal.csv
│       └── all.csv
├── error_analysis/
│   ├── ground_truth_template.csv      # 사람 어노테이션용 템플릿
│   ├── ground_truth_annotated.csv     # 사람이 채운 정답
│   ├── error_report.csv               # 샘플별 오류율
│   └── error_summary.json             # 집계: {substitution: 0.08, ...} + 수렴 정보
├── augmented/
│   ├── llm_fewshot.csv                # LLM 생성 피싱 표현
│   └── asr_noised.csv                 # ASR 노이즈 주입 결과
└── final/
    ├── train.csv
    ├── val.csv
    ├── test.csv
    └── dataset_stats.json             # 라벨 분포, 증강 비율
```

## CSV 공통 컬럼
`id, text, label, category, source, filename`

- `label`: 1(phishing), 0(normal)
- `source`: original / llm_fewshot / asr_noise
- `category`: 대출 사기형, 수사기관 사칭형, 등

## 실행 순서

### P1 — 즉시 착수
1. `audio_processor.py` 수정 (device/compute_type 파라미터 추가)
2. `pipeline_config.py` 작성
3. `batch_transcribe.py` 실행 (`--variant gpu_small`)
   - `preprocessing/transcriptions/gpu_small/*.csv` 생성
   - 예상 소요: GPU 약 30~60분
4. `sample_ground_truth.py` 실행
   - 카테고리별 10개씩 총 60개 샘플 추출
   - `ground_truth_template.csv` 생성 (STT 결과 자동 채움)
5. [수동] 사람이 60개 음성을 듣고 `human_transcription` 컬럼 채움
   - `ground_truth_annotated.csv` 저장
6. `whisper_error_analysis.py` 실행
   - 문자 단위 편집거리(CER)로 치환/삭제/삽입/띄어쓰기 오류율 산출
   - `error_summary.json` 생성
   - 수렴 확인: 10개 단위 슬라이딩 윈도우, 연속 3구간 ±0.01 이하

### P2 — P1 완료 후 착수
7. `augment_llm_fewshot.py` 실행
   - 피싱 전사 결과에서 카테고리별 seed 문장 선정
   - LLM API로 구어체 변형 생성 (피싱당 2~3배 증강)
   - 품질 필터링 (길이, 한국어 비율, 중복 제거)
   - `llm_fewshot.csv` 생성
8. `augment_asr_noise.py` 실행
   - `error_summary.json`의 오류 확률 기반
   - 혼합 확률 모델: 오류 유형 샘플링 → 세부 변형 적용
   - 한국어 음운 유사 치환표 (ㄱ/ㅋ, ㄷ/ㅌ, ㅂ/ㅍ, 유사 모음 등)
   - 원본 + LLM 증강 텍스트 모두에 노이즈 주입
   - `asr_noised.csv` 생성
9. `build_training_dataset.py` 실행
   - 원본 전사 + LLM 증강 + ASR 노이즈 합산
   - 화자/파일 단위 분할 (같은 원본의 증강 데이터는 같은 split)
   - train(80%) / val(10%) / test(10%)
   - 클래스 균형 확인 및 보고

## 핵심 설계 결정
1. AudioProcessor 재사용
   - `batch_transcribe.py`는 기존 `AudioProcessor.process_file()`을 그대로 호출한다.
   - 학습 데이터가 추론과 동일한 파이프라인을 거치도록 보장하기 위함.

2. `audio_processor.py` 최소 수정
   - `__init__`에 `device=None`, `compute_type=None` 파라미터만 추가.
   - `None`이면 기존처럼 자동감지. 기존 코드 전혀 깨지지 않음.

```python
# 변경 전
def __init__(self, whisper_model_size="deepdml/faster-whisper-large-v3-turbo-ct2"):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if self.device == "cuda" else "int8"

# 변경 후
def __init__(self, whisper_model_size="deepdml/faster-whisper-large-v3-turbo-ct2",
             device=None, compute_type=None):
    self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    compute_type = compute_type or ("float16" if self.device == "cuda" else "int8")
```

3. 한국어 CER 사용
   - 한국어는 영어처럼 단어 경계가 명확하지 않으므로 WER 대신 문자 단위 편집거리(CER)로 오류 측정.

4. 데이터 누수 방지
   - `build_training_dataset.py`에서 원본 파일 ID를 기준으로 분할.
   - 원본 파일에서 파생된 모든 증강 데이터(LLM, ASR)는 원본과 같은 split에 배치.

5. 증강 목표
   - 피싱 원본 515개 → LLM 증강 500~800개 → ASR 노이즈 적용
   - 목표: 피싱 총 1,000~1,500개 (normal 2,001개와 유사 수준)

## Whisper 모델 버전
| 변형 | 모델 | 디바이스 | Compute Type | 용도 |
| --- | --- | --- | --- | --- |
| gpu_small | `Systran/faster-whisper-small` | cuda | float16 | GPU 기본 선택 |
| cpu_base | `Systran/faster-whisper-base` | cpu | int8 | CPU 환경 대안 |

학습 데이터와 추론에서 반드시 같은 Whisper 모델을 사용해야 함. 데모 환경이 GPU라면 `gpu_small`로 통일, CPU라면 `cpu_base`로 통일.

## 검증 방법
- `batch_transcribe.py` 실행 후 CSV 열어서 전사 품질 육안 확인
- `whisper_error_analysis.py` 수렴 그래프 확인 (matplotlib 출력)
- `augment_llm_fewshot.py` 생성물 중 10~20% 무작위 검수 (의도 변질 여부)
- `build_training_dataset.py` 완료 후 `dataset_stats.json`의 라벨 분포 확인
- 최종 `train.csv`를 기존 학습 코드에 넣어 학습 가능 여부 확인

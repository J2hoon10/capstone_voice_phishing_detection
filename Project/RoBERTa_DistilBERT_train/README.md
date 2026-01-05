이 문서는 **Teacher-Student 모델 구조, 지식 증류(Knowledge Distillation) 전략, 데이터 증강 및 학습 파이프라인**을 포괄합니다.

---

# 🎓 보이스피싱 탐지 모델 학습 실험 보고서

## 1. 개요 (Overview)

본 실험은 보이스피싱 탐지 성능을 극대화하기 위해 **Teacher-Student 지식 증류(Knowledge Distillation)** 기법을 적용한 학습 파이프라인을 구축하고 검증하는 것을 목표로 합니다.
대규모 언어 모델인 **RoBERTa(Teacher)**의 지식을 경량화된 **Student 모델**로 전이하여, **높은 정확도**와 **빠른 추론 속도**를 동시에 달성하고자 합니다.

---

## 2. 시스템 환경 및 하이퍼파라미터 (`config.py`)

### 2.1. 기본 설정

* **Base Model**: `klue/roberta-base` (한국어 특화 Pre-trained 모델)
* **Device**: NVIDIA CUDA (AMP Mixed Precision 적용)
* **Seed**: 42 (재현성 확보)

### 2.2. 학습 파라미터

| 항목 | 설정값 | 설명 |
| --- | --- | --- |
| **Epochs** | 10 | 전체 데이터셋 반복 횟수 |
| **Batch Size** | 16 | 메모리 효율성을 고려한 배치 크기 |
| **Learning Rate** | 1e-5 | 안정적인 미세 조정(Fine-tuning)을 위한 낮은 학습률 |
| **Window / Stride** | 5 / 2 | 문맥 파악을 위한 슬라이딩 윈도우 설정 |
| **Student Layers** | **6** | Teacher(12층) 대비 **50% 압축** |

---

## 3. 데이터 처리 파이프라인 (Data Pipeline)

### 3.1. 데이터 로드 및 분할 (`utils.py`, `dataset.py`)

* **누수 방지 분할**: `GroupShuffleSplit`을 사용하여 **통화 ID(Session)** 단위로 Train/Test(8:2)를 분리합니다. 이는 동일한 통화 내의 문장이 학습셋과 테스트셋에 섞이는 **Data Leakage**를 원천 차단합니다.
* **슬라이딩 윈도우**: 긴 통화 내용을 `Window Size=5` (문장), `Stride=2`로 잘라내어 시퀀스 데이터를 생성합니다.

### 3.2. 텍스트 증강 (`dataset.py`)

데이터 부족 문제를 해결하고 일반화 성능을 높이기 위해 **KoEDA** 라이브러리를 활용합니다.

* **적용 시점**: `__getitem__`에서 학습(`train`) 모드일 때만 **50% 확률**로 적용.
* **기법**: 유의어 교체(SR), 삽입(RI), 교체(RS), 삭제(RD)를 무작위로 수행.

### 3.3. 클래스 불균형 해소 (`train_teacher.py`)

보이스피싱 데이터(Class 1)가 정상 데이터(Class 0)보다 적은 불균형 문제를 해결하기 위해 **가중치 손실(Weighted Loss)**을 적용합니다.

* **Weight Calculation**: 클래스 빈도의 역수(Inverse Frequency)로 가중치 계산.
* **Weight Clipping**: 가중치가 너무 커져 학습이 불안정해지는 것을 막기 위해 `np.clip(1.0 ~ 5.0)`으로 제한.

---

## 4. 모델 아키텍처 및 초기화 (`architecture.py`)

### 4.1. DistillableRoBERTaModel

* **Backbone**: `AutoModel` (RoBERTa)
* **Output**: Logits뿐만 아니라 지식 증류를 위해 **Hidden States(중간층 벡터)**까지 반환하도록 설계.
* **Head**: `Linear` → `GELU` → `Dropout(0.1)` → `Linear` 구조의 분류기.

### 4.2. Student 초기화 전략 (Dynamic Layer Mapping)

학습 전, Student 모델이 Teacher의 지식을 빠르게 흡수하도록 가중치를 똑똑하게 초기화합니다.

* **전략**: Teacher 레이어(12개)를 Student 레이어(6개) 비율에 맞춰 **등간격으로 복사**합니다.
* 예: Teacher의 0, 2, 4, 6... 번째 레이어 가중치를 Student의 0, 1, 2, 3... 번째로 이식.


* **Embeddings**: 입력 임베딩 층은 그대로 복사.

---

## 5. 학습 전략 (Training Strategy)

### 5.1. Teacher 학습 (`train_teacher.py`)

* **목표**: 최고의 정확도를 가진 "선생님" 모델 생성.
* **Loss**: `CrossEntropyLoss` (Class Weights 적용).
* **Optimization**: `AdamW` + `Linear Warmup Scheduler`.
* **Tech**: `GradScaler`를 이용한 **AMP(자동 혼합 정밀도)** 학습으로 메모리 절약 및 속도 향상.

### 5.2. Student 학습 (Knowledge Distillation) (`train_student.py`, `loss_fun.py`)

* **목표**: Teacher의 성능을 유지하며 크기가 작은 Student 학습.
* **Triple Loss Function (`loss_fun.py`)**:
1. **Student Loss ()**: 정답 레이블과의 오차 (가중치 )
2. **Distillation Loss ()**: Teacher의 Soft Label()을 모방 (KL-Divergence, 가중치 )
3. **Cosine Loss ()**: Teacher와 Student의 Hidden State 벡터 방향 일치 (가중치 )


* **Process**: 미리 학습된 `teacher_best.pt`를 로드하고 가중치를 고정(Freeze)한 상태에서 Student만 업데이트.

---

## 6. 프로젝트 파일 구조

```text
📂 Project Root
│
├── 📜 config.py            # [설정] 모든 하이퍼파라미터 및 경로 통합 관리
├── 📜 dataset.py           # [데이터] CSV 로드, 전처리, KoEDA 증강, 슬라이딩 윈도우
├── 📜 architecture.py      # [모델] RoBERTa 모델 정의 및 가중치 초기화 로직
├── 📜 loss_fun.py          # [손실함수] Distillation을 위한 복합 Loss 정의
├── 📜 utils.py             # [유틸] 시드 고정, 데이터 분할(Split), 로깅(Logging)
│
├── 📜 train_teacher.py     # [실행] Teacher 모델 학습 스크립트 (Weighted Loss)
├── 📜 train_student.py     # [실행] Student 모델 지식 증류 스크립트
├── 📜 trainer.py           # (참고용) 학습 루프 클래스화 모듈
└── 📜 requirements.txt     # 의존성 패키지 목록

```

## 7. 기대 효과

1. **일반화 성능 향상**: 데이터 증강과 가중치 손실 함수를 통해 데이터 불균형 및 부족 문제 완화.
2. **경량화 및 가속화**: Student 모델은 Teacher 대비 연산량이 약 50% 감소하여 실시간 탐지에 적합.
3. **안정적 학습**: Layer Mapping 초기화와 Cosine Loss를 통해 Student가 Teacher의 내부 표현력까지 효과적으로 학습.

---

*작성일: 2026-01-03*
*작성자: Gemini (AI Assistant)*
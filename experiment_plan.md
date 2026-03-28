# Streaming Dialogue Classifier with Compressive Memory
## 실험 계획서 (Experiment Plan)

---

## 1. 실험 개요

본 실험은 긴 대화 스크립트(씬)를 세그먼트 단위로 스트리밍하여 처리하면서, Compressive Memory를 활용해 이전 세그먼트의 정보를 누적하고 매 스텝마다 multi-label 분류를 수행하는 모델의 로직 및 성능을 검증한다.

| 항목 | 내용 |
|---|---|
| 실험 목적 | 스트리밍 방식의 Compressive Memory 기반 분류 모델 로직 검증 |
| 인코더 | RoBERTa-base (roberta-base, fine-tuning 전체 학습) |
| 데이터셋 | MELD (Friends TV series, 영어 대화 스크립트) |
| 태스크 | 씬(Scene) 단위 Multi-label 분류 (감정 7종 + 감성 3종 = 10차원) |
| 최종 목표 | 동일 아키텍처에서 인코더만 KoELECTRA로 교체 후 한국어 데이터 적용 |

---

## 2. 데이터셋 구성

### 2-1. MELD 원본 구조

MELD는 Friends TV 시리즈에서 추출한 1,433개의 씬(Dialogue), 13,708개의 발화(Utterance)로 구성된다. 각 발화에는 감정(Emotion)과 감성(Sentiment) 레이블이 하나씩 부착되어 있다.

| Split | 씬 수 | 발화 수 | 평균 턴 수 |
|---|---|---|---|
| Train | 1,039 | 9,989 | 약 9.6턴 |
| Validation | 114 | 1,109 | 약 9.7턴 |
| Test | 280 | 2,610 | 약 9.3턴 |

### 2-2. 레이블 구조

| 레이블 종류 | 클래스 | 차원 |
|---|---|---|
| Emotion (감정) | Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise | 7차원 |
| Sentiment (감성) | Positive, Negative, Neutral | 3차원 |
| **최종 레이블** | Emotion + Sentiment 합산 | **10차원 multi-hot 벡터** |

### 2-3. 씬 단위 Multi-hot 변환 방법

원본 MELD는 발화(utterance) 단위로 단일 레이블이 부착되어 있다. 이를 씬 단위 multi-hot 벡터로 변환하는 과정은 다음과 같다.

1. `Dialogue_ID`를 기준으로 발화를 그룹핑하여 하나의 씬으로 묶는다.
2. 발화를 `Utterance_ID` 순으로 정렬한 뒤 `"Speaker: Utterance"` 형식으로 이어붙여 하나의 텍스트 스크립트를 생성한다.
3. 씬 내 모든 발화의 Emotion과 Sentiment를 수집하여 10차원 multi-hot 벡터를 생성한다. 해당 씬에 한 번이라도 등장한 클래스는 1, 없으면 0으로 표시한다.
4. 스크립트 텍스트와 multi-hot 레이블 벡터를 쌍으로 저장한다.

**변환 예시 (씬 #6):**

```
발화 감정:  Joy, Joy, Surprise, Anger, Neutral, Sadness

multi-hot:
  Anger=1, Disgust=0, Fear=0, Joy=1, Neutral=1,
  Sadness=1, Surprise=1, Positive=1, Negative=1, Neutral_sent=1
  → [1, 0, 0, 1, 1, 1, 1, 1, 1, 1]
```

### 2-4. 세그먼테이션

씬 텍스트를 RoBERTa 토크나이저로 토크나이징한 뒤 슬라이딩 윈도우 방식으로 세그먼트를 생성한다.

| 파라미터 | 값 | 비고 |
|---|---|---|
| 세그먼트 크기 (seg_size) | 200 토큰 | CLS, SEP 포함 시 202 |
| 이동 폭 (shift) | 50 토큰 | 인접 세그먼트 150토큰 중첩 |
| 최대 길이 (max_len) | 512 토큰 | 패딩 포함 |

---

## 3. 모델 아키텍처

### 3-1. 전체 구조

모델은 RoBERTa 인코더, Linear Projection, Compressive Memory Module, Attention, Classification Head의 5단계로 구성된다. 각 세그먼트를 순차적으로 처리하면서 메모리를 업데이트하고 매 스텝마다 분류값을 출력한다.

```
세그먼트 segₜ 입력
      ↓
RoBERTa Encoder (fine-tuning)
      ↓  pooler_output (768차원)
Linear Projection (768 → 128)
      ↓  segment_repr sₜ (128차원)
┌─────────────────────────────────────┐
│       Compressive Memory            │
│                                     │
│  Fine Memory (FM)    최근 r=3개     │
│  fm₁, fm₂, fm₃      (128차원)      │
│        ↓ (가득 차면)                │
│  1D Conv 압축 함수                  │
│        ↓                            │
│  Compressed Memory (CM)  k=4개 고정 │
│  cm₁, cm₂, cm₃, cm₄  (128차원)    │
└─────────────────────────────────────┘
      ↓
Attention (query=sₜ, key/value=[CM;FM])
      ↓  context (128차원)
Classification Head (128 → 64 → 10)
      ↓
multi-hot 출력 (Sigmoid, 10차원)
```

| # | 모듈 | 입/출력 차원 | 역할 |
|---|---|---|---|
| 1 | RoBERTa Encoder | 입력: 202토큰 → 출력: 768 | 세그먼트를 문맥 표현으로 인코딩 (fine-tuning) |
| 2 | Linear Projection | 768 → 128 | 차원 축소 및 메모리 차원 통일 |
| 3 | Compressive Memory | FM: r×128 / CM: k×128 | 최근 세그먼트 상세 보존 + 오래된 세그먼트 압축 보존 |
| 4 | Attention | (r+k)×128 → 128 | 현재 세그먼트를 query로 메모리 전체에서 관련 정보 추출 |
| 5 | Classification Head | 128 → 64 → 10 | multi-hot 출력 (Sigmoid 활성화) |

### 3-2. Compressive Memory 하이퍼파라미터

| 파라미터 | 값 | 의미 |
|---|---|---|
| r (Fine Memory 크기) | 3 | 최근 3개 세그먼트 = 약 600토큰 상세 보존 |
| k (Compressed Memory 크기) | 4 | 고정 4개 슬롯에 압축된 과거 정보 보존 |
| slot_dim | 128 | 메모리 슬롯 1개의 차원 수 |
| 압축 함수 | 1D Conv | kernel_size=3, 순서 정보 보존하며 압축 |

### 3-3. 알고리즘 순서 (세그먼트 루프 t = 1, 2, ..., N)

```
Step 1. 세그먼트 인코딩
  segₜ → RoBERTa → pooler_output(768) → Projection → sₜ(128)

Step 2. Fine Memory 업데이트
  FM 맨 뒤에 sₜ 추가
  FM = [fm₁, ..., fmᵣ₋₁, sₜ]

Step 3. Fine Memory 크기 확인
  [FM 크기 ≤ r] → 압축 없이 Step 4 진행
  [FM 크기 > r] →
    fm₁(가장 오래된 것) FM에서 제거
    1D Conv([cm_last, fm₁]) → c_new(128)
    CM에서 cm₁ 제거, c_new 추가 (k개 고정 유지)

Step 4. Context 구성 및 분류
  [CM; FM] 이어붙이기  →  (k+r)개의 128차원 벡터
  Attention(query=sₜ, key/value=[CM;FM])  →  context(128)
  Classification Head(context)  →  logits(10)
  Sigmoid(logits)  →  multi-hot 확률값

Step 5. 다음 세그먼트로 이동 → Step 1 반복
```

---

## 4. 학습 설정

| 항목 | 설정값 |
|---|---|
| Loss 함수 | BCEWithLogitsLoss (multi-label 이진 분류) |
| Optimizer | AdamW |
| RoBERTa Learning Rate | 2e-5 (fine-tuning) |
| 상위 모듈 Learning Rate | 1e-3 (Memory, Attention, Head) |
| Batch Size | 16 (씬 단위) |
| Epoch | 10 (Early Stopping patience=3) |
| 학습 출력 기준 | 마지막 세그먼트(segₙ)의 출력만 Loss 계산에 사용 |
| 클래스 불균형 처리 | BCEWithLogitsLoss의 pos_weight로 희귀 클래스 가중치 부여 |

---

## 5. 평가 방식

### 5-1. 기본 평가 지표

입력 1개(씬 1개)에 대한 출력값이 정답 레이블 벡터와 얼마나 일치하는지를 기준으로 평가한다. threshold=0.5 기준으로 이진화한 뒤 아래 지표를 계산한다.

| 지표 | 수식 | 의미 |
|---|---|---|
| **Subset Accuracy (Exact Match)** | 예측 벡터 == 정답 벡터인 샘플 수 / 전체 | 10개 레이블을 모두 정확히 맞춘 씬의 비율 (가장 엄격) |
| Micro F1 | 전체 TP/FP/FN 합산 후 F1 | 클래스 불균형에 강함, 전체 성능 대표 |
| Macro F1 | 클래스별 F1의 단순 평균 | 희귀 클래스(Fear, Disgust) 성능 반영 |
| Hamming Loss | 잘못 예측한 레이블 수 / 전체 레이블 수 | 낮을수록 좋음, 부분 정답 반영 |

**주요 보고 지표:** Subset Accuracy (메인), Micro F1 (보조), Hamming Loss (보조)

### 5-2. 스트리밍 동작 정성 평가

추론 시 매 세그먼트마다 출력을 기록하여, 세그먼트가 누적될수록 분류 확신도(Sigmoid 출력값)가 정답 방향으로 수렴하는지 확인한다.

- seg₁ 출력 확신도 vs seg_N 출력 확신도 비교
- 틀린 예측이 발생하는 구간 분석 (초반부 세그먼트에서의 오류 비율)

---

## 6. Ablation 실험

| # | 실험 | 변경 조건 | 목적 |
|---|---|---|---|
| A | Baseline (하한선) | 메모리 없음, 마지막 세그먼트만 분류 | 메모리 유무의 성능 차이 확인 |
| B | Fine Memory만 사용 | CM 제거, FM만 유지 (r=3) | 압축 메모리의 기여도 측정 |
| **C** | **제안 모델 (Full)** | **FM(r=3) + CM(k=4) + 1D Conv** | **메인 실험** |
| D | 압축 함수 비교 | Mean Pooling vs 1D Conv | 1D Conv의 우위 검증 |
| E | k Ablation | k = 1, 2, 4, 8 | 최적 슬롯 수 탐색 |

---

## 7. 실험 로드맵

| 단계 | 작업 | 목표 산출물 |
|---|---|---|
| Day 1 | 데이터 전처리 | 씬 단위 multi-hot 변환 완료, 세그먼트 생성 확인 |
| Day 2 | Baseline(A) 학습 | 하한선 성능 수치 확보 |
| Day 3 | 제안 모델(C) 학습 | Subset Accuracy, Micro F1 측정 |
| Day 4 | Baseline vs 제안 모델 비교 | 메모리 효과 정량 확인 |
| Day 5 | Ablation (D, E) | 최적 압축 함수 및 k 결정 |
| Day 6 | 스트리밍 정성 분석 | 세그먼트별 확신도 수렴 그래프 |
| Day 7+ | 인코더 교체 (KoELECTRA) | 한국어 데이터로 동일 실험 재현 |

---

## 8. 언어 확장 계획 (한국어)

MELD 실험에서 검증된 아키텍처를 그대로 유지하고, 인코더(토큰 임베딩 레이어)만 교체하여 한국어 데이터에 적용한다. `pooler_output` 차원이 두 모델 모두 768로 동일하므로 Projection 레이어 이후의 모든 구성 요소는 수정 없이 재사용 가능하다.

| 구성 요소 | 영어 실험 | 한국어 실험 |
|---|---|---|
| 인코더 | `roberta-base` | `monologg/koelectra-base-v3-discriminator` |
| Projection 이후 모듈 | - | 동일 (변경 없음) |
| 데이터셋 | MELD (Friends) | AI Hub 한국어 SNS 멀티턴 대화 |
| pooler_output 차원 | 768 | 768 (동일) |

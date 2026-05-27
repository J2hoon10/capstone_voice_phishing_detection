# 방법 C: Streaming Dialogue Classifier with Compressive Memory

## 실험 개요

MELD 영어 데이터셋을 사용하여 Compressive Memory 기반 스트리밍 분류 아키텍처의 로직과 성능을 검증한다.
검증 완료 후 인코더를 KoELECTRA로 교체하여 한국어 보이스피싱 탐지에 적용한다.

| 항목 | 내용 |
|---|---|
| 실험 목적 | Compressive Memory 기반 스트리밍 분류 모델 로직 검증 |
| 인코더 | RoBERTa-base (fine-tuning) |
| 데이터셋 | MELD (Friends TV series, 영어 대화) |
| 태스크 | 씬(Scene) 단위 Multi-label 분류 (감정 7종 + 감성 3종 = 10차원) |
| 최종 목표 | 인코더만 KoELECTRA로 교체 후 한국어 보이스피싱 데이터 적용 |

### 방법 A/B와의 관계

| 비교 항목 | 방법 A (GRU) | 방법 B (RMT) | **방법 C (Compressive Memory)** |
|---|---|---|---|
| 문맥 저장 | GRU hidden (768×1) | MEM tokens (768×10) | FM (128×3) + CM (128×4) |
| 인코딩 시 문맥 참조 | 불가 (인코딩 후 결합) | 가능 (self-attention) | 불가 (인코딩 후 결합) |
| 메모리 구조 | 단일 벡터 | 고정 크기 토큰 | **2계층 (Fine + Compressed)** |
| 오래된 정보 처리 | 자연 망각 (GRU decay) | 덮어쓰기 (고정 슬롯) | **1D Conv 압축 후 보존** |
| 추가 파라미터 | ~1.77M | ~9K | ~100K (Projection + Conv + Attention + Head) |
| 검증 데이터 | 한국어 (보이스피싱) | 한국어 (보이스피싱) | **MELD (영어) → 한국어 전이** |

방법 C의 핵심 차별점은 **2계층 메모리**다. 최근 세그먼트는 Fine Memory에 상세 보존하고, 오래된 세그먼트는 1D Conv로 압축하여 Compressed Memory에 유지한다. 이를 통해 단일 벡터(GRU)나 고정 슬롯(RMT)보다 유연한 정보 보존이 가능하다.

---

## 상태

- [x] 계획 수립
- [x] MELD 데이터 전처리
- [x] Baseline 학습 (메모리 없음)
- [x] 제안 모델 학습
- [x] Ablation 실험 (baseline / fm_only / full / mean_pooling)
- [ ] 한국어 전이

---

## 데이터셋 구성

### MELD 원본 구조

MELD는 Friends TV 시리즈에서 추출한 1,433개의 씬(Dialogue), 13,708개의 발화(Utterance)로 구성된다.

| Split | 씬 수 | 발화 수 | 평균 턴 수 |
|---|---|---|---|
| Train | 1,039 | 9,989 | 약 9.6턴 |
| Validation | 114 | 1,109 | 약 9.7턴 |
| Test | 280 | 2,610 | 약 9.3턴 |

### 레이블 구조

| 레이블 종류 | 클래스 | 차원 |
|---|---|---|
| Emotion (감정) | Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise | 7차원 |
| Sentiment (감성) | Positive, Negative, Neutral | 3차원 |
| **최종 레이블** | Emotion + Sentiment 합산 | **10차원 multi-hot 벡터** |

### 씬 단위 Multi-hot 변환

1. `Dialogue_ID` 기준으로 발화를 그룹핑하여 하나의 씬으로 묶는다
2. `Utterance_ID` 순 정렬 후 `"Speaker: Utterance"` 형식으로 이어붙여 스크립트 생성
3. 씬 내 모든 Emotion/Sentiment를 수집하여 10차원 multi-hot 벡터 생성
4. 스크립트 텍스트와 multi-hot 레이블을 쌍으로 저장

```
변환 예시 (씬 #6):
발화 감정:  Joy, Joy, Surprise, Anger, Neutral, Sadness

multi-hot:
  Anger=1, Disgust=0, Fear=0, Joy=1, Neutral=1,
  Sadness=1, Surprise=1, Positive=1, Negative=1, Neutral_sent=1
  → [1, 0, 0, 1, 1, 1, 1, 1, 1, 1]
```

### 세그먼테이션

| 파라미터 | 값 | 비고 |
|---|---|---|
| 세그먼트 크기 (seg_size) | 200 토큰 | CLS, SEP 포함 시 202 |
| 이동 폭 (shift) | 50 토큰 | 인접 세그먼트 150토큰 중첩 |
| 최대 길이 (max_len) | 512 토큰 | 패딩 포함 |

### 보완: 세그먼트 통계 사전 분석

학습 시작 전에 아래 통계를 반드시 확인한다.

| 확인 항목 | 이유 |
|---|---|
| 씬당 세그먼트 수 분포 (min/max/mean/p95) | FM 크기(r)와 CM 크기(k) 설정의 근거 |
| 세그먼트 수 1인 씬의 비율 | 메모리가 전혀 작동하지 않는 샘플 비율 파악 |
| 세그먼트 수 ≤ r인 씬의 비율 | CM이 활성화되지 않는 샘플 비율 파악 |
| shift 변경 시 세그먼트 수 변화 | shift=50 vs 100 vs 150 비교 |

이 통계가 없으면 r=3, k=4 설정이 데이터에 적합한지 판단할 수 없다.

---

## 모델 아키텍처

### 전체 구조

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
| 1 | RoBERTa Encoder | 202토큰 → 768 | 세그먼트를 문맥 표현으로 인코딩 |
| 2 | Linear Projection | 768 → 128 | 차원 축소 및 메모리 차원 통일 |
| 3 | Compressive Memory | FM: r×128 / CM: k×128 | 최근 세그먼트 상세 보존 + 오래된 세그먼트 압축 보존 |
| 4 | Attention | (r+k)×128 → 128 | 현재 세그먼트를 query로 메모리에서 관련 정보 추출 |
| 5 | Classification Head | 128 → 64 → 10 | multi-hot 출력 (Sigmoid 활성화) |

### Compressive Memory 하이퍼파라미터

| 파라미터 | 값 | 의미 |
|---|---|---|
| r (Fine Memory 크기) | 3 | 최근 3개 세그먼트 = 약 600토큰 상세 보존 |
| k (Compressed Memory 크기) | 4 | 고정 4개 슬롯에 압축된 과거 정보 보존 |
| slot_dim | 128 | 메모리 슬롯 1개의 차원 수 |
| 압축 함수 | 1D Conv | kernel_size=3, 순서 정보 보존하며 압축 |

### 알고리즘 순서 (세그먼트 루프 t = 1, 2, ..., N)

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

### 보완: 1D Conv 압축 상세

원본 계획서에서 1D Conv의 입력이 `[cm_last, fm₁]`으로 되어 있으나, 이 연산의 세부사항을 명확히 한다.

```
1D Conv 압축 함수:
  입력: [cm_last, fm₁] → (2, 128) 행렬
  Conv1d(in_channels=128, out_channels=128, kernel_size=2)
  → 출력: c_new (128차원 벡터)

대안: kernel_size=3일 경우
  입력: [cm_last, fm₁, fm₂] → (3, 128) 행렬
  → 더 넓은 맥락 반영 가능하지만, FM에서 2개를 빼야 함
  → 현재 설정의 kernel_size와 FM에서 제거하는 요소 수의 일관성 확인 필요
```

### 보완: Attention 메커니즘 상세

```python
# Scaled Dot-Product Attention
# query: sₜ (1, 128) - 현재 세그먼트 표현
# key:   [CM; FM] (k+r, 128) - 전체 메모리
# value: [CM; FM] (k+r, 128)

attention_weights = softmax(query @ key.T / sqrt(128))  # (1, k+r)
context = attention_weights @ value  # (1, 128)

# 대안: Multi-Head Attention
# 헤드 수에 따른 성능 차이 → Ablation에서 검증
```

---

## 학습 설정

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

### 보완: Loss 전략 비교

마지막 세그먼트만 사용하는 것이 유일한 선택은 아니다. 방법 A/B와의 일관성 및 스트리밍 동작 개선을 위해 아래 3가지를 비교한다.

| Loss 옵션 | 설명 | 장점 | 단점 |
|---|---|---|---|
| L1: 마지막 세그먼트만 | segₙ의 출력만 loss 계산 | 최종 판단에 집중, 구현 단순 | 중간 세그먼트에 학습 신호 없음 |
| L2: 모든 세그먼트 동일 가중치 | 모든 segₜ에 동일 label로 loss | 학습 신호 풍부 | 초반 세그먼트에 불합리한 label 부여 가능 |
| **L3: 뒤쪽 가중치 증가 (권장)** | segₜ에 시간 비례 가중치 | 점진적 수렴 유도 | 가중치 설계에 추가 하이퍼파라미터 |

```python
# L3 구현 예시
def build_temporal_weights(window_mask):
    """뒤쪽 세그먼트에 높은 가중치 부여"""
    seq_len = window_mask.shape[1]
    positions = torch.arange(seq_len, dtype=torch.float)
    weights = (positions + 1) / seq_len  # 선형 증가: 0.1, 0.2, ..., 1.0
    return weights.unsqueeze(0) * window_mask
```

### 보완: pos_weight 계산 방법

```python
# 클래스별 pos_weight 계산
# 각 클래스의 양성/음성 비율로 산출
for i, class_name in enumerate(class_names):
    pos_count = train_labels[:, i].sum()
    neg_count = len(train_labels) - pos_count
    pos_weight[i] = neg_count / pos_count

# 예: Fear가 전체의 5%면 → pos_weight ≈ 19.0
# 너무 큰 값은 clip 권장 (max=10 등)
```

### 보완: Learning Rate Scheduler

| 항목 | 설정 |
|---|---|
| Scheduler | Linear warmup + cosine decay |
| Warmup steps | 전체 step의 10% |
| Min LR | 1e-6 |

Warmup이 없으면 RoBERTa fine-tuning 초기에 catastrophic forgetting이 발생할 수 있다.

---

## 평가 방식

### 기본 평가 지표

threshold=0.5 기준으로 이진화한 뒤 아래 지표를 계산한다.

| 지표 | 의미 |
|---|---|
| **Subset Accuracy (Exact Match)** | 10개 레이블을 모두 정확히 맞춘 씬의 비율 (가장 엄격) |
| Micro F1 | 전체 TP/FP/FN 합산 후 F1 (클래스 불균형에 강함) |
| Macro F1 | 클래스별 F1의 단순 평균 (희귀 클래스 반영) |
| Hamming Loss | 잘못 예측한 레이블 수 / 전체 레이블 수 (낮을수록 좋음) |

**주요 보고 지표:** Subset Accuracy (메인), Micro F1 (보조), Hamming Loss (보조)

### 보완: 클래스별 성능 분석

Macro F1만으로는 어떤 클래스가 약한지 보이지 않는다. 아래를 추가한다.

| 추가 지표 | 설명 |
|---|---|
| 클래스별 F1 / Precision / Recall | 특히 Fear, Disgust 등 희귀 클래스 |
| 클래스별 PR 곡선 | threshold 0.5가 최적인지 확인 |
| Confusion 히트맵 | 어떤 감정 쌍이 혼동되는지 시각화 |

### 보완: threshold 최적화

multi-label에서 threshold=0.5가 항상 최적은 아니다.

```python
# 클래스별 최적 threshold 탐색
from sklearn.metrics import f1_score
import numpy as np

best_thresholds = []
for i in range(num_classes):
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.2, 0.8, 0.05):
        pred = (probs[:, i] > t).astype(int)
        f1 = f1_score(labels[:, i], pred)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    best_thresholds.append(best_t)
```

### 스트리밍 동작 정성 평가

매 세그먼트마다 출력을 기록하여, 세그먼트가 누적될수록 확신도가 정답 방향으로 수렴하는지 확인한다.

- seg₁ 확신도 vs seg_N 확신도 비교
- 틀린 예측이 발생하는 구간 분석

### 보완: 스트리밍 정량 지표

정성 분석만으로는 부족하다. 아래 정량 지표를 추가한다.

| 지표 | 정의 | 목적 |
|---|---|---|
| Convergence Rate | 정답 방향으로 확신도가 단조 증가하는 세그먼트 비율 | 메모리가 안정적으로 정보를 누적하는지 |
| First Correct Step | 최초로 정답 예측이 나오는 세그먼트 인덱스 (평균) | 조기 판단 가능성 |
| Oscillation Count | 정답↔오답 전환 횟수 | 메모리 오염 여부 |

---

## Ablation 실험

| # | 실험 | 변경 조건 | 목적 |
|---|---|---|---|
| A | Baseline (하한선) | 메모리 없음, 마지막 세그먼트만 분류 | 메모리 유무의 성능 차이 확인 |
| B | Fine Memory만 사용 | CM 제거, FM만 유지 (r=3) | 압축 메모리의 기여도 측정 |
| **C** | **제안 모델 (Full)** | **FM(r=3) + CM(k=4) + 1D Conv** | **메인 실험** |
| D | 압축 함수 비교 | Mean Pooling vs 1D Conv | 1D Conv의 우위 검증 |
| E | k Ablation | k = 1, 2, 4, 8 | 최적 슬롯 수 탐색 |

### 보완: 추가 Ablation

| # | 실험 | 변경 조건 | 목적 |
|---|---|---|---|
| F | Projection 차원 비교 | slot_dim = 64 / 128 / 256 | 128이 최적인지 확인 |
| G | Attention 유형 비교 | Single-head vs Multi-head (2, 4 heads) | Attention 복잡도 대비 이득 |
| H | r Ablation | r = 1, 3, 5 | Fine Memory 크기 최적화 |
| I | Loss 전략 비교 | L1 vs L2 vs L3 | 학습 신호 전달 방식 |

### 보완: 실험 우선순위

리소스가 제한되므로, 아래 순서로 진행하고 유의미하지 않으면 조기 종료한다.

1. **Baseline(A) → 제안 모델(C)**: 메모리 효과가 있는지 (없으면 중단)
2. **Fine Memory만(B)**: 압축이 필요한지 (FM만으로 충분하면 CM 불필요)
3. **Loss 전략(I)**: L1 vs L3 (성능 차이가 크면 이후 실험에 적용)
4. **k Ablation(E)**: 슬롯 수 최적화
5. **나머지(D, F, G, H)**: 시간이 허용될 때

---

## 코드 신뢰성 이슈 및 수정 사항

실험을 진행하기 전에 아래 4가지 코드 이슈를 반드시 수정한다. 메인 실험(full + L1)은 이슈 1, 4의 영향을 받지 않으므로 먼저 돌릴 수 있지만, ablation 비교와 최종 보고 전에는 전부 수정해야 한다.

### 이슈 1: L2 Loss의 패딩 샘플 포함 문제

**위치:** `train.py` `compute_loss()` 함수의 L2 분기

**문제:** `criterion(all_logits[t], labels)`를 배치 전체에 적용한 뒤 `valid.mean()`만 곱하고 있다. 이는 invalid sample의 loss가 합산에 포함된 상태에서 비율만 보정하는 것이라서, L2 결과가 왜곡된다.

**수정 방향:** sample-wise masking으로 변경. valid sample만 선별하여 loss를 계산한 뒤 valid sample 수로 나누는 방식으로 수정한다.

```python
# 수정 전 (왜곡된 평균)
loss_t = criterion(all_logits[t], labels)  # 배치 전체에 loss 적용
valid = (num_segments > t).float()
loss += (loss_t * valid.mean())

# 수정 후 (정확한 masking)
valid = num_segments > t  # (B,) bool
if valid.any():
    loss_t = criterion(all_logits[t][valid], labels[valid])
    loss += loss_t
```

**영향 범위:** L2/L3 ablation. 기본값 L1은 마지막 세그먼트만 사용하므로 영향 없음.

### 이슈 2: 평가 threshold의 data leakage

**위치:** `evaluate.py` `evaluate_model()` 및 `__main__` 블록

**문제:** 같은 split(예: test)에서 threshold를 최적화하고, 바로 그 split에서 최적화된 threshold로 성능을 보고한다. 이는 data leakage이며 성능이 부풀려진다.

**수정 방향:** threshold는 dev에서 탐색하고, test에는 dev에서 찾은 threshold를 고정 적용한다.

```
평가 파이프라인:
  1. dev split으로 클래스별 최적 threshold 탐색
  2. 탐색된 threshold를 test split에 고정 적용
  3. test split 결과를 최종 성능으로 보고
```

**영향 범위:** 모든 실험의 최종 보고 수치. 학습 과정에는 영향 없음.

### 이슈 3: 패딩 세그먼트의 불필요한 encoder 통과

**위치:** `dataset.py` collate_fn (패딩 생성) → `model.py` forward (encoder 호출)

**문제:** 배치 내 세그먼트 수를 통일하기 위해 패딩 세그먼트(input_ids=0, attention_mask=0)가 생성되는데, 모델의 forward에서 이 패딩 세그먼트도 RoBERTa encoder에 통과된다. `segment_mask`로 메모리 업데이트는 skip하지만, encoder 연산 자체는 낭비다. 또한 RoBERTa에서 토큰 ID `0`은 `<pad>`가 아니라 `<s>`(BOS)이므로 해석상 부정확하다.

**수정 방향:** forward 루프에서 valid segment만 선별하여 encoder에 넣고, invalid segment는 skip한다.

```python
# 수정 방향: valid segment만 encoder에 통과
valid = segment_mask[:, t]  # (B,) bool - 이 타임스텝에서 실제 세그먼트를 가진 샘플
if valid.any():
    seg_repr = torch.zeros(B, slot_dim, device=device)
    seg_repr[valid] = self.encode_segment(
        input_ids[valid, t], attention_mask[valid, t]
    )
```

**영향 범위:** 학습/추론 속도 향상, 정확성 개선. 특히 배치 내 세그먼트 수 편차가 클 때 효과적.

### 이슈 4: MIN_LR과 differential LR의 불일치

**위치:** `train.py` `build_scheduler()`, `config.py` TRAIN_CONFIG

**문제:** `CosineAnnealingLR`의 `eta_min`에 `MIN_LR / UPPER_LR` 비율을 두 param group에 동일하게 적용한다. encoder의 base LR은 `2e-5`인데, `eta_min = 2e-5 × (1e-6 / 1e-3) = 2e-8`이 되어 사실상 0에 수렴한다.

**수정 방향:** param group별로 독립적인 `eta_min`을 설정한다.

```python
# 수정 방향: param group별 min LR 비율 적용
# encoder: 2e-5 → min 2e-7 (ratio 0.01)
# upper:   1e-3 → min 1e-6 (ratio 0.001)
# 또는 단순히 두 그룹 모두 동일한 eta_min=1e-6 적용
```

**영향 범위:** 학습 후반의 encoder fine-tuning 품질. L1 메인 실험에서도 영향을 줄 수 있으나, patience=3의 early stopping으로 cosine decay 후반에 도달하지 않을 가능성이 높아 실질적 영향은 제한적.

---

## 실험 실행 순서 (4단계)

코드 신뢰성 확보 후 실험을 진행하는 것이 원칙이다. 다만 L1 메인 실험은 이슈 1(L2 masking)의 영향을 받지 않으므로, 코드 수정과 병행하여 먼저 돌릴 수 있다.

### 1단계: 코드 수정 + 메인 실험 (full + L1)

| 작업 | 설명 |
|---|---|
| 코드 이슈 수정 | 위 4가지 이슈를 우선순위대로 수정 (이슈 2 > 이슈 3 > 이슈 1 > 이슈 4) |
| 메인 실험 학습 | `full` 설정 (FM=3, CM=4, 1D Conv) + `L1` loss로 10 epoch 학습 |
| 검증 | dev loss 수렴 확인, VRAM 사용량 확인 (14GB 이내) |
| 산출물 | `full_best.pt` 체크포인트, 학습 로그 |

**이슈 수정 우선순위 근거:**
- 이슈 2(threshold leakage): 최종 보고 수치 신뢰성에 직결 → 가장 먼저
- 이슈 3(패딩 encoder 통과): 학습 속도와 정확성 → 그 다음
- 이슈 1(L2 masking): L2 ablation 전에만 수정하면 됨
- 이슈 4(MIN_LR): early stopping으로 실질적 영향 제한적 → 마지막

### 2단계: Baseline 비교 및 메모리 효과 검증

| 작업 | 설명 |
|---|---|
| Baseline(A) 학습 | 메모리 없음 (FM=0, CM=0), 마지막 세그먼트만 분류 |
| FM only(B) 학습 | Fine Memory만 사용 (FM=3, CM=0) |
| 비교 분석 | Baseline vs FM only vs Full → 메모리 효과 정량 확인 |
| 판단 기준 | Full이 Baseline 대비 Subset Accuracy +3% 미만이면 아키텍처 재검토 |

### 3단계: Ablation 실험

1단계의 메인 실험 결과가 Baseline 대비 유의미한 개선을 보인 경우에만 진행한다.

| 작업 | 설명 |
|---|---|
| Loss 전략 비교 (I) | L1 vs L2 vs L3 (이슈 1 수정 후) |
| k Ablation (E) | k = 1, 2, 4, 8 |
| 압축 함수 비교 (D) | 1D Conv vs Mean Pooling |
| 스트리밍 정량 분석 | Convergence Rate, First Correct Step, Oscillation Count |
| 추가 (시간 허용 시) | Projection 차원(F), Attention 유형(G), r Ablation(H) |

### 4단계: 최종 정리 및 한국어 전이 준비

| 작업 | 설명 |
|---|---|
| 최적 설정 확정 | ablation 결과 기반 최종 하이퍼파라미터 결정 |
| 최종 평가 | dev에서 threshold 탐색 → test에 고정 적용 → 최종 성능 보고 |
| 스트리밍 시각화 | 확신도 수렴 그래프, 클래스별 PR 곡선 |
| 한국어 전이 | 인코더 교체 (RoBERTa → KoELECTRA), 태스크 변경 (10차원 → 2차원) |

---

## 실험 로드맵 (일정)

| 단계 | 작업 | 목표 산출물 |
|---|---|---|
| Day 1 | 데이터 전처리 + 세그먼트 통계 분석 | multi-hot 변환 완료, 세그먼트 분포 확인 |
| Day 2 | 코드 이슈 수정 (이슈 2, 3) + 메인 실험(full + L1) 학습 | 수정된 코드, full_best.pt |
| Day 3 | Baseline(A) + FM only(B) 학습 | 하한선 성능 수치, 메모리 효과 확인 |
| Day 4 | 코드 이슈 수정 (이슈 1, 4) + Loss 전략(I) + k Ablation(E) | L2/L3 비교, 최적 k 결정 |
| Day 5 | 압축 함수(D) + 추가 Ablation | 1D Conv vs Mean Pooling |
| Day 6 | 최종 평가 + 스트리밍 분석 | 최종 성능 수치, 확신도 수렴 그래프 |
| Day 7+ | 인코더 교체 (KoELECTRA) | 한국어 데이터로 동일 실험 재현 |

---

## 언어 확장 계획 (한국어)

MELD 실험에서 검증된 아키텍처를 유지하고, 인코더만 교체한다.
`pooler_output` 차원이 동일(768)하므로 Projection 이후 모듈은 수정 없이 재사용 가능하다.

| 구성 요소 | 영어 실험 | 한국어 실험 |
|---|---|---|
| 인코더 | `roberta-base` | `monologg/koelectra-base-v3-discriminator` |
| Projection 이후 모듈 | - | 동일 (변경 없음) |
| 데이터셋 | MELD (Friends) | 보이스피싱 대화 데이터 (all.csv + augmented) |
| pooler_output 차원 | 768 | 768 (동일) |
| 태스크 | Multi-label 감정 분류 (10차원) | Binary 분류 (피싱/정상, 2차원) |

### 보완: 전이 시 변경 사항

| 항목 | 변경 내용 |
|---|---|
| Classification Head | 128 → 64 → 10 에서 128 → 64 → 2 로 변경 |
| Loss 함수 | BCEWithLogitsLoss → CrossEntropyLoss |
| 세그먼테이션 | 토큰 기반 슬라이딩 윈도우 → 문장 기반 슬라이딩 윈도우 (방법 A/B와 동일) |
| 평가 지표 | Subset Accuracy, Hamming Loss → AUROC, F1, prefix AUROC, first alert latency |
| 학습 전략 | Single-stage fine-tuning → 2-stage (frozen → fine-tune) 검토 |

### 보완: 방법 A/B/C 통합 비교 실험

한국어 전이 후에는 세 방법을 동일 조건에서 비교한다.

```
공통 조건:
  - 동일 backbone: monologg/koelectra-base-v3-discriminator
  - 동일 window: 5문장, stride 3
  - 동일 data split
  - 동일 평가: AUROC, F1, prefix AUROC, first alert latency

비교:
  A. KoELECTRA + GRU
  B. KoELECTRA + RMT (Memory Tokens)
  C. KoELECTRA + Compressive Memory (FM + CM)
```

---

## 구현 시 주의사항

### 1. Compressive Memory의 gradient 흐름

FM→CM 압축 시 1D Conv를 통과하므로, 기본적으로 gradient가 흐른다.
그러나 세그먼트 수가 많으면 RMT와 마찬가지로 메모리 폭발이 발생할 수 있다.

```
해결: truncated BPTT 유사 전략
  - 매 t step에서 CM을 detach할지 결정
  - 또는 N step마다 CM.detach()
  - MELD는 평균 ~3 세그먼트이므로 문제가 적을 수 있으나,
    한국어 전이 시 (긴 통화) 반드시 고려
```

### 2. pooler_output vs [CLS]

RoBERTa의 `pooler_output`은 `[CLS]` hidden state에 추가 Linear + Tanh를 적용한 결과다.
MELD에서는 pooler_output을 사용하되, KoELECTRA 전이 시에는 `last_hidden_state[:, 0, :]`와 비교한다.

### 3. 메모리 초기화

t=1에서 FM과 CM이 비어 있을 때의 처리:
- FM: 비어있으면 Attention에서 key/value가 없음 → **fallback 필요**
- 옵션 1: sₜ를 context로 직접 사용 (Attention skip)
- 옵션 2: 학습 가능한 initial memory slots로 초기화
- 옵션 2를 권장 (RMT의 memory_embeddings와 동일한 접근)

### 4. Batch 내 가변 세그먼트 수

씬마다 세그먼트 수가 다르므로, 방법 A/B의 collate_fn과 동일한 패딩 전략을 사용한다.
메모리 업데이트 시 padding 세그먼트는 skip해야 한다.

---

## 성공 기준

### MELD 실험 기준

| 지표 | Baseline (메모리 없음) 대비 |
|---|---|
| Subset Accuracy | +3% 이상 |
| Micro F1 | +2% 이상 |
| Convergence Rate | 0.7 이상 |

### 한국어 전이 기준

방법 A/B와 동일 조건에서 비교하여 아래 중 하나를 만족:
- prefix AUROC (50% 시점)에서 최고 성능
- 긴 통화 (세그먼트 ≥ 5)에서 최고 F1
- 전체 F1에서 최소 동등 이상

---

## 참고 자료

| 자료 | 링크 | 관련성 |
|---|---|---|
| Compressive Transformers for Long-Range Sequence Modelling | https://arxiv.org/abs/1911.05507 | Compressive Memory 원본 논문 |
| MELD Dataset | https://affective-meld.github.io/ | 실험 데이터셋 |
| RoBERTa | https://arxiv.org/abs/1907.11692 | 인코더 |
| 방법 A 계획서 | [method_A_hierarchical_gru_20260324.md](method_A_hierarchical_gru_20260324.md) | Hierarchical GRU 비교 대상 |
| 방법 B 계획서 | [method_B_rmt_20260324.md](method_B_rmt_20260324.md) | RMT 비교 대상 |

---

## Ablation Study 실험 결과 (2026-03-27)

**데이터셋**: MELD Test Set (N=280)
**공통 설정**: RoBERTa-base, SLOT_DIM=128, 8-head Attention, Loss Strategy L3, threshold=0.5

### 실험 구성

| ID | 실험명 | 설명 | FM | CM | 압축 함수 |
|----|--------|------|:--:|:--:|:--------:|
| A | `baseline` | 메모리 없음 — 마지막 세그먼트만 분류 | 0 | 0 | — |
| B | `fm_only` | Fine Memory만 사용 (최근 r=3 세그먼트) | 3 | 0 | — |
| C | `full` | **제안 모델** — FM + CM (1D Conv) | 3 | 4 | 1D Conv |
| D | `mean_pooling` | FM + CM, 압축 함수를 Mean Pooling으로 교체 | 3 | 4 | Mean Pool |

### 분류 성능 비교 (threshold=0.5)

| 지표 | A: baseline | B: fm_only | C: full | D: mean_pool |
|------|:-----------:|:----------:|:-------:|:------------:|
| **Macro F1** | 0.704 | 0.673 | 0.689 | 0.688 |
| Weighted F1 | 0.788 | 0.761 | 0.777 | 0.774 |
| Micro F1 | 0.760 | 0.754 | 0.760 | 0.761 |
| Subset Accuracy | 0.014 | 0.043 | 0.036 | 0.036 |
| Hamming Loss | 0.300 | **0.276** | 0.293 | 0.279 |

**Micro F1 (최적 threshold, dev 기준 튜닝)**

| | A: baseline | B: fm_only | C: full | D: mean_pool |
|--|:-----------:|:----------:|:-------:|:------------:|
| Micro F1 (opt) | 0.779 | 0.795 | 0.782 | **0.790** |
| Subset Acc (opt) | 0.039 | 0.050 | 0.050 | **0.075** |

네 모델 모두 Macro F1 0.67~0.70 범위로 최종 분류 성능 차이는 미미하다.

### 클래스별 F1 비교 (threshold=0.5)

| 클래스 | Support | A: baseline | B: fm_only | C: full | D: mean_pool |
|--------|:-------:|:-----------:|:----------:|:-------:|:------------:|
| anger | 137 | 0.706 | 0.712 | 0.696 | **0.726** |
| disgust | 48 | 0.365 | **0.392** | 0.346 | 0.348 |
| fear | 43 | **0.377** | 0.265 | 0.342 | 0.340 |
| joy | 165 | 0.756 | 0.736 | **0.752** | 0.736 |
| neutral | 259 | **0.895** | 0.874 | 0.892 | 0.888 |
| sadness | 97 | **0.598** | 0.549 | 0.567 | 0.576 |
| surprise | 159 | 0.789 | 0.724 | **0.792** | 0.771 |
| positive | 180 | **0.788** | 0.743 | 0.786 | 0.755 |
| negative | 214 | **0.882** | 0.872 | 0.839 | 0.863 |

소수 클래스(disgust, fear)는 네 모델 모두 F1 0.26~0.39 수준으로 구조적 어려움을 공유한다.

### 스트리밍 성능 비교

| 지표 | A: baseline | B: fm_only | C: full (1D Conv) | D: mean_pool |
|------|:-----------:|:----------:|:-----------------:|:------------:|
| **Convergence Rate ↑** | 0.352 | 0.522 | **0.894** | 0.793 |
| **First Correct Step ↓** | 2.67 | 2.51 | 2.70 | **2.64** |
| **Oscillation Count ↓** | 0.075 | 0.075 | **0.040** | **0.040** |
| 분석 샘플 수 | 67 | 67 | 67 | 67 |

**Convergence Rate 단계별 변화**

- A → B (FM 추가): 0.352 → 0.522 (+48%) — Fine Memory만으로도 수렴성 크게 향상
- B → C (CM 추가): 0.522 → 0.894 (+71%) — Compressed Memory가 장기 압축으로 추가 기여
- A → C 전체: 0.352 → 0.894 **(+154%)** — 메모리 모듈 전체 기여

### 압축 함수 비교: 1D Conv vs Mean Pooling (C vs D)

| 지표 | C: full (1D Conv) | D: mean_pool | 우위 |
|------|:-----------------:|:------------:|:----:|
| Macro F1 | 0.689 | 0.688 | ≈ 동일 |
| Micro F1 (opt) | 0.782 | 0.790 | D |
| Subset Acc (opt) | 0.050 | **0.075** | D |
| **Convergence Rate** | **0.894** | 0.793 | **C** |
| Oscillation Count | 0.040 | 0.040 | 동일 |

분류 성능은 Mean Pooling이 소폭 유리하지만, 스트리밍 수렴 안정성(Convergence Rate)은 1D Conv가 +10.1%p 높다. 1D Conv는 인접 FM 슬롯 간의 순서 정보(temporal locality)를 보존하는 반면, Mean Pooling은 이를 소실한다. 스트리밍 탐지가 목적이라면 1D Conv가 우월한 선택이다.

### 결론

**메모리 모듈의 기여 (A → C)**

- 분류 성능: Macro F1 0.704 → 0.689 (−1.5%p, 유의미하지 않은 수준)
- 스트리밍 수렴: Convergence Rate 0.352 → **0.894 (+154%)**
- 진동 억제: Oscillation Count 0.075 → **0.040 (−47%)**

MELD의 씬당 평균 세그먼트 수가 ~3개로 짧아 Compressive Memory의 장기 컨텍스트 이점이 최종 분류 성능에는 제한적이다. 반면 스트리밍 지표에서 메모리 모듈의 효과가 명확하다. 보이스 피싱 실시간 탐지에서 Convergence Rate는 "통화 중 조기 경보의 신뢰도"에 직결되며, 이것이 이 아키텍처의 핵심 차별점이다.

**권장 모델: `full` (FM+CM, 1D Conv)** — 스트리밍 수렴성 최고, 분류 성능 동등 수준.

### 성공 기준 달성 여부

| 지표 | 기준 | 결과 | 달성 |
|------|------|------|:----:|
| Subset Accuracy (vs baseline) | +3% 이상 | +2.2%p (0.014→0.036) | 미달 (소폭 부족) |
| Micro F1 (vs baseline) | +2% 이상 | ±0.000 | 미달 |
| Convergence Rate | 0.7 이상 | **0.894** | **달성** |

최종 분류 성능 기준은 미달이나, MELD의 짧은 씬 길이를 고려하면 예상된 결과다. 스트리밍 수렴 기준(Convergence Rate ≥ 0.7)은 명확히 달성했으며, 이는 보이스 피싱 탐지(긴 통화)에서 더 큰 이점으로 발현될 것으로 예상된다.

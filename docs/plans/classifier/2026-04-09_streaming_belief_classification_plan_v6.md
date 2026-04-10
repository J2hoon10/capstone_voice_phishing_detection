# Streaming Belief-State Classification Architecture
> Recurrent Belief-State Approximation 기반 실시간 텍스트 스트리밍 분류 시스템 설계 계획서
> **v6: 실험 데이터셋 명세 추가 / 이중 손실 함수 설계 (Focal Loss ↔ Hierarchical Prototype Loss) / Mamba State 초기화 전략 명시**

---

## 변경 이력

| 버전 | 주요 변경 사항 |
|------|--------------|
| v3 | Sliding Window BERT + `[CLS]` 추출 + Mamba SSM 기본 구조 |
| v4 | `[CLS]` 추출 → **Attention-weighted Pooling** 교체 / λ_t 초반 강조 스케줄로 역전 / Causal 범위 명시 / 클래스 수 정정(20→5) |
| v5 | BERT 인코더를 **KoELECTRA-base** 로 명시 / Mamba 레이어 수 **2개로 확정** 및 근거 추가 / 2-레이어 내부 역할 분리 명시 |
| **v6** | 아키텍처 명칭 수정 (Bayesian → Recurrent Belief-State Approximation) / **실험 데이터셋 명세 추가 (20-class)** / **이중 손실 함수 설계 (Mode A: Focal / Mode B: Hierarchical Prototype)** / **Mamba State 대화 경계 초기화 전략 명시** / 대화 단위 배치 구성 전략 추가 |

---

## 1. 태스크 정의

### 1.1 입력 / 출력 명세

| 항목 | 내용 |
|------|------|
| **입력** | 원본 대화(음성/텍스트)를 실시간으로 처리하는 연속 스트림 |
| **단위** | 이전 청크와 일부 겹치는(Overlap) 슬라이딩 윈도우 단위 텍스트 청크 $c_1, c_2, \ldots, c_T$ |
| **연속성** | 청크들은 시간순으로 배열되며, 겹치는 구간을 통해 단어 끊김(Boundary) 손실을 방지 |
| **출력** | 매 청크(Window) 수신 시마다 즉각적으로 업데이트되는 레이블 확률 분포 $p_t(y)$ |
| **제약 조건** | 1. 실시간 스트리밍(미래 정보 참조 불가) <br> 2. 청크 간 강한 맥락 연결 <br> 3. 대화 경계에서 Mamba State 완전 초기화 필수 |

### 1.2 핵심 설계 철학

> **"로컬 문맥은 Transformer가 깊고 강하게 압축하고, 전체 시간의 흐름(Belief)은 Mamba가 빠르고 가볍게 추적한다."**

두 모델의 치명적 단점을 서로 상쇄시키는 최적의 결합:
- **Transformer 단점 극복:** 무거운 시퀀스 길이를 감당할 필요 없이, 가장 잘하는 '짧은 문장 독해 및 압축'만 수행.
- **Mamba 단점 극복:** 노이즈가 많은 단어 길이를 직접 스캔하지 않고, 정제된 '청크 단위' 요약본만 넘겨받아 효율적 문맥 추적 수행.

> **아키텍처 명칭 근거:** 본 시스템의 Mamba SSM은 Kalman Filter와 동일한 선형 SSM 계보에 있으며, Kalman Filter가 선형 가우시안 가정 하에서 Bayesian Filtering의 exact solution임은 잘 알려진 사실이다 (Solin, 2016). 본 시스템은 Mamba의 입력 의존적 상태 갱신 ($\Delta$, $B$, $C$가 입력에 따라 결정)을 통해 이 구조를 비선형 대화 도메인으로 확장하는 **학습 기반 근사(Learned Approximation)** 로 해석할 수 있다. 따라서 "Sequential Bayesian Updating" 대신 **"Recurrent Belief-State Approximation"** 으로 명명한다.

### 1.3 Causal 처리 범위 명세

실시간 스트리밍 제약("미래 정보 참조 불가")의 적용 범위를 명확히 정의한다.

| 범위 | 방향성 | 근거 |
|------|--------|------|
| **청크 내부 (Intra-chunk)** | 양방향 허용 | 청크 전체가 버퍼에 쌓인 뒤 KoELECTRA에 입력되므로, 청크 내 미래 토큰 참조는 실시간 제약 위반이 아님 |
| **청크 간 (Inter-chunk)** | 단방향 보장 (Causal) | Mamba는 $h_{t-1} \rightarrow h_t$ 방향으로만 상태를 전파하므로, 청크 $t$ 처리 시 미래 청크 $t+1, t+2, \ldots$ 정보는 절대 참조되지 않음 |
| **대화 간 (Inter-dialogue)** | 완전 차단 | 대화 경계에서 $h^{(1)}, h^{(2)}$를 **반드시 0으로 초기화**. 이전 대화의 state가 다음 대화로 유출되는 것을 구조적으로 차단 |

> **결론:** 실시간 인과성(Causality)은 **청크 레벨에서 Mamba가 보장**하며, 청크 내부의 양방향 처리는 버퍼 기반 배치 처리로 정당화된다. 대화 간 독립성은 **State Reset**으로 보장한다.

### 1.4 모듈 역할 분리 (핵심 설계 원칙)

| 모듈 | 담당 역할 | 기술 (적용 기법) |
|------|----------|------|
| STT | 음성을 텍스트 스트림으로 변환 | Whisper / STT API |
| **Sliding Window** | 스트림을 겹치게 잘라 단어 끊김 현상 방지 | Overlap Window Array |
| **KoELECTRA-base** | "현재 윈도우(청크)가 무슨 의미인가?" **압축** | KoELECTRA-base (12 레이어, hidden 768, 하위 10레이어 동결 또는 LoRA) + Attention-weighted Pooling |
| **Attention Pooling** | "이 청크에서 어떤 토큰이 분류 판단에 중요한가?" **선택적 집약** | Learnable Attention Weight ($W_a$, $W_b$) |
| **Mamba SSM** | **"과거부터 지금까지의 누적 상태 추적"** | Recurrent Step 모드 (대화 내 상태 유지 / 대화 경계에서 초기화) |
| Classification | "현재 시점 레이블 확률 분포 출력" | MLP + 선택적 손실 함수 (Mode A / B) |

---

## 2. 실험 데이터셋

### 2.1 Phase 1 — 일반 성능 검증용 (20-Class 멀티턴 대화셋)

모델의 **범용 분류 성능 및 Mamba State 추적 기능**을 빠르게 검증하기 위한 공개 벤치마크를 사용한다. 클래스 수가 많고 구조가 단순하지 않아 모델의 기본 표현력을 확인하기에 적합하다.

| 항목 | 내용 |
|------|------|
| **클래스 수** | 20 (대화 의도 / 주제 분류) |
| **데이터 구조** | 다수의 턴(utterance)으로 구성된 멀티턴 대화 시퀀스 |
| **레이블 방식** | 대화 단위 단일 레이블 (20개 카테고리 중 하나) |
| **사용 목적** | Mamba의 시간 흐름 추적 기능, KoELECTRA 인코딩 품질, 전체 파이프라인 통합 동작 검증 |
| **손실 함수 모드** | **Mode A (Focal Loss)** 적용 |
| **출력 차원** | $p_t(y) \in \mathbb{R}^{20}$ |

> **Phase 1의 목적은 아키텍처의 일반 동작 확인이다.** 계층 구조 레이블이 없으므로 Hierarchical Prototype Loss는 적용하지 않는다. 이 단계에서 검증된 아키텍처를 Phase 2 도메인 데이터에 이식한다.

### 2.2 Phase 2 — 도메인 특화 적용 (계층 레이블 데이터셋)

Phase 1에서 검증된 모델을 계층적 위험도 레이블이 부착된 도메인 데이터셋에 적용한다.

| 항목 | 내용 |
|------|------|
| **클래스 수** | 5 (위험도 단계: 0=정상 / 1=주의 / 2=의심 / 3=위험 / 4=확정) |
| **레이블 방식** | 대화 단위 서수(Ordinal) 계층 레이블 |
| **사용 목적** | 계층 구조를 임베딩 공간에 반영하는 Prototype Loss 검증 |
| **손실 함수 모드** | **Mode B (Hierarchical Prototype Loss)** 전환 |
| **출력 차원** | $p_t(y) \in \mathbb{R}^{5}$ |

---

## 3. 전체 아키텍처 개요

### 3.1 단일 청크 처리 파이프라인 (1회 윈도우 루프)

```text
[연속 텍스트/음성 스트림]
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Overlapping Sliding Window 분할                     │
│                                                             │
│  ... [이전 윈도우 끝단 20토큰] + [새로운 음성 80토큰] ...       │
│       (끊김없는 매끄러운 100토큰 윈도우 생성)                 │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: KoELECTRA-base 인코딩 + Attention-weighted Pooling  │
│                                                             │
│  [CLS] tokens [SEP]                                         │
│        ↓ KoELECTRA-base (12레이어, 청크 내 양방향 집중 분석) │
│  H^(t) ∈ R^(W × 768)                                       │
│        ↓ [CLS], [SEP] 제외 → 실제 의미 토큰 W'개 대상        │
│  e_i = W_a · tanh(W_b · h_i)  ← 토큰별 중요도 점수         │
│  α_i = softmax(e_i)            ← 정규화된 가중치             │
│  x_t = Σ α_i · h_i  ∈ R^(768) ← 핵심 토큰 강조 벡터        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: Mamba SSM (2-Layer) Belief(상태) 갱신               │
│                                                             │
│  ★ 대화 시작 시: h^(1), h^(2) = zeros  ← 반드시 초기화      │
│  ★ 청크 루프 중: h^(1), h^(2) 유지 (덮어쓰지 않음)           │
│                                                             │
│  ┌── Mamba Layer 1: 저수준 상태 변화 감지 ──────────────┐   │
│  │  h_t^(1) = Mamba_Step(h_{t-1}^(1), x_t)            │   │
│  │  y_t^(1) = C^(1) · h_t^(1)  ∈ R^(768)              │   │
│  └────────────────────────────────────────────────────┘   │
│                       ↓                                     │
│  ┌── Mamba Layer 2: 고수준 Belief 통합 ──────────────────┐  │
│  │  h_t^(2) = Mamba_Step(h_{t-1}^(2), y_t^(1))         │  │
│  │  y_t = C^(2) · h_t^(2)  ∈ R^(768)                   │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: 즉각 분류 (Real-time Classification)                │
│                                                             │
│  p_t = softmax(MLP(y_t))  ∈ R^C  (C = Phase에 따라 결정)   │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 수학적 수식 흐름

### 4.1 표기법 정의

| 기호 | 의미 | 차원 |
|------|------|------|
| $W$ | 윈도우 크기 (토큰 수, [CLS]/[SEP] 포함) | 예: 100 |
| $W'$ | 실제 의미 토큰 수 ([CLS], [SEP] 제외) | $W - 2 = 98$ |
| $S$ | 스트라이드 (이동 간격) | 예: 80 (20 토큰 오버랩) |
| $c_t$ | 슬라이딩 윈도우 $t$의 텍스트 | - |
| $H^{(t)}$ | KoELECTRA 출력 전체 hidden state | $\mathbb{R}^{W \times d}$ |
| $\alpha_i$ | $i$번째 토큰의 Attention-weighted Pooling 가중치 | 스칼라 |
| $x_t$ | Attention-weighted Pooling 결과 벡터 | $\mathbb{R}^{d}$ ($d=768$) |
| $h_t^{(l)}$ | $t$번째 시점 $l$번째 레이어의 Mamba Hidden State | $\mathbb{R}^{N \times d}$ |
| $y_t$ | $t$번째 시점의 Mamba Layer 2 출력 벡터 | $\mathbb{R}^{d}$ |
| $p_t$ | $t$번째 시점 분류 확률 | $\mathbb{R}^{C}$ |
| $C$ | 클래스 수 | Phase 1: 20 / Phase 2: 5 |
| $N$ | Mamba SSM State 차원 | 예: 16 |
| $W_b$ | Pooling 중요도 변환 행렬 (학습 파라미터) | $\mathbb{R}^{d \times d}$ |
| $W_a$ | Pooling 중요도 스칼라 투영 (학습 파라미터) | $\mathbb{R}^{d \times 1}$ |
| $\boldsymbol{\mu}_k$ | 클래스 $k$의 프로토타입 벡터 (Mode B 전용) | $\mathbb{R}^{d}$ |

---

### 4.2 Step 1 & 2: 윈도우 분할 및 KoELECTRA + Attention-weighted Pooling

**[Step 1] 슬라이딩 윈도우 분할**

$$c_t = \text{tokens}[(t-1) \cdot S \;:\; (t-1) \cdot S + W]$$

오버랩 구간 $W - S = 20$ 토큰이 이전 청크와 겹쳐 단어 경계 손실을 방지한다.

**[Step 2-A] KoELECTRA-base 인코딩**

사용 모델: `monologg/koelectra-base-v3-discriminator`

| 항목 | 사양 |
|------|------|
| 아키텍처 | ELECTRA Discriminator (Transformer 기반) |
| 레이어 수 | 12 |
| Hidden 차원 | 768 |
| Attention Heads | 12 |
| 최대 입력 길이 | 512 토큰 |
| 사전학습 언어 | 한국어 |

$$H^{(t)} = \text{KoELECTRA-base}(E^{(t)}) \in \mathbb{R}^{W \times d}$$

> **동결 전략:** 하위 1~10 레이어는 동결(Freeze)하고 상위 11~12 레이어만 학습하거나, 전 레이어에 LoRA (rank=8, alpha=16)를 적용한다. Attention-weighted Pooling의 $W_a$, $W_b$는 항상 학습 대상으로 유지한다.

**[Step 2-B] Attention-weighted Pooling**

`[CLS]`(위치 0)와 `[SEP]`(마지막 위치)를 제외한 실제 의미 토큰 $W' = W-2$개만을 대상으로 학습 가능한 중요도 가중치를 산출한다.

$$e_i = W_a \cdot \tanh(W_b \cdot h_i^{(t)}), \quad i = 1, \ldots, W'$$

$$\alpha_i = \frac{\exp(e_i)}{\displaystyle\sum_{j=1}^{W'} \exp(e_j)}$$

$$x_t = \sum_{i=1}^{W'} \alpha_i \cdot h_i^{(t)} \in \mathbb{R}^{d}$$

> **추가 파라미터:** $W_b$ (768×768 ≈ 589K개) + $W_a$ (768×1 = 768개) = **약 59만 개**.
> KoELECTRA 전체 파라미터(~1억 개) 대비 **0.6% 수준**으로 과적합 부담이 극히 낮다.

---

### 4.3 Step 3: Mamba 2-Layer Recurrent State Update ★스트리밍 핵심

**Layer 1 — 저수준 상태 변화 감지:**

$$h_t^{(1)},\; y_t^{(1)} = \text{Mamba\_Step}^{(1)}\!\left(h_{t-1}^{(1)},\; x_t\right)$$

**Layer 2 — 고수준 Belief 통합:**

$$h_t^{(2)},\; y_t = \text{Mamba\_Step}^{(2)}\!\left(h_{t-1}^{(2)},\; y_t^{(1)}\right)$$

각 레이어의 선택적 상태 갱신 수식 ($l \in \{1, 2\}$):

- $\Delta^{(l)} = \text{softplus}(W_\Delta^{(l)} \cdot \text{input}^{(l)})$
- $\bar{A}^{(l)} = \exp(\Delta^{(l)} \cdot A^{(l)})$
- $\bar{B}^{(l)} = (\Delta^{(l)} \cdot A^{(l)})^{-1}(\exp(\Delta^{(l)} \cdot A^{(l)}) - I) \cdot \Delta^{(l)} \cdot B^{(l)}$
- $h_t^{(l)} = \bar{A}^{(l)} \cdot h_{t-1}^{(l)} + \bar{B}^{(l)} \cdot \text{input}^{(l)}$
- $y_t^{(l)} = C^{(l)} \cdot h_t^{(l)}$

> 추론 시 메모리에 $h_{t-1}^{(1)}$과 $h_{t-1}^{(2)}$ 두 state만 유지하면 되므로, 대화 길이와 무관하게 $O(1)$ 레이턴시가 보장된다.

---

### 4.4 Step 4: 매 순간(Every Moment) 레이블 출력

$$p_t(y) = \text{softmax}(W_{c2} \cdot \text{ReLU}(W_{c1} \cdot y_t + b_1) + b_2) \in \mathbb{R}^{C}$$

---

## 5. 학습 전략

### 5.1 KoELECTRA-base 부분 동결 (Freeze / LoRA)

처음부터 KoELECTRA-base 전체 파라미터(약 1억 1천만 개)를 튜닝하면 과적합이 발생한다.

- **[옵션 A]** 하위 1~10 레이어는 Freeze(동결)하고 상위 11~12 레이어만 학습.
- **[옵션 B]** 전 레이어에 **LoRA** (rank=8, alpha=16)를 적용하여 가벼운 파인튜닝 수행.
- Attention-weighted Pooling의 $W_a$, $W_b$는 옵션과 무관하게 항상 학습 대상으로 유지한다.

### 5.2 Mamba 정규화 전략

- **Mamba Layer 수: 2개로 확정.** {1, 2} ablation으로 최종 검증 권장.
- **Dropout (0.1~0.2)** 를 각 Mamba 레이어 출력 직후에 적용.
- **Weight Decay** (AdamW, $\lambda = 0.01$).
- **Early Stopping** 검증셋 F1 기준 (patience = 5 epoch).
- `d_state` ($N$) 최적값은 $\{16, 32, 64\}$ 범위에서 ablation으로 결정.

### 5.3 이중 손실 함수 설계

손실 함수는 **데이터셋의 레이블 구조에 따라 두 가지 Mode 중 하나를 선택**하여 사용한다. 두 Mode는 동일한 `y_t` 벡터를 입력으로 받으며, MLP 헤드와 출력 차원만 Phase에 따라 교체된다.

---

#### Mode A — Focal Loss (Phase 1 기본 모드)

**적용 시점:** 계층 구조가 없는 일반 다중 분류 데이터셋 (Phase 1, 20-class)

**손실 함수:**

$$\mathcal{L}_{\text{Focal}}(p_t, y^*) = -\alpha_{y^*}(1 - p_t^{(y^*)})^\gamma \log(p_t^{(y^*)})$$

**시간 가중치 $\lambda_t$:**

보이스피싱 등 조기 탐지 목표가 있는 경우 통화 초반 청크에 더 높은 가중치를 부여한다. Phase 1에서는 $\lambda_t = 1$ (균등)로 시작하고, 이후 도메인 요구에 따라 활성화한다.

$$\lambda_t = \exp(-\alpha \cdot (t-1)), \quad \alpha \in \{0.0,\; 0.05,\; 0.1,\; 0.2\}$$

| $t$ | $\alpha=0.0$ (균등) | $\alpha=0.05$ | $\alpha=0.1$ | $\alpha=0.2$ |
|:---:|:---:|:---:|:---:|:---:|
| 1 (초반) | 1.00 | 1.00 | 1.00 | 1.00 |
| 5 | 1.00 | 0.82 | 0.67 | 0.45 |
| 10 | 1.00 | 0.64 | 0.37 | 0.14 |
| 20 (후반) | 1.00 | 0.39 | 0.14 | 0.02 |

**최종 Loss (Mode A):**

$$\mathcal{L}_{\text{A}} = \sum_{t=1}^{T} \lambda_t \cdot \mathcal{L}_{\text{Focal}}(p_t, y^*)$$

---

#### Mode B — Hierarchical Prototype Loss (Phase 2 전환 모드)

**적용 시점:** 위험도 단계와 같이 클래스 간 서수(Ordinal) 계층 구조가 존재하는 데이터셋 (Phase 2, 5-class)

**설계 목적:** 클래스 간 순서 정보를 임베딩 공간의 기하학적 거리로 강제한다. Focal Loss는 레이블 1을 레이블 2로 잘못 예측한 것과 레이블 1을 레이블 4로 잘못 예측한 것을 동일한 오답으로 취급하는 반면, Hierarchical Prototype Loss는 위험도 차이에 비례하는 페널티를 부여한다.

**프로토타입 정의:**

$$\boldsymbol{\mu}_k = \frac{1}{|\mathcal{S}_k|} \sum_{i \in \mathcal{S}_k} y_T^{(i)}, \quad k \in \{0, 1, 2, 3, 4\}$$

여기서 $y_T^{(i)}$는 대화 $i$의 마지막 청크에서 Mamba Layer 2가 출력한 벡터다. 어노테이션이 없는 환경에서는 마지막 청크의 state가 전체 대화 정보를 가장 완전하게 누적하고 있으므로 프로토타입 기준점으로 사용한다.

**프로토타입 Momentum 업데이트 (배치마다):**

$$\boldsymbol{\mu}_k \leftarrow (1 - \eta) \cdot \boldsymbol{\mu}_k + \eta \cdot \frac{1}{|\mathcal{B}_k|} \sum_{i \in \mathcal{B}_k} y_T^{(i)}, \quad \eta \in \{0.05, 0.1\}$$

단, $|\mathcal{B}_k| = 0$ (배치 내 클래스 $k$ 샘플 없음)인 경우에는 해당 클래스의 $\boldsymbol{\mu}_k$를 갱신하지 않고 이전 값을 유지한다.

**분류항 (FocalProto):**

거리 기반 softmax와 Focal 항을 통합한 단일 손실이다.

$$p_t^{(k)} = \frac{\exp(-\|y_t - \boldsymbol{\mu}_k\|^2)}{\displaystyle\sum_{j=0}^{4} \exp(-\|y_t - \boldsymbol{\mu}_j\|^2)}$$

$$\mathcal{L}_{\text{cls}}(y_t, y^*) = -(1 - p_t^{(y^*)})^\gamma \log p_t^{(y^*)}$$

**계층 정규화항:**

프로토타입 간 거리가 위험도 차이에 비례하도록 강제한다.

$$\mathcal{L}_{\text{hier}}(\{\boldsymbol{\mu}_k\}) = \sum_{0 \leq j < k \leq 4} \max\!\left(0,\; m \cdot |j - k| - \|\boldsymbol{\mu}_j - \boldsymbol{\mu}_k\|\right)^2$$

여기서 $m > 0$은 위험도 1단계 차이당 보장해야 할 최소 프로토타입 거리(margin)이다.

| 프로토타입 쌍 | 최소 요구 거리 ($m=1.0$ 기준) |
|:---:|:---:|
| $\mu_0$ — $\mu_1$ (인접 클래스) | 1.0 |
| $\mu_0$ — $\mu_2$ | 2.0 |
| $\mu_1$ — $\mu_3$ | 2.0 |
| $\mu_0$ — $\mu_4$ (최원거리) | 4.0 |

**최종 Loss (Mode B):**

$$\mathcal{L}_{\text{B}} = \sum_{t=1}^{T} \lambda_t \cdot \mathcal{L}_{\text{cls}}(y_t, y^*) + \beta \cdot \mathcal{L}_{\text{hier}}(\{\boldsymbol{\mu}_k\})$$

$\beta$는 두 항의 균형을 잡는 하이퍼파라미터로, 학습 초반에는 작게 시작해서 점진적으로 증가시키는 warm-up 방식을 적용한다 (예: $\beta: 0.01 \to 1.0$, 10 epoch에 걸쳐).

---

#### Mode 선택 요약

| 항목 | Mode A (Focal Loss) | Mode B (Hierarchical Prototype Loss) |
|------|:---:|:---:|
| 적용 Phase | Phase 1 | Phase 2 |
| 클래스 수 | 20 | 5 |
| 클래스 간 순서 정보 반영 | ❌ | ✅ |
| 추가 관리 요소 | 없음 | 프로토타입 벡터 5개 (`mu`, shape `(5, 768)`) |
| 하이퍼파라미터 추가 | 없음 | $m$ (margin), $\beta$ (균형), $\eta$ (momentum) |
| 구현 복잡도 | 낮음 | 중간 |

---

### 5.4 Mamba State 관리 전략 ★학습의 핵심 제약

Mamba의 hidden state $h^{(1)}, h^{(2)}$는 **대화(Dialogue) 단위의 문맥 정보**를 담는다. 아래 두 규칙이 반드시 지켜져야 한다.

**규칙 1 — 대화 경계에서 State 초기화:**
서로 다른 대화의 정보가 Mamba state를 통해 섞이는 것을 방지한다. 새로운 대화가 시작될 때마다 $h^{(1)}, h^{(2)}$를 0으로 리셋한다.

**규칙 2 — 대화 내 청크 루프에서 State 유지:**
같은 대화에 속하는 청크 $c_1 \to c_2 \to \cdots \to c_T$를 처리하는 동안에는 state를 덮어쓰거나 초기화하지 않는다. $h_{t-1}$에서 $h_t$로의 연속적 갱신이 곧 대화 문맥의 누적이다.

**학습 루프 구조 (Pseudo-code):**

```python
# ── 에폭 시작 전: 대화 단위로 셔플 ──────────────────────────
random.shuffle(dialogue_list)
# 대화 순서만 섞음. 각 대화 내 청크 순서(c_1→c_T)는 절대 유지.

# ── 배치 루프 ─────────────────────────────────────────────
for batch in dataloader:          # batch: B개의 대화 묶음
    # ★ 규칙 1: 새 대화 배치마다 state 초기화
    h1 = torch.zeros(B, N, d).to(device)   # Layer 1 state
    h2 = torch.zeros(B, N, d).to(device)   # Layer 2 state

    loss = 0.0
    y_T_list = []   # Mode B 프로토타입 업데이트용

    # ── 청크 루프 ─────────────────────────────────────────
    for t, chunks_t in enumerate(batch.chunk_sequence):
        # STEP 1-2: KoELECTRA + Attention Pooling
        x_t = encoder(chunks_t)              # (B, 768)

        # STEP 3: Mamba — ★ 규칙 2: state 유지하며 갱신
        h1, y1 = mamba_layer1.step(h1, x_t) # h1 in-place 갱신
        h2, y_t = mamba_layer2.step(h2, y1) # h2 in-place 갱신

        # STEP 4: 분류
        p_t = classifier(y_t)                # (B, C)

        # 손실 누적
        lam = compute_lambda(t)
        loss += lam * loss_fn(p_t, batch.labels)

        if t == batch.T - 1:                 # 마지막 청크
            y_T_list.append(y_t.detach())    # Mode B 프로토타입용

    # Mode B: 프로토타입 momentum 업데이트
    if mode == 'B':
        loss += beta * hierarchical_loss(prototypes)
        update_prototypes(y_T_list, batch.labels, eta)

    # 역전파
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

> **주의:** `h1`, `h2`는 배치 루프 최상단에서 초기화되고, 청크 루프 안에서는 절대 초기화하지 않는다. 이것이 "대화 간 독립 + 대화 내 연속성"을 동시에 보장하는 유일한 방법이다.

**Train / Validation / Test 분할 원칙:**

같은 대화의 청크가 train과 test에 동시에 들어가면 data leakage가 발생한다. **반드시 대화 단위로 분할**한다.

```python
from sklearn.model_selection import train_test_split

dialogue_ids = list(range(len(all_dialogues)))
train_ids, temp = train_test_split(dialogue_ids, test_size=0.3, random_state=42)
val_ids, test_ids = train_test_split(temp, test_size=0.5, random_state=42)
```

### 5.5 학습 프로세스 (Training vs Inference)

**학습 시:** $T$개의 청크를 한 배치로 묶어 Mamba의 Parallel Scan 연산을 활용해 VRAM 효율을 극대화한다. 단, 반드시 대화 내 청크 순서를 유지한 상태로 입력해야 한다.

**추론 시:** Step 단위로 하나씩 처리하여 실시간 레이턴시 $O(1)$을 보장한다.

---

## 6. 구현 코드

### 6.1 Attention-weighted Pooling

```python
import torch
import torch.nn as nn

class AttentionWeightedPooling(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.W_b = nn.Linear(hidden_dim, hidden_dim)
        self.W_a = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, H, attention_mask):
        # H              : (batch, seq_len, 768)
        # attention_mask : (batch, seq_len)
        H_tokens    = H[:, 1:-1, :]
        mask_tokens = attention_mask[:, 1:-1]
        e = self.W_a(torch.tanh(self.W_b(H_tokens)))
        e = e.masked_fill(mask_tokens.unsqueeze(-1) == 0, float('-inf'))
        alpha = torch.softmax(e, dim=1)
        return (alpha * H_tokens).sum(dim=1)   # (batch, 768)
```

### 6.2 Mode A — Focal Loss

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, num_classes=20):
        super().__init__()
        self.gamma = gamma
        # alpha: 클래스별 가중치 텐서 (None이면 균등)
        self.alpha = alpha

    def forward(self, logits, targets):
        # logits  : (B, C)
        # targets : (B,)
        ce = F.cross_entropy(logits, targets, reduction='none')  # (B,)
        p  = torch.exp(-ce)
        focal = (1 - p) ** self.gamma * ce
        if self.alpha is not None:
            focal = self.alpha[targets] * focal
        return focal.mean()
```

### 6.3 Mode B — Hierarchical Prototype Loss

```python
class HierarchicalPrototypeLoss(nn.Module):
    def __init__(self, num_classes=5, hidden_dim=768,
                 gamma=2.0, margin=1.0, momentum=0.05):
        super().__init__()
        self.gamma   = gamma
        self.margin  = margin
        self.eta     = momentum
        # 프로토타입: 모델 파라미터가 아닌 외부 버퍼로 관리
        self.register_buffer('mu', torch.zeros(num_classes, hidden_dim))

    # ── 분류항 ────────────────────────────────────────────
    def classification_loss(self, y_t, labels):
        # y_t    : (B, 768)
        # labels : (B,)
        # 거리 행렬: (B, C)
        dists = torch.cdist(y_t, self.mu)          # 유클리드 거리
        logits = -dists ** 2
        p = torch.softmax(logits, dim=-1)          # (B, C)

        p_true = p[torch.arange(len(labels)), labels]
        focal  = -(1 - p_true) ** self.gamma * torch.log(p_true + 1e-8)
        return focal.mean()

    # ── 계층 정규화항 ──────────────────────────────────────
    def hierarchical_loss(self):
        loss = 0.0
        C = self.mu.shape[0]
        for j in range(C):
            for k in range(j + 1, C):
                dist    = torch.norm(self.mu[j] - self.mu[k])
                required = self.margin * abs(j - k)
                loss    += torch.clamp(required - dist, min=0.0) ** 2
        return loss

    # ── 프로토타입 Momentum 업데이트 ──────────────────────
    @torch.no_grad()
    def update_prototypes(self, y_T, labels):
        # y_T    : (B, 768) — 각 대화의 마지막 청크 y_t
        # labels : (B,)
        for k in range(self.mu.shape[0]):
            mask = (labels == k)
            if mask.sum() == 0:
                continue                           # 해당 클래스 없으면 스킵
            batch_mean = y_T[mask].mean(dim=0)
            self.mu[k] = (1 - self.eta) * self.mu[k] + self.eta * batch_mean

    def forward(self, y_t, labels, beta=1.0):
        l_cls  = self.classification_loss(y_t, labels)
        l_hier = self.hierarchical_loss()
        return l_cls + beta * l_hier
```

### 6.4 손실 함수 선택 래퍼

```python
def build_loss_fn(mode: str, num_classes: int, **kwargs):
    """
    mode = 'A' : Focal Loss  (Phase 1, 20-class)
    mode = 'B' : Hierarchical Prototype Loss  (Phase 2, 5-class)
    """
    if mode == 'A':
        return FocalLoss(gamma=kwargs.get('gamma', 2.0),
                         num_classes=num_classes)
    elif mode == 'B':
        return HierarchicalPrototypeLoss(
            num_classes=num_classes,
            hidden_dim=kwargs.get('hidden_dim', 768),
            gamma=kwargs.get('gamma', 2.0),
            margin=kwargs.get('margin', 1.0),
            momentum=kwargs.get('momentum', 0.05)
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

# ── 사용 예시 ──────────────────────────────────────────────
# Phase 1
loss_fn = build_loss_fn(mode='A', num_classes=20)

# Phase 2 전환
loss_fn = build_loss_fn(mode='B', num_classes=5, margin=1.0, momentum=0.05)
```

---

## 7. 하이퍼파라미터 가이드

| 파라미터 | Phase 1 권장값 | Phase 2 권장값 | 사유 |
|----------|:---:|:---:|------|
| **Window Size (W)** | 100 ~ 128 토큰 | 동일 | 실시간 응답성과 문맥 분석력의 균형 |
| **Stride (S)** | 80 ~ 100 토큰 | 동일 | Overlap 20~30 토큰으로 단어 경계 보호 |
| **Mamba Layers** | **2** | **2** | {1, 2} ablation 권장 |
| **Mamba d_state ($N$)** | {16, 32, 64} ablation | 동일 | 검증셋 F1 기준 결정 |
| **Dropout** | 0.1 ~ 0.2 | 동일 | 각 Mamba 레이어 출력 직후 적용 |
| **Weight Decay** | 0.01 | 동일 | AdamW |
| **Early Stopping** | patience = 5 | 동일 | 검증셋 F1 기준 |
| **Loss Mode** | **A (Focal)** | **B (HProto)** | 데이터 레이블 구조에 따라 선택 |
| **$\gamma$ (Focal)** | 2.0 | 2.0 | - |
| **$\lambda_t$ ($\alpha$)** | 0.0 (균등) | {0.05, 0.1, 0.2} | Phase 1에선 균등; Phase 2에선 조기탐지 목표에 따라 ablation |
| **Margin $m$** | — | {0.5, 1.0} ablation | Mode B 전용 |
| **$\beta$ (계층항 균형)** | — | 0.01 → 1.0 warm-up | Mode B 전용; 초반 분류 학습 안정화 후 증가 |
| **Prototype Momentum $\eta$** | — | 0.05 ~ 0.1 | Mode B 전용; 느린 업데이트로 배치 노이즈 흡수 |
| **KoELECTRA Trainable** | Last 2 Layers or LoRA | 동일 | 과적합 방지 |
| **Pooling $W_a$, $W_b$** | 항상 학습 | 항상 학습 | 파라미터 수 적어 항상 업데이트 가능 |

---

## 8. 기대 효과

- **실시간 반응속도 (Latency):** $O(1)$의 초고속 추론 스피드 보장
- **문맥 보존력:** Sliding Overlap 덕에 언어적 단절 문제 해소
- **핵심 토큰 집중:** Attention-weighted Pooling이 분류 핵심 표현에 자동 집중
- **대화 독립성 보장:** 대화 경계 State Reset으로 cross-dialogue 오염 구조적 차단
- **단계적 확장성:** Phase 1(일반 검증) → Phase 2(계층 레이블 특화)로 점진적 전환 가능
- **계층 구조 반영 (Phase 2):** Hierarchical Prototype Loss가 클래스 간 서수 관계를 임베딩 공간에 구조화하여 인접 클래스 혼동 페널티를 자연스럽게 차별화

# Streaming Belief-State Classification Architecture
> Sequential Bayesian Updating 기반 실시간 텍스트 스트리밍 분류 시스템 설계 계획서
> **v2: BERT + Mamba SSM 하이브리드 아키텍처**

---

## 1. 태스크 정의

### 1.1 입력 / 출력 명세

| 항목 | 내용 |
|------|------|
| **입력** | 실시간으로 순차 수신되는 음성 스트림 → STT → 텍스트 청크 |
| **단위** | 고정 크기 또는 의미 단위로 분할된 청크 c₁, c₂, ..., cT |
| **연속성** | 청크들은 독립적이지 않고 하나의 이어지는 내용을 구성 |
| **출력** | 각 청크 수신 시마다 업데이트되는 분류 확률 분포 p_t(y) |
| **최종 판정** | 마지막 청크 cT 수신 후 argmax(p_T(y)) |

### 1.2 핵심 설계 철학

> **"판단은 한 번에 내리는 것이 아니라, 증거가 쌓일수록 점진적으로 확정된다"**

멀티홉 사실 검증(Multi-hop Fact Verification)과 동일한 구조:

```
멀티홉 검증:
  Query → Doc₁ 검색 → 증거 불충분 → Doc₂ 검색 → 증거 보강 → ... → 판정

스트리밍 분류:
  Prior → Chunk₁ 수신 → 확률 업데이트 → Chunk₂ 수신 → 확률 업데이트 → ... → 최종 판정
```

두 태스크 모두 **불완전한 증거를 순차적으로 누적하며 판단을 확정**하는 구조다.

### 1.3 모듈 역할 분리 (핵심 설계 원칙)

| 모듈 | 담당 질문 | 기술 |
|------|----------|------|
| STT | "음성을 텍스트로" | Whisper 등 |
| BERT Encoder | "이 청크가 무슨 의미인가?" | KoELECTRA / KoBERT |
| **Mamba SSM** | **"지금까지 들은 내용을 종합하면 어떤 판단인가?"** | Selective SSM |
| Classification Head | "현재 시점의 분류 확률은?" | Linear + Softmax |

---

## 2. 전체 아키텍처 개요

### 2.1 단일 청크 처리 파이프라인 (1회 루프)

```
[음성 입력]
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: 음성 → 텍스트 (STT)                                │
│                                                             │
│  🎤 "안녕하세요, 저는 금융감독원 직원인데요..."               │
│       ↓ STT (Whisper 등)                                    │
│  📝 "안녕하세요 저는 금융감독원 직원인데요"  ← 청크 c_t      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: BERT 인코딩                                         │
│                                                             │
│  ["안녕하세요", "저는", "금융감독원", "직원", "인데요"]       │
│       ↓ BERT (KoELECTRA / KoBERT)                          │
│  H^(t) ∈ R^(L × d)   ← 각 토큰마다 d차원 벡터              │
│                                                             │
│  예: "금융감독원" → [0.23, -0.11, 0.87, ...] (768개 숫자)   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: Mamba SSM으로 Belief 업데이트                       │
│                                                             │
│  이전 상태: h_{t-1}  (직전 청크까지의 누적 기억)             │
│  새 입력:   H^(t)    (현재 청크의 BERT 출력)                 │
│                                                             │
│  h_t = Mamba(h_{t-1}, H^(t))                               │
│                                                             │
│  ← Mamba가 "이전 기억 중 뭘 유지하고, 새 정보 중 뭘 흡수    │
│     할지" 선택적으로 결정                                    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: 분류 및 불확실도 계산                               │
│                                                             │
│  p_t = softmax(W_c · mean(h_t))                            │
│  H(p_t) = -Σ p_t · log(p_t)   ← Shannon Entropy           │
│                                                             │
│  예: p("보이스피싱") = 0.31   ← 아직 불확실                  │
│      p("정상")      = 0.69                                  │
└─────────────────────────────────────────────────────────────┘
    │
    ├──► LOG 저장 { t, h_t, p_t, entropy, conf }
    │
    └──► h_t → 다음 루프의 h_{t-1}
```

### 2.2 전체 루프 구조

```
 [음성 스트림]
      │ STT
      ▼ (청크 단위 분할)
 ┌────────────────────────────────────────────────────────┐
 │                   루프 (t = 1, 2, ..., T)              │
 │                                                        │
 │   h_{t-1} ──────────────────────────────┐             │
 │   (이전 Mamba Hidden State)              │             │
 │                                          ▼             │
 │   c_t ──► STT ──► Tokenizer ──► BERT ──► Mamba SSM    │
 │   (현재 청크)        E^(t)       H^(t)      h_t        │
 │                                          │             │
 │                         ┌───────────────┤             │
 │                         │               │             │
 │                         ▼               ▼             │
 │                  Classification    LOG 저장            │
 │                    Head            { t, h_t, p_t,     │
 │                 p_t = softmax       entropy, conf }    │
 │                 (W · mean(h_t))                        │
 │                         │                             │
 │                         └──► h_t ──► 다음 루프 h_{t-1}│
 └────────────────────────────────────────────────────────┘
      │
      ▼ (t = T 도달 또는 조기 종료 조건 충족)
 [최종 분류 판정: argmax(p_T(y))]
```

---

## 3. 수학적 수식 흐름

### 3.1 표기법 정의

| 기호 | 의미 | 차원 |
|------|------|------|
| $c_t$ | t번째 청크 텍스트 | - |
| $E^{(t)}$ | 청크 $c_t$의 토큰 임베딩 | $\mathbb{R}^{L \times d}$ |
| $H^{(t)}$ | BERT Encoder 출력 hidden state | $\mathbb{R}^{L \times d}$ |
| $h_t$ | t번째 Mamba Hidden State (Belief) | $\mathbb{R}^{N \times d}$ |
| $p_t(y)$ | t번째 시점의 분류 확률 분포 | $\mathbb{R}^{C}$ (C: 클래스 수) |
| $d$ | BERT hidden size (d_model) | 768 (BERT-base 기준) |
| $N$ | Mamba SSM state 차원 (하이퍼파라미터) | 예: 16~64 |
| $L$ | 청크 길이 (토큰 수) | 예: 64~256 |
| $A, B, C$ | Mamba SSM 파라미터 행렬 | 학습 가능 |
| $\Delta$ | 입력 의존적 시간 스텝 (discretization) | $\mathbb{R}^{L \times d}$ |

---

### 3.2 Step 0: 초기화

$$h_0 = \mathbf{0}^{N \times d} \quad \text{(zero prior)} \tag{1}$$

또는 학습 가능한 초기 상태:

$$h_0 = \text{LearnableInit} \in \mathbb{R}^{N \times d} \tag{2}$$

---

### 3.3 Step 1: BERT 인코딩

t번째 청크를 BERT에 입력하여 토큰별 hidden state 획득:

$$H^{(t)} = \text{BERT}(E^{(t)}) \in \mathbb{R}^{L \times d} \tag{3}$$

- $E^{(t)}$: 현재 청크 $c_t$의 토큰 임베딩 시퀀스
- BERT가 청크 내부의 **양방향 문맥** 파악 담당

> **직관:** BERT는 "이 청크만" 집중해서 토큰 간 관계를 파악한다. 청크 간 기억은 Mamba가 담당.

---

### 3.4 Step 2: Mamba SSM Belief 업데이트 ★핵심

Mamba의 Selective State Space Model 수식:

**Discretization (연속 → 이산 변환):**

$$\bar{A} = \exp(\Delta \cdot A) \tag{4}$$

$$\bar{B} = (\Delta \cdot A)^{-1}(\exp(\Delta \cdot A) - I) \cdot \Delta \cdot B \tag{5}$$

**선택적 상태 업데이트 (토큰 $l$마다 순차 적용):**

$$h_t^{(l)} = \bar{A} \cdot h_t^{(l-1)} + \bar{B} \cdot H^{(t)}_l \tag{6}$$

$$y^{(l)} = C \cdot h_t^{(l)} \tag{7}$$

**청크 전체 처리 후 최종 상태:**

$$h_t = h_t^{(L)} \quad \text{(L번째 토큰 처리 후 상태)} \tag{8}$$

> **핵심 특성 — Selective Mechanism:**
> $\Delta, B, C$가 입력 $H^{(t)}$에 의존하여 **동적으로 결정**됨:
> $$\Delta = \text{softplus}(W_\Delta \cdot H^{(t)}), \quad B = W_B \cdot H^{(t)}, \quad C = W_C \cdot H^{(t)}$$
> → $\Delta$가 클수록 새 입력을 강하게 흡수, 작을수록 이전 상태 $h_{t-1}$ 유지

**청크 간 상태 전달:**

$$h_t^{(0)} = h_{t-1} \tag{9}$$

이를 통해 이전 청크의 기억이 현재 청크 처리의 초기 상태로 연결됨.

---

### 3.5 Step 3: 분류 확률 계산

Mamba 출력의 mean pooling 후 분류:

$$\bar{h}_t = \frac{1}{L} \sum_{l=1}^{L} y^{(l)} \in \mathbb{R}^d \tag{10}$$

$$\text{logit}_t = W_c \cdot \bar{h}_t + \text{bias} \in \mathbb{R}^C \tag{11}$$

$$p_t(y) = \text{softmax}(\text{logit}_t) \in \mathbb{R}^C \tag{12}$$

---

### 3.6 Step 4: 불확실도 계산 및 로그 저장

**Shannon Entropy** 로 현재 시점의 불확실도 측정:

$$\mathcal{H}(p_t) = -\sum_{y=1}^{C} p_t(y) \log p_t(y) \tag{13}$$

- $\mathcal{H}(p_t) \to \log C$: 최대 불확실 (균등 분포)
- $\mathcal{H}(p_t) \to 0$: 최대 확신 (한 클래스에 집중)

**로그 저장 항목:**

```
log[t] = {
    step        : t,
    chunk_text  : c_t,
    hidden      : h_t          ∈ R^(N×d),
    mamba_out   : ȳ_t          ∈ R^(L×d),
    class_prob  : p_t(y)       ∈ R^C,
    pred_class  : argmax(p_t),
    entropy     : H(p_t),
    confidence  : max(p_t)
}
```

---

### 3.7 Step 5: 조기 종료 조건

$$\text{if } \mathcal{H}(p_t) < \tau \Rightarrow \text{최종 판정 출력} \tag{14}$$

- $\tau$: 엔트로피 임계값 (하이퍼파라미터)
- 충분한 증거가 쌓이면 마지막 청크를 기다리지 않고 조기 확정

---

### 3.8 전체 수식 흐름 요약

```
h₀ = 0  (또는 Learnable Init)
         │
         ▼
For t = 1 to T:
│
│  [BERT 인코딩]
│  H^(t) = BERT(E^(t))                             ... (3)
│
│  [Mamba SSM Belief 업데이트]
│  h_t^(0) = h_{t-1}                               ... (9)
│  for l = 1 to L:
│    h_t^(l) = Ā · h_t^(l-1) + B̄ · H^(t)_l        ... (6)
│    y^(l)   = C · h_t^(l)                         ... (7)
│  h_t = h_t^(L)                                   ... (8)
│
│  [분류 확률]
│  ȳ_t = mean(y^(1..L))
│  p_t = softmax(W_c · ȳ_t)                        ... (10)~(12)
│
│  [불확실도 & 로그]
│  H(p_t) = -Σ p_t · log(p_t)                      ... (13)
│  LOG ← { t, h_t, p_t, H(p_t), conf }
│
│  [조기 종료 체크]
│  if H(p_t) < τ → 최종 판정 반환                  ... (14)
│
└──► h_t → 다음 루프 h_{t-1}

최종 판정: argmax(p_T(y))
```

---

## 4. 여러 청크에 걸친 상태 변화 예시

보이스피싱 탐지 시나리오에서 Mamba hidden state가 어떻게 누적되는지:

```
t=1: 청크 c_1 = "안녕하세요 저는 금융감독원 직원인데요"
     BERT → H^(1): "금융감독원", "직원" 등 토큰 의미 인코딩
     Mamba: h_0(=0) + H^(1) → h_1  (기관 사칭 패턴 약하게 기록)
     p("보이스피싱") = 0.31  H = 0.93  → 불확실, 계속 청취

t=2: 청크 c_2 = "고객님 계좌에 문제가 발생했습니다"
     BERT → H^(2): "계좌", "문제" 등 위험 어휘 인코딩
     Mamba: h_1 + H^(2) → h_2  (t=1 기억 + t=2 위험신호 누적)
     p("보이스피싱") = 0.67  H = 0.61  → 의심 상승, 계속 청취

t=3: 청크 c_3 = "지금 바로 안전계좌로 이체해주세요"
     BERT → H^(3): "이체", "안전계좌" 강한 피싱 키워드 인코딩
     Mamba: h_2 + H^(3) → h_3  (3개 청크 누적 → 강한 확신)
     p("보이스피싱") = 0.94  H = 0.24 < τ → 조기 종료
     ★ 최종 판정: 보이스피싱
```

**Mamba hidden state의 역할:**
- `h_1`: "금융기관 사칭" 패턴 희미하게 기록
- `h_2`: "계좌 문제" 맥락 추가 누적 → 이전 기억(h_1)과 결합
- `h_3`: "이체 요구" 패턴이 h_2의 누적 기억과 결합 → 피싱 확신

> Mamba의 $\Delta$ (시간 스텝)가 각 청크에서 **얼마나 많은 정보를 흡수할지** 자동 결정.
> 위험 키워드가 나올수록 $\Delta$가 커져 상태 업데이트가 강해지는 효과.

---

## 5. 학습 설계

### 5.1 레이블 구성 방식

스트림 전체에 **하나의 최종 레이블 y*** 부여:
- 각 청크 단위가 아니라 **누적 문맥 기준**으로 레이블링
- Annotator는 전체 스트림을 보고 레이블 결정

### 5.2 손실 함수

모든 시점의 예측을 학습에 활용 (가중 누적 손실):

$$\mathcal{L} = \sum_{t=1}^{T} \lambda_t \cdot \text{CrossEntropy}(p_t(y),\ y^*) \tag{15}$$

**가중치 $\lambda_t$ 설계:**

$$\lambda_t = \frac{t}{T} \quad \text{또는} \quad \lambda_t = e^{\alpha(t/T - 1)} \tag{16}$$

| 청크 위치 | $\lambda_t$ 크기 | 의미 |
|----------|-----------------|------|
| 초기 청크 | 작음 | 불완전한 예측에 관대 |
| 후반 청크 | 큼 | 확정적 판단 강하게 요구 |

> 모델이 자연스럽게 **"초반엔 불확실해도 되고, 갈수록 확신을 높여야 한다"** 는 패턴을 학습

### 5.3 BPTT (Backpropagation Through Time)

Mamba SSM은 청크 간 hidden state $h_t$가 연결되어 있으므로, 학습 시 **청크 시퀀스 전체를 통한 역전파**가 필요:

$$\frac{\partial \mathcal{L}}{\partial \theta} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}_t}{\partial h_t} \cdot \frac{\partial h_t}{\partial \theta}$$

- 긴 시퀀스에서 gradient vanishing 우려 → Mamba의 구조적 설계로 완화됨
- 필요 시 **truncated BPTT** (일정 청크 수만큼만 역전파) 적용 가능

---

## 6. 멀티홉 검증과의 구조적 대응

| 멀티홉 사실 검증 | 스트리밍 분류 |
|----------------|-------------|
| 초기 쿼리 | $h_0$ (zero/learnable prior) |
| 문서 검색 & 읽기 | 청크 수신 & BERT 인코딩 |
| 검색 쿼리 업데이트 | Mamba SSM$(h_{t-1}, H^{(t)}) \to h_t$ |
| 중간 증거 불충분 판정 | $\mathcal{H}(p_t)$ 높음 (불확실) |
| 증거 충분 → 최종 판정 | $\mathcal{H}(p_t) < \tau$ → 조기 종료 |
| 추가 문서 필요 결정 | 다음 청크 대기 |

---

## 7. BeliefUpdate 방안 비교

| 방안 | 수식 | 특징 | 추천도 |
|------|------|------|--------|
| **A. Mamba SSM ★** | $h_t = \text{Mamba}(h_{t-1}, H^{(t)})$ | 선택적 상태 업데이트, O(L) 선형 복잡도, 스트리밍 최적 | ★★★★★ |
| B. Cross-Attention | $b_t = \text{LN}(b_{t-1} + \text{Attn}(b_{t-1}, H^{(t)}))$ | 중요 부분 명시적 선택, O(m×L) 복잡도 | ★★★★ |
| C. Gated (GRU 스타일) | $h_t = (1-z) \odot h_{t-1} + z \odot \tilde{h}$ | 구현 단순, 표현력 제한 | ★★★ |
| D. Logit-space Additive | $\text{logit}_t = \text{logit}_{t-1} + f(H^{(t)})$ | 베이즈 업데이트 직접 근사, 해석 용이 | ★★ |

**Mamba SSM이 Cross-Attention보다 유리한 이유:**
- Cross-Attention은 belief가 청크를 **명시적으로 조회**하는 구조 → O(m×L)
- Mamba SSM은 입력 흐름을 순방향으로 처리하며 **자동으로 선택적 업데이트** → O(L)
- 스트리밍 추론 시 **O(1) 메모리** (hidden state 크기 고정)
- 청크 수 T가 많아질수록 Mamba의 효율성 우위 증가

---

## 8. 구현 시 고려사항

### 8.1 BERT 인코더 설정

| 항목 | 권장 |
|------|------|
| 모델 | KoELECTRA-base 또는 KoBERT |
| Fine-tuning | 초기엔 freeze, 이후 LoRA 또는 full fine-tune 실험 |
| 청크 입력 | 각 청크를 독립적으로 인코딩 (청크 간 BERT attention 없음) |
| [CLS] 토큰 | 사용 안 함 — 전체 토큰 hidden state H^(t) 사용 |

### 8.2 Mamba SSM 설정

| 항목 | 권장 |
|------|------|
| 모델 | `mamba-ssm` 패키지의 Mamba 블록 1~2개 |
| input dim | BERT hidden size (768) |
| state dim N | 16 ~ 64 |
| 초기 hidden state | zero 또는 learnable |
| 청크 간 state 전달 | `h_t` detach 여부 실험 필요 (truncated BPTT) |

### 8.3 하이퍼파라미터

| 파라미터 | 권장 범위 | 영향 |
|----------|----------|------|
| $N$ (SSM state 차원) | 16 ~ 64 | 클수록 기억 용량↑, 파라미터↑ |
| $L$ (청크 길이) | 64 ~ 256 토큰 | 클수록 문맥↑, 레이턴시↑ |
| $\tau$ (조기종료 임계값) | 0.1 ~ 0.5 | 작을수록 엄격한 조기종료 |
| $\alpha$ (λ 가중치 기울기) | 1.0 ~ 3.0 | 클수록 후반 청크 중시 |
| Mamba layers | 1 ~ 2 | 깊을수록 표현력↑, 속도↓ |

### 8.4 학습 파라미터 목록

BERT freeze 시 학습되는 파라미터:
- Mamba SSM 블록 전체 ($A, B, C, \Delta$ 관련 가중치)
- $W_c$ (Classification Head)
- $h_0$ (Learnable Init, 선택)

BERT unfreeze 시 추가:
- BERT 전체 또는 LoRA adapter

---

## 9. 예상 성능 특성

### 청크 누적에 따른 확률 업데이트 예시

```
예시 스트림: "오늘 날씨가 너무 좋아서 | 친구한테 연락했는데 | 걔가 바쁘다고 하더라고"
정답 레이블: [실망/아쉬움]

t=1: p("중립")=0.45, p("긍정")=0.40, p("실망")=0.15   H=1.52 (불확실)
t=2: p("중립")=0.35, p("긍정")=0.25, p("실망")=0.40   H=1.33 (감소)
t=3: p("중립")=0.10, p("긍정")=0.08, p("실망")=0.82   H=0.51 (확신↑) → 판정
```

### 기대 효과

| 항목 | 기대 효과 |
|------|---------|
| 정확도 | 초기 청크만으로 판단 불가능한 케이스에서 향상 |
| 레이턴시 | 조기 종료로 절감 가능 |
| 메모리 | O(1) 추론 메모리 (Mamba hidden state 고정 크기) |
| 해석 가능성 | 로그를 통한 판단 근거 추적 가능 |
| 확장성 | 청크 수 T가 늘어도 선형 복잡도 유지 |

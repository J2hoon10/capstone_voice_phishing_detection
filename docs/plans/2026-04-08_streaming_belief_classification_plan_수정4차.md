# Streaming Belief-State Classification Architecture
> Sequential Bayesian Updating 기반 실시간 텍스트 스트리밍 분류 시스템 설계 계획서
> **v5: Sliding Window KoELECTRA-base (Compression) + Attention-weighted Pooling + Mamba SSM 2-Layer (State Tracking) & 롱테일 대응 구조**

---

## 변경 이력

| 버전 | 주요 변경 사항 |
|------|--------------|
| v3 | Sliding Window BERT + `[CLS]` 추출 + Mamba SSM 기본 구조 |
| v4 | `[CLS]` 추출 → **Attention-weighted Pooling** 교체 / λ_t 초반 강조 스케줄로 역전 / Causal 범위 명시 / 클래스 수 정정(20→5) |
| **v5** | BERT 인코더를 **KoELECTRA-base** 로 명시 / Mamba 레이어 수 **2개로 확정** 및 근거 추가 / 2-레이어 내부 역할 분리 명시 |

---

## 1. 태스크 정의

### 1.1 입력 / 출력 명세

| 항목 | 내용 |
|------|------|
| **입력** | 원본 대화(음성/텍스트)를 실시간으로 처리하는 연속 스트림 |
| **단위** | 이전 청크와 일부 겹치는(Overlap) 슬라이딩 윈도우 단위 텍스트 청크 $c_1, c_2, \ldots, c_T$ |
| **연속성** | 청크들은 시간순으로 배열되며, 겹치는 구간을 통해 단어 끊김(Boundary) 손실을 방지 |
| **출력** | 매 청크(Window) 수신 시마다 즉각적으로 업데이트되는 레이블 확률 분포 $p_t(y)$ |
| **제약 조건** | 1. 실시간 스트리밍(미래 정보 참조 불가) <br> 2. 청크 간 강한 맥락 연결 <br> 3. 소규모 롱테일 데이터(약 3만 개) 환경 |

### 1.2 핵심 설계 철학

> **"로컬 문맥은 Transformer가 깊고 강하게 압축하고, 전체 시간의 흐름(Belief)은 Mamba가 빠르고 가볍게 추적한다."**

두 모델의 치명적 단점을 서로 상쇄시키는 최적의 결합:
- **Transformer 단점 극복:** 무거운 시퀀스 길이를 감당할 필요 없이, 가장 잘하는 '짧은 문장 독해 및 압축'만 수행.
- **Mamba 단점 극복:** 노이즈가 많은 단어 길이를 직접 스캔하지 않고, 정제된 '청크 단위' 요약본만 넘겨받아 효율적 문맥 추적 수행.

### 1.3 Causal 처리 범위 명세

실시간 스트리밍 제약("미래 정보 참조 불가")의 적용 범위를 명확히 정의한다.

| 범위 | 방향성 | 근거 |
|------|--------|------|
| **청크 내부 (Intra-chunk)** | 양방향 허용 | 청크 전체가 버퍼에 쌓인 뒤 BERT에 입력되므로, 청크 내 미래 토큰 참조는 실시간 제약 위반이 아님 |
| **청크 간 (Inter-chunk)** | 단방향 보장 (Causal) | Mamba는 $h_{t-1} \rightarrow h_t$ 방향으로만 상태를 전파하므로, 청크 $t$ 처리 시 미래 청크 $t+1, t+2, \ldots$ 정보는 절대 참조되지 않음 |

> **결론:** 실시간 인과성(Causality)은 **청크 레벨에서 Mamba가 보장**하며, 청크 내부의 양방향 처리는 버퍼 기반 배치 처리로 정당화된다.

### 1.4 모듈 역할 분리 (핵심 설계 원칙)

| 모듈 | 담당 역할 | 기술 (적용 기법) |
|------|----------|------|
| STT | 음성을 텍스트 스트림으로 변환 | Whisper / STT API |
| **Sliding Window** | 스트림을 겹치게 잘라 단어 끊김 현상 방지 | Overlap Window Array |
| **KoELECTRA-base** | "현재 윈도우(청크)가 무슨 의미인가?" **압축** | KoELECTRA-base (12 레이어, hidden 768, 하위 10레이어 동결 또는 LoRA) + Attention-weighted Pooling |
| **Attention Pooling** | "이 청크에서 어떤 토큰이 분류 판단에 중요한가?" **선택적 집약** | Learnable Attention Weight ($W_a$, $W_b$) |
| **Mamba SSM** | **"과거부터 지금까지의 의심도 누적 상태 추적"** | Recurrent Step 모드 (상태 유지) |
| Classification | "현재 시점 레이블 확률 분포 출력" | MLP + Focal Loss |

---

## 2. 전체 아키텍처 개요

### 2.1 단일 청크 처리 파이프라인 (1회 윈도우 루프)

```text
[연속 텍스트/음성 스트림]
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Overlapping Sliding Window 분할                     │
│                                                             │
│  ... [이전 윈도우 끝단 20토큰] + [새로운 음성 80토큰] ...       │
│  📝 "카드 번호를 알려... / 주시면 한도 조회를 해드릴게요"       │
│       (끊김없는 매끄러운 100토큰 윈도우 생성)                 │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: KoELECTRA-base 인코딩 + Attention-weighted Pooling  │
│                                                             │
│  [CLS] "카드번호를 알려주시면..." [SEP]                      │
│        ↓ KoELECTRA-base (12레이어, 청크 내 양방향 집중 분석) │
│  H^(t) ∈ R^(100 × 768)                                     │
│        ↓ [CLS], [SEP] 제외 → 실제 의미 토큰 98개 대상        │
│  e_i = W_a · tanh(W_b · h_i)  ← 토큰별 중요도 점수         │
│  α_i = softmax(e_i)            ← 정규화된 가중치             │
│  x_t = Σ α_i · h_i  ∈ R^(768) ← 분류 핵심 토큰 강조 벡터   │
│                                                             │
│  ★ "카드번호", "계좌이체" 등 핵심 위험 토큰에                │
│     높은 α가 자동으로 집중됨 (end-to-end 학습)               │
└─────────────────────────────────────────────────────────────┘
    │ (길이가 1/98 로 압축, [CLS] 대비 정보 손실 최소화)
    ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: Mamba SSM (2-Layer) Belief(상태) 갱신               │
│                                                             │
│  ┌── Mamba Layer 1: 저수준 상태 변화 감지 ──────────────┐   │
│  │  "이번 청크가 이전 청크 대비 얼마나 달라졌는가?"       │   │
│  │  h_t^(1) = Mamba_Step(h_{t-1}^(1), x_t)            │   │
│  │  y_t^(1) = C^(1) · h_t^(1)  ∈ R^(768)              │   │
│  └────────────────────────────────────────────────────┘   │
│                       ↓                                     │
│  ┌── Mamba Layer 2: 고수준 Belief 통합 ──────────────────┐  │
│  │  "지금까지의 전체 흐름이 보이스피싱 패턴인가?"          │  │
│  │  h_t^(2) = Mamba_Step(h_{t-1}^(2), y_t^(1))         │  │
│  │  y_t = C^(2) · h_t^(2)  ∈ R^(768)                   │  │
│  └────────────────────────────────────────────────────┘  │
│                                                             │
│  ★ O(1) 메모리와 시간으로 즉시 업데이트                      │
│  ★ 청크 간 단방향(Causal) 상태 전파 보장                     │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: 즉각 분류 (Real-time Classification)                │
│                                                             │
│  p_t = softmax(MLP(y_t))  ∈ R^5  (5개 클래스)              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 수학적 수식 흐름

### 3.1 표기법 정의

| 기호 | 의미 | 차원 |
|------|------|------|
| $W$ | 윈도우 크기 (토큰 수, [CLS]/[SEP] 포함) | 예: 100 |
| $W'$ | 실제 의미 토큰 수 ([CLS], [SEP] 제외) | $W - 2 = 98$ |
| $S$ | 스트라이드 (이동 간격) | 예: 80 (20 토큰 오버랩) |
| $c_t$ | 슬라이딩 윈도우 $t$의 텍스트 | - |
| $H^{(t)}$ | BERT 출력 전체 hidden state | $\mathbb{R}^{W \times d}$ |
| $\alpha_i$ | $i$번째 토큰의 Attention-weighted Pooling 가중치 | 스칼라 |
| $x_t$ | Attention-weighted Pooling 결과 벡터 | $\mathbb{R}^{d}$ ($d=768$) |
| $h_t$ | $t$번째 시점의 Mamba Hidden State | $\mathbb{R}^{N \times d}$ |
| $y_t$ | $t$번째 시점의 Mamba 출력 벡터 | $\mathbb{R}^{d}$ |
| $p_t$ | $t$번째 시점 분류 확률 | $\mathbb{R}^{C}$ ($C=5$ 클래스) |
| $N$ | Mamba SSM State 차원 | 예: 16 |
| $W_b$ | Pooling 중요도 변환 행렬 (학습 파라미터) | $\mathbb{R}^{d \times d}$ |
| $W_a$ | Pooling 중요도 스칼라 투영 (학습 파라미터) | $\mathbb{R}^{d \times 1}$ |

---

### 3.2 Step 1 & 2: 윈도우 분할 및 BERT + Attention-weighted Pooling

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

$t$번째 슬라이딩 윈도우 $c_t$에 대해 토큰화를 진행하고 KoELECTRA-base를 통과시킨다.

$$H^{(t)} = \text{KoELECTRA-base}(E^{(t)}) \in \mathbb{R}^{W \times d}$$

청크 내부에서는 양방향 self-attention이 허용된다 (§1.3 Causal 범위 참조).

> **동결 전략:** 하위 1~10 레이어는 동결(Freeze)하고 상위 11~12 레이어만 학습하거나, 전 레이어에 LoRA를 적용한다. Attention-weighted Pooling의 $W_a$, $W_b$는 항상 학습 대상으로 유지한다.

**[Step 2-B] Attention-weighted Pooling** *(v4 핵심 변경)*

`[CLS]`(위치 0)와 `[SEP]`(마지막 위치)를 제외한 실제 의미 토큰 $W' = W-2$개만을 대상으로 학습 가능한 중요도 가중치를 산출한다.

$$e_i = W_a \cdot \tanh(W_b \cdot h_i^{(t)}), \quad i = 1, \ldots, W'$$

$$\alpha_i = \frac{\exp(e_i)}{\displaystyle\sum_{j=1}^{W'} \exp(e_j)}$$

$$x_t = \sum_{i=1}^{W'} \alpha_i \cdot h_i^{(t)} \in \mathbb{R}^{d}$$

> **[CLS] 제외 근거:** Attention-weighted Pooling은 $W_a$, $W_b$가 중요도를 직접 학습하므로 `[CLS]`의 사전 요약 능력이 불필요하다. 오히려 `[CLS]`의 압축된 hidden state가 softmax 경쟁에 끼어들어 $\alpha$ 분포를 왜곡할 수 있어 제외한다.

> **효과:** "카드 번호", "계좌 이체", "금감원 직원" 등 보이스피싱 탐지에 결정적인 토큰에 높은 $\alpha$가 자동으로 집중되도록 end-to-end 학습된다.

> **추가 파라미터:** $W_b$ (768×768 ≈ 589K개) + $W_a$ (768×1 = 768개) = **약 59만 개**.
> KoELECTRA 전체 파라미터(~1억 개) 대비 **0.6% 수준**으로 과적합 부담이 극히 낮다.

---

### 3.3 Step 3: Mamba 2-Layer Recurrent State Update ★스트리밍 핵심

#### Mamba 레이어 수 결정 근거

이 태스크에서 Mamba가 처리하는 시퀀스는 토큰이 아닌 **청크 단위 압축 벡터**이며, 5분 통화 기준 $T \approx 30$~$60$개 수준이다. Mamba가 학습해야 하는 패턴의 구조는 아래와 같은 **단조로운 상태 전이**다.

```
[정상 대화] → [신뢰 구축] → [개인정보 요청] → [긴박감 조성]
```

| 레이어 수 | 판단 |
|:---------:|------|
| 1 | 단순 상태 전이는 포착 가능하나 중간 단계 패턴 구분력 부족 가능성 |
| **2** ✅ | 1층(저수준 변화 감지) + 2층(고수준 belief 통합)으로 역할 자연 분리, 30K 데이터에서 과적합 위험 낮음 |
| 3+ | 30K 소규모 데이터 환경에서 과적합 위험, 이미 KoELECTRA가 고품질 입력을 제공하므로 불필요 |

> **확정: 2 레이어.** 단, {1, 2} ablation을 통해 검증셋 F1 기준으로 최종 확인을 권장한다.

#### 2-레이어 수식

병렬 스캔(Parallel Scan)을 하지 않고, 실시간 환경에 맞게 RNN처럼 1-step 연산을 수행한다.

**Layer 1 — 저수준 상태 변화 감지:**

$$h_t^{(1)},\; y_t^{(1)} = \text{Mamba\_Step}^{(1)}\!\left(h_{t-1}^{(1)},\; x_t\right)$$

**Layer 2 — 고수준 Belief 통합:**

$$h_t^{(2)},\; y_t = \text{Mamba\_Step}^{(2)}\!\left(h_{t-1}^{(2)},\; y_t^{(1)}\right)$$

각 레이어의 선택적 상태 갱신 수식 ($l \in \{1, 2\}$):

- $\Delta^{(l)} = \text{softplus}(W_\Delta^{(l)} \cdot \text{input}^{(l)})$
- $\bar{A}^{(l)} = \exp(\Delta^{(l)} \cdot A^{(l)})$
- $\bar{B}^{(l)} = (\Delta^{(l)} \cdot A^{(l)})^{-1}(\exp(\Delta^{(l)} \cdot A^{(l)}) - I) \cdot \Delta^{(l)} \cdot B^{(l)}$
- $h_t^{(l)} = \bar{A}^{(l)} \cdot h_{t-1}^{(l)} + \bar{B}^{(l)} \cdot \text{input}^{(l)}$
- $y_t^{(l)} = C^{(l)} \cdot h_t^{(l)}$ &nbsp;&nbsp; ($\Delta^{(l)}, B^{(l)}, C^{(l)}$는 해당 레이어 입력에 의존)

> 추론 시 메모리에 $h_{t-1}^{(1)}$과 $h_{t-1}^{(2)}$ 두 state만 유지하면 되므로, 통화 시간과 무관하게 $O(1)$ 레이턴시가 보장된다.

> **청크 간 Causal 보장:** 각 레이어의 $h_t^{(l)}$는 오직 $h_{t-1}^{(l)}$과 해당 레이어 입력만으로 계산되므로 미래 청크 정보가 유입되지 않는다.

---

### 3.4 Step 4: 매 순간(Every Moment) 레이블 출력

$$p_t(y) = \text{softmax}(W_{c2} \cdot \text{ReLU}(W_{c1} \cdot y_t + b_1) + b_2) \in \mathbb{R}^5$$

---

## 4. 소규모 롱테일 데이터(3만 개) 극복 아키텍처 (학습 전략)

가장 큰 난제인 3만 개 롱테일 불균형 데이터(5개 클래스)를 해결하기 위한 학습 설계.

### 4.1 KoELECTRA-base 부분 동결 (Freeze / LoRA)
- 처음부터 KoELECTRA-base 전체 파라미터(약 1억 1천만 개)를 튜닝하면 과적합이 발생한다.
- **[옵션 A]** 하위 1~10 레이어는 Freeze(동결)하고 상위 11~12 레이어만 학습.
- **[옵션 B]** 전 레이어에 **LoRA** (rank=8, alpha=16)를 적용하여 가벼운 파인튜닝 수행. 추가 학습 파라미터를 수백만 개 이하로 제한.
- Attention-weighted Pooling의 $W_a$, $W_b$는 옵션과 무관하게 항상 학습 대상으로 유지한다.

### 4.2 Mamba 정규화 전략

과적합 방지는 state 크기 축소가 아닌 **정규화 기법**으로 처리한다.

- **Mamba Layer 수: 2개로 확정** (§3.3 근거 참조). {1, 2} ablation으로 최종 검증.
- **Dropout (0.1~0.2)** 를 각 Mamba 레이어 출력 직후에 적용.
- **Weight Decay** 를 옵티마이저에 설정 (AdamW, $\lambda = 0.01$).
- **Early Stopping** 을 검증셋 F1 기준으로 적용 (patience = 5 epoch 권장).
- `d_state` ($N$) 최적값은 고정하지 않고 $\{16, 32, 64\}$ 범위에서 ablation으로 결정한다.

### 4.3 손실 함수: Focal Loss + 초반 강조 시간 가중치

**Focal Loss** (롱테일 불균형 처리):

$$\mathcal{L}_{\text{Focal}}(p_t, y^*) = -\alpha_{y^*}(1 - p_t^{(y*)})^\gamma \log(p_t^{(y*)})$$

**시간 가중치 $\lambda_t$** *(v4 수정: 후반 강조 → 초반 강조로 역전)*

보이스피싱 조기 탐지 목표에 맞게, 통화 초반 청크에 더 높은 가중치를 부여한다.

- **[기본안] 지수 감쇠 (어노테이션 불필요):**

$$\lambda_t = \exp(-\alpha \cdot (t-1)), \quad \alpha \in \{0.05,\; 0.1,\; 0.2\}$$

| $t$ (청크 순서) | $\alpha=0.05$ | $\alpha=0.1$ | $\alpha=0.2$ |
|:--------------:|:-------------:|:------------:|:------------:|
| 1 (초반) | 1.00 | 1.00 | 1.00 |
| 5 | 0.82 | 0.67 | 0.45 |
| 10 | 0.64 | 0.37 | 0.14 |
| 20 (후반) | 0.39 | 0.14 | 0.02 |

- **[확장안] 전환 시점 집중 (t* 어노테이션 구축 후 적용 가능):**

$$\lambda_t = \exp(-\beta \cdot |t - t^*|), \quad \beta \in \{0.2,\; 0.3,\; 0.5\}$$

여기서 $t^*$는 각 통화에서 정상 → 보이스피싱 의심으로 레이블이 처음 전환되는 청크 인덱스다. 이 어노테이션이 존재하면 모델이 **전환 시점을 정확히 포착**하도록 집중 학습된다.

> **현재 권장:** 어노테이션 미구축 상태에서는 지수 감쇠($\alpha=0.1$)를 기본으로 적용하고, $\alpha$는 검증셋 F1 기준으로 선택한다.

**최종 Loss:**

$$\mathcal{L}_{\text{Total}} = \sum_{t=1}^{T} \lambda_t \cdot \mathcal{L}_{\text{Focal}}(p_t, y^*)$$

### 4.4 학습 프로세스 (BPTT vs Parallel)

**추론** 때는 Step 단위로 하나씩 처리하지만, **학습** 시에는 $T$개의 윈도우를 한꺼번에 $x_{1:T}$로 추출한 뒤 Mamba 모델에 던져, 고속의 Parallel Scan 연산을 통해 VRAM 효율을 극대화한다.

---

## 5. 구현 시 요약 및 권장 스펙

### 5.1 Attention-weighted Pooling 구현 코드

```python
import torch
import torch.nn as nn

class AttentionWeightedPooling(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.W_b = nn.Linear(hidden_dim, hidden_dim)          # (768 → 768)
        self.W_a = nn.Linear(hidden_dim, 1, bias=False)       # (768 → 1)

    def forward(self, H, attention_mask):
        # H              : (batch, seq_len, 768)  ← BERT 전체 출력
        # attention_mask : (batch, seq_len)        ← 패딩 위치 구분

        # [CLS](위치 0), [SEP](마지막 위치) 제외
        H_tokens    = H[:, 1:-1, :]              # (batch, seq_len-2, 768)
        mask_tokens = attention_mask[:, 1:-1]    # (batch, seq_len-2)

        # 토큰별 중요도 점수 계산
        e = self.W_a(torch.tanh(self.W_b(H_tokens)))          # (batch, seq_len-2, 1)

        # 패딩 토큰 마스킹 후 softmax
        e = e.masked_fill(mask_tokens.unsqueeze(-1) == 0, float('-inf'))
        alpha = torch.softmax(e, dim=1)                        # (batch, seq_len-2, 1)

        # 가중합으로 최종 벡터 생성
        x_t = (alpha * H_tokens).sum(dim=1)                   # (batch, 768)
        return x_t
```

### 5.2 하이퍼파라미터 가이드

| 파라미터 | 권장 설정 | 사유 |
|----------|----------|------|
| **Window Size (W)** | 100 ~ 128 토큰 | 너무 짧으면 KoELECTRA 분석력 하락, 길면 실시간 응답성 저하 |
| **Stride (S)** | 80 ~ 100 토큰 | Overlap(20~30토큰)을 두어 단어와 문맥 단절 보호 |
| **Mamba Layers** | **2** (1 vs 2 ablation) | 1층: 저수준 변화 감지 / 2층: 고수준 belief 통합. 3층 이상은 30K 데이터에서 과적합 위험 |
| **Mamba d_state ($N$)** | {16, 32, 64} ablation | 데이터 규모 고려, 검증셋 F1 기준으로 최적값 결정 |
| **Dropout** | 0.1 ~ 0.2 | 각 Mamba 레이어 출력 직후 적용, 정규화 핵심 수단 |
| **Weight Decay** | 0.01 | AdamW 옵티마이저 설정 |
| **Early Stopping** | patience = 5 | 검증셋 F1 기준 |
| **Loss Function** | Focal Loss ($\gamma=2.0$) | Long-Tail 분포 해소를 위한 소수 클래스 집중 학습 |
| **λ_t 감쇠 계수 ($\alpha$)** | {0.05, 0.1, 0.2} ablation | 검증셋 F1 기준으로 최적값 결정 |
| **KoELECTRA Trainable** | Last 2 Layers or LoRA (rank=8) | 풀튜닝 시 과적합 방지 |
| **Pooling W_a, W_b** | 항상 학습 | 소규모 추가 파라미터(~59만 개), 항상 업데이트 |

### 5.3 기대 효과

- **실시간 반응속도 (Latency):** $O(1)$의 초고속(수십 ms 내) 추론 스피드 보장
- **문맥 보존력:** 단방향(Causal) 한계에도 불구하고 슬라이딩 오버랩 덕에 언어적 단절 문제 완전 해소
- **핵심 토큰 집중:** Attention-weighted Pooling이 "카드 번호", "계좌 이체" 등 탐지 핵심 표현에 자동으로 집중하여 Mamba에 전달되는 입력 품질 향상
- **안정적 성능:** BERT의 뛰어난 단기 독해력과 Mamba의 탁월한 장기 기억력(RNN-like)이 역할 충돌 없이 극대화
- **소규모 데이터 승리:** LoRA + Dropout + Early Stopping + 초반 강조 Loss 결합으로 3만 개 데이터에서 안정적 수렴 기대

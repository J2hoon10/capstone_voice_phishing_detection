# 보이스 피싱 대화 탐지 모델 아키텍처 및 손실 함수 설계서

본 문서는 KoELECTRA와 Mamba SSM을 결합한 보이스 피싱 탐지 모델의 구조와 데이터의 계층적 및 순차적 특성을 반영한 손실 함수 적용 계획을 상술합니다.

---

## 1. 용어 정의 (Definitions)
답변의 일관성을 위해 모델링에 사용되는 주요 용어를 다음과 같이 정의합니다.

* **위험도 단계 (Risk Level, 3-class)**: 개별 발화(Segment) 내에 포함된 위협의 강도를 나타내며, 순서적 관계(Ordinal relationship)를 가집니다. (예: 안전 < 의심 < 위험)
* **최종 클래스 (Final Class, 5-class)**: 전체 대화의 목적을 분류하며, 크게 '일반'과 '피싱'으로 나뉘는 계층적 구조(Hierarchical structure)를 가집니다.
    * **일반 클래스**: 일상 대화, 공적 업무, 단순 문의 등 (3종)
    * **피싱 클래스**: 대출 사기, 공공기관 사칭 등 (2종)

---

## 2. 모델 아키텍처 (Model Architecture)

제안된 모델은 문장 단위의 의미 추출과 전체 대화의 시계열적 맥락 파악을 동시에 수행하는 Multi-task Learning 구조를 취합니다.

### 2.1 인코더 및 시퀀스 모델링

1. **Sliding Window 기반 세그먼트 분할**
   * 실시간 탐지(Streaming)를 위해 STT(음성 인식) 텍스트 스트림을 일정한 윈도우 크기(`WINDOW_SIZE=128`)와 스트라이드(`STRIDE=100`)로 분할하여 모델에 입력합니다.
   * **데이터 입력 길이 설정 근거 (WINDOW_SIZE=128, STRIDE=100):**
     * **문맥 보존의 최소 단위**: 한국어 구어체 기준 128 토큰은 약 3~5개의 발화(Turn) 또는 1~2개의 긴 문장을 포함합니다. 이는 피싱범이 피해자를 압박하거나 속이기 위해 사용하는 '위협의 단기적 맥락(예: 신분 사칭 -> 행동 요구)'을 포착하기 위한 최소한의 유효 정보량입니다.
     * **실시간 탐지 지연(Latency) 방지**: 윈도우 크기를 256이나 512로 늘리면 세그먼트 단위 처리를 위한 STT 버퍼 대기 시간이 길어져, 즉각적인 경고가 생명인 보이스 피싱 방지 시스템의 실시간성이 크게 훼손됩니다.
     * **오버랩(Overlap)을 통한 문맥 단절 방지**: 128 크기의 윈도우를 100칸씩(`STRIDE=100`) 이동시켜 28 토큰의 겹침(Overlap) 구간을 둠으로써, 핵심 단어가 윈도우 경계에서 잘려 의미가 손실되는 문제를 예방합니다.
   * **검증을 위한 실험 계획 (Ablation Study):**
     * **Window Size 실험**: 크기를 64, 128, 256으로 변경하며 (1) Classification 성능(Macro F1), (2) 세그먼트 생성 대기 시간(Latency), (3) KoELECTRA 인코더 연산 속도를 측정하여 최적의 trade-off 지점 도출.
     * **Stride 비율 실험**: Overlap 비율(Window Size 대비 10%, 25%, 50%)을 조절하며 문맥 절단에 따른 정확도 하락분 검증.

2. **KoELECTRA Encoder**
   * 한국어 구어체 및 문맥 파악에 특화된 ELECTRA 모델을 사용하여 각 토큰의 고차원(768차원) 임베딩을 추출합니다.

3. **Mamba SSM (State Space Model) 도입 상세 근거**
   * 대화의 조각(Segment)들을 단순 분석하는 것을 넘어, **"스트리밍(Streaming) 환경에서의 끊김 없는 대화 흐름 파악"**을 위해 도입되었습니다.
   * **[핵심 의문] KoELECTRA(Transformer) 단일 모델로는 왜 흐름 파악이 불가능한가?**
     * **단절된 문장 분석의 한계**: KoELECTRA는 한 번에 최대 512 토큰만 볼 수 있어, 긴 대화를 여러 개의 짧은 윈도우로 잘라서 분석해야 합니다. 이 윈도우들의 결과를 단순히 평균(Mean/Max Pooling) 내버리면, **'신분 사칭 $\rightarrow$ 공포감 조성 $\rightarrow$ 금전 요구'**로 이어지는 보이스 피싱 특유의 시나리오 흐름(시간적 선후 관계)이 완전히 단절되고 소실됩니다.
     * **Streaming 환경에서의 구조적 모순**: 만약 Transformer로 실시간 누적 대화를 계속 파악하려 한다면, 매 시점마다 누적된 모든 토큰에 대해 Self-Attention을 재계산해야 하므로 연산량이 $\mathcal{O}(L^2)$로 폭발합니다 (KV Cache를 쓰더라도 최소 $\mathcal{O}(L)$로 증가). 즉, Transformer는 애초에 끝을 알 수 없는 실시간 스트리밍 음성 인식을 추적하도록 설계되지 않았습니다.
   * **Mamba SSM의 진정한 가치: Streaming Belief State**
     * Mamba의 진짜 강점은 단순히 '긴 문맥을 잘 본다'는 것이 아니라, 고정된 크기의 은닉 상태(**Hidden State, $h_t \in \mathbb{R}^d$**)만 유지하면서 매 스텝마다 $\mathcal{O}(1)$의 일정한 연산량만으로 전체 대화의 맥락을 누적해 나간다는 구조적 성질에 있습니다.
     * 즉, Mamba의 Hidden State 자체가 화자의 의도와 위험도를 실시간으로 추적하는 **'스트리밍 상태(Streaming Belief State)'**로 작동합니다.
     * 텍스트 윈도우가 들어올 때마다 KoELECTRA가 "현재 문장의 조각난 의미"를 분석하여 던져주면, Mamba는 과거의 대화 흐름(Hidden State)에 이 새로운 정보를 선택적으로 업데이트(Selective State Space)하여 대화의 큰 줄기(시나리오)를 끝까지 추적합니다.
   * **가성비(효율성) 극대화**: Mamba 레이어 추가로 늘어나는 파라미터는 2~5M 수준으로, KoELECTRA(약 110M) 대비 5% 미만의 미미한 증가에 불과합니다. 단 5%의 무게 추가만으로 **"순간의 의미(단절된 조각)만 파악하는 모델"**을 **"시간의 흐름(대화 시나리오)을 실시간으로 꿰뚫어 보는 모델"**로 진화시킬 수 있으므로, 스트리밍 환경에서 Mamba의 결합은 구조적 필연성을 가집니다.

### 2.2 풀링 및 분류 헤드 (Pooling & Heads)
* **Streaming Branch**:
    * **Running Max**: 시퀀스 내에서 가장 강한 특징값을 보존합니다.
    * **Last Belief**: 가장 최근의 상태 정보를 통해 현재 대화의 흐름을 반영합니다.
    * **Concat**: 위 두 특징을 결합하여 **Main Classifier**를 통해 최종 **5-class**를 분류합니다.
* **Auxiliary Branch**:
    * **Aux Head**: Mamba의 중간 출력을 받아 문장 단위의 **위험도(3-class)**를 예측하여 학습의 보조 신호로 활용합니다.

---

## 3. 손실 함수 설계 (Loss Function Strategy)

### 3.1 보조 태스크: Ordinal Regression Loss
위험도 단계(3-class)는 클래스 간의 '순서'가 중요하므로, 일반적인 Cross-Entropy 대신 서수 회귀(Ordinal Regression) 기법을 적용합니다.

**공식:**
각 단계 $k$에 대해 이진 분류 문제를 수행하는 효과를 줍니다.
$$\mathcal{L}_{aux} = -\sum_{k=1}^{K-1} [y > k] \log(\sigma(f(x)_k)) + [y \leq k] \log(1 - \sigma(f(x)_k))$$
(여기서 $\sigma$는 시그모이드 함수, $K$는 클래스 수)

**기대 효과:**
* '안전'을 '위험'으로 분류할 때 더 큰 오차를 발생시켜 모델이 위험도의 강도를 선형적으로 학습하도록 유도합니다.

### 3.2 주 태스크: Hierarchical Cross-Entropy (HCE)
최종 5개 클래스는 계층적 구조를 가지므로, 상위 범주(일반/피싱)를 먼저 구분하고 하위 범주를 예측하는 방식으로 손실을 계산합니다.

**공식:**
$$\mathcal{L}_{main} = \mathcal{L}_{superclass}(y_{super}, \hat{y}_{super}) + \lambda \sum \mathcal{L}_{subclass}(y_{sub}, \hat{y}_{sub} | y_{super})$$
* $\mathcal{L}_{superclass}$: 일반 대화인가 피싱 대화인가에 대한 이진 분류 손실.
* $\mathcal{L}_{subclass}$: 해당 상위 범주 내에서의 세부 클래스 분류 손실.

**기대 효과:**
* 피싱 범주 내에서 오분류가 발생하더라도, 이를 '일반 대화'로 판단하는 치명적 오류(Critical Error)를 방지하여 모델의 신뢰성을 높입니다.

---

## 4. 추가 제언 (Points to Explore)

1.  **Loss Balancing**: $\mathcal{L}_{total} = \alpha \mathcal{L}_{main} + \beta \mathcal{L}_{aux}$ 설정 시, 학습 초기에는 $\beta$를 높여 문장 특징을 먼저 익히게 하고, 후기에는 $\alpha$를 높이는 Curriculum Learning 기법 검토.
2.  **Class Imbalance**: 피싱 데이터의 희소성을 해결하기 위해 하위 계층 손실 함수에 **Focal Loss** 가중치 적용 권장.
3.  **Explainability**: Aux Head의 출력을 통해 대화 중 어느 지점에서 위험도가 급증했는지 시각화하여 탐지 근거 제공 가능성 확인.

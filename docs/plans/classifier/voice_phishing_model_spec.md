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
1.  **Sliding Window**: 실시간 탐지(Streaming)를 위해 입력 대화를 일정 윈도우 단위로 분할하여 처리합니다.
2.  **KoELECTRA Encoder**: 한국어 구어체 및 문맥 파악에 특화된 ELECTRA 모델을 사용하여 각 토큰의 고차원 임베딩을 추출합니다.
3.  **Mamba SSM (State Space Model)**: RNN의 효율성과 Transformer의 병렬성을 결합한 구조로, 긴 대화 시퀀스 내의 장기 의존성(Long-range dependency)을 $\mathcal{O}(N)$의 복잡도로 처리합니다.

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

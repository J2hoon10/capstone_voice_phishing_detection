
---

# [Final Plan] Unpadded ModernBERT Pure Distillation

본 문서는 **ModernBERT의 효율성(FlashAttention, Unpadding)**을 유지하며, 동시에 **구조적 축소(, Hidden Size )**를 수행하기 위한 최종 증류(Distillation) 계획이다. 학습은 Hard Label 없이 **오직 Teacher 모델의 정보(Hidden State, Logits)**만을 사용하여 진행한다.

---

## 1. Student 모델 초기화 (Scientific Initialization)

Student 모델을 무작위로 초기화(Random Init)하지 않고, **데이터 기반 분석**을 통해 Teacher의 핵심 정보를 보존한 상태로 조립하여 초기 성능을 확보한다.

### **A. Layer 선택 (Depth): **

* **방법:** **BI (Block Influence) Score** 측정
* 입력과 출력의 코사인 유사도가 낮아 정보 변환이 활발하게 일어나는 상위 11개 레이어를 선택한다.


* **근거 논문:**
> Men et al., *"ShortGPT: Layers in LLMs are More Redundant Than You Think"*, arXiv 2024.



### **B. Head/Neuron 선택 (Width): Hidden Size **

* **방법:** **Gradient-based Importance Score**
* 소량의 데이터에 대한 Loss 민감도를 측정하여 상위 50%의 Attention Head와 MLP Neuron을 선택(Pruning)한다.


* **근거 논문:**
> Michel et al., *"Are Sixteen Heads Really Better than One?"*, NeurIPS 2019.



---

## 2. 학습 전략 (Training Strategy)

초기화 과정에서 끊어진 신경망 연결을 복구하기 위해 모델 전체를 미세 조정한다.

### **A. 학습 방법: Full Fine-tuning (No LoRA)**

* **설명:** Pruning(가지치기) 후 구조적 손상을 입은 모델의 성능 회복을 위해 모델 전체를 재학습하는 **"Healing"** 과정을 수행한다.
* **근거 논문:**
> Xia et al., *"Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning"*, ICLR 2024.



### **B. 데이터 처리: Unpadding & FlashAttention**

* **설명:** Padding 없이 유효 토큰만 일렬로 나열한 **1D Tensor**를 사용하여 ModernBERT의 추론 속도 이점을 극대화한다.
* **근거 논문:**
> Dao et al., *"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"*, NeurIPS 2022.
> Lai et al., *"ModernBERT: A Modern BERT for the Modern Web"*, 2024.



---

## 3. 손실 함수 아키텍처 (Loss Architecture)

Teacher와 Student 간의 차원 불일치를 해결하기 위해 **투영(Projection) 레이어**를 사용하며, Hard Label(정답) 없이 진행한다.

* **구조:** **Learnable Linear Projection ()**
* **수식:**


* *Shape Transformation:* 


* **근거 논문:**
> Jiao et al., *"TinyBERT: Distilling BERT for Natural Language Understanding"*, Findings of EMNLP 2020.



---

## 4. 실험 계획 (Experimental Design)

본 연구는 **"중간층 정보()"**가 주가 되었을 때, **"최종 출력 정보()"**의 추가가 성능에 미치는 영향을 단계별로 분석한다.

### **실험 1: Hidden State Loss Metric 최적화 (단독 사용)**

 없이, $L_{hidden}$만을 사용하여 어떤 Metric이 내부 표현 학습에 가장 효과적인지 탐색한다.

* **설정:** 
* **비교군 (Variants):**
1. **MSE (Mean Squared Error):** 값과 크기(Norm)를 모두 맞춤. *(Ref: TinyBERT)*
2. **Cosine Similarity Loss:** 벡터의 방향(의미) 일치도에 집중. *(Ref: DistilBERT, Sentence-BERT)*
3. **KL-Divergence:** Hidden State를 확률 분포로 변환하여 비교. *(Ref: PKD - Patient Knowledge Distillation)*



### **실험 2: Soft Target 효과 검증 (Soft Target 추가)**

실험 1에서 선정된 최적의 Hidden Loss()에 Soft Target을 더했을 때의 성능 변화를 관찰한다.

* **목적:** "정답지(Hard Label) 없이 Teacher의 확률 분포(Soft Target)만으로도 학습이 충분히 강화되는가?" 검증.
* **비교군:**
* **Baseline:** 

* **Experiment:** 



* **근거 논문:**
> Hinton et al., *"Distilling the Knowledge in a Neural Network"*, NIPS 2015 Workshop.

---

## 5. Expected Computational Complexity (예상 연산 복잡도)

본 연구에서 사용하는 **Teacher 모델(ModernBERT-Base)**은 22개 레이어를 가지며 약 1.49억 개의 파라미터를 보유한다. 이를 기반으로 제안하는 **Student 모델**은 Depth(깊이)와 Width(너비)가 모두 50%씩 축소되었다.

### **A. 파라미터 수 (Model Size) 비교**

* **Teacher Model:** ModernBERT-Base ()  **149M Params**
* **Student Model:** Custom Small ()
* **파라미터 감소 분석:**
* **Non-Embedding Params:**  비례  약 **1/8**로 감소.
* **Embedding Params:**  비례  **1/2**로 감소.
* **예상 Student 크기:** **약 25M - 30M (2,500만 ~ 3,000만)** 수준의 초경량 모델.



### **B. 연산량 (FLOPs) 및 학습 속도**

* **Speed-up Factor:**
* 핵심 행렬 연산(Matrix Multiplication)에서 이론적으로 **약 8배의 FLOPs 감소**가 발생한다.
* 모델 사이즈가 MobileBERT(25M) 수준으로 작아지므로, CPU 환경이나 엣지 디바이스(모바일)에서도 실시간 추론이 가능하다.


* **Unpadding & FlashAttention 효과:**
* 패딩 연산 제거()와 FlashAttention의 결합으로, 짧은 문장이 많은 대화 데이터셋에서 기존 BERT 대비 압도적인 처리 속도를 보인다.



### **C. Loss Calculation Overhead**

* **Projection 연산:** 
* **분석:** Projection Layer의 파라미터는 약 30만 개()로, 전체 학습 부하에 미치는 영향은 미미하다(Ignorable).

### **D. 요약 비교 (Summary Table)**

| Metric | Teacher (Base) | Student (Ours) | Reduction Ratio |
| --- | --- | --- | --- |
| **Layers** | 22 | 11 | **50% (1/2)** |
| **Hidden Size** | 768 | 384 | **50% (1/2)** |
| **Parameters** | **~149M** | **~28M (Est.)** | **~81% Reduction** |
| **Training FLOPs** |  |  | **~87% Reduction** |
| **Inference** | Fast | Extremely Fast | **Mobile-Ready** |

---

## 6. Reference List (핵심 근거 논문)

| Category | Reference |
| --- | --- |
| **Depth Pruning** | Men et al., *"ShortGPT: Layers in LLMs are More Redundant Than You Think"*, arXiv 2024. |
| **Width Pruning** | Michel et al., *"Are Sixteen Heads Really Better than One?"*, NeurIPS 2019. |
| **Recovery Training** | Xia et al., *"Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning"*, ICLR 2024. |
| **Distillation Arch** | Jiao et al., *"TinyBERT: Distilling BERT for Natural Language Understanding"*, Findings of EMNLP 2020. |
| **Efficiency** | Dao et al., *"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"*, NeurIPS 2022. |
| **Soft Target** | Hinton et al., *"Distilling the Knowledge in a Neural Network"*, NIPS 2015 Workshop. |


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

본 연구에서 제안하는 Student 모델은 Teacher 모델 대비 Depth(깊이)와 Width(너비)가 모두 축소되었습니다. 이에 따른 이론적 가속 효과와 메모리 절감 효과를 분석한다.

### **A. 파라미터 수 (Model Size) 비교**

Transformer 모델의 파라미터 수는 주로 **Hidden Size()의 제곱**에 비례한다.

* **Teacher Model:**  (가정)
* **Student Model:** 
* **Transformer Block 파라미터 비율:**


* *Note:* Embedding Layer는 에 비례()하므로, 전체 모델 크기는 **약 15%~20% 수준(약 1/5 ~ 1/6)**으로 대폭 감소할 것으로 예상된다. (약 40M~50M 파라미터 추정)



### **B. 연산량 (FLOPs) 및 학습 속도**

학습 및 추론 시의 연산량(FLOPs) 역시 $O(L \cdot H^2)$에 지배된다.

* **Speed-up Factor:**
위 파라미터 계산과 동일하게, 핵심 연산(Matrix Multiplication)에서 이론적으로 **약 8배의 FLOPs 감소**가 발생한다.
* **Unpadding & FlashAttention 효과:**
* 기존 Attention 복잡도 $O(B \cdot S^2)$에서 Padding 연산을 제거하여, 유효 토큰 수()에 선형적인 $O(N_{valid})$에 가까운 효율을 보인다.
* Student 모델은 Teacher 대비 Head 개수가 절반이므로, Attention 연산 자체도 물리적으로 2배 빠르다.



### **C. Loss Calculation Overhead (Projection Layer)**

Distillation을 위해 추가된 Projection Layer의 연산 비용은 전체 학습 파이프라인에서 무시할 수 있는 수준이다.

* **Projection 연산:** 
* **복잡도:** 
* 이는 Transformer 내부의 FFN(Feed-Forward Network) 연산량의 일부에 불과하므로, 학습 속도 저하를 거의 유발하지 않는다.



### **D. 요약 비교 (Summary Table)**

| Metric | Teacher (Baseline) | Student (Proposed) | Reduction Ratio |
| --- | --- | --- | --- |
| **Layers** | 22 | 11 | **50% (1/2)** |
| **Hidden Size** |  (1024) |  (512) | **50% (1/2)** |
| **Parameters** | ~300M+ (Est.) | ~50M (Est.) | **~16% (1/6)** |
| **Training FLOPs** |  |  | **~87% Reduction** |
| **Inference Latency** | Baseline | Very Fast | **Max 8x Speedup** |

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

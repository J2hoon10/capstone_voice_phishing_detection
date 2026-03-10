
---

# 온디바이스 보이스피싱 탐지를 위한 ModernBERT 경량화 실험 설계서

**작성일:** 2026-01-06
**베이스 모델:** ModernBERT-base
**목표:** Teacher Assistant(TA) 기반의 단계적 지식 증류를 통한 고성능 온디바이스 탐지 모델 구축

---

## 1. 개요 (Overview)

본 문서는 보이스피싱 탐지 모델을 모바일 환경(Android/iOS)에서 효율적으로 구동하기 위한 구조 설계 및 실험 계획을 기술한다. 고성능 모델인 ModernBERT를 모바일에 최적화하기 위해, **Teacher Assistant (TA)** 모델을 중간 단계에 도입하는 계단식 증류(Cascaded Distillation) 전략을 채택한다. 또한, 모델 간 구조적 차이를 극복하고 추론 성능을 보존하기 위해 **MiniLM**의 Deep Self-Attention Distillation 방식을 적용한다.

## 2. 핵심 아키텍처 전략 (Core Architecture Strategy)

### 2.1. 하이브리드 어텐션 메커니즘 (Hybrid Attention Mechanism)

학습 효율성과 추론 호환성을 분리하여 접근한다.

* **학습 시 (Training):** `Flash Attention` 적용 -> windows 환경에서 적용 어려움이 있어서 없이 진행, flash Attention은 연산 속도 향상에만 도움이 있고 성능과는 무관
* 목적: GPU 메모리 사용량 최소화 및 학습 속도 가속.
* 설정: Head Dimension을 **64**로 고정하여 하드웨어 가속 호환성 확보.


* **추론 시 (Inference):** `Standard (Math) Attention` 적용
* 목적: ONNX, TFLite 변환 호환성 확보 및 NPU 가속 지원.



### 2.2. Inverted Bottleneck 구조 도입

모바일 AP의 메모리 대역폭(Memory Bandwidth) 병목을 해결하기 위한 설계.

* **구조:** `Input(128) -> Expand(384) -> Output(128)`
* **효과:** 메모리 I/O 비용 감소 및 캐시 히트율 향상.
* **설정:** 입출력 차원은 **128**, 내부 연산 차원은 **384**로 설정.

---

## 3. 단계적 실험 설계 (Cascaded Experimental Design)

직접적인 증류(Teacher  Student)로 인한 성능 급락을 방지하기 위해, **Teacher  TA  Student**로 이어지는 순차적 파이프라인을 구축한다.

### 3.1. 모델 계층 구조 비교

| 구분 | **Stage 1: Teacher** | **Stage 2: Teacher Assistant (TA)** | **Stage 3: Student (Target)** |
| --- | --- | --- | --- |
| **역할** | 지식 원천 (Source) | 지식 가교 (Bridge) | 최종 온디바이스 모델 |
| **Layers (Depth)** | **22** | **22** (Teacher와 동일) | **11** (TA의 50%) |
| **Hidden Size** | 768 | **384** (축소) | **384** (TA와 동일) |
| **Bottleneck I/O** | - | **128** | **128** |
| **Attn Heads** | 12 | **6** | **6** |
| **전략 키워드** | Base Training | Width Compression | Depth Compression |

### 3.2. 단계별 상세 전략

#### **Step 1. Teacher Assistant (TA) 학습**

* **전략:** 너비(Width) 축소, 깊이(Depth) 유지.
* **의도:** Teacher와 레이어 수가 동일하여 1:1 매핑 학습이 유리하다. Hidden Size를 줄여 모델 용량을 1차적으로 경량화한다.
* **기대 효과:** Teacher와 Student 사이의 급격한 구조 차이를 완화하여, Student가 학습하기 좋은 정제된 타겟을 제공한다.

#### **Step 2. Student (Target) 학습**

* **전략:** 너비(Width) 유지, 깊이(Depth) 축소.
* **의도:** TA 모델과 벡터 차원이 동일하여 매핑이 용이하다. 레이어 수를 절반으로 줄여 실질적인 추론 속도(Latency)를 확보한다.
* **특이사항:** MiniLM 방식을 적용하므로 레이어 간 1:1 매핑이나 스킵 매핑 없이, **마지막 레이어(Last Layer)**의 지식을 집중적으로 증류한다.
* **기대 효과:** TA가 정제한 정보를 바탕으로 학습하여 수렴 속도가 빠르고 정확도 손실을 최소화한다.

---

## 4. 지식 증류 전략 (Knowledge Distillation Strategy)

구조가 다른 모델 간의 효과적인 지식 전달을 위해 **MiniLM (Deep Self-Attention Distillation)** 방식을 적용한다. 단순한 출력값(Logits) 모방을 넘어, 내부의 정보 처리 흐름을 모방한다.

### 4.1. Loss Function 구성

최종 Loss는 어텐션 분포와 값의 관계를 모두 고려하여 합산한다.


### 4.2. Attention Distribution Transfer ()

* **개념:** Teacher가 문맥에서 "어떤 단어를 중요하게 보는지"를 모방한다.
* **방법:** **마지막 레이어(Last Layer)**의 Query()와 Key() 내적 분포(Softmax 결과) 간의 KL-Divergence를 계산한다.



### 4.3. Self-Attention Value-Relation Transfer ()

* **개념:** 정보()들이 서로 "어떻게 결합되는지"에 대한 관계(Relation)를 모방한다.
* **방법:** **마지막 레이어(Last Layer)**의 Value() 벡터들 간의 내적 분포 간의 KL-Divergence를 계산한다.


* **선정 이유 및 장점:**
1. 
**레이어 매핑 불필요:** 마지막 레이어의 정보만 사용하므로, 학생 모델의 레이어 깊이에 상관없이 유연한 학습이 가능하다.


2. **차원 불일치 해결:**  연산 결과 행렬은 `Sequence Length x Sequence Length` 크기를 가지므로, Hidden Size가 달라도(768 vs 384) 추가적인 선형 변환(Projection Matrix) 없이 직접 비교가 가능하다.
3. **성능 향상:** 단순히 어디를 볼지(Attention) 뿐만 아니라, 정보의 합성 방식(Relation)까지 학습하므로 더 깊은 수준의 문맥 이해가 가능하다.



---

## 5. 학습 파이프라인 (Pipeline)

1. **데이터셋 구성:**
* 보이스피싱 오디오 전처리 데이터 및 일반 대화 데이터 (KorCCVi_v2).


2. **Phase 0: Teacher Training**
* ModernBERT-base 모델 Fine-tuning.


3. **Phase 1: TA Distillation**
* Teacher  TA (22L, 384H)
* Loss: 
* *Note: TA는 Teacher와 깊이가 같으므로 Hidden State MSE를 추가하여 학습 효율을 높일 수 있음.*


4. **Phase 2: Student Distillation**
* TA  Student (11L, 384H)
* Loss:  (Last Layer Only)
* *Note: 레이어 수 차이로 인한 매핑 문제를 피하기 위해 MiniLM 본연의 방식인 마지막 레이어 증류만 수행.*



## 6. 평가 지표 (Evaluation Metrics)

1. **Detection Performance:** F1-Score, AUC (Teacher 모델 대비 성능 보존율 95% 이상 목표)

---
# KoELECTRA-Mamba 구조 개선 계획서 (Last State → Average Pooling)

## 1. 배경 및 문제 정의
최근 `koelectra_mamba_phishing5` 모델을 학습 및 평가한 결과, Mamba의 뛰어난 장기 기억력(Long-term Memory)과 대화 흐름(Flow) 반영 능력에도 불구하고, 단순한 Average Pooling을 사용한 베이스라인(`koelectra_phishing5`)보다 전체적인 분류 성능이 하락하는 현상(Macro F1: 0.8090 → 0.7438)이 발생했습니다. 

분석 결과, 이는 Mamba 모델 구조 자체의 결함이 아니라 **"Mamba의 출력값을 가져다 쓰는 방식(Architecture)"**에서 비롯된 **최신 편향(Recency Bias) 및 정보 병목 현상** 때문임이 밝혀졌습니다.

## 2. 현재 구조의 한계점
현재 Mamba 기반 분류기의 아키텍처는 전형적인 `Many-to-One` RNN 분류기 구조를 따르고 있습니다.
1. **과정:** `[KoELECTRA]` ➡️ `[Mamba로 순차적 상태 업데이트]` ➡️ `[마지막 시점(Last State)의 Logit 추출]` ➡️ `[최종 판단]`
2. **한계점:** 보이스피싱 대화는 중간에 피싱 의도(계좌 요구 등)가 등장하고, 대화의 마지막은 평범한 종료 인사("네, 알겠습니다. 감사합니다.")로 끝나는 경우가 대부분입니다. 현재 구조는 오직 '마지막 상태'에 의존하므로, 모델이 대화 후반부의 무의미한 일상 대화에 압도되어 중간에 등장한 결정적 피싱 단서(지뢰)를 희석시키거나 망각하게 됩니다. 정상 클래스 간의 극심한 오분류(Confusion)도 종결 어미가 유사하기 때문에 발생한 현상입니다.

## 3. 구조 변경 제안 (Mamba + Average Pooling)
지훈님의 초기 의도인 **"시계열 흐름을 반영하기 위한 Mamba"**의 장점을 살리면서, 동시에 **"어느 시점에서든 피싱 키워드가 등장하면 놓치지 않는 탐지력"**을 결합하기 위해 구조를 아래와 같이 변경합니다.

* **변경 후 아키텍처:** 
  `[KoELECTRA]` ➡️ `[Mamba로 순차적 상태 업데이트]` ➡️ **`[Mamba가 출력한 모든 시점(T)의 출력값들을 Average Pooling (평균)]`** ➡️ `[최종 판단]`

### 주요 로직 변경
- 기존: Mamba를 통과한 결과 $Y = [y_1, y_2, ..., y_T]$ 중 $y_T$ (마지막 상태)만 추출하여 Classifier 통과.
- 수정: $Y$의 유효한(Padding 제외) 모든 상태를 평균 내어 $y_{avg} = \frac{1}{T} \sum y_t$ 를 구한 뒤 Classifier 통과.
- 효과: Mamba가 $y_t$를 만들 때 앞선 문맥($y_{t-1}, y_{t-2}$)을 참고하여 더 풍부한 흐름 정보를 담게 되며, Average Pooling이 적용되어 대화 초/중/후반 어디서 피싱 키워드가 등장하더라도 그 신호가 최종 분류기로 안전하게 전달됩니다.

## 4. 기대 효과
1. **상호 보완적 시너지 극대화:** 트랜스포머(KoELECTRA)가 잡아낸 '지역적 문맥'을 Mamba가 '시간적 흐름'으로 연결해주고, Average Pooling이 '전역적 피싱 탐지력'을 보장하는 완전체 아키텍처 완성.
2. **정상 대화 오분류 해소:** 마지막 인사말의 형태원(State)에만 의존하지 않고 전체 대화 내용을 고르게 반영하므로, 상품 가입/이체/잔고 등 정상 대화 간의 오분류가 획기적으로 감소할 것으로 예상.

## 5. 실행 계획 (Action Items)
1. `models/classifier/experiments/koelectra_mamba_phishing5/model.py` 수정
   - `_gather_last_logits` 메서드 호출부를 제거.
   - `segment_mask`를 활용하여 유효한 시점(T)의 Mamba 출력물(`y`)에 대해 Masked Average Pooling 수행.
2. `train.py` 실행하여 새 구조로 학습 및 평가.
3. 베이스라인(F1 0.8090)과의 성능 재비교.

# 음성 의도 분류 파이프라인 구축 작업 계획

> 목표: 음성 입력 → Whisper STT → BERT 이진 의도 분류 파이프라인 구축  
> 현재 상황: 학습용 음성 데이터 원본 보유, 수량 부족으로 학습 불가 상태

---

## 작업 우선순위 요약

| 우선순위 | 작업 | 이유 |
|:---:|---|---|
| 🔴 P1 | Whisper 모델 선정 (CPU/GPU 두 버전) | 모든 후속 작업의 기준이 됨 |
| 🔴 P1 | 보유 음성 데이터 Whisper 전사 | 데이터셋 구성의 시작점 |
| 🔴 P1 | Whisper 오류 분포 측정 (버전별 별도) | 노이즈 주입 확률값의 근거 |
| 🟠 P2 | LLM Few-shot 데이터 증강 | 표현 다양성 확보 |
| 🟠 P2 | 혼합 확률 모델 기반 ASR 노이즈 주입 | Whisper 전사 스타일 반영 |
| 🟡 P3 | 데이터셋 분할 및 검수 | 학습 준비 완료 |
| 🟡 P3 | BERT 모델 선정 및 파인튜닝 | 분류 모델 학습 |
| 🟢 P4 | 전체 파이프라인 통합 및 검증 | 서비스 연결 |

---

## P1 — 즉시 착수

### 1-1. Whisper 모델 선정

**왜 가장 먼저 해야 하는가**  
학습 데이터 생성에 사용하는 Whisper 모델과 실제 서비스에서 사용하는 모델이 반드시 동일해야 한다.
모델이 달라지면 전사 스타일(띄어쓰기, 오인식 패턴)이 달라지고, 학습 데이터와 추론 입력 간 도메인 갭이 발생한다.

**선택 기준**

| 모델 | 크기 | 한국어 WER | CPU 추론 (5초 발화) | GPU 추론 (5초 발화) | 권장 환경 |
|---|---|---|---|---|---|
| tiny (int8) | 39MB | ~18% | ~0.4s | ~0.1s | 빠른 프로토타이핑 |
| base (int8) | 74MB | ~10% | ~0.8s | ~0.2s | **CPU 기본 선택** |
| small (int8) | 244MB | ~6% | ~2.5s | ~0.4s | **GPU 기본 선택** |
| medium (int8) | 769MB | ~4% | ~6s | ~0.8s | GPU 여유 있을 때 |
| large-v3 | 1.5GB | ~3% | 비현실적 | ~1.2s | GPU 최고 품질 |

**CPU / GPU 두 버전 병행 개발 전략**

데모 환경이 확정되지 않았으므로, 두 버전을 처음부터 분리하여 개발한다.
핵심 원칙은 **데이터셋은 하나, 모델 가중치는 두 벌**이다.
두 버전 모두 동일한 학습 데이터를 사용하되, 추론 시 로드하는 Whisper 모델만 달라진다.

| 구분 | CPU 버전 | GPU 버전 |
|---|---|---|
| Whisper 모델 | `base` (int8) | `small` (float16) |
| 데이터셋 생성 기준 | CPU 버전 전사 결과 | GPU 버전 전사 결과 |
| BERT 학습 데이터 | CPU용 별도 구성 | GPU용 별도 구성 |
| 비고 | 데모 환경 GPU 없을 때 | 데모 환경 GPU 있을 때 |

> ⚠️ 주의: CPU 버전과 GPU 버전은 Whisper 전사 결과가 다르므로 학습 데이터를 공유하면 안 된다.  
> 각 버전의 Whisper로 전사한 데이터로 각각 별도의 BERT를 파인튜닝해야 한다.

```python
from faster_whisper import WhisperModel

def load_whisper(use_gpu: bool):
    if use_gpu:
        return WhisperModel("small", device="cuda", compute_type="float16")
    else:
        return WhisperModel("base", device="cpu", compute_type="int8")

# 사용 예시
model = load_whisper(use_gpu=False)  # 환경에 따라 True/False 전환
segments, _ = model.transcribe(audio_path, language="ko")
text = " ".join([s.text for s in segments])
```

---

### 1-2. 보유 음성 데이터 Whisper 전사

**목적**: 정답 레이블 부착의 기준 텍스트 생성 + 오류 분포 측정용 기반 데이터 확보

**작업 순서**
1. 보유한 음성 파일 전체를 1-1에서 선정한 Whisper 모델로 일괄 전사
2. 전사 결과를 `(파일명, 전사 텍스트)` 형태로 저장
3. 일부 샘플(약 50개)은 사람이 직접 정답 텍스트도 병행 작성 → 오류 분포 측정에 사용

```python
import csv
from faster_whisper import WhisperModel

model = WhisperModel("base", device="cpu", compute_type="int8")

results = []
for audio_path in audio_files:
    segments, _ = model.transcribe(audio_path, language="ko")
    text = " ".join([s.text for s in segments])
    results.append({"file": audio_path, "transcription": text})

# CSV로 저장
with open("transcriptions.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "transcription"])
    writer.writeheader()
    writer.writerows(results)
```

---

### 1-3. Whisper 오류 분포 측정

**왜 필요한가**  
룰 기반 ASR 노이즈 주입의 확률값을 임의로 설정하면 실제 Whisper 전사 스타일과 달라진다.
실제 측정값 기반으로 확률을 설정해야 증강 데이터가 실제 추론 입력과 같은 분포를 가진다.

> ⚠️ CPU 버전 / GPU 버전 각각에 대해 별도로 오류 분포를 측정해야 한다.  
> 같은 음성이라도 두 모델의 전사 결과가 다르므로 오류 패턴도 다르다.

**시작 샘플 수 및 클래스 비율 기준**

오류 분포 측정용 정답 쌍은 보유 데이터의 **자연 클래스 비율을 그대로 반영**해서 구성한다.
측정 대상은 의도 클래스가 아니라 Whisper의 전사 오류이므로, 클래스 균형을 억지로 맞출 필요가 없다.
단, 클래스별로 오류 패턴이 다를 수 있으므로 어느 한 클래스에만 치우치지 않도록 최소 기준을 지킨다.

| 항목 | 권장값 | 이유 |
|---|---|---|
| 시작 샘플 수 | 클래스별 최소 30개 (총 60개~) | 30개 미만은 통계적으로 불안정 |
| 증가 단위 | 10개씩 | 수렴 곡선을 세밀하게 관찰하기 위함 |
| 목표 총 샘플 수 | 200~300개 | 이 구간에서 ±0.01 이내 수렴 일반적 |
| 클래스 비율 | 보유 데이터 자연 비율 유지 | 측정 목적이 전사 오류이므로 클래스 균형 불필요 |
| 수렴 기준 | 연속 3구간(30개) 동안 오류율 변화 ±0.01 이하 | 이 조건 충족 시 측정 종료 |

```python
from collections import defaultdict
import jiwer
import matplotlib.pyplot as plt

def measure_error_distribution(ground_truth_list, whisper_output_list):
    error_counts = defaultdict(int)
    total_tokens = 0
    convergence_log = []  # 수렴 확인용

    for i, (ref, hyp) in enumerate(zip(ground_truth_list, whisper_output_list)):
        result = jiwer.process_words(ref, hyp)
        error_counts['substitution'] += result.substitutions
        error_counts['deletion']     += result.deletions
        error_counts['insertion']    += result.insertions
        total_tokens += len(ref.split())

        # 10개 단위로 현재 오류율 기록
        if (i + 1) % 10 == 0:
            current_probs = {k: v / total_tokens for k, v in error_counts.items()}
            convergence_log.append((i + 1, current_probs.copy()))

    return error_counts, total_tokens, convergence_log

error_counts, total_tokens, log = measure_error_distribution(refs, hyps)
error_probs = {k: v / total_tokens for k, v in error_counts.items()}

# 수렴 시각화
steps = [x[0] for x in log]
sub_probs = [x[1].get('substitution', 0) for x in log]
plt.plot(steps, sub_probs, label='substitution')
plt.axhline(y=sub_probs[-1], color='r', linestyle='--', label='수렴값')
plt.xlabel('샘플 수')
plt.ylabel('오류 확률')
plt.legend()
plt.title('Whisper 오류 분포 수렴 확인')
plt.show()
```

**수렴 기준 자동 판별**

```python
def is_converged(log, window=3, threshold=0.01):
    """최근 window 구간(각 10개 단위)의 변화가 threshold 이하면 수렴으로 판단"""
    if len(log) < window + 1:
        return False
    recent = [entry[1].get('substitution', 0) for entry in log[-(window+1):]]
    return max(recent) - min(recent) < threshold

print("수렴 여부:", is_converged(log))
```

---

## P2 — P1 완료 후 착수

### 2-1. LLM Few-shot 데이터 증강

**목적**: 같은 의도를 표현하는 다양한 구어체 문장 생성 → 표현 다양성 확보

**작업 순서**
1. 전사 데이터에 의도 레이블 부착 (이진: 클래스 A / 클래스 B)
2. 클래스별로 seed 문장 선정
3. LLM API로 각 seed에서 구어체 변형 생성
4. 생성 결과 샘플링 검수 (클래스당 10~20% 확인, 의도 변질 여부 체크)

```python
import openai, json

def augment_with_llm(seed_text, intent_label, n=10):
    prompt = f"""
다음 문장과 동일한 의도를 가진 한국어 구어체 표현 {n}개를 생성해줘.
조건:
- 실제 사람이 말하는 것처럼 자연스럽게
- 문어체, 격식체 사용 금지
- 표현 방식만 바꾸고 의도는 절대 바꾸지 말 것
- 각 문장은 서로 최대한 다르게

원본: {seed_text}
의도: {intent_label}

출력 형식: JSON 배열만 반환, 설명 없이
예시: ["표현1", "표현2", ...]
"""
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(response.choices[0].message.content)
```

**주의사항**
- 생성된 텍스트가 너무 깔끔하면 다음 단계(노이즈 주입)에서 처리됨 → 과도한 정제 불필요
- 의도가 바뀐 샘플은 즉시 제거
- 원본:증강 비율은 **4:6 ~ 5:5** 유지

---

### 2-2. ASR 노이즈 주입 방식 선정 및 적용

**목적**: LLM이 생성한 깔끔한 텍스트에 Whisper 전사 스타일 적용  
**입력값**: 1-3에서 측정한 실제 오류 확률

#### 노이즈 주입 방식 비교

일반적으로 사용되는 세 가지 방식이 있다. 현재 태스크에는 **혼합 확률 모델**이 가장 적합하다.

| 방식 | 원리 | 장점 | 단점 | 적합성 |
|---|---|---|---|---|
| 독립 확률 주입 (Stochastic) | 각 토큰에 독립적으로 오류 확률 적용 | 구현 단순 | 오류 간 상관관계 미반영 | 보통 |
| 마르코프 체인 기반 | 이전 토큰 오류 여부가 다음에 영향 | 연속 오류 패턴 모사 | 파라미터 추정 복잡 | 낮음 |
| 혼합 확률 모델 (Mixture) | 오류 유형 먼저 샘플링 → 세부 변형 적용 | 실제 ASR 오류 분포에 가장 가까움 | 구현 다소 복잡 | **높음** ✓ |

#### 혼합 확률 모델 (권장 방식) 상세

혼합 확률 모델은 두 단계로 작동한다.

1. **오류 유형 결정**: 측정된 오류 확률을 가중치로 삼아 이번 토큰에 어떤 유형의 오류를 적용할지 먼저 샘플링
2. **세부 변형 적용**: 선택된 오류 유형 내에서 구체적인 변형 방식을 다시 샘플링

```python
import random
import numpy as np

class ASRNoiseMixer:
    """
    혼합 확률 모델 기반 ASR 노이즈 주입기
    error_probs: 1-3에서 측정한 실제 오류 분포
    예: {'substitution': 0.08, 'deletion': 0.04, 'insertion': 0.02, 'spacing': 0.12}
    """

    # 음운 유사 대체 사전 (실제 Whisper 오인식 패턴 기반으로 확장 필요)
    PHONETIC_MAP = {
        '되': '돼', '돼': '되',
        '안': '않', '않': '안',
        '에': '애', '애': '에',
        '의': '에', '왜': '왜',
        '로': '으로', '으로': '로',
    }
    PARTICLES = ['을', '를', '이', '가', '은', '는', '도', '에서', '에게']

    def __init__(self, error_probs: dict):
        self.error_probs = error_probs
        # 오류 없음 확률 = 1 - 모든 오류 확률의 합
        total_error = sum(error_probs.values())
        self.no_error_prob = max(0.0, 1.0 - total_error)

    def _sample_error_type(self) -> str:
        """오류 유형을 확률에 따라 샘플링"""
        types = list(self.error_probs.keys()) + ['none']
        weights = list(self.error_probs.values()) + [self.no_error_prob]
        return random.choices(types, weights=weights, k=1)[0]

    def _apply_substitution(self, text: str) -> str:
        """음운 유사 단어로 대체"""
        for src, tgt in self.PHONETIC_MAP.items():
            if src in text:
                return text.replace(src, tgt, 1)
        return text

    def _apply_deletion(self, text: str) -> str:
        """조사/어미 탈락"""
        for p in self.PARTICLES:
            if p in text:
                return text.replace(p, '', 1)
        return text

    def _apply_insertion(self, text: str) -> str:
        """불필요한 어절 삽입 (Whisper가 환각으로 생성하는 패턴)"""
        filler_words = ['그', '어', '음', '저']
        words = text.split()
        if words:
            i = random.randint(0, len(words))
            words.insert(i, random.choice(filler_words))
        return ' '.join(words)

    def _apply_spacing(self, text: str) -> str:
        """띄어쓰기 오류 (붙여쓰기 또는 불필요한 분리)"""
        words = text.split()
        if len(words) < 2:
            return text
        if random.random() < 0.7:  # 70% 확률로 붙여쓰기
            i = random.randint(0, len(words) - 2)
            words[i] = words[i] + words[i+1]
            words.pop(i+1)
        else:  # 30% 확률로 불필요한 분리
            i = random.randint(0, len(words) - 1)
            w = words[i]
            if len(w) > 2:
                split_at = random.randint(1, len(w) - 1)
                words[i] = w[:split_at]
                words.insert(i+1, w[split_at:])
        return ' '.join(words)

    def apply(self, text: str) -> str:
        """오류 유형 샘플링 후 해당 변형 적용"""
        error_type = self._sample_error_type()
        if error_type == 'substitution':
            return self._apply_substitution(text)
        elif error_type == 'deletion':
            return self._apply_deletion(text)
        elif error_type == 'insertion':
            return self._apply_insertion(text)
        elif error_type == 'spacing':
            return self._apply_spacing(text)
        return text  # 'none': 변형 없음


# 사용 예시
error_probs = {'substitution': 0.08, 'deletion': 0.04,
               'insertion': 0.02, 'spacing': 0.12}  # 1-3 측정값 사용

mixer = ASRNoiseMixer(error_probs)
augmented_texts = [mixer.apply(text) for text in llm_generated_texts]
```

#### 검증: 증강 데이터 분포가 실제 분포와 얼마나 가까운가

노이즈 주입 후 증강 데이터의 오류 분포를 다시 측정하여 실제 Whisper 오류 분포와 비교한다.

```python
from scipy.stats import entropy
import numpy as np

def kl_divergence(real_probs: dict, aug_probs: dict) -> float:
    """KL divergence로 두 분포의 유사도 측정. 0에 가까울수록 유사."""
    keys = list(real_probs.keys())
    p = np.array([real_probs.get(k, 1e-9) for k in keys])
    q = np.array([aug_probs.get(k, 1e-9) for k in keys])
    p /= p.sum()
    q /= q.sum()
    return float(entropy(p, q))

kl = kl_divergence(error_probs, measured_aug_probs)
print(f"KL divergence: {kl:.4f}")
# 0.05 이하면 실제 분포와 충분히 유사한 것으로 판단
```

> 💡 PHONETIC_MAP은 1-3 측정에서 발견된 실제 Substitution 오류 쌍으로 지속적으로 확장해야 한다.  
> 일반적인 음운 지식 기반보다 실제 데이터에서 추출한 패턴이 훨씬 정확하다.

---

## P3 — P2 완료 후 착수

### 3-1. 데이터셋 분할 및 최종 검수

**분할 기준**: 반드시 **화자 단위**로 분리 (같은 화자가 Train/Test에 동시에 포함되면 과적합 평가 오류 발생)

| 분할 | 비율 | 조건 |
|---|---|---|
| Train | 70~80% | 다양한 화자, 다양한 노이즈 수준 |
| Validation | 10~15% | Train과 화자 겹침 없음 |
| Test | 10~15% | Train/Val과 화자 겹침 없음, 녹음 환경도 다르게 |

**클래스 균형 확인**
- 목표 비율: 50:50 ~ 70:30
- 불균형 심할 경우: 다수 클래스 언더샘플링 또는 소수 클래스 추가 증강

---

### 3-2. BERT 모델 선정 및 파인튜닝

**후보 모델 비교** (이진분류 + 짧은 발화 기준)

| 모델 | 특징 | 권장 상황 |
|---|---|---|
| `klue/roberta-small` | 68M, 빠름, 한국어 특화 | 속도 우선, 리소스 제한 |
| `klue/bert-base` | 110M, 범용 한국어 | 균형 잡힌 선택 |
| `snunlp/KR-ELECTRA` | Replaced Token Detection, 분류 강점 | 정확도 우선 |

> 참고: BERT 계열 모델은 입력 길이 확장이 아닌 사전학습 목표 재설계(RoBERTa, ELECTRA)와  
> Attention 메커니즘 개선(DeBERTa)으로 성능을 향상시켜 왔음.  
> 짧은 발화에서는 시퀀스 길이보다 각 토큰 간 관계를 얼마나 정밀하게 이해하느냐가 중요.

**파인튜닝 기본 구성**

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "snunlp/KR-ELECTRA-discriminator"  # 또는 klue/roberta-small

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2  # 이진분류
)

# 신뢰도 기반 fallback
def predict(text, threshold=0.6):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    score, label = probs.max(dim=-1)
    if score.item() < threshold:
        return "unknown"  # fallback
    return label.item()
```

---

## P4 — 전체 파이프라인 통합

### 최종 파이프라인 구성

```
마이크 / 음성 파일
    ↓
VAD (Silero VAD) — 발화 구간 감지, 침묵 스킵
    ↓
Whisper (선정한 모델, int8) — 음성 → 텍스트
    ↓
텍스트 전처리 — 필러어 제거, 정규화
    ↓
BERT 분류 모델 — 의도 레이블 + 신뢰도 점수
    ↓
출력 {"intent": "...", "score": 0.94}
    (score < 0.6 → fallback 처리)
```

### 검증 체크리스트

- [ ] Test셋 Accuracy / F1 score 목표치 설정 및 달성 여부 확인
- [ ] 실제 음성으로 end-to-end 지연 시간 측정 (목표: 1.5초 이하)
- [ ] 신뢰도 낮은 샘플(score < 0.6) 비율 확인
- [ ] 학습 데이터에 없는 새로운 표현에 대한 robustness 테스트

---

## 참고: 핵심 원칙 요약

1. **Whisper 모델을 먼저 고정하라** — CPU/GPU 버전 각각 고정, 데이터셋도 버전별로 별도 구성
2. **학습 데이터는 Whisper 전사 결과 기준으로 만들어라** — 깔끔한 정답 텍스트가 아님
3. **ASR 오류 패턴은 실측값 기반 혼합 확률 모델로 주입하라** — 임의 확률값, 독립 확률 주입 지양
4. **오류 분포 측정은 클래스별 최소 30개(총 60개~)에서 시작하라** — 수렴 기준: 연속 3구간 ±0.01 이하
5. **데이터셋 분할은 화자 단위로 하라** — 랜덤 분할은 과적합 평가 오류 유발
6. **모델 크기보다 데이터 품질이 먼저다** — 이진분류에서 SOTA 대형 모델은 과투자

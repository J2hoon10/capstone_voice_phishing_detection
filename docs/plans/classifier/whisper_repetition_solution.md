# Faster Whisper STT 단어 반복(Repetition Loop) 문제 해결 방안

## 1. 문제 정의

Whisper 계열 모델에서 발음이 뭉개지거나 말이 빨라지는 구간을 처리할 때, 디코더가 동일 토큰에 갇혀 같은 단어를 반복 출력하는 현상이 발생한다. 이는 Whisper의 autoregressive 디코딩 구조에서 비롯되는 알려진 문제로, 입력 신호가 불명확할 때 모델이 가장 확률이 높은 토큰을 반복 선택하면서 루프에 빠지는 것이 원인이다.

### 발생 조건

- 화자의 발음이 불명확하거나 뭉개지는 구간
- 말 속도가 급격히 빨라지는 구간
- 배경 잡음이 심한 구간
- 작은 모델(small 이하)에서 더 빈번하게 발생

---

## 2. 해결 방안

### 2.1 디코딩 파라미터 튜닝 (1차 대응)

Faster Whisper의 `transcribe()` 호출 시 아래 파라미터를 조정하여 반복을 억제할 수 있다.

#### 핵심 파라미터

| 파라미터 | 기본값 | 권장값 | 설명 |
|---------|--------|--------|------|
| `repetition_penalty` | 1.0 | 1.1 ~ 1.3 | 이미 출력된 토큰의 확률에 페널티를 부여하여 반복을 억제 |
| `no_repeat_ngram_size` | 0 | 2 ~ 3 | 지정된 크기의 n-gram이 반복 출력되지 않도록 차단 |
| `condition_on_previous_text` | True | False | 이전 세그먼트 텍스트를 다음 세그먼트 컨텍스트로 사용하지 않음 |
| `compression_ratio_threshold` | 2.4 | 2.0 ~ 2.2 | 반복 감지 민감도 조절 (낮을수록 민감) |
| `log_prob_threshold` | -1.0 | -0.8 ~ -1.0 | 품질 낮은 세그먼트 필터링 기준 |

#### 보조 파라미터

| 파라미터 | 기본값 | 권장값 | 설명 |
|---------|--------|--------|------|
| `temperature` | 0 | (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) | 튜플 설정 시 품질 미달 세그먼트를 높은 temperature로 자동 재시도 |
| `beam_size` | 5 | 3 ~ 5 | 빔 서치 크기. 낮추면 속도 향상, 너무 낮추면(1) 반복 증가 가능 |
| `best_of` | 1 | 1 ~ 3 | 후보 수. 높이면 품질 향상되나 속도 저하 |

#### 적용 예시

```python
from faster_whisper import WhisperModel

model = WhisperModel("medium", device="cuda", compute_type="float16")

segments, info = model.transcribe(
    audio_path,
    language="ko",

    # 반복 방지 핵심
    repetition_penalty=1.15,
    no_repeat_ngram_size=3,
    condition_on_previous_text=False,

    # 품질 제어
    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold=2.2,
    log_prob_threshold=-0.8,

    # 디코딩 설정
    beam_size=3,
    vad_filter=True,
    vad_parameters=dict(
        min_silence_duration_ms=300,
        speech_pad_ms=200,
    ),
)
```

#### 파라미터별 상세 설명

**repetition_penalty (1.1 ~ 1.3)**

가장 직접적인 해결책이다. 디코딩 과정에서 이미 생성된 토큰의 logit 값을 이 수치로 나눠 확률을 낮춘다. 1.0은 페널티 없음, 1.15 정도가 한국어 통화 음성에서 반복 억제와 자연스러운 출력 사이의 균형점이다. 1.3을 초과하면 필요한 반복(예: "네 네")까지 억제될 수 있으므로 주의해야 한다.

**no_repeat_ngram_size (2 ~ 3)**

지정한 크기의 n-gram이 출력에서 두 번 이상 나타나지 않도록 강제한다. 값이 2이면 같은 2-gram(연속 2개 토큰)이 반복되지 않고, 3이면 3-gram 반복이 차단된다. 값이 너무 크면 정상적인 문장 구조까지 제한될 수 있으므로 2~3이 적당하다.

**condition_on_previous_text = False**

Whisper는 기본적으로 이전 세그먼트의 출력 텍스트를 다음 세그먼트의 프롬프트로 사용한다. 반복이 한 세그먼트에서 발생하면 이 메커니즘을 통해 다음 세그먼트로 전파될 수 있다. False로 설정하면 전파를 차단하지만, 세그먼트 간 문맥 연결이 약해질 수 있다는 트레이드오프가 존재한다.

**compression_ratio_threshold (2.0 ~ 2.2)**

Whisper는 생성된 텍스트의 압축률(gzip 기반)을 계산하여 반복 여부를 판단한다. 반복이 많으면 압축률이 높아지므로, 이 임계값을 초과하면 해당 세그먼트를 "실패"로 판정하고 더 높은 temperature로 재시도한다. 기본값 2.4에서 2.0~2.2로 낮추면 반복 감지가 더 민감해진다.

**temperature 튜플**

튜플로 지정하면 첫 번째 값으로 시도 후, compression_ratio나 log_prob 기준을 충족하지 못하면 다음 값으로 재시도한다. 0.0(greedy)에서 시작하여 점진적으로 랜덤성을 높이는 방식으로, 반복 루프에 빠진 세그먼트를 자동으로 복구할 수 있다.

---

### 2.2 후처리를 통한 반복 제거 (2차 대응)

파라미터 튜닝으로 완전히 해결되지 않는 경우를 대비한 안전망이다.

#### 정규식 기반 반복 제거

```python
import re

def remove_repetitions(text: str) -> str:
    """STT 결과에서 연속 반복되는 단어/구문을 제거"""

    # 1) 동일 어절 3회 이상 연속 반복 → 1회로 축소
    #    예: "네 네 네 네" → "네"
    text = re.sub(r'(\S+)(\s+\1){2,}', r'\1', text)

    # 2) 동일 2어절 구문 2회 이상 반복 → 1회로 축소
    #    예: "그래서 말인데 그래서 말인데" → "그래서 말인데"
    text = re.sub(r'((\S+\s+\S+)\s+)\1+', r'\1', text)

    # 3) 동일 3어절 구문 반복 → 1회로 축소
    #    예: "제가 생각하기에는 제가 생각하기에는" → "제가 생각하기에는"
    text = re.sub(r'((\S+\s+\S+\s+\S+)\s+)\1+', r'\1', text)

    return text.strip()
```

#### 신뢰도 기반 필터링

Whisper는 세그먼트별 `avg_logprob`을 제공한다. 이 값이 낮은 세그먼트는 모델이 확신하지 못하는 구간이므로 별도 표시하거나 필터링할 수 있다.

```python
def filter_low_confidence_segments(segments, threshold=-0.8):
    """신뢰도 낮은 세그먼트에 [불명확] 태그 추가"""
    results = []
    for seg in segments:
        text = seg.text.strip()
        if seg.avg_logprob < threshold:
            text = f"[불명확] {text}"
        results.append({
            "start": seg.start,
            "end": seg.end,
            "text": text,
            "confidence": round(2 ** seg.avg_logprob, 3)
        })
    return results
```

---

### 2.3 VAD(Voice Activity Detection) 활용 (입력 단계 최적화)

반복 문제는 무음이나 잡음 구간에서도 자주 발생한다. VAD를 사용하여 실제 발화 구간만 Whisper에 전달하면 불필요한 구간에서의 반복 생성을 원천적으로 방지할 수 있다.

#### Faster Whisper 내장 VAD 사용

```python
segments, info = model.transcribe(
    audio_path,
    language="ko",
    vad_filter=True,
    vad_parameters=dict(
        min_silence_duration_ms=300,   # 300ms 이상 무음 시 세그먼트 분리
        speech_pad_ms=200,             # 발화 전후 200ms 패딩
        threshold=0.5,                 # VAD 감도 (낮을수록 민감)
    ),
)
```

#### 외부 VAD (Silero VAD)를 사용한 더 정밀한 분리

통화 음성처럼 잡음이 많은 환경에서는 Silero VAD로 발화 구간을 먼저 추출한 뒤 각 구간만 Whisper에 전달하는 방식이 더 안정적이다.

```python
import torch
import numpy as np

def extract_speech_segments(audio: np.ndarray, sr: int = 16000):
    """Silero VAD로 발화 구간만 추출"""
    model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')
    get_speech_timestamps = utils[0]

    audio_tensor = torch.from_numpy(audio).float()
    timestamps = get_speech_timestamps(
        audio_tensor, model,
        sampling_rate=sr,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=300,
    )

    segments = []
    for ts in timestamps:
        start_sec = ts['start'] / sr
        end_sec = ts['end'] / sr
        segment_audio = audio[ts['start']:ts['end']]
        segments.append({
            "audio": segment_audio,
            "start": start_sec,
            "end": end_sec,
        })

    return segments
```

---

### 2.4 모델 스케일업 (근본적 개선)

작은 모델일수록 불명확한 입력에서 반복 루프에 빠지기 쉽다. 환경이 허용하는 범위에서 모델을 키우는 것이 가장 근본적인 해결책이다.

#### 모델별 반복 문제 발생 빈도 (한국어 통화 음성 기준)

| 모델 | 파라미터 | VRAM (float16) | 반복 문제 빈도 | 권장 환경 |
|------|----------|----------------|---------------|----------|
| tiny | 39M | ~0.5GB | 매우 빈번 | 프로토타입용 |
| base | 74M | ~0.7GB | 빈번 | 경량 환경 |
| small | 244M | ~1.5GB | 종종 발생 | CPU / 저사양 GPU |
| medium | 769M | ~3.5GB | 크게 감소 | RTX 3060 이상 |
| large-v3 | 1.5B | ~6.5GB | 거의 없음 | RTX 3080 이상 |

#### 양자화를 통한 속도 보완

모델을 키우면서도 속도를 유지하려면 CTranslate2의 양자화 옵션을 활용한다.

```python
# medium 모델 + int8_float16 양자화
# → small 수준의 속도에 근접하면서 medium의 정확도 유지
model = WhisperModel("medium", device="cuda", compute_type="int8_float16")
```

| compute_type | 속도 (상대값) | 정확도 영향 | VRAM |
|-------------|-------------|------------|------|
| float16 | 1.0x (기준) | 없음 | 기준 |
| int8_float16 | ~1.3x 빠름 | 거의 없음 | ~20% 절약 |
| int8 | ~1.5x 빠름 | 미미한 저하 | ~30% 절약 |

---

## 3. 권장 적용 순서

반복 문제 해결을 위한 단계별 적용 전략이다. 이전 단계에서 충분히 해결되면 다음 단계는 생략해도 된다.

```
[1단계] 파라미터 튜닝
  - repetition_penalty=1.15, no_repeat_ngram_size=3 적용
  - compression_ratio_threshold=2.2로 조정
  - condition_on_previous_text=False 설정
         ↓
  반복 문제가 여전히 발생하는가?
         ↓
[2단계] VAD 적용
  - vad_filter=True 또는 Silero VAD로 발화 구간만 추출
  - 무음/잡음 구간에서의 불필요한 반복 제거
         ↓
  반복 문제가 여전히 발생하는가?
         ↓
[3단계] 후처리 적용
  - 정규식 기반 반복 구문 제거
  - 신뢰도 기반 저품질 세그먼트 태깅
         ↓
  품질이 전반적으로 부족한가?
         ↓
[4단계] 모델 스케일업
  - small → medium (또는 medium + int8_float16 양자화)
  - 하드웨어 여유가 있다면 large-v3까지 고려
```

---

## 4. 통합 적용 코드

위의 모든 방안을 하나의 파이프라인으로 통합한 예시이다.

```python
from faster_whisper import WhisperModel
import re
import numpy as np


def create_optimized_model(
    model_size: str = "medium",
    device: str = "cuda",
    compute_type: str = "float16"
) -> WhisperModel:
    """최적화된 Whisper 모델 생성"""
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def remove_repetitions(text: str) -> str:
    """후처리: 연속 반복 단어/구문 제거"""
    text = re.sub(r'(\S+)(\s+\1){2,}', r'\1', text)
    text = re.sub(r'((\S+\s+\S+)\s+)\1+', r'\1', text)
    text = re.sub(r'((\S+\s+\S+\s+\S+)\s+)\1+', r'\1', text)
    return text.strip()


def transcribe_with_anti_repetition(
    model: WhisperModel,
    audio_input,
    language: str = "ko"
) -> list[dict]:
    """반복 방지 파라미터가 적용된 STT 수행"""

    segments, info = model.transcribe(
        audio_input,
        language=language,

        # 반복 방지
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        condition_on_previous_text=False,

        # 품질 제어
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold=2.2,
        log_prob_threshold=-0.8,

        # 디코딩
        beam_size=3,

        # VAD
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=300,
            speech_pad_ms=200,
        ),
    )

    results = []
    for seg in segments:
        text = remove_repetitions(seg.text.strip())
        confidence = round(2 ** seg.avg_logprob, 3)

        results.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": f"[불명확] {text}" if seg.avg_logprob < -0.8 else text,
            "confidence": confidence,
        })

    return results


# 사용 예시
if __name__ == "__main__":
    model = create_optimized_model("medium", "cuda", "float16")
    results = transcribe_with_anti_repetition(model, "audio.wav")

    for r in results:
        print(f"[{r['start']:.1f}s ~ {r['end']:.1f}s] "
              f"(신뢰도: {r['confidence']}) {r['text']}")
```

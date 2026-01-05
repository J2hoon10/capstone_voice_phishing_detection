이 문서는 현재 구현된 시스템의 **아키텍처, 최적화 기법, 실험 설정 및 로직**을 포괄합니다.

---

# 🧪 실시간 보이스피싱 탐지 추론 실험 보고서

## 1. 개요 (Overview)

본 실험은 **실시간 통화 데이터**를 스트리밍 방식으로 처리하여 보이스피싱 위험을 탐지하는 시스템의 성능과 속도를 검증하기 위해 수행되었습니다.
고성능의 **Whisper(STT)** 모델과 경량화된 **RoBERTa(NLP)** 모델을 결합하여, **정확도(Accuracy)**를 유지하면서 **실시간성(Low Latency)**을 확보하는 데 중점을 두었습니다.

---

## 2. 시스템 환경 및 모델 설정 (System Configuration)

### 2.1. 하드웨어 및 기반 설정 (`config.py`)

* **Device**: NVIDIA CUDA (GPU 가속 사용)
* **Base Path**: `test_inference` (기준 디렉토리)
* **Output Dir**: `test_inference/model_weight`

### 2.2. 사용 모델

| 구분 | 모델명 / 설정 | 역할 | 비고 |
| --- | --- | --- | --- |
| **STT (음성인식)** | `deepdml/faster-whisper-large-v3-turbo-ct2` | 음성을 텍스트로 변환 | CTranslate2 기반 최적화 모델 |
| **NLP (텍스트분류)** | `klue/roberta-base` (Student Model) | 텍스트의 피싱 위험도 분류 | Teacher(12층) → **Student(6층)** 지식 증류(Distillation) 적용 |

---

## 3. 핵심 최적화 기술 (Optimization Techniques)

실시간 처리를 위해 **병목(Bottleneck)**을 제거하고 추론 속도를 극대화한 기법들입니다.

### ⚡ 3.1. STT 가속화 (`audio_processor.py`)

* **Precision**: `float16` (Half Precision) 적용으로 메모리 사용량 감소 및 연산 가속.
* **Decoding Strategy**: `beam_size=1` (Greedy Search) 설정.
* 기본값(5) 대비 약 3~5배 속도 향상.
* 실시간 탐지 특성상 미세한 정확도 저하보다 속도가 우선됨.


* **VAD (Voice Activity Detection)**:
* 외부 VAD 라이브러리(Silero) 대신 **Whisper 내부 `vad_filter=True**` 옵션 사용.
* `no_speech_threshold=0.6`: 비언어(잡음) 구간 자동 필터링.



### ⚡ 3.2. 오디오 전처리 경량화 (`audio_enhancer.py`)

* **Disk I/O 제거**: 오디오 데이터를 파일로 저장했다가 다시 읽는 과정을 제거하고, **Numpy Array를 메모리에서 직접 전달**하는 파이프라인 구축.
* **Heavy Processing 비활성화**:
* 병목이 심한 외부 **VAD(`vad_trim`)** 비활성화.
* **Bandpass Filter** 및 **Normalization**만 유지하여 최소한의 음질 보정 수행.



### ⚡ 3.3. NLP 추론 가속 (`inference.py`)

* **Mixed Precision**: `torch.amp.autocast('cuda')`를 적용하여 RoBERTa 추론 시 FP16 연산 수행.
* **Batch Processing**: 윈도우 단위로 묶어서 배치 처리 (현재 윈도우 크기에 따라 가변적).

---

## 4. 실험 로직 (Experiment Logic)

### 4.1. 슬라이딩 윈도우 (`test_simulation.py`)

실시간 스트리밍을 모사하기 위해 긴 오디오 파일을 일정 간격으로 잘라서 처리합니다.

* **Window Size**: **15초** (한 번에 분석하는 오디오 길이)
* **Stride**: **5초** (다음 분석을 위해 이동하는 간격)
* *특징*: **10초의 중첩(Overlap)** 구간이 발생하여 대화의 문맥을 놓치지 않고 촘촘하게 검사함.



### 4.2. 위험 점수 산정 알고리즘 (`inference.py` - Leaky Bucket)

단건 추론의 불안정성(튀는 값)을 보정하기 위해 누적 점수 시스템을 도입했습니다.

* **초기 점수**: 0점 (범위: 0 ~ 100점)
* **점수 갱신 규칙**:
* 🚨 **High Risk** (확률 > 0.8): **+20점**
* ⚠️ **Medium Risk** (0.5 < 확률 ≤ 0.8): **+10점**
* ✅ **Normal** (확률 ≤ 0.5): **-10점** (점수 차감)


* **경고 레벨**:
* **LEVEL 2 (Warning)**: 60점 이상 → 🚨 즉각 경고 (진동/알림)
* **LEVEL 1 (Caution)**: 30점 이상 → ⚠️ 주의



---

## 5. 프로젝트 파일 구조 및 역할

```text
📂 test_inference/
│
├── 📜 test_simulation.py   # [메인] 전체 시뮬레이션 실행, 속도(FPS) 측정 및 결과 출력
├── 📜 inference.py         # [핵심] 보이스피싱 탐지기 클래스 및 점수 관리(Scorer) 로직
├── 📜 audio_processor.py   # [STT] Whisper 모델 관리, 텍스트 정제(PII 마스킹 등)
├── 📜 audio_enhancer.py    # [전처리] 오디오 필터링, 정규화, 타입 변환(File <-> Numpy)
├── 📜 config.py            # [설정] 모델 경로, 하이퍼파라미터 통합 관리
└── 📜 architecture.py      # [모델] DistillableRoBERTaModel 구조 정의 (KD 지원)

```

## 6. 성능 지표 (Performance Metrics)

실험 실행 시 콘솔에 다음과 같은 지표가 실시간으로 출력됩니다.

1. **Latency (지연 시간)**: 오디오 1개 청크(Chunk)를 처리하는 데 걸리는 시간 (초).
* 목표: Stride(5초)보다 현저히 낮아야 함 (권장: < 0.5초).


2. **FPS (Frames Per Second)**: 초당 처리 횟수.
3. **Risk Score**: 현재 누적된 위험 점수.

### 6.1 프로파일링 (Profiling)

각 단계별 소요 시간을 로그로 출력하여 병목 구간을 모니터링합니다.

* `[Profile] Enhance`: 오디오 전처리 시간
* `[Profile] Whisper STT`: 음성 인식 시간
* `[Profile] RoBERTa NLP`: 텍스트 분류 시간

---

*작성일: 2026-01-03*
*작성자: Gemini (AI Assistant)*
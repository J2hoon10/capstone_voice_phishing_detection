# 스트리밍 추론 아키텍처

> 기준 구현: `models/main/model_architecture/streaming_inference.py`,
> `streaming_test/run_pipeline.py`  
> 최종 업데이트: 2026-05-27

---

## 1. 전체 파이프라인

```
┌─────────────────────────────────────────────────────────────┐
│ 음성 입력                                                     │
│  - 파일(mp3/wav) 또는 실시간 스트림                           │
│  - N초 단위 청크로 분할 (권장: 15초 청크 + 8초 오버랩)         │
└───────────────────────────┬─────────────────────────────────┘
                            │ audio slice (numpy array, 16 kHz mono)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Whisper STT  (faster-whisper small, 한국어, VAD 필터)         │
│  - 오버랩 구간은 컨텍스트로만 사용, 분류기에는 미전달           │
│  - 출력: 세그먼트 단위 텍스트 + 타임스탬프                     │
└───────────────────────────┬─────────────────────────────────┘
                            │ 문장 단위 텍스트 (str)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 슬라이딩 윈도우 생성  (build_segments)                        │
│  WINDOW_SIZE=64 tokens, STRIDE=32, MAX_SEQ_LEN=128           │
│  누적 텍스트 전체를 토큰화 후 윈도우 재분할                    │
└───────────────────────────┬─────────────────────────────────┘
                            │ 신규 윈도우 목록 [{input_ids, attention_mask}, ...]
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ RoBERTa 인코더  (klue/roberta-base)                          │
│  신규 윈도우만 인코딩 → AttentionWeightedPooling              │
│  출력: 윈도우 벡터 (1, 768)  ← 인코딩 캐시에 추가             │
└───────────────────────────┬─────────────────────────────────┘
                            │ 전체 캐시 벡터 스택 (1, T, 768)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Mamba SSM  (2-layer, d_model=768)                            │
│  전체 캐시로 매번 재실행 (병렬 스캔 구조상 step caching 불가)  │
│  출력: 시퀀스 표현 (1, T, 768)                               │
└───────────────────────────┬─────────────────────────────────┘
                            │ 마지막 위치 벡터 (1, 768)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ HierarchicalHead + 온도 스케일링                              │
│  상위 분류: 일반(0) vs 피싱(1)                               │
│  서브 분류: 상담/일상  또는  대출/수사                        │
│  4-class 확률 = P(super) × P(sub|super)                      │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │  4-class 분류 결과       │
              │  상담 대화               │
              │  일상 대화               │
              │  대출 사기형  ← 피싱     │
              │  수사기관 사칭형 ← 피싱  │
              └─────────────────────────┘
```

---

## 2. 음성 입력 처리

### 2.1 청크 + 오버랩 전략

```
시간축:  0s    8s   15s   23s   30s   38s   45s
         ├─────┼─────┼─────┼─────┼─────┼─────┤

청크 0: [■■■■■■■■■■■■■■■]  0~15s (오버랩 없음, 전체 사용)
청크 1:      [▒▒▒▒▒▒▒■■■■■■■■■■■■■■■■]  7~30s
                      ↑ 15s 이후 세그먼트만 분류기에 전달
청크 2:                   [▒▒▒▒▒▒▒■■■■■■■■■■■■■■■■]  23~45s

 ▒ = Whisper 컨텍스트 오버랩 (단어 경계 보호용, 분류기 미전달)
 ■ = 신규 처리 구간 (분류기에 전달)
```

**오버랩 목적**: Whisper는 오디오 경계에서 단어를 잘못 분리하는 경향이 있음.  
이전 M초를 컨텍스트로 포함해 전사 품질을 높이되,  
분류기에는 신규 구간 세그먼트만 전달하여 중복 처리를 방지.

### 2.2 실행 모드

| 모드 | 설명 | 권장 설정 |
|------|------|-----------|
| 전체 파일 모드 | 파일 전체를 Whisper에 한 번에 입력 | 단발성 분석 |
| 청크 모드 | N초 단위 분할, 오버랩 적용 | **실시간 스트리밍** |
| 인터랙티브 모드 | 문장을 한 줄씩 수동 입력 | 디버깅/시연 |

---

## 3. STT: Whisper

| 항목 | 설정 |
|------|------|
| 구현체 | faster-whisper |
| 모델 크기 | small (권장) |
| 입력 형식 | 16 kHz mono numpy array |
| 언어 | 한국어 (`language="ko"`) |
| VAD 필터 | 활성화 (`vad_filter=True`) |
| 출력 | 세그먼트 리스트 (텍스트 + 시작/종료 타임스탬프) |

---

## 4. 슬라이딩 윈도우 (`build_segments`)

새 Whisper 세그먼트가 도착할 때마다 누적 텍스트 전체를 재토큰화하여 윈도우를 재생성합니다.

```
누적 텍스트 전체 토큰 시퀀스:
  [t0, t1, t2, ..., t_n]

윈도우 분할 (WINDOW_SIZE=64, STRIDE=32):
  윈도우 1: [CLS] t0 ~ t61  [SEP]  → 64 tokens (content=62)
  윈도우 2: [CLS] t32 ~ t93 [SEP]
  윈도우 3: [CLS] t64 ~ t125[SEP]
  ...
```

- 문장 경계(`.?!\n`)를 기준으로 윈도우 끝점 정렬 → 문맥 단절 최소화
- 각 윈도우는 MAX_SEQ_LEN=128로 패딩
- 이전에 처리한 윈도우 이후로 생긴 **신규 윈도우만** 인코딩 대상

---

## 5. RoBERTa 인코더 + AttentionWeightedPooling

### 인코딩 흐름

```
윈도우 (input_ids, attention_mask)  shape: (1, 128)
    │
    ▼  klue/roberta-base
last_hidden_state  shape: (1, 128, 768)
    │
    ▼  AttentionWeightedPooling
       - [CLS], [SEP] 제외한 실제 토큰에만 Attention 가중치 부여
       - e_i = W_a(tanh(W_b(h_i)))  →  softmax  →  가중합
윈도우 벡터  shape: (1, 768)
    │
    ▼  인코딩 캐시에 append
```

### 캐싱 전략 (Incremental 추론의 핵심)

- **신규 윈도우만** RoBERTa 인코딩 → 계산 중복 제거
- 기존 윈도우 벡터는 캐시에서 재사용
- `encoded_cache: list[Tensor(1,768)]`로 관리

---

## 6. Mamba SSM (2-layer)

```python
x_stacked = torch.stack(encoded_cache, dim=1)  # (1, T, 768)
y = x_stacked
for mamba, dropout in zip(model.mamba_layers, model.mamba_dropouts):
    y = dropout(mamba(y))                        # (1, T, 768)
```

| 파라미터 | 값 |
|----------|-----|
| d_model | 768 |
| d_state | 16 |
| d_conv | 4 |
| expand | 2 |
| num_layers | 2 |
| dropout | 0.1 |

**Mamba는 매 스텝 전체 캐시로 재실행**됩니다.  
병렬 스캔(associative scan) 구조상 step-by-step caching이 불가능하여,  
새 윈도우가 추가될 때마다 전체 시퀀스를 다시 처리합니다.  
그러나 Mamba의 처리 속도(~18–73 win/s)가 STT 속도보다 훨씬 빠르므로 병목이 되지 않습니다.

---

## 7. HierarchicalHead

```
Mamba 마지막 위치 출력  (1, 768)
    │
    ▼  LayerNorm → Linear(768, 64) → GELU → Dropout
    │
    ├──▶ superclass_head  Linear(64, 2)  →  [일반, 피싱]
    ├──▶ normal_head      Linear(64, 2)  →  [상담 대화, 일상 대화]
    └──▶ phishing_head    Linear(64, 2)  →  [대출 사기형, 수사기관 사칭형]

4-class 확률 결합:
  P(상담 대화)       = P(super=일반)  × P(sub=상담|일반)
  P(일상 대화)       = P(super=일반)  × P(sub=일상|일반)
  P(대출 사기형)     = P(super=피싱)  × P(sub=대출|피싱)
  P(수사기관 사칭형) = P(super=피싱)  × P(sub=수사|피싱)
```

---

## 8. 온도 스케일링 (Temperature Scaling)

초반 윈도우가 적을 때는 텍스트 정보가 부족해 예측이 불안정합니다.  
이를 완화하기 위해 초반 구간에서 temperature를 높여 확률 분포를 평탄화합니다.

```
T(k) = max_temp - (max_temp - 1.0) × (k - 1) / (warmup - 1)  (k < warmup)
T(k) = 1.0                                                      (k ≥ warmup)

기본값: warmup=8, max_temp=4.0

k=1: T=4.0  (매우 평탄)
k=4: T=2.5
k=8: T=1.0  (이후 고정)
```

```
윈도우 수 k에 따른 temperature:
  1  ████████████████  4.0
  2  ████████████     3.1
  4  ████████         2.5
  6  █████            1.9
  8  ███              1.0  ← 이후 고정
```

---

## 9. Incremental 추론 흐름 (step-by-step)

새 Whisper 세그먼트가 도착할 때마다 아래 순서로 실행됩니다.

```
① 새 텍스트 누적
   state["accumulated"].append(new_text)
   full_text = " ".join(accumulated)

② 슬라이딩 윈도우 재생성
   segments = build_segments(tokenizer, full_text)
   new_segs = segments[processed_seg_count:]

   ┌ new_segs가 없으면 → 결과 없음, 다음 세그먼트 대기

③ 신규 윈도우만 RoBERTa 인코딩
   for seg in new_segs:
       encoded_cache.append(encode_window(model, seg))
       processed_seg_count += 1

④ 전체 캐시로 Mamba 실행
   x = stack(encoded_cache)   # (1, T, 768)
   y = mamba_layers(x)

⑤ 예측 출력 (신규 윈도우별)
   for i, seg_idx in enumerate(new_segs):
       temp   = get_temperature(seg_idx + 1)
       probs  = HierarchicalHead(y[:, seg_idx, :])  →  softmax(logits / temp)
       label  = argmax(probs)
```

---

## 10. 파이프라인 성능

> 실험 환경: 20.mp3 (209.9초, 실제 보이스피싱 통화)  
> STT: faster-whisper small | 분류기: roberta_mamba_freeze_init_4class | GPU (float16)

### 청크 크기별 비교

| 설정 | 총 처리 시간 | RTF | 파이프라인 FPS | 첫 판단 시점 |
|------|------------|-----|----------------|------------|
| 30초 + 10초 오버랩 | 62.3초 | 0.297x | 0.39 win/s | 수신 후 ~9초 |
| **15초 + 8초 오버랩** | **14.1초** | **0.067x** | **1.56 win/s** | **수신 후 ~1초** |

**RTF < 1.0**: 실시간 처리 가능 (15초 청크 기준 실시간보다 **15배 빠름**)

### 처리 시간 분포 (15초 청크, GPU)

```
STT  (faster-whisper)  ████████████████████████████████████  95.8%
분류기 (RoBERTa-Mamba) ██                                     4.2%
```

병목은 STT이며, 분류기는 ~37–73 win/s로 STT 대비 충분히 빠릅니다.

### 모델 성능 (roberta_mamba_freeze_init_4class, Test)

| 클래스 | F1 | Precision | Recall |
|--------|-----|-----------|--------|
| 상담 대화 | 1.0000 | 1.0000 | 1.0000 |
| 일상 대화 | 1.0000 | 1.0000 | 1.0000 |
| 대출 사기형 | 0.9876 | 0.9803 | 0.9950 |
| 수사기관 사칭형 | 0.9874 | 0.9949 | 0.9800 |
| **Macro F1** | **0.9937** | | |

---

## 11. 관련 파일

| 파일 | 역할 |
|------|------|
| [`models/main/model_architecture/streaming_inference.py`](../models/main/model_architecture/streaming_inference.py) | 텍스트 기반 스트리밍 추론 (배치/인터랙티브 모드) |
| [`models/main/model_architecture/model.py`](../models/main/model_architecture/model.py) | `StreamingBeliefClassifier` 모델 정의 |
| [`models/main/model_architecture/dataset.py`](../models/main/model_architecture/dataset.py) | `build_segments` 슬라이딩 윈도우 생성 |
| [`models/main/model_architecture/config.py`](../models/main/model_architecture/config.py) | 윈도우/모델 하이퍼파라미터 |
| [`streaming_test/run_pipeline.py`](../streaming_test/run_pipeline.py) | 음성 파일 → Whisper → 분류기 파이프라인 (속도 측정) |
| [`streaming_test/run_pipeline_watch.py`](../streaming_test/run_pipeline_watch.py) | 파일 감시 기반 2스레드 병렬 파이프라인 |
| [`streaming_test/PIPELINE_REPORT.md`](../streaming_test/PIPELINE_REPORT.md) | 파이프라인 실험 상세 보고서 |

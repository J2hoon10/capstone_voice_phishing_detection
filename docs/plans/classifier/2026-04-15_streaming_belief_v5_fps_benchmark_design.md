# Streaming Belief v5 — FPS 추론 속도 벤치마크 설계 및 지표 해석

작성일: 2026-04-15  
대상 실험: `v5_lora_r8`  
스크립트: `models/classifier/experiments/streaming_belief_v5/benchmark.py`

---

## 1. FPS의 정의

이 모델에서 **FPS(Frames Per Second)** 는 영상의 프레임이 아니라  
**128-토큰 슬라이딩 윈도우 세그먼트를 초당 몇 개 처리하는가**를 의미한다.

```
FPS = 처리된 세그먼트 수 / 추론 소요 시간 (초)
```

### 세그먼트란?

대화 텍스트를 128-토큰, stride 100으로 슬라이딩한 청크 하나.  
음성 phishing 감지에서 새로운 발화가 축적될수록 세그먼트가 하나씩 추가된다.  
즉, **세그먼트 = 모델이 belief를 한 번 업데이트하는 단위**.

---

## 2. 측정 모드

벤치마크는 두 가지 시나리오를 구분해 측정한다.

### 2-A. batch 모드 (처리량 측정)

```
DataLoader → [seg₁…segₙ, seg₁…segₘ, …] 배치 단위 → GPU 병렬 처리
```

- DataLoader로 test 셋 전체를 순회하며 배치 단위로 forward
- **GPU 병렬화가 최대로 활용**되는 조건
- 데이터셋 전체를 얼마나 빠르게 훑을 수 있는지 측정 (오프라인 평가 시나리오)

```
FPS_batch = 총 세그먼트 수 / 전체 순회 시간
```

### 2-B. streaming 모드 (실시간 시뮬레이션)

```
샘플 1개 → 세그먼트 1개 → forward → 세그먼트 2개 → forward → … → 세그먼트 N개 → forward
```

- 배치 크기 1, 세그먼트를 1개씩 추가하며 매번 forward
- **실제 서비스 환경과 동일한 조건** — 새 발화가 도착할 때마다 belief 업데이트
- 1 세그먼트 추가 시 latency가 실시간 처리 가능 여부를 결정

```
FPS_streaming = 1 / seg_latency_mean
```

---

## 3. 타이밍 측정 방법

정확한 GPU 추론 시간을 얻기 위해 아래 두 가지를 적용한다.

| 처리 | 이유 |
|---|---|
| `torch.cuda.synchronize()` 호출 후 타이머 시작/종료 | GPU는 비동기 실행이므로 sync 없이 측정하면 CPU 스케줄링 시간만 잡힘 |
| warm-up 배치/샘플 제외 | GPU의 첫 번째 실행은 커널 컴파일·캐시 초기화로 인해 느림 — 안정화 후 측정 |

```python
torch.cuda.synchronize()
t0 = time.perf_counter()
_ = model(...)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
```

---

## 4. 출력 지표 및 해석

### 4-A. batch 모드 지표

| 지표 | 단위 | 설명 |
|---|---|---|
| `avg_fps_segments_per_sec` | seg/s | **핵심 FPS** — 전체 소요 시간 기준 평균 |
| `avg_samples_per_sec` | sample/s | 대화 샘플 처리량 |
| `batch_fps_mean` | seg/s | 배치별 FPS의 평균 |
| `batch_fps_std` | seg/s | 배치별 FPS의 표준편차 (안정성 지표) |
| `batch_fps_p50` | seg/s | 중간값 FPS |
| `batch_fps_p95` | seg/s | 상위 5% 빠른 배치의 FPS (최대 성능 근사) |
| `batch_fps_min` / `_max` | seg/s | 최솟값 / 최댓값 |
| `total_segments` | 개 | 측정에 사용된 총 세그먼트 수 |
| `total_time_sec` | 초 | 전체 측정 시간 |
| `num_batches_measured` | 개 | warm-up 제외 측정 배치 수 |

#### 읽는 법

```
avg_fps_segments_per_sec ≈ batch_fps_p50   →  분포가 안정적
avg_fps_segments_per_sec  < batch_fps_p50  →  일부 느린 배치(긴 시퀀스 등)가 평균을 끌어내림
batch_fps_std / batch_fps_mean > 0.2       →  배치 간 처리 시간 편차가 큰 것 (버킷 샘플러 효과 확인)
```

---

### 4-B. streaming 모드 지표

| 지표 | 단위 | 설명 |
|---|---|---|
| `avg_fps_segments_per_sec` | seg/s | **핵심 FPS** — `1 / seg_latency_mean` |
| `seg_latency_mean_ms` | ms | 세그먼트 1개 추가 시 평균 처리 시간 |
| `seg_latency_std_ms` | ms | latency 표준편차 |
| `seg_latency_p50_ms` | ms | 중간값 latency |
| `seg_latency_p95_ms` | ms | 95%의 요청이 이 시간 이내에 처리됨 |
| `seg_latency_p99_ms` | ms | 99%의 요청이 이 시간 이내에 처리됨 |
| `seg_latency_max_ms` | ms | 최악 케이스 latency |
| `sample_latency_mean_ms` | ms | 대화 1개 전체를 스트리밍 처리하는 평균 시간 |
| `total_segments_measured` | 개 | warm-up 제외 측정 세그먼트 수 |

#### 읽는 법

**실시간 처리 가능 여부 판단**

새 세그먼트는 음성 발화 간격마다 생성된다. 발화 1개의 최소 지속 시간을 기준으로:

```
seg_latency_p99_ms < 발화 최소 지속 시간(ms)  →  실시간 처리 가능
```

예: 발화 최소 지속 시간 200 ms, p99 latency 69 ms → 실시간 처리 가능

**latency 분포 해석**

```
p99 - p50 이 크다   →  긴 세그먼트 시퀀스(t가 클 때)에서 latency 급등 가능성
seg_latency_std > 15 ms  →  세그먼트 위치(t)에 따른 latency 편차가 유의미함
```

---

## 5. batch vs streaming FPS 비교

두 모드의 FPS는 항상 크게 차이난다. 이는 버그가 아니라 **측정 조건의 차이**다.

| 항목 | batch | streaming |
|---|---|---|
| 배치 크기 | 8 (config 기본) | 1 |
| 세그먼트 패킹 | 배치 내 여러 샘플 병렬 | 1샘플 × 1세그먼트씩 순차 |
| GPU 활용률 | 높음 | 낮음 |
| 측정 목적 | 데이터셋 처리 처리량 | 온라인 서비스 응답 속도 |

```
FPS_batch / FPS_streaming ≈ 5 ~ 10배  →  정상 범위
```

---

## 6. 실험 명령어 요약

```bash
# v5_lora_r8 — batch 모드 (기본)
python models/classifier/experiments/streaming_belief_v5/benchmark.py --experiment v5_lora_r8

# v5_lora_r8 — streaming 모드
python models/classifier/experiments/streaming_belief_v5/benchmark.py --experiment v5_lora_r8 --mode streaming

# 다른 ablation과 비교
python models/classifier/experiments/streaming_belief_v5/benchmark.py --experiment v5_default
python models/classifier/experiments/streaming_belief_v5/benchmark.py --experiment v5_unfreeze_all
```

결과 JSON은 `models/classifier/experiments/streaming_belief_v5/logs/` 에 저장된다.

---

## 7. 결과 기록 예시

실험 후 아래 표에 결과를 채워 비교한다.

| experiment | mode | avg FPS (seg/s) | p50 latency (ms) | p99 latency (ms) |
|---|---|---|---|---|
| v5_lora_r8 | batch | — | — | — |
| v5_lora_r8 | streaming | — | — | — |
| v5_default | batch | — | — | — |
| v5_default | streaming | — | — | — |

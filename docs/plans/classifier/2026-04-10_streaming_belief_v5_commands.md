# streaming_belief_v5 실험 명령어 가이드

## 일반적인 실험 순서

```bash
# 1. 전처리
python models/classifier/experiments/streaming_belief_v5/data_preprocessing.py --source raw --overwrite

# 2. 학습
python models/classifier/experiments/streaming_belief_v5/train.py --experiment v5_default

# 3. 평가
python models/classifier/experiments/streaming_belief_v5/evaluate.py --experiment v5_default --split test
```

---

## 1. 데이터 전처리 — `data_preprocessing.py`

`Data/Training`, `Data/Validation` 원시 데이터를 128토큰 overlap sliding-window 청크로 변환해 캐시 파일로 저장한다.

```bash
# 원시 데이터로 슬라이딩 윈도우 캐시 빌드
python models/classifier/experiments/streaming_belief_v5/data_preprocessing.py --source raw

# 기존 캐시가 있어도 덮어쓰기
python models/classifier/experiments/streaming_belief_v5/data_preprocessing.py --source raw --overwrite

# compressive_memory_kr 실험의 turn-level 캐시를 그대로 복사 (재처리 없음)
python models/classifier/experiments/streaming_belief_v5/data_preprocessing.py --source compressive
```

| 옵션 | 설명 |
|---|---|
| `--source raw` | 원시 JSON 파일에서 슬라이딩 윈도우 캐시를 새로 생성 |
| `--source compressive` | compressive_memory_kr 실험 캐시를 복사 (turn-level 청크 그대로 사용) |
| `--overwrite` | 대상 디렉토리에 이미 파일이 있어도 덮어씀 |

출력 파일: `data/train.json`, `data/val.json`, `data/test.json`, `data/label_map.json`, `data/class_weight.json`, `data/preprocess_meta.json`

---

## 2. 학습 — `train.py`

```bash
python models/classifier/experiments/streaming_belief_v5/train.py --experiment <실험명> [옵션]
```

### Ablation 실험 목록

| `--experiment` 값 | 설명 |
|---|---|
| `v5_default` | KoELECTRA 하위 10층 freeze + attention pooling + Mamba 2층 (d_state=16) — 기본 구성 |
| `v5_mamba1` | Mamba 층수를 1개로 줄인 ablation |
| `v5_dstate32` | Mamba SSM 상태 크기 d_state=32 ablation |
| `v5_dstate64` | Mamba SSM 상태 크기 d_state=64 ablation |
| `v5_lora_r8` | encoder 전체 unfreeze + LoRA (rank=8, alpha=16) 적용 ablation |
| `v5_unfreeze_all` | encoder 완전 fine-tuning (LoRA 없음) ablation |
| `v5_lambda005` | 시간적 focal 가중치 alpha=0.05 (초반 segment 약하게 강조) |
| `v5_lambda01` | 시간적 focal 가중치 alpha=0.1 |
| `v5_lambda02` | 시간적 focal 가중치 alpha=0.2 |

### 추가 옵션

```bash
# lambda_alpha 값을 config 무시하고 직접 지정
python train.py --experiment v5_default --lambda-alpha 0.05

# 중단된 학습 재개 (최신 체크포인트 자동 탐색)
python train.py --experiment v5_default --resume

# 특정 체크포인트 경로로 재개
python train.py --experiment v5_default --resume --resume-path checkpoints/v5_default_best.pt

# 특정 run-id로 재개
python train.py --experiment v5_default --resume --resume-run-id <run_id>
```

| 옵션 | 설명 |
|---|---|
| `--experiment` | 실험 구성 선택 (기본값: `v5_default`) |
| `--lambda-alpha` | temporal 가중치 alpha 값을 config 값 대신 직접 지정 |
| `--resume` | 가장 최근 체크포인트에서 학습 재개 |
| `--resume-path` | 재개할 체크포인트 파일 경로 직접 지정 |
| `--resume-run-id` | 재개할 run-id 지정 |

---

## 3. 평가 — `evaluate.py`

```bash
python models/classifier/experiments/streaming_belief_v5/evaluate.py --experiment <실험명> [옵션]
```

```bash
# test 셋으로 평가 (기본값)
python models/classifier/experiments/streaming_belief_v5/evaluate.py --experiment v5_default

# val 셋으로 평가
python models/classifier/experiments/streaming_belief_v5/evaluate.py --experiment v5_default --split val

# 특정 run-id의 체크포인트 로드
python models/classifier/experiments/streaming_belief_v5/evaluate.py --experiment v5_default --run-id <run_id>

# dataloader 워커 수 지정
python models/classifier/experiments/streaming_belief_v5/evaluate.py --experiment v5_default --num-workers 4
```

| 옵션 | 설명 |
|---|---|
| `--experiment` | 평가할 실험 구성 선택 (기본값: `v5_default`) |
| `--split` | 평가 데이터 선택: `test` (기본) 또는 `val` |
| `--run-id` | 특정 run-id의 체크포인트를 지정해 로드 |
| `--num-workers` | DataLoader 워커 수 |

`--experiment` 옵션은 `train.py`와 동일한 9가지 ablation 이름을 모두 사용할 수 있다.

---

## 4. Loss Function 설정

### 기본 구성: Temporal Weighted Focal Loss

기본 loss는 **Cross Entropy가 아닌 Focal Loss**이며, 세그먼트 시간 순서에 따른 가중치가 추가된다.

```
loss_t = -alpha[label] * (1 - pt)^gamma * log(pt)
lambda_t = exp(-lambda_alpha * t)          # t = 세그먼트 인덱스
final_loss = sum(loss_t * lambda_t) / sum(lambda_t)
```

- `alpha`: 클래스 불균형 보정 가중치 (데이터 분포에서 자동 계산, `class_weight.json`)
- `gamma`: focal 강도 조절 파라미터 (`FOCAL_GAMMA`, 기본값 2.0)
- `lambda_alpha`: 시간 가중치 감쇠율 (`LAMBDA_ALPHA`, 기본값 0.0 = 균등)

### Focal Gamma 조절 (`config.py` → `TRAIN_CONFIG["FOCAL_GAMMA"]`)

| gamma 값 | 효과 |
|---|---|
| `0.0` | Weighted Cross Entropy (focal 효과 없음) |
| `1.0` | 약한 focal 효과 |
| `2.0` (기본) | 표준 Focal Loss — 어려운 샘플 강조 |
| `3.0+` | 더 강한 어려운 샘플 집중 |

```python
# config.py
TRAIN_CONFIG = {
    "FOCAL_GAMMA": 2.0,   # 0.0으로 설정 시 Weighted CE와 동일
    ...
}
```

### Temporal 가중치 조절 (`LAMBDA_ALPHA`)

`config.py` 수정 또는 `--lambda-alpha` 옵션으로 지정한다.

| lambda_alpha 값 | 효과 |
|---|---|
| `0.0` (기본) | 모든 세그먼트 균등 가중치 |
| `0.05` | 초반 세그먼트 약하게 강조 |
| `0.1` | 초반 세그먼트 중간 강조 |
| `0.2` | 초반 세그먼트 강하게 강조 |

```bash
# 명령줄에서 직접 오버라이드
python train.py --experiment v5_default --lambda-alpha 0.0
python train.py --experiment v5_default --lambda-alpha 0.1
```

### Loss 조합 요약

| 원하는 loss | 설정 |
|---|---|
| Weighted Cross Entropy | `FOCAL_GAMMA=0.0`, `LAMBDA_ALPHA=0.0` |
| Focal Loss | `FOCAL_GAMMA=2.0`, `LAMBDA_ALPHA=0.0` |
| Temporal Focal Loss (초반 강조) | `FOCAL_GAMMA=2.0`, `--lambda-alpha 0.05~0.2` |

> Focal / Temporal Focal Loss 외의 loss (예: Label Smoothing CE, ArcFace 등)를 사용하려면 `train.py`의 `focal_loss_per_sample()` 함수를 직접 수정해야 한다.

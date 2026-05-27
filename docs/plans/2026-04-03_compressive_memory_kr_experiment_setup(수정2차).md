# Compressive Memory KR 실험 구성 정리

## 1. 문서 목적
- `normal_data_reference/` 폴더의 SNS 일상대화 데이터셋을 활용하여 Compressive Memory(FM/CM) 구조로 대화 주제(subject) 분류를 수행하는 실험 계획과 코드 구현 상태를 한 문서로 정리한다.
- 실험에 사용한 데이터셋 정보, 모델 정보, 파일 구성 및 실행 흐름을 명확히 기록한다.

## 2. 실험 목표 및 배경
- 목표: Compressive Memory(FM/CM) 구조가 SNS 멀티턴 일상대화에서 맥락 누적 효과를 통해 대화 주제(subject) 20-class 분류 성능을 개선하는지 검증.
- 변경 축:
  1. 인코더 변경: `roberta-base` → `monologg/koelectra-base-v3-discriminator`
  2. 데이터 변경: MELD(scene-level multi-label) → SNS 일상대화(dialogue-level 20-class single-label)
  3. 분류 단위 변경: 턴별(per-turn) 분류 → 대화 전체(dialogue-level) 분류
  4. 레이블 변경: 치료자 발화 화행 8-class → 대화 주제 subject 20-class

## 3. 사용 데이터셋 정보

### 3.1 SNS 일상대화 데이터셋
- 이름: 한국어 SNS 일상대화 데이터셋
- 출처: `normal_data_reference/Training/` 및 `normal_data_reference/Validation/` (로컬 보유)
- 성격: 카카오톡·페이스북·인스타그램·밴드·네이트온 등 SNS 멀티턴 대화
- 단위: JSON 파일 1개 = 대화(dialogue) 1개
- 레이블: `info[0].annotations.subject` 필드 (대화 수준 단일 레이블)
- 원본 규모: Training 87,690개 / Validation 10,962개
- 플랫폼 분포: 카카오톡 81.7%, 페이스북 9.1%, 인스타그램 5.5%, 밴드 1.9%, 네이트온 1.8% (플랫폼 간 편향이 있으나 현 실험에서는 별도 처리하지 않음)

### 3.2 데이터 구조(JSON 스키마)
```
{
  "info": [{
    "annotations": {
      "subject": "<레이블>",
      "speaker_type": "다자간 대화" 또는 "1:1",
      "lines": [
        {
          "id": <int>,
          "text": "<speaker_id> : <원문>",
          "norm_text": "<정규화 발화>",
          "speaker": { "id": "<1번|2번|...>", "sex": "...", "age": "..." },
          "speechAct": "...",
          "morpheme": "..."
        }, ...
      ]
    }
  }]
}
```
- `speaker_type`이 `"1:1"`인 대화와 `"다자간 대화"`인 대화가 혼재하나, 구분 없이 동일하게 처리한다.

### 3.3 레이블 정의(20-class subject)
| idx | subject |
|---|---|
| 0 | 가족 |
| 1 | 건강 |
| 2 | 게임 |
| 3 | 계절/날씨 |
| 4 | 교육 |
| 5 | 교통 |
| 6 | 군대 |
| 7 | 미용 |
| 8 | 반려동물 |
| 9 | 방송/연예 |
| 10 | 사회이슈 |
| 11 | 상거래 전반 |
| 12 | 스포츠/레저 |
| 13 | 식음료 |
| 14 | 여행 |
| 15 | 영화/만화 |
| 16 | 연애/결혼 |
| 17 | 주거와 생활 |
| 18 | 타 국가 이슈 |
| 19 | 회사/아르바이트 |

- 품질 이슈: `상거래전반`(띄어쓰기 누락) 소수 샘플 → 전처리 시 `상거래 전반`으로 정규화하여 통합

### 3.4 데이터 분할 설계 (3-Split)

기존 폴더 분할을 재구성하여 **Train / Val / Test 3-split**을 구성한다.

| Split | 출처 | 구성 방법 | 샘플 수(예상) |
|---|---|---|---:|
| **Train** | `normal_data_reference/Training/` | 원본 Training에서 **90%** 무작위 추출 (label stratified) | ~78,921 |
| **Val** | `normal_data_reference/Training/` | 원본 Training에서 **10%** 무작위 추출 (label stratified) | ~8,769 |
| **Test** | `normal_data_reference/Validation/` | 원본 Validation 폴더 **전체** | 10,962 |

- 분할 근거:
  - 원본 Validation은 수집 시점이 다른 독립 데이터이므로, **최종 성능 보고용 Test set으로 보존**한다.
  - Train에서 10%를 떼어내 Val을 구성하면, Early Stopping 및 하이퍼파라미터 선택에 사용하고 Test와 분리된다.
  - label stratified split (seed=42)으로 Train/Val 간 클래스 비율을 유지한다.
- 역할:
  - **Train**: 모델 학습
  - **Val**: Early Stopping 기준 (macro F1), 하이퍼파라미터 튜닝
  - **Test**: 최종 성능 보고 전용 (학습 중 사용하지 않음)

### 3.5 클래스 분포

| subject | 원본 Training | 비율 | 원본 Validation(=Test) | 비율 |
|---|---:|---:|---:|---:|
| 가족 | 4,521 | 5.16% | 548 | 5.00% |
| 건강 | 4,454 | 5.08% | 546 | 4.98% |
| 게임 | 4,352 | 4.96% | 552 | 5.04% |
| 계절/날씨 | 4,348 | 4.96% | 550 | 5.02% |
| 교육 | 4,575 | 5.22% | 549 | 5.01% |
| 교통 | 4,474 | 5.10% | 550 | 5.02% |
| 군대 | 4,151 | 4.73% | 549 | 5.01% |
| 미용 | 4,675 | 5.33% | 545 | 4.97% |
| 반려동물 | 4,234 | 4.83% | 548 | 5.00% |
| 방송/연예 | 4,323 | 4.93% | 549 | 5.01% |
| 사회이슈 | 4,281 | 4.88% | 547 | 4.99% |
| 상거래 전반 | 4,518 | 5.15% | 550 | 5.01% |
| 스포츠/레저 | 4,642 | 5.29% | 546 | 4.98% |
| 식음료 | 4,132 | 4.71% | 549 | 5.01% |
| 여행 | 4,464 | 5.09% | 545 | 4.97% |
| 영화/만화 | 4,503 | 5.14% | 547 | 4.99% |
| 연애/결혼 | 4,356 | 4.97% | 550 | 5.02% |
| 주거와 생활 | 4,259 | 4.86% | 547 | 4.99% |
| 타 국가 이슈 | 3,969 | 4.53% | 548 | 5.00% |
| 회사/아르바이트 | 4,459 | 5.08% | 547 | 4.99% |
| **합계** | **87,690** | **100%** | **10,962** | **100%** |

- 분포 특성: 클래스 간 분포가 매우 균형적(약 5% 내외). Class-Balanced 손실함수 적용 필요성은 낮음.
- `상거래전반`(5개) → `상거래 전반`으로 통합. 위 테이블은 통합 후 수치.

### 3.6 전처리 설계

전처리 단계에서 모든 JSON 파일을 한 번에 로드/토큰화하여 **통합 JSON 파일로 캐싱**한다.
개별 JSON 파일을 매 에폭 재접근하는 I/O 오버헤드를 제거한다.

#### 전처리 흐름
```
1) normal_data_reference/Training/**/*.json 재귀 로드 (87,690개)
   → 레이블 정규화 (상거래전반 → 상거래 전반)
   → label stratified split (90/10, seed=42) → Train / Val

2) normal_data_reference/Validation/**/*.json 재귀 로드 (10,962개)
   → 레이블 정규화 → Test 그대로 사용

3) 각 대화에 대해:
   - info[0].annotations.lines에서 턴 추출
   - 각 턴 포맷: "{speaker_id}: {norm_text}" (예: "1번: 와 그런 사람들 되게 많다더라.")
   - speaker_type 구분 없이 모든 화자 턴을 동등 처리
   - KoELECTRA tokenizer로 토큰화 (max_length=128, padding=max_length, truncation=True)
   - subject → label index (0~19) 매핑

4) 산출물 저장:
   - train.json: [{dialogue_id, subject_label, num_turns, turns: [{input_ids, attention_mask}]}]
   - val.json: 동일 구조
   - test.json: 동일 구조
   - label_map.json: {subject_name → idx} 매핑
   - class_weight.json: train 기준 inverse frequency weight (필요 시)
```

#### 대화 길이(턴 수) 특성
- 범위: 20~60턴 (평균 약 35~40턴)
- `MAX_TURNS` 하드 컷오프는 두지 않음 (최대 60턴이므로 별도 제한 불필요)

#### Bucket Batching 전략
- 턴 패딩 오버헤드를 최소화하기 위해 **대화 길이 기준 정렬 배치(Bucket Sampler)** 적용
- 대화를 `num_turns` 기준으로 정렬 후, 유사 길이끼리 배치를 구성
- 배치 내 턴 수 차이를 최소화하여 패딩 턴이 1~5턴 수준으로 감소
- Training에서는 버킷 단위 내 셔플을 유지하여 학습 무작위성 보존

## 4. 사용 모델 정보

### 4.1 전체 구조
- `KoELECTRA Encoder` → `Linear Projection(768→128)` → `Compressive Memory(FM+CM)` → `Final Memory Pooling` → `Classification Head(20-class logits)`
- **분류 단위**: 대화 전체 - 모든 턴을 순차적으로 Compressive Memory에 통과시킨 후, 마지막 턴의 메모리 상태(FM 슬롯 평균 풀링)를 사용하여 대화 수준 subject 분류

### 4.2 핵심 설정
- Encoder: `monologg/koelectra-base-v3-discriminator`
- Encoder hidden: `768`
- PAD token id: `0`
- Encoder freeze: 기본 `True`
- Memory:
  1. `SLOT_DIM=128`
  2. `FM_SIZE=3`
  3. `CM_SIZE=4`
  4. `CONV_KERNEL_SIZE=2`
  5. 압축함수: `conv`(기본), ablation에서 `mean` 지원
- Attention: `NUM_HEADS=8`, `DROPOUT=0.1`
- Head: `128 → 64 → 20`
- Final Pooling: 마지막 턴 처리 후 FM 슬롯 평균 풀링 → 분류 헤드 입력

### 4.3 학습/손실
- 분류 단위: 대화 전체(dialogue-level) → 대화 1개당 logit 1개 생성
- 기본 Loss: `CrossEntropyLoss` (균형 분포이므로 기본 CE 우선 적용)
- 선택 가능 Loss: `cross_entropy` / `focal` / `weighted_ce`
- 최적화:
  1. AdamW 차등 LR (encoder `2e-5`, upper `5e-4`)
  2. warmup + cosine decay
  3. gradient accumulation(`2`)
  4. gradient clipping(`1.0`)
  5. early stopping(patience `7`, macro F1 기준, **Val set 기준**)
- 평가 지표: Accuracy, Macro F1, Weighted F1, per-class F1

### 4.4 KMI 대비 모델 변경 포인트
- `therapist_mask`, `turn_labels` (per-turn) 제거 → `dialogue_label` (per-dialogue) 사용
- 모든 화자 턴을 동등하게 처리 (역할 구분 없음)
- 모든 턴을 순차 처리한 후 **최종 메모리 상태에서 1개 logit** 생성
- 손실 계산: 대화 단위 CE (`logit` vs `dialogue_label`)
- Head 출력: 8-class → 20-class

## 5. 코드 구현 요약

### 5.1 구현 대상 파일
- `models/classifier/experiments/compressive_memory_kr/config.py`
- `models/classifier/experiments/compressive_memory_kr/data_preprocessing.py`
- `models/classifier/experiments/compressive_memory_kr/dataset.py`
- `models/classifier/experiments/compressive_memory_kr/model.py`
- `models/classifier/experiments/compressive_memory_kr/train.py`
- `models/classifier/experiments/compressive_memory_kr/evaluate.py`
- `models/classifier/experiments/compressive_memory_kr/requirements.txt`

### 5.2 파일별 주요 변경 사항

#### `config.py`
- `KMI_CONFIG` → `SNS_CONFIG`로 변경
  - `LABELS`: 20-class subject 목록
  - `NUM_LABELS`: 20
  - `SPLIT_RATIOS`: `{"train": 0.9, "val": 0.1}` (원본 Training에서 분할)
  - `SPLIT_SEED`: 42
  - `UTTERANCE_FORMAT`: `"{speaker_id}: {norm_text}"`
  - `ROLE_MAP` 제거 (치료자/내담자 구분 없음)
- `DATA_ROOT`: `PROJECT_ROOT / "normal_data_reference"` 추가 (원본 JSON 경로)
- `TRAIN_CONFIG.LOSS_FN`: 기본값 `"cross_entropy"`로 변경

#### `data_preprocessing.py`
- `normal_data_reference/Training/**/*.json`, `normal_data_reference/Validation/**/*.json` 재귀 로드
- `상거래전반` → `상거래 전반` 레이블 정규화
- 원본 Training → label stratified split (90/10, seed=42) → `train.json` / `val.json`
- 원본 Validation → `test.json`
- 각 대화: 턴별 토큰화 + subject label 매핑 후 통합 JSON 저장
- `label_map.json`, `class_weight.json` 생성

#### `dataset.py`
- `KMIDialogueDataset` → `SNSDialogueDataset`로 변경
  - `therapist_mask`, `turn_labels` 제거
  - `dialogue_label` (int) 반환
- `collate_fn` 변경:
  - `therapist_mask`, `turn_labels` 제거
  - `dialogue_labels` (B,) 반환
- **`BucketBatchSampler` 구현**: `num_turns` 기준 정렬 후 유사 길이끼리 배치 구성
- `create_dataloaders`: train/val/test 3-split 로드, Bucket Sampler 적용

#### `model.py`
- `forward()` 변경:
  - `therapist_mask` 인자 제거
  - 모든 턴 순차 처리 후 **마지막 턴의 메모리 상태에서 FM 슬롯 평균 풀링**
  - 단일 `(B, num_labels)` logit 반환 (per-turn logit 리스트가 아님)
- `_forward_baseline()` 변경:
  - 모든 턴 인코딩의 평균 풀링 → 단일 logit 반환
- `num_labels`: 8 → 20
- `build_compressive_memory_model()`: `KMI_CONFIG` → `SNS_CONFIG` 참조

#### `train.py`
- `compute_loss()` 대폭 단순화:
  - per-turn 마스킹/집계 로직 제거
  - `logits (B, 20)` vs `dialogue_labels (B,)` → 단일 `F.cross_entropy` 호출
  - `therapist_mask`, `LOSS_STRATEGY`, `L3` 시간 가중치 제거
- `validate()`:
  - per-turn 집계 제거 → 대화 단위 `y_true/y_pred` 수집
  - `KMI_CONFIG["LABELS"]` → `SNS_CONFIG["LABELS"]`
- `create_dataloaders` 호출: `"dev"` → `"val"` split명 변경

#### `evaluate.py`
- 대화 수준 `y_true/y_pred/prob` 수집
- per-turn context benefit → **대화 길이 구간별 정확도** 분석으로 변경
- `--split` 옵션: `test`(기본), `val`

### 5.3 ablation 구성(유지)
| 실험명 | 설명 |
|---|---|
| `baseline` | 메모리 없음, 턴별 독립 인코딩 후 평균 풀링으로 대화 분류 |
| `fm_only` | Fine-grained Memory만 사용 |
| `full` | FM + CM 전체 사용 |
| `mean_pooling` | 압축함수 conv → mean으로 변경 |
| `cm_k1` | CM_SIZE=1 |
| `cm_k2` | CM_SIZE=2 |
| `cm_k8` | CM_SIZE=8 |

## 6. 실행 절차(권장)
1. `cd models/classifier/experiments/compressive_memory_kr`
2. `pip install -r requirements.txt`
3. `python data_preprocessing.py`
   - `normal_data_reference/Training/` → train.json(~78.9K) + val.json(~8.8K)
   - `normal_data_reference/Validation/` → test.json(10,962)
4. `python train.py --experiment baseline --loss-fn cross_entropy`
5. `python train.py --experiment full --loss-fn cross_entropy`
6. `python evaluate.py --experiment full --split test`
7. 비교 실험(옵션): `python train.py --experiment full --loss-fn focal`

## 7. 현재 상태
- 데이터셋 변경 계획 확정(KMI 8-class → SNS 일상대화 subject 20-class)
- 분류 단위 변경 확정(per-turn → dialogue-level)
- 3-split 구성 확정: Train(원본 Training 90%) / Val(원본 Training 10%) / Test(원본 Validation 전체)
- 전처리 캐싱 전략 확정: 통합 JSON 파일로 토큰화 결과 저장
- Bucket Batching 전략 확정: 턴 수 기준 정렬 배치로 패딩 최소화
- speaker_type 구분 없이 통일 처리 확정
- 코드 수정 필요 파일: `config.py`, `data_preprocessing.py`, `dataset.py`, `model.py`, `train.py`, `evaluate.py`
- 문법 검증 및 실험 실행은 코드 수정 후 진행 예정

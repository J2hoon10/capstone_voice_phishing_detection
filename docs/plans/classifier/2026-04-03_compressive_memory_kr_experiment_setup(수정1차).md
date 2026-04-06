# Compressive Memory KR 실험 구성 정리

## 1. 문서 목적
- 기존 `experiments/compressive_memory`(영어 MELD) 실험을 한국어 KMI 환경으로 이식한 계획과 실제 코드 구현 상태를 한 문서로 정리한다.
- 실험에 사용한 데이터셋 정보, 모델 정보, 파일 구성 및 실행 흐름을 명확히 기록한다.

## 2. 실험 목표 및 배경
- 목표: Compressive Memory(FM/CM) 구조가 한국어 상담 대화(KMI)에서도 맥락 누적 효과를 통해 치료자 발화 의도 분류 성능을 개선하는지 검증.
- 변경 축:
1. 인코더 변경: `roberta-base` -> `monologg/koelectra-base-v3-discriminator`
2. 데이터 변경: MELD(scene-level multi-label) -> KMI(turn-level 8-class single-label)

## 3. 사용 데이터셋 정보

### 3.1 KMI 데이터셋
- 이름: Korean Motivational Interviewing (KMI)
- 출처: `hjkim811/KMI` (GitHub 공개 데이터)
- 성격: 한국어 동기면담 대화 데이터
- 규모: 약 1,000개 대화
- 카테고리: 7개(`정신건강`, `대인관계`, `자아·성격`, `진로·취업`, `학업·시험`, `중독·집착`, `가족`)
- 분류 태스크: 치료자 턴의 화행 8-class single-label

### 3.2 레이블 정의(8-class)
1. Simple Reflection
2. Complex Reflection
3. Open Question
4. Closed Question
5. Affirm
6. Give Information
7. Advise
8. General

### 3.3 전처리/분할 설계
- 입력 원본: `kmi.json`
- 권장 위치: `models/classifier/experiments/compressive_memory_kr/data/kmi.json`
- 보완 구현: 현재 코드에서 루트 `kmi.json`도 fallback으로 자동 탐색
- 분할: **label 기반 StratifiedGroupKFold** `80/10/10` (seed=42)
- stratify 기준: 치료자 턴 레이블
- group 기준: dialogue(대화 단위 보존, split 간 대화 분리)
- 단위: 대화의 각 턴 1개를 세그먼트 1개로 사용
- 토큰화:
1. 포맷: `"{role}: {utterance}"` (`Therapist`/`Client` -> `치료자`/`내담자`)
2. tokenizer: KoELECTRA tokenizer
3. `max_length=128`, `padding=max_length`, `truncation=True`
- 라벨 처리:
1. 치료자 턴: `0~7`
2. 내담자 턴: `-1` (`ignore_index`)
- 산출물:
1. `train.json`
2. `dev.json`
3. `test.json`
4. `class_weight.json` (train 기준)

### 3.4 KMI 클래스 분포(실제 전처리 산출 기준)
- 기준 파일: `models/classifier/experiments/compressive_memory_kr/data/{kmi,train,dev,test}.json`
- 기준 단위: 치료자 턴(label이 있는 턴)

| Label | Raw(kmi) | Train | Dev | Test |
|---|---:|---:|---:|---:|
| Simple Reflection | 1269 | 1016 | 129 | 124 |
| Complex Reflection | 3055 | 2445 | 304 | 306 |
| Open Question | 3305 | 2647 | 331 | 327 |
| Closed Question | 109 | 87 | 11 | 11 |
| Affirm | 914 | 731 | 91 | 92 |
| Give Information | 87 | 70 | 9 | 8 |
| Advise | 43 | 34 | 4 | 5 |
| General | 776 | 618 | 79 | 79 |
| **Total** | **9558** | **7648** | **958** | **952** |

- 분포 특성: `Open Question`/`Complex Reflection` 비중이 높고 `Advise`/`Give Information`/`Closed Question`이 매우 희소한 강한 불균형 구조.
- split 후에도 비율이 거의 유지되어, 기존 category stratified 대비 label 관점의 분포 보존성이 개선됨.

## 4. 사용 모델 정보

### 4.1 전체 구조
- `KoELECTRA Encoder` -> `Linear Projection(768->128)` -> `Compressive Memory(FM+CM)` -> `Memory Attention` -> `Classification Head(8-class logits)`

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
- Head: `128 -> 64 -> 8`

### 4.3 학습/손실
- 기본 Loss: `Class-Balanced Focal Loss(cb_focal)` (`alpha=focal_alpha`, `gamma=2.0`)
- 선택 가능 Loss: `cb_focal` / `focal` / `cross_entropy`
- 손실 계산: 모든 턴 logits 생성 후 `치료자 턴`만 마스킹하여 계산
- `LOSS_STRATEGY`:
1. `equal` (기본)
2. `L3` (치료자 턴의 대화 후반 가중치 증가)
- 저장 가중치(`class_weight.json`):
1. `class_weight` (balanced CE/Focal alpha용)
2. `focal_alpha` (effective number 기반 CB-Focal alpha)
3. `class_count`, `focal_beta`
- 최적화:
1. AdamW 차등 LR (encoder `2e-5`, upper `5e-4`)
2. warmup + cosine decay
3. gradient accumulation(`2`)
4. gradient clipping(`1.0`)
5. early stopping(patience `7`, macro F1 기준)

## 5. 계획 대비 실제 코드 구현 요약

### 5.1 구현 완료 파일
- `models/classifier/experiments/compressive_memory_kr/config.py`
- `models/classifier/experiments/compressive_memory_kr/data_preprocessing.py`
- `models/classifier/experiments/compressive_memory_kr/dataset.py`
- `models/classifier/experiments/compressive_memory_kr/model.py`
- `models/classifier/experiments/compressive_memory_kr/train.py`
- `models/classifier/experiments/compressive_memory_kr/evaluate.py`
- `models/classifier/experiments/compressive_memory_kr/requirements.txt`
- `models/classifier/experiments/compressive_memory_kr/.gitignore`
- `models/classifier/experiments/compressive_memory_kr/data/.gitkeep`

### 5.2 계획 반영 포인트
- MELD 전용 multi-label 학습/평가를 KMI 전용 multi-class 방식으로 변환 완료
- 치료자 턴만 학습/평가 대상으로 처리 완료
- confusion matrix 및 context benefit(초반/후반 치료자 턴 정확도) 평가 추가 완료
- ablation 구성(`baseline`, `fm_only`, `full`, `mean_pooling`, `cm_k1/k2/k8`) 반영 완료

### 5.3 구현 중 보완된 사항
- `data_preprocessing.py`: `sklearn.compute_class_weight` 기반 class weight 계산으로 정합성 개선
- `data_preprocessing.py`: `kmi.json` 파일 탐색 경로 fallback 추가(루트 파일도 인식)
- `model.py`: baseline 경로에서도 `turn_mask`, `therapist_mask` 반환 일관성 확보
- `data_preprocessing.py`: split 로직을 `category stratified`에서 `label stratified + group(dialogue)`로 전환
- `data_preprocessing.py`: `class_weight`뿐 아니라 `focal_alpha`까지 함께 계산/저장하도록 확장
- `train.py`: 기본 손실함수를 `cb_focal`로 변경하고 `--loss-fn` CLI 옵션 추가

## 6. 코드 파일 구성 설명

### 6.1 `config.py`
- KMI/KoELECTRA 실험 전용 상수 관리
- 데이터셋, 인코더, 메모리, 어텐션, 학습, 평가, ablation 설정 포함

### 6.2 `data_preprocessing.py`
- `kmi.json` 로드
- label stratified group split
- 턴별 토큰화 및 라벨 인코딩
- 통계 출력(턴 분포/토큰 길이/레이블 분포)
- `class_weight.json` 생성(`class_weight`, `focal_alpha`, `class_count` 포함)

### 6.3 `dataset.py`
- `KMIDialogueDataset` 구현
- 대화 길이(턴 수) 가변 배치 패딩 `collate_fn`
- `turn_mask`, `therapist_mask`, `turn_labels`를 포함한 DataLoader 구성

### 6.4 `model.py`
- Compressive Memory 핵심 모듈(Projection/Memory/Attention/Head)
- 턴 순차 처리 `forward` 구현
- 모든 턴 logits 생성 후 학습/평가에서 치료자 턴만 선택 가능하도록 출력
- baseline 모드(메모리 없는 독립 분류) 지원

### 6.5 `train.py`
- CB-Focal/CE 선택형 학습 루프
- AMP/BF16 + gradient accumulation
- validation 지표(Accuracy, Macro F1, Weighted F1, per-class F1)
- early stopping 및 checkpoint/log 저장

### 6.6 `evaluate.py`
- 체크포인트 로드 및 split 평가
- 치료자 턴 기준 `y_true/y_pred/prob` 수집
- Accuracy/Macro F1/Weighted F1/per-class/confusion matrix 계산
- context benefit(early vs late therapist turns) 계산
- JSON 결과 저장

### 6.7 `requirements.txt`
- `torch`, `transformers`, `datasets`, `numpy`, `pandas`, `scikit-learn`

## 7. 실행 절차(권장)
1. `cd models/classifier/experiments/compressive_memory_kr`
2. `pip install -r requirements.txt`
3. `python data_preprocessing.py`
4. `python train.py --experiment baseline --loss-fn cb_focal`
5. `python train.py --experiment full --loss-fn cb_focal`
6. `python evaluate.py --experiment full --split test`
7. 비교 실험(옵션): `python train.py --experiment full --loss-fn cross_entropy`

## 8. 현재 상태
- 실험 코드 구조 및 핵심 로직 구현 완료
- 문법 검증(`py_compile`) 완료
- 라벨 기반 split 및 CB-Focal loss 적용 반영 완료
- `capstone` 환경에서 실험 필수 라이브러리 설치 상태 확인 완료

# 정상 데이터(label=0) 증강 계획 — OpenAI Batch API 기반

> 작성일: 2026-05-15
> 관련 파일: `models/classifier/preprocessing/augmented_pipeline/`

---

## 1. 배경 및 목적

### 문제 상황
- 보이스피싱 탐지 모델 학습 시 정상 대화(label=0) 데이터의 **다양성 부족**
- 기존 원본 정상 데이터:
  - `일상 대화` (VS_01~VS_05, 5개 플랫폼): 10,962건, 평균 553자 (중간값 508자)
  - `민원 상담` (콜센터): 9,773건 (train), 1,260건 (val), 평균 1,332자 (중간값 1,307자)
- 보이스피싱 관련 키워드(수사관, 대출, 통장, 명의 등)가 **정상 맥락**에서도 등장하는 훈련 샘플이 부족
  → 모델이 키워드 단독 패턴에 과적합될 위험 존재

### 증강 목표
- 정상 대화 스크립트 **550건** 추가 생성 (label=0)
- 보이스피싱 연관 키워드가 **범죄 맥락이 아닌 일상적 맥락**에서 자연스럽게 등장하는 샘플 확보
- 두 가지 스타일 커버:
  - **트랙 A**: 콜센터 상담 스타일 (금융기관 실제 업무 통화)
  - **트랙 B**: 일상 통화 스타일 (가족·친구·동료 간 대화)

---

## 2. 참조 데이터 현황

### 레퍼런스 CSV 파일 위치

| 파일 | 경로 | 역할 |
|---|---|---|
| `normal_scripts.csv` | `preprocessing/transcriptions/normal_clean/` | 트랙 B few-shot 소스 |
| `minsang_train.csv` | `preprocessing/transcriptions/normal_clean/` | 트랙 A few-shot 소스 |

### 글자수 통계 (스크립트 본문 기준)

| 통계 | normal_scripts.csv (B트랙) | minsang_train.csv (A트랙) |
|---|---|---|
| 건수 | 10,962 | 9,773 |
| 평균 | 553자 | 1,332자 |
| 중간값 (Q2) | 508자 | 1,307자 |
| Q1 | 436자 | 1,026자 |
| Q3 | 617자 | 1,593자 |
| 최소 | 29자 | 590자 |
| 최대 | 1,837자 | 5,573자 |

> **주의**: 생성 목표 길이(800~3000자)가 B트랙 원본 평균(553자)보다 길게 설정되어 있음.
> 테스트 결과에 따라 `normal_system_prompt.txt`의 길이 조건 조정 검토 필요.

### 레퍼런스 CSV 생성 스크립트

- `preprocessing/build_normal_scripts_csv.py` — 일상 대화 txt → `normal_scripts.csv`
- `preprocessing/build_minsang_csv.py` — 민원 상담 JSON → `minsang_train.csv`, `minsang_val.csv`

---

## 3. 증강 파이프라인 개요

```
[레퍼런스 CSV 2종]
  minsang_train.csv  →  트랙 A few-shot (5개 무작위 샘플)
  normal_scripts.csv →  트랙 B few-shot (5개 무작위 샘플)
         │
         ▼
[generate_normal_augmented.py --mode submit]
  55회 루프 × (A×5 + B×5) = 550건 생성 요청
  JSONL 배치 파일 생성 → OpenAI Batch API 제출
         │
         ▼
[generate_normal_augmented.py --mode check]
  배치 완료 여부 폴링
         │
         ▼
[generate_normal_augmented.py --mode collect]
  결과 수집 → JSON 파싱 → output/normal_augmented.csv 저장
```

---

## 4. 세부 설계

### 4.1 모델 및 API 설정

| 항목 | 값 |
|---|---|
| 모델 | `gpt-5.4-mini` |
| API 방식 | OpenAI Batch API (비동기, 24h completion window) |
| 비용 절감 | 실시간 API 대비 **50% 절감** |
| `response_format` | `{"type": "json_object"}` — 구조화 JSON 응답 강제 |
| `temperature` | 1.0 — 다양성 극대화 |
| `max_completion_tokens` | 8192 |
| SEED | 42 (재현 가능) |

### 4.2 루프 설계

```
N_LOOPS = 55
호출당 생성: 트랙 A 5건 + 트랙 B 5건 = 10건
총 생성량: 55 × 10 = 550건
```

루프마다 `random.Random(SEED)` 기반으로 서로 다른 `random_state` 값을 생성하여
매 호출마다 **다른 few-shot 샘플 조합**이 선택됨 (deterministic, 재현 가능).

### 4.3 시나리오 및 필수 키워드

#### 트랙 A — 콜센터 상담 (5개 시나리오)

| # | 시나리오 | 필수 키워드 |
|---|---|---|
| A1 | 대출 심사 통과 후 정상적인 상환 방법과 이율을 안내하는 상황 | 대출, 승인, 자금, 상환 |
| A2 | 은행 전산 시스템 오류로 인해 고객의 납부 처리가 지연되고 있다고 양해를 구하는 상황 | 전산, 접수, 납부, 금액 |
| A3 | 비대면 계좌 개설 과정에서 필요한 개인정보 및 명의 확인 절차를 설명하는 상황 | 명의, 개설, 정보, 개인 |
| A4 | 고객의 카드 한도 조회 및 해외 거래 가능 여부를 상담해 주는 상황 | 카드, 이용, 거래, 가능 |
| A5 | 대출금 입금 요청이 접수되었으나 본인 확인 통화 절차가 필요하다고 안내하는 상황 | 대출, 입금, 요청, 통화 |

#### 트랙 B — 일상 통화 (5개 시나리오)

| # | 시나리오 | 필수 키워드 |
|---|---|---|
| B1 | 아침 뉴스에서 본 끔찍한 명의도용 사건을 지인 또는 가족에게 이야기하며 조심하라고 당부하는 상황 | 사건, 수사, 명의, 도용 |
| B2 | 모르는 번호로 온 수사관 사칭 전화를 방금 끊고 황당해하며 지인에게 말하는 상황 | 수사관, 범죄, 연루, 조사 |
| B3 | 최근 유행하는 불법 대포통장 개설 범죄에 관한 다큐멘터리 감상평을 나누는 상황 | 불법, 대포, 개설, 피해자 |
| B4 | 서울에서 대규모 보이스피싱 일당(남성)이 검거되었다는 인터넷 기사를 단톡방에 공유하는 상황 | 검거, 서울, 남성, 발견 |
| B5 | 회사 보안 교육에서 배운 녹취 방법과 개인정보 보호의 중요성에 대해 동료와 푸념하는 상황 | 녹취, 진술, 압수, 정보 |

> **설계 의도**: B트랙 시나리오는 보이스피싱 연관 키워드(수사관, 명의, 대포, 녹취 등)를
> **뉴스 시청 / 지인 대화 / 다큐 감상** 맥락에 배치해 모델이 키워드만으로 오분류하지 않도록 훈련.

### 4.4 프롬프트 설계

**시스템 프롬프트** (`augmented_pipeline/prompts/normal_system_prompt.txt`):

```
[트랙 구분]
- 트랙 A (콜센터): 은행/카드사/통신사 콜센터 상담 스타일. 전문적·정중하되 친근한 말투.
- 트랙 B (일상대화): 가족·친구·동료 사이 전화 통화 스타일. 격식 없고 구어체 풍부.

[작성 규칙]
- 발신자/수신자 라벨 없이 연속 텍스트로 작성
- 한국어 구어체, 실제 전화 통화 녹취록처럼 작성
- 줄바꿈 문자 사용 금지 (한 줄 연속 텍스트)
- 길이 800~3000자 (항목마다 다양하게)
- 영어 혼용 금지, 외래어는 한국어 발음 표기

[구어체 필수 반영]
- "음", "어", "저", "그러니까", "아" 등 망설임 표현 자연스럽게 삽입
- 자기 수정 패턴, 상대 반응 표현 ("아 그래요?", "맞아맞아") 포함
- 필수 키워드는 자연스러운 문맥 속에 반드시 포함
```

**유저 프롬프트 구조** (`build_user_prompt()` 함수):
1. 트랙 A few-shot 예시 5건 (minsang_train.csv에서 샘플링, 최대 400자 발췌)
2. 트랙 B few-shot 예시 5건 (normal_scripts.csv에서 샘플링, 최대 400자 발췌)
3. 생성 요청: 트랙별 시나리오 + 필수 키워드 명시
4. 출력 형식 강제: `{"A": [...], "B": [...]}`

### 4.5 텍스트 전처리 (`clean_script()`)

few-shot 예시 삽입 전 화자 레이블 제거:
- 숫자 레이블: `"1 : "`, `"2 : "` 등
- 역할 레이블: `"고객:"`, `"상담사:"`, `"손님:"`, `"상담원:"`
- 줄바꿈 → 공백으로 압축 후 사용

---

## 5. 실행 방법

### 사전 준비

```bash
conda activate capstone
pip install openai
export OPENAI_API_KEY=sk-...
```

### 단건 테스트 (품질 확인 먼저)

```bash
cd models/classifier/preprocessing/augmented_pipeline
python test_single_call.py
```

- 출력: `output/test_single_call.csv`
- 컬럼: `id, track, scenario_idx, scenario, keywords, script, label`
- 토큰 사용량 및 각 스크립트 200자 미리보기 출력

### 전체 배치 실행

```bash
# 1단계: 배치 제출 (JSONL 생성 → API 업로드 → 배치 시작)
python generate_normal_augmented.py --mode submit

# 2단계: 완료 확인
python generate_normal_augmented.py --mode check

# 3단계: 결과 수집 및 CSV 저장
python generate_normal_augmented.py --mode collect
```

### 배치 ID 직접 지정 (선택)

```bash
python generate_normal_augmented.py --mode check --batch-id batch_xxx
python generate_normal_augmented.py --mode collect --batch-id batch_xxx
```

---

## 6. 출력 파일

| 파일 | 경로 | 설명 |
|---|---|---|
| `normal_batch_input.jsonl` | `augmented_pipeline/output/` | API 제출용 배치 입력 (55건) |
| `normal_batch_meta.json` | `augmented_pipeline/output/` | 배치 ID, 파일 ID, 상태 저장 |
| `normal_augmented.csv` | `augmented_pipeline/output/` | 최종 증강 결과 (550건) |

### `normal_augmented.csv` 컬럼

| 컬럼 | 설명 |
|---|---|
| `id` | `normal_aug_0001` ~ `normal_aug_0550` |
| `script` | 생성된 스크립트 본문 |
| `label` | 0 (정상) |
| `track` | A (콜센터) 또는 B (일상대화) |
| `scenario_idx` | 시나리오 번호 (1~5) |
| `loop_idx` | 배치 루프 인덱스 (0~54) |

---

## 7. 파일 구조

```
models/classifier/preprocessing/
├── augmented_pipeline/
│   ├── generate_normal_augmented.py  # 메인 배치 파이프라인 (submit/check/collect)
│   ├── test_single_call.py           # 단건 테스트 스크립트
│   ├── prompts/
│   │   └── normal_system_prompt.txt  # 정상 대화 생성 시스템 프롬프트
│   └── output/
│       ├── normal_batch_input.jsonl  # 배치 입력 JSONL
│       ├── normal_batch_meta.json    # 배치 메타 (ID, 상태)
│       ├── normal_augmented.csv      # 최종 증강 결과 (550건)
│       └── test_single_call.csv      # 단건 테스트 결과
├── transcriptions/
│   └── normal_clean/
│       ├── normal_scripts.csv        # 일상 대화 레퍼런스 (10,962건)
│       └── minsang_train.csv         # 민원 상담 레퍼런스 (9,773건)
├── build_normal_scripts_csv.py       # normal_scripts.csv 생성
└── build_minsang_csv.py              # minsang_train/val.csv 생성
```

---

## 8. 에러 처리 및 재처리

`collect_results()`의 파싱 실패 처리:
1. `json.loads(raw_text)` 직접 파싱 시도
2. 실패 시 정규식으로 JSON 블록 추출: `re.search(r"\{.*\}", raw_text, re.DOTALL)`
3. 여전히 실패하면 `parse_errors` 리스트에 `custom_id` 기록 후 건너뜀
4. 최종 출력 시 파싱 실패 요청 수 및 ID 목록 출력

파싱 실패 요청은 `custom_id` (`loop_0003` 등)를 확인 후 개별 재호출 처리 가능.

---

## 9. 진행 상황 및 남은 작업

- [x] `normal_scripts.csv` 생성 (10,962건)
- [x] `minsang_train.csv` 생성 (9,773건)
- [x] `generate_normal_augmented.py` 작성 (submit/check/collect)
- [x] `normal_system_prompt.txt` 작성
- [x] 시나리오 및 키워드 확정 (A×5 + B×5)
- [x] `test_single_call.py` 작성 및 단건 테스트 실행
- [ ] 테스트 결과(`test_single_call.csv`) 검수
  - 생성 스크립트 품질 확인 (키워드 포함 여부, 자연스러움, 길이)
  - 필요 시 `normal_system_prompt.txt`의 길이 조건 조정 (현재 800~3000자 vs B트랙 원본 평균 553자)
- [ ] 전체 배치 실행: `python generate_normal_augmented.py --mode submit`
- [ ] 배치 완료 확인 및 결과 수집
- [ ] `output/normal_augmented.csv` 검수 (550건 확인, 트랙별 분포)
- [ ] 최종 학습 데이터셋에 통합 (`build_training_dataset.py`)

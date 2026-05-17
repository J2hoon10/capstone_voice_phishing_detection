# 4클래스 학습 데이터셋 레퍼런스

> 작성일: 2026-05-17  
> 대상 파일: `models/classifier/preprocessing/output/4class/{train,val,test}.csv`  
> 생성 스크립트: `models/classifier/preprocessing/build_4class_dataset.py`

---

## 1. 클래스 정의

| label | binary_label | category | 설명 |
|-------|-------------|----------|------|
| 0 | 0 | 상담 대화 | 콜센터 금융 상담 (Track A) |
| 1 | 0 | 일상 대화 | 일상 전화 통화 (Track B) |
| 2 | 1 | 대출 사기형 | 보이스피싱 — 대출 사기 |
| 3 | 1 | 수사기관 사칭형 | 보이스피싱 — 검찰·경찰 사칭 |

`binary_label`: 0 = 정상, 1 = 보이스피싱

---

## 2. 최종 데이터셋 규모

| split | 건수 |
|-------|------|
| train | 2,407 |
| val   | 802  |
| test  | 804  |
| **합계** | **4,013** |

분할 방식: 클래스별 stratified split (60 / 20 / 20), seed = 42

### 클래스별 분포

| label | category | total | train | val | test |
|-------|----------|------:|------:|----:|-----:|
| 0 | 상담 대화 | 1,000 | 600 | 200 | 200 |
| 1 | 일상 대화 | 1,000 | 600 | 200 | 200 |
| 2 | 대출 사기형 | 1,013 | 607 | 202 | 204 |
| 3 | 수사기관 사칭형 | 1,000 | 600 | 200 | 200 |

> 대출 사기형이 13건 더 많은 이유: 피싱 LLM 증강량(823건 vs 677건)의 차이에서 기인.  
> 피싱 원본 자체가 대출 사기형 190건 / 수사기관 사칭형 323건으로 불균형 → LLM 증강으로 수량 역전.

### 소스별 분포 (전체 4,013건)

| source | 건수 | 비율 | 설명 |
|--------|----:|-----:|------|
| normal_asr_llm | 1,200 | 29.9% | 일반 LLM 생성 + ASR 노이즈 (트랙당 600) |
| phishing_original | 513 | 12.8% | 피싱 원본 Whisper 전사 |
| phishing_asr_llm | 999 | 24.9% | 피싱 LLM 생성 + ASR 노이즈 |
| normal_clean_llm | 600 | 14.9% | 일반 LLM 생성 클린 (트랙당 300) |
| phishing_clean_llm | 501 | 12.5% | 피싱 LLM 생성 클린 |
| normal_asr_orig | 200 | 5.0% | 일반 원본 + ASR 노이즈 (트랙당 100) |

---

## 3. 원본 데이터

### 3-1. 피싱 원본 (`transcriptions/gpu_small/phishing.csv`)

- **건수**: 513건
- **전사 도구**: Whisper small (GPU, float16)
- **형식**: 화자 분리 없는 연속 텍스트, 구어체 한국어
- **카테고리 구성**:

| category | 건수 | 주요 특징 |
|----------|----:|---------|
| 대출 사기형 | 190 | 금리·한도 안내 → 편법 대환 → 선납금 요구 |
| 수사기관 사칭형 | 323 | 검찰·경찰 사칭, 대포통장 발견, 계좌 동결 협박 |

> **"바로 이 목소리" 재분류 이력 (2026-05-11)**: 기존 "바로 이 목소리" 카테고리(94건)를 LLM으로 내용 분석 후 두 카테고리로 분류하여 병합.  
> - 수사기관 사칭형: **89건** (검찰·수사관 사칭, 대포통장 발견, 계좌 동결 협박 패턴)  
> - 대출 사기형: **5건** (대출 한도·금리 안내, 선납금 요구 패턴) — ID: `_5, _72, _73, _74, _75`

- **등장 기관명** (빈도 분석 기반):
  - 수사기관 사칭형: 검찰청, 서울중앙지방검찰청, 금융감독원, 경찰청
  - 대출 사기형: 농협, 국민은행, KB저축은행, 햇살론, 삼성카드

### 3-2. 일반 원본 데이터

4class 데이터셋에 실제로 사용되는 일반 원본은 `data/normal/` 하위 두 폴더이며,  
`build_minsang_csv.py` / `build_normal_scripts_csv.py`를 통해 CSV로 변환된 뒤 사용된다.

#### Track A — 민원 상담 (`data/normal/민원 상담/`)

- **데이터셋 원출처**: **AIHub — 민간 민원 상담 LLM 사전학습 및 Instruction Tuning 데이터**
- **원본 형식**: JSON 파일 (11,033개)
- **출처 기관**: 카드사·통신사 콜센터 상담 녹취록
  - `TS_하나카드`: 5,757건
  - `TS_엘지유플러스`: 3,216건
  - `TS_액티벤처`: 800건
- **변환 스크립트**: `stt_tools/build_minsang_csv.py`
- **변환 결과**: `transcriptions/normal_clean/minsang_train.csv` (9,773건 train)
- **사용 방법**: LLM 생성 시 Track A few-shot 소스 (5건 무작위 샘플링) + 원본 기반 ASR 노이즈 대상

**상담 카테고리 예시** (상위 빈도): 선결제/즉시출금 825건, 이용내역 안내 801건, 요금 안내 619건, 요금 납부 489건 등

#### Track B — 일상 대화 (`data/normal/일상 대화/`)

- **데이터셋 원출처**: **AIHub — 주제별 텍스트 일상 대화 데이터**
- **원본 형식**: 텍스트 파일 (10,962개)
- **출처 플랫폼**: 소셜 플랫폼 일상 대화 (VS_01~05)
  - VS_01. KAKAO / VS_02. FACEBOOK / VS_03. INSTAGRAM / VS_04. BAND / VS_05. NATEON
- **변환 스크립트**: `stt_tools/build_normal_scripts_csv.py` (별도 확인 필요)
- **변환 결과**: `transcriptions/normal_clean/normal_scripts.csv` (10,962건)
- **사용 방법**: LLM 생성 시 Track B few-shot 소스 (5건 무작위 샘플링)

---

> **참고**: `transcriptions/gpu_small/normal.csv` (2,001건, 카테고리: 상품 가입 및 해지·이체 출금 대출서비스·잔고 및 거래내역)는  
> `build_4class_dataset.py`에서 **직접 사용되지 않음**. 어떤 원본에서 왔는지 별도 확인 필요.

---

## 4. 증강 파이프라인 전체 흐름

```
[원본 음성 데이터]
       │
       ▼
[Stage 0] batch_transcribe.py
  Whisper small (GPU) → phishing.csv, normal.csv
       │
       ├──────────────────────────────────────┐
       ▼ (피싱)                               ▼ (일반)
[Stage 1-피싱]                          [Stage 1-일반]
  phishing_augmentation/                normal_augmentation/
  augment_phishing.py                   generate_normal_augmented.py
  LLM 텍스트 생성                         LLM 텍스트 생성
  → phishing_augmented.csv (1500건)     → normal_augmented.csv (1800건)
       │                                       │
       ▼                                       ▼
[Stage 2-피싱]                          [Stage 2-일반]
  phishing_augmentation/                normal_augmentation/
  augment_asr_noise.py                  augment_asr_noise.py
  ASR 노이즈 주입                         ASR 노이즈 주입
  → asr_noised.csv (1500건)             → normal_asr_noised.csv (2000건)
       │                                       │
       └────────────────┬─────────────────────┘
                        ▼
[Stage 3] build_4class_dataset.py
  원본 + LLM 증강 + ASR 노이즈 조합
  → output/4class/train.csv (2,407)
  → output/4class/val.csv   (  802)
  → output/4class/test.csv  (  804)
  → output/4class/dataset_stats.json
```

---

## 5. Stage 1: LLM 텍스트 생성

### 5-1. 피싱 LLM 증강

- **스크립트**: `phishing_augmentation/augment_phishing.py`
- **LLM 모델**: `gpt-5.4-mini` (일부 `gpt-5.5`)
- **방식**: few-shot 배치 생성 (few-shot k=10, 호출당 5건 생성)
- **생성량**: 카테고리별 ~750건 → 총 1,500건
  - 수사기관 사칭형: 677건
  - 대출 사기형: 823건
- **출력**: `phishing_augmentation/output/phishing_augmented.csv`
  - 컬럼: `id, text, label, category, source`
  - `source = "augmented_llm"`

**카테고리별 필수 포함 요소**:

| category | 필수 요소 |
|----------|---------|
| 수사기관 사칭형 | ① 대포통장 발견 ② 수사관·검사 신분 강조 ③ 계좌 동결·압수 ④ 긴박감 조성 ⑤ 제3자 차단 |
| 대출 사기형 | ① 금리·한도 구체 안내 ② 부결·신용점수 문제 ③ 편법 대환대출 제안 ④ 선납금 요구 ⑤ 마감 긴박감 |

**생성 규칙**:
- 화자 분리 없는 연속 텍스트
- 한국어 구어체 (문장부호 최소화, 받아쓰기 스타일)
- 길이: 1,000~2,000자
- 허용 기관명: 실제 phishing.csv 빈도 분석 기반으로 제한

### 5-2. 일반 LLM 증강

- **스크립트**: `normal_augmentation/generate_normal_augmented.py`
- **LLM 모델**: `gpt-5.4-mini`
- **API 방식**: OpenAI Batch API (비동기, 50% 비용 절감)
- **생성량**: 트랙당 900건 → 총 1,800건
  - Track A (상담 대화): 900건
  - Track B (일상 대화): 900건
- **루프 설계**: 900회 루프 × 트랙별 2건 = 1,800건
- **출력**: `normal_augmentation/output/normal_augmented.csv`
  - 컬럼: `id, script, label, track, scenario_idx, loop_idx`

**few-shot 소스**:
| Track | few-shot 소스 | 원데이터 | 건수 | 평균 길이 |
|-------|-------------|---------|----:|------:|
| A (상담) | minsang_train.csv (민원 상담 JSON 전사) | AIHub — 민간 민원 상담 LLM 사전학습 및 Instruction Tuning 데이터 | 9,773건 | 1,332자 |
| B (일상) | normal_scripts.csv (일상 대화 txt) | AIHub — 주제별 텍스트 일상 대화 데이터 | 10,962건 | 553자 |

**시나리오 설계 의도**: B트랙 시나리오에 보이스피싱 연관 키워드(수사관, 대포통장, 계좌 동결 등)를  
뉴스 시청 / 지인 대화 / 다큐 감상 등 **정상 맥락**에 배치 → 모델의 키워드 과적합 방지

**Track A 시나리오 예시** (총 10종):
- 대출 심사 통과 후 상환 방법·이율 안내
- 비대면 계좌 개설 시 명의 확인 절차 설명
- 경찰청 민원콜센터에 보이스피싱 피해 신고
- 금융감독원 소비자보호센터 불법 금융상품 피해 상담

**Track B 시나리오 예시** (총 10종):
- 명의도용 사건을 지인에게 설명하며 계좌 동결·압수 피해 사례 당부
- 검사 사칭 전화를 끊고 지인에게 통화 내용 재현
- 대출 사기 전화에 속을 뻔한 가족 경험 공유

---

## 6. Stage 2: ASR 노이즈 주입

### 6-1. 목적

LLM이 생성한 깨끗한 텍스트에 실제 Whisper STT 오류 패턴을 주입하여,  
모델이 실제 서비스 환경(STT 오류 포함)에서도 강건하게 동작하도록 학습.

### 6-2. 노이즈 파라미터 (실측값, `error_analysis/error_summary.json`)

| 오류 유형 | 파라미터 | 실측값 | 설명 |
|---------|---------|------:|------|
| 음절 치환 | p_sub | 7.76% | 유사 음소로 치환 (ㄱ↔ㅋ, ㅐ↔ㅔ, ㅏ↔ㅓ 등) |
| 음절 삭제 | p_del | 8.21% | 무작위 음절 제거 |
| 필러 삽입 | p_ins | 3.65% | "음", "어", "아", "그", "저", "네", "요" 삽입 |
| 공백 오류 | p_space | 4.17% | 공백 삭제 또는 추가 |
| **전체 CER** | — | **19.6%** | 측정 샘플 1,778건 기준 |

### 6-3. 스크립트 및 출력

- **피싱**: `phishing_augmentation/augment_asr_noise.py`
  - 입력: `phishing_augmented.csv` (1,500건)
  - 출력: `phishing_augmentation/output/asr_noised.csv` (1,500건)
  - ID 형식: `asr_aug_수사기관 사칭형_0001_1` (원본 ID + 인덱스)

- **일반**: `normal_augmentation/augment_asr_noise.py`
  - 입력: `normal_augmented.csv` (LLM 1,800건) + `minsang_train.csv` (원본 일부)
  - 출력: `normal_augmentation/output/normal_asr_noised.csv` (2,000건)
    - LLM 기반 noised (`asr_normal_aug_*`): 1,800건
    - 원본 기반 noised (`asr_minsang_*`): 200건
  - ID 형식: `asr_normal_aug_0001` (LLM 기반), `asr_minsang_836` (원본 기반)

---

## 7. Stage 3: 4클래스 데이터셋 조합 (`build_4class_dataset.py`)

### 7-1. 일반 데이터 조합 비율 (트랙당 고정)

| 소스 | 건수 | 비율 | source 태그 |
|------|----:|-----:|-----------|
| 원본 기반 ASR 노이즈 | 100 | 10% | `normal_asr_orig` |
| LLM 기반 ASR 노이즈 | 600 | 60% | `normal_asr_llm` |
| LLM 클린 | 300 | 30% | `normal_clean_llm` |
| **합계** | **1,000** | **100%** | |

- 클린 풀에서 LLM 노이즈 선택에 사용된 소스 ID는 중복 제외 (동일 텍스트의 클린본과 노이즈본이 동시에 포함되지 않도록)
- 트랙당 총 1,000건 × 2트랙 = **2,000건**

### 7-2. 피싱 데이터 조합 비율 (카테고리별 동적)

| 소스 | 설명 | source 태그 |
|------|------|-----------|
| 피싱 원본 전체 | phishing.csv 전체 사용 (자연 노이즈 이미 보유) | `phishing_original` |
| LLM 기반 ASR 노이즈 (2/3) | asr_noised.csv 중 LLM 기반 (`asr_aug_*`) | `phishing_asr_llm` |
| LLM 클린 (1/3) | phishing_augmented.csv 중 노이즈 미사용 행 | `phishing_clean_llm` |

- LLM 노이즈:클린 = 2:1 (가용 LLM 전체 기준 동적 계산)
- 클린 풀에서 노이즈 선택에 사용된 소스 ID 중복 제외

**실제 결과**:

| category | 원본 | LLM_ASR | LLM_Clean | 합계 |
|----------|----:|--------:|----------:|----:|
| 대출 사기형 | 190 | ~549 | ~274 | 1,013 |
| 수사기관 사칭형 | 323 | ~450 | ~227 | 1,000 |

### 7-3. 전처리 (clean_text)

- 한글·영문·숫자·공백 외 문자 제거 (특수문자, 이모지 등)
- 줄바꿈 → 공백
- 연속 공백 → 단일 공백

---

## 8. CSV 컬럼 스펙

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `id` | string | 고유 식별자 |
| `text` | string | 전처리된 전사 텍스트 |
| `label` | int | 4클래스 레이블 (0~3) |
| `binary_label` | int | 이진 레이블 (0=정상, 1=피싱) |
| `category` | string | 카테고리명 |
| `source` | string | 데이터 출처 태그 |
| `filename` | string | 원본 음성 파일명 (증강 데이터는 빈 문자열) |
| `segment_risks` | string | 세그먼트별 위험도 JSON 배열 |

**source 값 목록**:
- `normal_asr_orig` — 일반 원본 + ASR 노이즈
- `normal_asr_llm` — 일반 LLM 생성 + ASR 노이즈
- `normal_clean_llm` — 일반 LLM 생성 클린
- `phishing_original` — 피싱 원본 Whisper 전사
- `phishing_asr_llm` — 피싱 LLM 생성 + ASR 노이즈
- `phishing_clean_llm` — 피싱 LLM 생성 클린

---

## 9. 관련 파일 위치

```
models/classifier/preprocessing/
├── build_4class_dataset.py                     # [Stage 3] 4클래스 데이터셋 생성
├── transcriptions/
│   └── gpu_small/
│       ├── phishing.csv                        # 원본 피싱 전사 (513건)
│       └── normal.csv                          # 원본 일반 전사 (2,001건)
├── normal_augmentation/
│   ├── generate_normal_augmented.py            # [Stage 1] 일반 LLM 생성
│   ├── augment_asr_noise.py                    # [Stage 2] 일반 ASR 노이즈
│   ├── prompts/
│   │   └── normal_system_prompt.txt            # 일반 생성 시스템 프롬프트
│   └── output/
│       ├── normal_augmented.csv                # LLM 생성 (1,800건)
│       ├── normal_asr_noised.csv               # ASR 노이즈 (2,000건)
│       ├── normal_segment_risks.csv            # 원본 기반 세그먼트 위험도
│       └── normal_aug_segment_risks.csv        # 증강 기반 세그먼트 위험도
├── phishing_augmentation/
│   ├── augment_phishing.py                     # [Stage 1] 피싱 LLM 생성
│   ├── augment_asr_noise.py                    # [Stage 2] 피싱 ASR 노이즈
│   └── output/
│       ├── phishing_augmented.csv              # LLM 생성 (1,500건)
│       └── asr_noised.csv                      # ASR 노이즈 (1,500건)
├── error_analysis/
│   └── error_summary.json                      # Whisper 실측 오류율
└── output/
    └── 4class/
        ├── train.csv                           # 학습 데이터 (2,407건)
        ├── val.csv                             # 검증 데이터 (802건)
        ├── test.csv                            # 테스트 데이터 (804건)
        └── dataset_stats.json                  # 분포 통계
```

---

## 10. 확인이 필요한 사항

다음 항목은 코드·파일 분석만으로는 확인이 불가능하여 추가 확인이 필요합니다.

1. **`transcriptions/gpu_small/normal.csv` 출처 및 사용 여부**  
   `상품 가입 및 해지`, `이체 출금 대출서비스`, `잔고 및 거래내역` 3개 카테고리(2,001건)가 어디서 온 것인지,  
   현재 4class 파이프라인에서 사용되지 않는 것이 확실한지 확인 필요

2. **segment_risks 생성 방법**  
   `normal_segment_risks.csv`, `normal_aug_segment_risks.csv`가 어떤 스크립트로 생성됐는지 추적되지 않음.  
   피싱 데이터의 `segment_risks` 컬럼도 생성 과정 불명확.

3. **normal_combined.csv 역할**  
   `normal_augmentation/output/normal_combined.csv`가 존재하나 `build_4class_dataset.py`에서 사용되지 않음.  
   이 파일이 어떤 용도로 생성된 것인지, 폐기 여부 확인 필요.

4. **피싱 LLM 카테고리 불균형 (대출 823건 vs 수사기관 677건)**  
   원본 불균형(대출 190 < 수사기관 323)이 LLM 증강 후 역전된 것이 의도된 설계인지,  
   아니면 LLM 생성 실패·파싱 오류 등의 결과인지 확인 필요.

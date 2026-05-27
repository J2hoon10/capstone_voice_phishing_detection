# 보이스피싱 탐지 학습 데이터 증강 계획

## 1. 배경 및 목적

### 문제 상황
- 보유 피싱 음성 데이터가 적어 모델 학습에 충분하지 않음
- 카테고리별 불균형: 수사기관 사칭형 329건(바로 이 목소리 94건 병합), 대출 사기형 185건

### 증강 목표
- 피싱 데이터를 카테고리별 +200건 이상 증강
- 증강 데이터가 실제 STT 전사 결과와 동일한 형식 및 분포를 가져야 함
- 모델이 실제 서비스 환경에서도 동일하게 동작하도록 분포 일치(distribution matching)

---

## 2. 원본 데이터 현황

### 원본 STT 전사 형식 특징
- **화자 분리 없음**: 발신자/수신자 구분 없이 하나의 연속된 텍스트로 전사됨
- **구어체 한국어**: 문장부호 최소화, 받아쓰기 스타일
- **파일 경로**: `transcriptions/gpu_small/phishing.csv`
- **컬럼**: `id, text, label, category, source, filename`

### 카테고리별 실제 데이터 분석 결과

> **클래스 병합 (2026-05-11)**: "바로 이 목소리" 94건을 "수사기관 사칭형"에 병합.
> 실제 내용 분석 결과 지인/가족 사칭 데이터 없음 — 수사기관 사칭 + 녹취 강조 패턴이 전부.

| 카테고리 | 건수 | 평균 텍스트 길이 | 1,000~2,000자 후보 |
|---|---|---|---|
| 수사기관 사칭형 | 329건 (구 235 + 바로이목소리 94) | — | — |
| 대출 사기형 | 185건 | 약 1,359자 | 71건 |

### 카테고리별 실제 등장 기관명 (빈도 분석 기반)

**수사기관 사칭형**
- 사칭 기관: 검찰청, 서울중앙지방검찰청, 지방검찰청, 금융감독원, 경찰청
- 피해자 계좌 기관: 농협, 하나은행, 국민은행, 기업은행

**대출 사기형**
- 농협, 국민은행, KB저축은행, 햇살론, 삼성카드, 기업은행, 하나은행

*(바로 이 목소리 카테고리는 수사기관 사칭형으로 병합됨)*

---

## 3. 증강 파이프라인 전체 흐름

```
[원본 음성 데이터]
       │
       ▼
[Stage 0] batch_transcribe.py
  Whisper STT → phishing.csv / normal.csv / all.csv
       │
       ▼
[Stage 1] augment_phishing.py          ← 핵심 증강 단계
  LLM API (OpenAI gpt-5.4-mini) 텍스트 생성
  few-shot 프롬프트 + 카테고리별 지시
  출력: augmented/phishing_augmented.csv
       │
       ▼
[Stage 2] augment_asr_noise.py
  ASR 노이즈 주입 (실측 오류율 기반)
  출력: augmented/asr_noised.csv
       │
       ▼
[Stage 3] build_training_dataset.py
  원본 + LLM 증강 + ASR 노이즈 합산
  파일 단위 train/val/test 분할 (8:1:1)
  출력: final/train.csv, val.csv, test.csv
```

---

## 4. 각 단계 상세

### Stage 1: LLM 기반 텍스트 생성 (`augment_phishing.py`)

#### 생성 방식: 카테고리별 배치 생성 (Method B)
- 원본 데이터에서 카테고리별로 few-shot 예시(k=10)를 무작위 샘플링
- 예시들을 프롬프트에 포함해 API 한 번에 여러 건 생성
- 원본당 1:1 생성(Method A)보다 API 호출 수가 적고, 다양한 변형 확보 가능

#### 프롬프트 설계 원칙
1. **화자 분리 없음**: 발신자/수신자 구분 없이 연속 텍스트로 생성
2. **실제 데이터 기반 제약**: 허용 기관명을 실제 phishing.csv 빈도 분석으로 제한
3. **필수 포함 요소 지정**: 카테고리별로 실제 피싱 수법의 필수 요소를 명시
4. **길이 제한**: 1,000~2,000자 (few-shot 예시의 길이 범위와 동일)
5. **[변형N] 태그**: 생성물 파싱을 위한 구분 태그 강제

#### 시스템 프롬프트 핵심 규칙
```
- 화자 구분 없이 연속된 하나의 텍스트로 작성
- 한국어 구어체 (문장부호 최소화, 받아쓰기 스타일)
- 길이: 1,000~2,000자
- 영어 단어 혼용 금지, 외래어는 한국어 발음으로만 표기
```

#### 카테고리별 필수 포함 요소

| 카테고리 | 필수 요소 | 선택 요소 |
|---|---|---|
| 수사기관 사칭형 | ① 피해자 명의 대포통장 발견 ② 수사관/검사 신분 강조 ③ 계좌 동결/압수 ④ 긴박감 조성 ⑤ 제3자 차단 | 녹취 강조 문구 ("이 통화는 녹취됩니다") |
| 대출 사기형 | ① 금리/한도 구체적 안내 ② 부결/신용점수 문제 ③ 편법 대환대출 제안 ④ 선납금 요구 ⑤ 마감 긴박감 | — |

#### 실행 명령어

```bash
# 사전 준비
pip install openai
set OPENAI_API_KEY=sk-...

# 소규모 테스트 (품질 확인)
python augment_phishing.py --n-per-category 10 --per-call 5

# 전체 실행 (카테고리별 200건)
python augment_phishing.py --n-per-category 200 --per-call 5

# 중단 후 재개
python augment_phishing.py --n-per-category 200 --resume

# 모델 변경 (기본: gpt-5.4-mini)
python augment_phishing.py --model gpt-5.5 --n-per-category 200
```

#### CLI 옵션 요약

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--n-per-category` | 200 | 카테고리별 생성 목표 건수 |
| `--per-call` | 5 | API 호출당 생성 건수 |
| `--few-shot-k` | 10 | few-shot 예시 수 |
| `--model` | gpt-5.4-mini | 사용할 OpenAI 모델 |
| `--seed` | 42 | 랜덤 시드 |
| `--resume` | False | 기존 결과에서 이어서 생성 |

#### 출력
- 파일: `augmented/phishing_augmented.csv`
- 컬럼: `id, text, label, category, source`
- `source` 값: `augmented_llm`

---

### Stage 2: ASR 노이즈 주입 (`augment_asr_noise.py`)

#### 목적
- LLM이 생성한 깨끗한 텍스트에 실제 Whisper STT 오류 패턴 적용
- 모델이 STT 오류에 강건(robust)해지도록 학습

#### 노이즈 파라미터 (실측 오류율 기반)
- `p_sub` (음절 치환): ~7.8% — 유사 음소(ㄱ↔ㅋ, ㅐ↔ㅔ 등)로 치환
- `p_del` (음절 삭제): ~8.2% — 무작위 음절 제거
- `p_ins` (삽입): ~3.6% — "음", "어", "아", "그" 등 필러 삽입
- `p_space` (공백 오류): ~4.2% — 공백 삭제 또는 추가

#### 오류 파라미터 출처
- `error_analysis/error_summary.json` — `whisper_error_analysis.py`로 측정한 실측값

#### 실행 명령어

```bash
python augment_asr_noise.py \
  --input-llm augmented/phishing_augmented.csv \
  --error-summary error_analysis/error_summary.json \
  --output augmented/asr_noised.csv
```

---

### Stage 3: 최종 데이터셋 생성 (`build_training_dataset.py`)

#### 처리 내용
1. 원본 STT(`all.csv`) + LLM 증강(`phishing_augmented.csv`) + ASR 노이즈(`asr_noised.csv`) 합산
2. **파일 단위 분할**: 원본 음성 파일명 기준으로 분할 → 동일 파일의 증강본이 같은 split에 배치됨 (데이터 누수 방지)
3. 분할 비율: train 80% / val 10% / test 10%

#### 실행 명령어

```bash
python build_training_dataset.py \
  --input-llm augmented/phishing_augmented.csv \
  --input-asr augmented/asr_noised.csv \
  --output-dir final/
```

#### 출력
- `final/train.csv`, `final/val.csv`, `final/test.csv`
- `final/dataset_stats.json` — 건수, 라벨 분포, 증강 비율 요약

---

## 5. 실제 서비스와의 일관성

### 서비스 시나리오
실제 서비스에서 실시간 음성이 입력될 때 Whisper VAD가 자연스럽게 발화를 문장 단위로 분절한다. 따라서 학습 데이터도 문장 단위로 분절하여 사용하는 것이 실제 배포 환경과 일치한다.

```
[실시간 음성 입력]
       │
  Whisper STT + VAD
       │
  발화 → 문장 단위 분절
       │
  분류 모델 입력 (streaming_belief_v5 형식)
```

### 학습-서비스 일관성 보장 사항
- STT 전사 형식 동일: 화자 분리 없는 연속 텍스트
- 노이즈 패턴 일치: 실측 Whisper 오류율로 증강 데이터 오염
- 문장 단위 입력: 학습/추론 모두 동일한 분절 단위 사용

---

## 6. 파일 구조

```
models/classifier/preprocessing/
├── augmented_pipeline/
│   ├── pipeline_config.py          # 증강 파이프라인 설정
│   ├── prompt_loader.py            # 프롬프트 로더
│   ├── prompts/
│   │   ├── system_prompt.txt       # 시스템 프롬프트
│   │   └── category_prompts.json   # 카테고리별 프롬프트
│   ├── clean_transcription/
│   │   ├── transcribe_one_sample.py        # 오디오 1개 샘플 전사 테스트
│   │   ├── build_clean_fewshot_from_audio.py # 오디오 재전사 기반 clean few-shot 생성
│   │   └── README.md                       # clean 전사 실행 가이드
│   ├── augment_phishing.py         # [Stage 1] LLM 텍스트 생성
│   ├── augment_asr_noise.py        # [Stage 2] ASR 노이즈 주입
│   ├── build_clean_fewshot_from_audio.py   # clean_transcription 래퍼
│   └── build_training_dataset.py   # [Stage 3] 최종 데이터셋 생성
├── augment_phishing.py             # 호환 래퍼
├── augment_asr_noise.py            # 호환 래퍼
├── build_training_dataset.py       # 호환 래퍼
├── batch_transcribe.py             # [Stage 0] 원본 음성 → STT
├── whisper_error_analysis.py       # STT 오류율 측정
├── transcriptions/
│   └── gpu_small/
│       ├── phishing.csv            # 원본 피싱 STT (514건)
│       ├── normal.csv              # 원본 일반 STT
│       └── all.csv                 # 합본
├── augmented/
│   ├── phishing_augmented.csv      # LLM 생성 텍스트
│   └── asr_noised.csv              # ASR 노이즈 적용본
├── error_analysis/
│   └── error_summary.json          # 실측 오류율
└── final/
    ├── train.csv
    ├── val.csv
    ├── test.csv
    └── dataset_stats.json
```

---

## 7. 남은 작업 (Todo)

- [ ] `pip install openai` 설치 및 `OPENAI_API_KEY` 설정
- [ ] `--n-per-category 10` 소규모 테스트로 생성 품질 확인
- [ ] 생성 샘플 검수 후 전체 규모 실행 (`--n-per-category 200`)
- [ ] `whisper_error_analysis.py` 실행하여 `error_summary.json` 갱신 확인
- [ ] `augment_asr_noise.py` 실행
- [ ] `build_training_dataset.py` 실행 → `final/train.csv` 생성
- [ ] (선택) 일반 데이터도 동일한 증강 파이프라인 적용 검토

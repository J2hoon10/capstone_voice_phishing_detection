# [Classifier] 학습 데이터 전처리 파이프라인 수정 계획

## 작성일
2026-03-10

## 상태
- [x] 계획 수립
- [x] 개발 진행 중
- [ ] 테스트 완료
- [ ] PR 생성
- [ ] 머지 완료

## 브랜치
`feature/classifier-preprocessing-cleanup`

## Context
학습 데이터 전처리 코드(`Project/Audio2Text/audio_processor.py`)에서 불필요하거나 부적합한 텍스트 후처리를 정리하고, Whisper 설정을 조정하여 반복 단어 문제를 해결한다. 데이터셋 구성 변경은 보류.

## 수정 대상 파일
- `Project/Audio2Text/audio_processor.py`

## 변경 사항

### 1. PII 마스킹 제거
키워드 기반 분류에 방해가 되므로 제거.

- `mask_pii()` 메서드 삭제 (33~48행)
- `clean_text_basic()` 내 `self.mask_pii(text)` 호출 제거 (74행)
- `is_valid_sentence()`에서 마스킹 토큰 필터 조건 제거 (93~95행)
- 파일 상단 주석(1~5행) 업데이트

### 2. 블랙리스트 필터 제거
STT 결과를 있는 그대로 보고 판단하기 위해 제거.

- `self.blacklist` 리스트 삭제 (24~30행)
- `is_valid_sentence()`에서 블랙리스트 체크 로직 제거 (84~86행)

### 3. 반복 단어 해결: Whisper 설정 조정
텍스트 후처리 대신 Whisper 파라미터로 원인 억제.

- `remove_phrase_repetition()` 메서드 삭제 (51~58행)
- `clean_text_basic()` 내 `self.remove_phrase_repetition(text)` 호출 제거 (77행)
- Whisper `transcribe()` 설정 변경 (125~133행):
  - `repetition_penalty`: 1.2 → **1.5** (반복 억제 강화)
  - `no_speech_threshold`: 0.6 → **0.4** (비음성 구간 감지 민감도 증가)

### 4. is_valid_sentence() 정리
위 제거 후 남는 로직:
- 최소 길이 2자 체크 (유지)
- 무의미한 반복 문자 체크 (유지: "네네네네네" 등 고유 문자 비율 < 10%)

## 변경하지 않는 것
- `audio_enhancer.py` — 오디오 향상 파이프라인 (현행 유지)
- `dataset_builder.py`, `csv_merger.py` — 데이터셋 구성 (보류)
- `clean_text_basic()`의 특수문자 제거, 구두점 정리, 공백 정리 (유지)
- `remove_duplicates()` — 순차 중복 문장 제거 (유지)

## 수정 후 clean_text_basic 흐름
```
텍스트 입력
  → 특수문자 제거 (한글/영문/숫자/기본구두점만 유지)
  → 구두점 정리
  → 공백 정리
  → 출력
```

## 수정 후 is_valid_sentence 흐름
```
문장 입력
  → 2자 미만이면 제거
  → 무의미한 반복 문자(고유 문자 비율 < 10%)이면 제거
  → 통과
```

## 테스트 계획
1. 수정 후 테스트 오디오로 `process_file()` 실행하여 정상 동작 확인
2. Whisper 반복 억제 효과 확인: 침묵 구간이 포함된 오디오로 테스트
3. PII/블랙리스트 제거 후 텍스트에 해당 내용이 그대로 유지되는지 확인

## 비고
- 데이터셋 구성 변경은 별도 계획서로 진행 예정

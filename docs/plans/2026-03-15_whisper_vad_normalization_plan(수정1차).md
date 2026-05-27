# PLAN-2026-03-15 Whisper VAD 통일 + 음량 정규화 개선

## 목적
- STT 반복/환각 억제 전략을 적용하기 위한 1단계로 VAD 정책을 단일화한다.
- 음량 편차를 완화해 STT 입력 품질을 안정화한다.

## 결정 사항
- VAD는 Whisper VAD만 사용한다.
- Silero VAD는 기본 비활성화한다(필요 시 실험 옵션으로만 사용).
- RMS 정규화에 저음량 증폭 제한과 간단 리미터를 추가한다.

## 적용 범위
- 서비스 추론: `models/classifier/audio_processor.py`
- 데이터 전처리: `models/classifier/preprocessing/batch_transcribe.py`
- STT 미리보기: `models/classifier/preprocessing/stt_preview.py`
- 전처리 공통: `models/classifier/audio_enhancer.py`

## 변경 요약
1. Silero VAD 기본 비활성화
2. 음량 정규화 로직 강화
   - min_rms 기준 도입
   - peak 리미터 추가
3. 반복 루프 억제 적용
   - Whisper 디코딩 파라미터 강화
   - STT 결과 반복 축약 후처리 추가

## 확인 방법
- `stt_preview.py`로 30~50개 샘플 전사
- 반복/환각 발생률과 문장 자연성 확인
- 처리 시간 변화 측정

## 후속 작업
- 반복 억제 파라미터 추가 적용
- 오류 분포 재측정 및 증강 규칙 재정렬

## 적용 상세
- `models/classifier/audio_enhancer.py`
  - Silero VAD 기본 비활성화
  - 정규화에 `min_rms` 기준과 peak 리미터 추가
- `models/classifier/audio_processor.py`
  - 반복 축약 후처리(`remove_repetitions`) 추가
  - STT 파라미터 추가:
    - `no_repeat_ngram_size=3`
    - `compression_ratio_threshold=2.2`
    - `log_prob_threshold=-0.8`
    - `temperature=(0.0, 0.2, 0.4, 0.6)`
- `models/classifier/preprocessing/stt_preview.py`
  - 고정 샘플 manifest 추가 (`error_analysis/stt_preview_manifest.csv`)
  - STT 파라미터 동일 적용
- `models/classifier/preprocessing/batch_transcribe.py`
  - STT 파라미터 동일 적용

## 파라미터 설명
- `repetition_penalty=1.2`
  - 반복 토큰에 패널티를 부여해 동일 단어의 반복 출력을 완화.
- `no_repeat_ngram_size=3`
  - 동일 3-gram 반복을 차단해 루프성 반복 문장 발생을 억제.
- `compression_ratio_threshold=2.2`
  - gzip 기반 압축률로 반복 여부를 감지. 낮출수록 반복 감지 민감도 증가.
- `log_prob_threshold=-0.8`
  - 평균 로그확률이 낮은 세그먼트를 재시도하도록 유도해 저신뢰 반복 구간 완화.
- `temperature=(0.0, 0.2, 0.4, 0.6)`
  - 낮은 온도에서 실패(압축률/로그확률 기준) 시 점진적으로 샘플링 다양성을 높여 반복 루프 탈출을 유도.

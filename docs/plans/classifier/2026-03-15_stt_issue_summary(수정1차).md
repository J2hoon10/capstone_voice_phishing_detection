# PLAN-2026-03-15 STT 문제 및 해결 정리

## 1. 문제 요약
- Whisper STT에서 **단어/문장 반복(반복 루프)** 현상이 발생.
- 무음/잡음 구간과 발음이 불명확한 구간에서 특히 심함.
- STT 결과 품질 저하로 분류 입력이 불안정해짐.

## 2. 원인 가설
- Whisper의 autoregressive 디코딩 특성상 불명확 입력에서 동일 토큰 반복 선택.
- 세그먼트 간 문맥 전파가 반복 루프를 증폭.
- 무음/잡음 구간에서 VAD 처리 미흡 시 반복 발생 가능.
- 음량 편차가 큰 데이터는 인식 품질을 더 악화시킴.

## 3. 해결 접근
### 3.1 VAD 정책 단일화
- Silero VAD와 Whisper VAD의 중복 사용은 과도한 절단/지연을 유발.
- **Whisper VAD만 사용**하도록 정책을 통일하고 Silero VAD는 기본 비활성화.

### 3.2 음량 정규화 개선
- RMS 정규화에 `min_rms` 기준과 peak 리미터를 추가해 저음량 과증폭과 클리핑 방지.

### 3.3 디코딩 파라미터 강화
- 반복 억제를 위한 파라미터를 추가해 반복 루프를 완화.
  - `repetition_penalty=1.2`
  - `no_repeat_ngram_size=3`
  - `compression_ratio_threshold=2.2`
  - `log_prob_threshold=-0.8`
  - `temperature=(0.0, 0.2, 0.4, 0.6)`

### 3.4 샘플 고정 테스트
- 랜덤 샘플 대신 **고정 샘플 manifest**를 도입해 변경 전후 비교 가능.
- `stt_preview_manifest.csv`로 동일 샘플 반복 테스트.

## 4. 적용 파일
- `models/classifier/audio_enhancer.py`
- `models/classifier/audio_processor.py`
- `models/classifier/preprocessing/stt_preview.py`
- `models/classifier/preprocessing/batch_transcribe.py`

## 5. 확인 방법
```powershell
cd models\classifier\preprocessing
python stt_preview.py
```

## 6. 남은 과제
- 필요 시 Whisper VAD vs Silero VAD 품질 A/B 비교.
- 오류 분포 재측정 및 증강 규칙 재정렬.

# Classifier 담당자 가이드 (담당자A)

## 담당 범위

```
models/classifier/
├── app.py              # FastAPI 앱 — 엔드포인트 정의 (/health, /predict)
├── inference.py         # 추론 파이프라인 — VoicePhishingDetector, PhishingRiskScorer
├── audio_processor.py   # 오디오 전처리 — 필터링, 노이즈 제거, VAD
├── audio_enhancer.py    # 오디오 품질 향상
├── architecture.py      # 모델 아키텍처 정의 (ModernBERT 기반)
├── config.py            # 설정 관리 — 경로, 디바이스, 임계값
├── requirements.txt     # Python 의존성
├── Dockerfile           # 컨테이너 빌드 정의
└── weights/             # 모델 가중치 (git 제외)
    └── student_best.pt
```

## 브랜치 네이밍

```
feature/classifier-<작업내용>
```

예시:
- `feature/classifier-noise-reduction` — 노이즈 제거 개선
- `feature/classifier-vad-tuning` — VAD 파라미터 조정
- `feature/classifier-risk-scoring` — 리스크 스코어링 로직 수정

## 주요 파일별 역할

### `inference.py`
핵심 추론 로직이 담긴 파일입니다.

- `VoicePhishingDetector`: Whisper로 음성→텍스트 변환 후 ModernBERT로 분류
- `PhishingRiskScorer`: Leaky Bucket 알고리즘 기반 리스크 점수 산출
  - Level 1 (주의): 30점 이상
  - Level 2 (경고): 60점 이상

### `audio_processor.py`
오디오 신호 전처리 파이프라인입니다.

- 대역 필터링 (Bandpass filter)
- 노이즈 제거 (noisereduce)
- VAD (Voice Activity Detection)
- 텍스트 후처리: 개인정보 마스킹, 반복 제거

### `config.py`
모든 설정값의 진입점입니다. 환경변수와 기본값을 관리합니다.

## 수정 시 주의사항

1. **`inference.py`의 `VoicePhishingDetector`를 수정할 때**
   - 모델 로딩 로직은 thread-safe 하게 유지 (lazy loading with lock)
   - 디바이스(cuda/cpu) 호환성 반드시 확인

2. **`audio_processor.py` 수정 시**
   - 오디오 포맷 호환성 확인 (wav, mp3, webm 등)
   - 처리 시간이 늘어나면 WebSocket 스트리밍 타임아웃에 영향

3. **`architecture.py` 수정 시**
   - 기존 가중치(`student_best.pt`)와의 호환성 반드시 확인
   - 레이어 구조 변경 시 팀에 공유

4. **의존성 추가 시**
   - `requirements.txt`에 버전 명시
   - `Dockerfile`에서 빌드 확인

## 로컬 테스트 방법

### 서비스 단독 실행

```bash
cd models/classifier
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

### API 테스트

```bash
# 헬스체크
curl http://localhost:8001/health

# 예측 요청
curl -X POST http://localhost:8001/predict \
  -F "audio=@/path/to/test.wav" \
  -F "threshold=0.5"
```

### Docker 단독 실행

```bash
docker compose up --build classifier
```

## 관련 의존성

주요 라이브러리 (자세한 목록은 `requirements.txt` 참고):

| 라이브러리 | 용도 |
|-----------|------|
| `fastapi`, `uvicorn` | API 서버 |
| `torch` | 모델 추론 |
| `transformers` | ModernBERT 토크나이저 |
| `faster-whisper` | 음성→텍스트 변환 |
| `librosa` | 오디오 처리 |
| `noisereduce` | 노이즈 제거 |

## 커밋 예시

```bash
git add models/classifier/audio_processor.py
git commit -m "[classifier] VAD 감도 파라미터 조정

기존 에너지 기반 VAD에서 오탐이 빈번하여
최소 음성 길이 임계값을 300ms → 500ms로 조정"
```

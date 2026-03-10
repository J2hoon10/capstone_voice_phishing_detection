# API 명세

## 1. Classifier 서비스 (담당자A)

> 포트: `8001` | 디렉터리: `models/classifier/`

### `GET /health`

서비스 상태 확인.

```json
{
  "status": "ok|degraded",
  "model_loaded": true,
  "device": "cuda",
  "model_path": "/app/weights/student_best.pt",
  "error": null
}
```

### `POST /predict`

음성 파일을 받아 보이스피싱 여부를 판별.

- **Content-Type**: `multipart/form-data`
- **필드**:
  - `audio`: 음성 파일 (wav, mp3 등)
  - `threshold`: float (0~1, 선택)
- **응답**:

```json
{
  "status": "success",
  "is_phishing": true,
  "max_risk_score": 87.3,
  "dangerous_segment": "계좌 동결 처리를 위해...",
  "filename": "sample.wav"
}
```

---

## 2. Guidance 서비스 (담당자B, C)

> 포트: `8002` | 디렉터리: `models/guidance/`

### `GET /health`

```json
{ "status": "ok" }
```

### `POST /guidance`

위험 점수와 텍스트를 기반으로 대응 지침을 생성.

- **요청**:

```json
{
  "risk_score": 75,
  "warning_level": "WARNING",
  "text": "검찰에서 계좌 동결..."
}
```

- **응답**:

```json
{
  "risk_score": 75,
  "warning_level": "WARNING",
  "text": "...",
  "guidance": {
    "matched_type": "impersonation_investigation",
    "matched_label": "수사기관 사칭형",
    "summary": "...",
    "actions": ["즉시 전화를 끊으세요", "..."],
    "emergency_contacts": [
      { "name": "경찰청", "phone": "112", "description": "긴급 신고" }
    ],
    "banks_notice": "..."
  }
}
```

---

## 3. Backend 게이트웨이

> 포트: `8000` | 디렉터리: `backend/`

### `GET /health`

```json
{ "status": "ok" }
```

### `POST /api/detect`

음성 파일 업로드 → classifier + guidance 통합 결과 반환.

- **Content-Type**: `multipart/form-data`
- **필드**: `audio`, `threshold` (선택)
- **응답**:

```json
{
  "status": "success",
  "is_phishing": true,
  "max_risk_score": 83.4,
  "dangerous_segment": "...",
  "warning_level": "WARNING",
  "guidance": { "..." },
  "raw": { "classifier 원본 응답" }
}
```

### `POST /api/guidance`

guidance 서비스에 직접 요청 전달.

### `WS /ws/stream`

실시간 마이크 스트리밍 엔드포인트.

- **클라이언트 → 서버**: 바이너리 오디오 청크 (`audio/webm`) 또는 JSON `{ "event": "reset" }`
- **서버 → 클라이언트** (`event=prediction`):

```json
{
  "event": "prediction",
  "status": "success",
  "risk_score": 64.0,
  "score": 70.0,
  "warning_level": "WARNING",
  "transcript": "...",
  "guidance": { "..." }
}
```

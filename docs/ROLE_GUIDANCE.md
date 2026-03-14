# Guidance 담당자 가이드 (담당자B, C)

## 담당 범위

```
models/guidance/
├── app.py                 # FastAPI 앱 — 엔드포인트 정의 (/health, /guidance)
├── guidance_engine.py      # 대응 지침 생성 엔진
├── knowledge_base/         # 피싱 유형 및 긴급 연락처 데이터
│   ├── phishing_types.json
│   └── emergency_contacts.json
├── requirements.txt        # Python 의존성
└── Dockerfile              # 컨테이너 빌드 정의
```

## 2명 협업 분담 전략

Guidance 서비스는 2명이 함께 작업하므로 **파일 단위로 분담**하여 충돌을 최소화합니다.

### 권장 분담 방식

| 담당자 | 주요 담당 파일 | 작업 내용 |
|--------|---------------|-----------|
| **담당자B** | `guidance_engine.py`, `app.py` | 엔진 로직, API 엔드포인트, 새 기능 개발 |
| **담당자C** | `knowledge_base/*.json` | 피싱 유형 데이터, 키워드, 긴급 연락처 관리 |

### 충돌 방지 원칙

1. **같은 파일을 동시에 수정하지 않기**
   - 작업 시작 전 Slack/카톡으로 "오늘 `guidance_engine.py` 수정합니다" 공유
2. **소규모 커밋 + 자주 push**
   - 큰 변경을 한번에 하지 말고, 작은 단위로 나눠 커밋

## 작업 브랜치

담당자B는 `feature/guidance-B`, 담당자C는 `feature/guidance-C`로 브랜치가 고정되어 있습니다. 새로 생성할 필요 없이 아래 명령어로 작업을 시작하세요.

### 작업 시작 전 (매번 필수)

```bash
# Step 1. 원격 저장소의 최신 변경 내역을 가져옵니다.
git fetch origin

# Step 2. 내 작업 브랜치로 이동합니다.
git checkout feature/guidance-B   # 담당자B
git checkout feature/guidance-C   # 담당자C

# Step 3. 원격 dev/final 최신 상태를 기준으로 재정렬합니다.
git rebase origin/dev/final
```

## 주요 파일별 역할

### `guidance_engine.py`

대응 지침 생성의 핵심 로직입니다.

- `GuidanceEngine` 클래스: 피싱 knowledge base를 로드하여 지침 생성
- `_match_type()`: 텍스트에서 키워드 기반으로 피싱 유형 매칭
- `_default_actions()`: 경고 레벨에 따른 기본 대응 행동 생성
- `build_guidance()`: 최종 대응 지침 구성

### `knowledge_base/phishing_types.json`

피싱 유형별 키워드, 요약, 권장 행동이 정의되어 있습니다.

현재 등록된 유형:
| 유형 ID | 한국어명 | 설명 |
|---------|---------|------|
| `loan_fraud` | 대출 사기형 | 대출/선수금 사기 |
| `impersonation_investigation` | 수사기관 사칭형 | 검찰/경찰 사칭 |
| `voice_spoofing` | 지인 사칭형 | 가족/지인 사칭 |

### `knowledge_base/emergency_contacts.json`

긴급 연락처 정보 (경찰, 금융감독원 등).

## 수정 시 주의사항

1. **`phishing_types.json` 수정 시**
   - JSON 문법 오류 주의 (쉼표, 괄호)
   - 새 유형 추가 시 `guidance_engine.py`의 매칭 로직과 호환되는지 확인
   - 키워드가 다른 유형과 중복되지 않는지 확인

2. **`guidance_engine.py` 수정 시**
   - `_match_type()` 로직 변경 시 기존 피싱 유형이 정상 매칭되는지 테스트
   - 새 필드 추가 시 `app.py`의 응답 스키마도 함께 수정

3. **`app.py` 수정 시**
   - 엔드포인트 인터페이스 변경 시 backend 서비스의 `guidance_client.py`도 확인
   - Pydantic 모델(스키마) 변경 시 [api_spec.md](api_spec.md) 문서 업데이트

## 로컬 테스트 방법

### 서비스 단독 실행

```bash
cd models/guidance
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8002 --reload
```

### API 테스트

```bash
# 헬스체크
curl http://localhost:8002/health

# 대응 지침 요청 (수사기관 사칭 시나리오)
curl -X POST http://localhost:8002/guidance \
  -H "Content-Type: application/json" \
  -d '{
    "risk_score": 75,
    "warning_level": "WARNING",
    "text": "검찰에서 나왔는데 계좌가 범죄에 연루되어 동결 처리해야 합니다"
  }'

# 대응 지침 요청 (대출 사기 시나리오)
curl -X POST http://localhost:8002/guidance \
  -H "Content-Type: application/json" \
  -d '{
    "risk_score": 50,
    "warning_level": "CAUTION",
    "text": "저금리 대출 받으시려면 먼저 수수료를 입금해주셔야 합니다"
  }'
```

### Docker 단독 실행

```bash
docker compose up --build guidance
```

### JSON 유효성 검사

knowledge_base JSON 수정 후 문법 오류 확인:

```bash
python -c "import json; json.load(open('knowledge_base/phishing_types.json', encoding='utf-8')); print('OK')"
python -c "import json; json.load(open('knowledge_base/emergency_contacts.json', encoding='utf-8')); print('OK')"
```

## 커밋 및 Push 예시

```bash
# 담당자B: 엔진 로직 수정 후 커밋 & push
git add models/guidance/guidance_engine.py
git commit -m "[guidance] 피싱 유형 매칭 시 복합 키워드 지원

단일 키워드 매칭에서 AND/OR 조합 매칭으로 확장하여
오탐률을 낮춤"
git push -u origin feature/guidance-B

# 담당자C: knowledge base 수정 후 커밋 & push
git add models/guidance/knowledge_base/phishing_types.json
git commit -m "[guidance] 보험 사기 유형 키워드 추가"
git push -u origin feature/guidance-C

# rebase 후 push가 거부될 경우 (히스토리 변경으로 인한 정상 현상)
git push --force-with-lease origin feature/guidance-B   # 담당자B
git push --force-with-lease origin feature/guidance-C   # 담당자C
```

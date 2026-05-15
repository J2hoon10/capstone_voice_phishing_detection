# AIShield Demo

캡스톤디자인 프로젝트
모바일 앱 형태의 보이스피싱 탐지 데모
음성 파일 업로드로 백엔드 `/api/detect`를 호출하고, 결과에 따라 안전 화면 또는 피싱 경고 화면으로 이동

## 실행

```bash
cd demo
npm install
npm run dev
```

기본 개발 서버: `http://localhost:5174`

## 백엔드 연결

개발 서버는 기본적으로 `/api` 요청을 `http://localhost:8000`으로 프록시함
백엔드 주소가 다르면 환경변수로 지정할 수 있음

```bash
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

## 5-label 모델 연결

데모는 다음 5개 라벨을 표시함

- 상품 가입 및 해지
- 이체 출금 대출서비스
- 잔고 및 거래내역
- 수사기관 사칭형
- 대출 사기형

classifier 서비스를 RoBERTa AvgPool 5-label 모델로 실행하려면 체크포인트 파일을 준비한 뒤 아래 환경변수를 지정

```bash
CLASSIFIER_MODEL_KIND=roberta_avgpool_5
FIVE_LABEL_MODEL_PATH=/app/weights/roberta_avgpool_20260514_225702_best.pt
```

Docker Compose를 쓸 때는 해당 `.pt` 파일을 `models/classifier/weights/` 아래에 두고 `.env`에 같은 값을 적으면 됨

현재 repo에는 5-label `.pt` 체크포인트가 포함되어 있지 않으므로, weight 파일이 없으면 classifier health가 `degraded`로 표시됨

## 포함된 흐름

- Home Dashboard
- 음성 파일 업로드 분석 화면
- 분석 중 상태
- 모델 서버 실패 시 샘플 fallback
- Phishing Alert
- Safe Result
- Phishing Detection Details
- 신고 완료 화면
- 최근 기록 localStorage 저장

## API 매핑

- `max_risk_score`: 위험도와 탐지 확신도
- `is_phishing`: 피싱 경고 화면 진입 여부
- `dangerous_segment`: 의심 문구
- `guidance.summary`: AI 분석 설명
- `guidance.actions`: 안전 수칙
- `pred_label`, `guidance.matched_label` 또는 `raw.class_label`: 예측 유형
- `class_probs` 또는 `raw.class_probs`: 5-class 확률

# [Guidance-C] 작업 계획서

## 작업명
<!-- 간단한 작업 제목 -->

## 상태
- [ ] 계획 수립
- [ ] 개발 진행 중
- [ ] 테스트 완료
- [ ] PR 생성
- [ ] 머지 완료

## 브랜치
`feature/guidance-<작업내용>`

## 목표
<!-- 이 작업으로 달성하려는 것 -->

## 변경 대상 파일
<!-- 수정할 파일 목록 — 담당자B와 겹치지 않는지 확인 -->
- `models/guidance/knowledge_base/phishing_types.json`
- `models/guidance/knowledge_base/emergency_contacts.json`

## 작업 내용
<!-- 구체적인 작업 단계 -->
1.
2.
3.

## JSON 변경 사항
<!-- 추가/수정하는 데이터 요약 -->

## 테스트 계획
```bash
# JSON 유효성 검사
python -c "import json; json.load(open('knowledge_base/phishing_types.json', encoding='utf-8')); print('OK')"

# API 테스트
curl -X POST http://localhost:8002/guidance \
  -H "Content-Type: application/json" \
  -d '{"risk_score":50,"warning_level":"CAUTION","text":"테스트 텍스트"}'
```

## 담당자B와 조율 사항
<!-- 엔진 로직 변경이 필요하면 여기에 기록하고 담당자B에게 공유 -->

## 예상 일정
- 시작일:
- 완료 예정일:

## 비고

# [Guidance-B] 작업 계획서

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
<!-- 수정할 파일 목록 — 담당자C와 겹치지 않는지 확인 -->
- `models/guidance/guidance_engine.py`
- `models/guidance/app.py`

## 작업 내용
<!-- 구체적인 작업 단계 -->
1.
2.
3.

## 테스트 계획
```bash
curl -X POST http://localhost:8002/guidance \
  -H "Content-Type: application/json" \
  -d '{"risk_score":75,"warning_level":"WARNING","text":"테스트 텍스트"}'
```

## 담당자C와 조율 사항
<!-- knowledge_base 변경이 필요하면 여기에 기록하고 담당자C에게 공유 -->

## 예상 일정
- 시작일:
- 완료 예정일:

## 비고

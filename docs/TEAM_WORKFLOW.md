# 팀 작업 가이드

## 팀 구성 및 역할 분담

| 담당자 | 담당 영역 | 디렉터리 | 주요 작업 |
|--------|-----------|----------|-----------|
| **담당자A** | Classifier 서비스 | `models/classifier/` | 음성 전처리, 모델 추론, 리스크 스코어링 |
| **담당자B** | Guidance 서비스 | `models/guidance/` | 대응 지침 엔진, 피싱 유형 로직 |
| **담당자C** | Guidance 서비스 | `models/guidance/` | knowledge base 관리, 긴급 연락처 |

> 담당자B, C의 세부 분담은 [ROLE_GUIDANCE.md](ROLE_GUIDANCE.md)를 참고하세요.

---

## Git 브랜치 전략

### 브랜치 구조

```text
main                          ← 안정 배포 브랜치 (직접 push 금지)
 └── dev/final                ← 통합 개발 브랜치 (PR 머지 대상)
      ├── feature/classifier-* ← 담당자A 작업 브랜치
      ├── feature/guidance-*   ← 담당자B, C 작업 브랜치
      └── fix/*               ← 버그 수정 브랜치
```

### 브랜치 네이밍 규칙

| 접두사 | 용도 | 예시 |
|--------|------|------|
| `feature/classifier-` | Classifier 기능 개발 | `feature/classifier-noise-reduction` |
| `feature/guidance-` | Guidance 기능 개발 | `feature/guidance-new-phishing-type` |
| `fix/` | 버그 수정 | `fix/classifier-memory-leak` |
| `docs/` | 문서 수정 | `docs/update-api-spec` |

### 브랜치 생성 방법

```bash
# 1. dev/final 브랜치를 최신 상태로 업데이트
git checkout dev/final
git pull origin dev/final

# 2. 작업 브랜치 생성
git checkout -b feature/classifier-noise-reduction
```

---

## 커밋 컨벤션

### 커밋 메시지 형식

```
[서비스명] 변경 요약

본문 (선택): 변경 이유나 상세 설명
```

### 서비스 태그

| 태그 | 사용 시점 |
|------|-----------|
| `[classifier]` | models/classifier 관련 변경 |
| `[guidance]` | models/guidance 관련 변경 |
| `[backend]` | backend 관련 변경 |
| `[frontend]` | frontend 관련 변경 |
| `[docs]` | 문서 변경 |
| `[config]` | Docker, .env 등 설정 변경 |

### 커밋 메시지 예시

```
[classifier] Whisper 모델 로딩 시 메모리 최적화

기존 eager 로딩에서 lazy 로딩으로 변경하여
첫 요청 시에만 모델을 메모리에 올리도록 수정
```

```
[guidance] 대출 사기 유형 키워드 추가
```

```
[docs] API 명세 업데이트
```

---

## 작업 플로우

### 전체 흐름

```
1. 브랜치 생성 (dev/final 기반)
       ↓
2. 로컬에서 코드 작성 & 테스트
       ↓
3. 커밋 (컨벤션에 맞게)
       ↓
4. 원격에 Push
       ↓
5. Pull Request 생성 (→ dev/final)
       ↓
6. 코드 리뷰 (최소 1명)
       ↓
7. 승인 후 Merge
       ↓
8. 작업 브랜치 삭제
```

### 상세 명령어

```bash
# === 1. 브랜치 생성 ===
git checkout dev/final
git pull origin dev/final
git checkout -b feature/guidance-new-type

# === 2~3. 작업 및 커밋 ===
# (코드 수정 후)
git add models/guidance/guidance_engine.py
git commit -m "[guidance] 새로운 피싱 유형 매칭 로직 추가"

# === 4. Push ===
git push -u origin feature/guidance-new-type

# === 5. PR 생성 ===
# GitHub 웹에서 Pull Request 생성
# - base: dev/final
# - compare: feature/guidance-new-type
# - 리뷰어 지정

# === 7~8. 머지 후 정리 ===
git checkout dev/final
git pull origin dev/final
git branch -d feature/guidance-new-type
```

---

## Pull Request 규칙

### PR 작성 시

1. **제목**: 커밋 메시지와 동일한 태그 사용 (예: `[guidance] 새로운 피싱 유형 추가`)
2. **본문**에 포함할 내용:
   - 무엇을 변경했는지
   - 왜 변경했는지
   - 테스트 방법
3. **base 브랜치**: 반드시 `dev/final`

### PR 템플릿

```markdown
## 변경 사항
-

## 변경 이유
-

## 테스트 방법
- [ ] 로컬에서 서비스 실행 확인
- [ ] curl로 API 정상 응답 확인
```

### 코드 리뷰

- PR은 **최소 1명**의 리뷰 승인 후 머지
- Classifier 변경 → 담당자B 또는 C가 리뷰
- Guidance 변경 → 담당자A가 리뷰 (또는 Guidance 다른 담당자)
- 리뷰어는 24시간 이내에 리뷰 완료

---

## 충돌 해결 방법

### 기본 절차

```bash
# 1. dev/final 최신 변경 가져오기
git checkout dev/final
git pull origin dev/final

# 2. 작업 브랜치로 돌아가서 rebase
git checkout feature/guidance-new-type
git rebase dev/final

# 3. 충돌 발생 시 수동 해결
# (충돌 파일 편집 후)
git add <충돌_해결한_파일>
git rebase --continue

# 4. 강제 push (rebase 후에는 필요)
git push --force-with-lease
```

### Guidance 담당자 간 충돌 방지

담당자B와 C가 같은 디렉터리에서 작업하므로 충돌 가능성이 높습니다.
자세한 분담 전략은 [ROLE_GUIDANCE.md](ROLE_GUIDANCE.md)를 참고하세요.

**핵심 원칙:**
- 같은 파일을 동시에 수정하지 않기
- 작업 시작 전 `git pull` 습관화
- 소규모 커밋으로 자주 push

---

## 주의사항

1. **`main` 브랜치에 직접 push하지 마세요** — 항상 `dev/final`로 PR
2. **`.env` 파일을 커밋하지 마세요** — `.gitignore`에 포함되어 있음
3. **모델 가중치(`.pt`)를 커밋하지 마세요** — `.gitignore`에 포함되어 있음
4. **작업 전에 항상 `dev/final`에서 최신 코드를 pull 하세요**

# Git 데이터 파일 추적 문제 정리

> 작성일: 2026-05-27  
> 현재 브랜치: `dev/final`

---

## 요약

현재 git 저장소에 원본 데이터 JSON 파일 **53,501개**가 커밋 히스토리에 포함되어 있음.  
결과적으로 `git objects` 총 용량이 **약 1.4 GB**에 달하며, `git clone` 시 이 데이터가 전부 다운로드됨.

---

## 1. 문제 원인

커밋 `991e77b` (mamba 실험 config 파일 수정) 시점에 `Data/Training/` 폴더의  
원본 데이터 JSON 파일들이 git에 추가된 채로 커밋됨.

이후 `.gitignore`에 `Data/`를 추가했지만,  
**이미 커밋된 파일은 `.gitignore`로 무시할 수 없기 때문에** 히스토리에 계속 남아있음.

현재 로컬 워킹트리에서는 파일이 삭제되어 있으나,  
git index에는 여전히 추적 중인 상태(` D` — unstaged deletion)임.

---

## 2. 현재 git 추적 파일 현황

### 🔴 문제: 원본 데이터 파일 (히스토리에 박혀있음)

| 경로 | 파일 수 | 상태 |
|------|---------|------|
| `Data/Training/TL_01. KAKAO(1)/` | JSON 다수 | 커밋됨, 로컬 삭제됨 (` D`) |
| `Data/Training/TL_01. KAKAO(2)/` | JSON 다수 | 커밋됨, 로컬 삭제됨 (` D`) |
| `Data/Training/TL_01. KAKAO(3)/` | JSON 다수 | 커밋됨, 로컬 삭제됨 (` D`) |
| **합계** | **53,501개** | git objects ≈ **1.4 GB** |

### ✅ 정상: `.gitignore`로 올바르게 제외된 경로

| 경로 | 내용 |
|------|------|
| `models/main/data_augmentation/output/` | 전처리 결과 CSV (train.csv, val.csv 등, 약 17 MB) |
| `models/main/model_architecture/data/` | 모델 학습용 데이터 |
| `models/classifier/preprocessing/output/` | 전처리 출력물 |

### ✅ 정상: 의도적으로 추적 중인 파일

| 경로 | 내용 |
|------|------|
| `models/experiments/**/logs/*_eval.json` | 실험 평가 결과 (`.gitignore` 예외 처리) |
| `models/experiments/**/checkpoints/*_latest.json` | 체크포인트 메타 (`.gitignore` 예외 처리) |
| `models/main/data_augmentation/error_analysis/*.csv` | STT 에러 분석 자료 |
| `models/main/data_augmentation/transcriptions/**/*.csv` | STT 변환 스크립트 원문 |
| `models/main/data_augmentation/phishing_augmentation/prompts/*.json` | LLM 프롬프트 |
| `models_analysis/output/*.json` | 모델 길이 분석 결과 |
| `streaming_test/results.json` | 스트리밍 테스트 결과 |

### 🟡 현재 수정됨 (unstaged modified)

아래 파일들은 마지막 커밋 이후 로컬에서 내용이 바뀐 상태 (아직 커밋 안 됨):

```
models/experiments/model_architecture/roberta_gru_freeze_init_4class/checkpoints/roberta_gru_freeze_init_4class_latest.json
models/experiments/model_architecture/roberta_lstm_freeze_init_4class/checkpoints/roberta_lstm_freeze_init_4class_latest.json
models/experiments/model_architecture/roberta_mamba_freeze_init_4class/checkpoints/roberta_mamba_freeze_init_4class_latest.json
models/experiments/model_architecture/roberta_mamba_w32_freeze_init_4class/checkpoints/roberta_mamba_w32_freeze_init_4class_latest.json
```

---

## 3. 해결 방법

### 방법 A — 간단한 방법: 삭제 커밋 (추적 해제만)

> 히스토리에서 파일은 유지되지만, 앞으로 추적하지 않음.  
> `git clone` 시 여전히 히스토리 포함 데이터가 다운로드됨.

```bash
git rm -r --cached Data/
git commit -m "chore: untrack Data/ raw dataset from git index"
```

### 방법 B — 근본 해결: 히스토리 전체 재작성 (권장)

> git 히스토리에서 `Data/` 폴더를 완전히 제거.  
> 저장소 크기를 대폭 줄일 수 있음.  
> **팀원 전체가 re-clone 필요.**

**BFG Repo Cleaner 사용 (빠르고 안전):**

```bash
# 1. BFG 설치 (없으면)
brew install bfg   # macOS
# 또는 https://rtyley.github.io/bfg-repo-cleaner/ 에서 .jar 다운로드

# 2. 히스토리에서 Data/ 폴더 완전 삭제
bfg --delete-folders Data --no-blob-protection .git

# 3. git reflog 정리 및 gc
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 4. 강제 push
git push origin --force --all
git push origin --force --tags
```

**git filter-repo 사용 (Python 도구):**

```bash
pip install git-filter-repo
git filter-repo --path Data/ --invert-paths
```

---

## 4. 재발 방지를 위한 `.gitignore` 현행 규칙 확인

현재 `.gitignore`에 아래 항목이 올바르게 설정되어 있음:

```gitignore
# Raw datasets
Data/
normal_data_reference/

# Preprocessed / augmented data
models/main/data_augmentation/output/
...

# Training logs (eval/latest JSON은 예외)
*_train.json
!**/*_eval.json
!**/*_latest.json
```

**주의:** `.gitignore`는 이미 커밋된 파일에는 적용되지 않음.  
새 파일을 추가하기 전 반드시 `git status`로 추적 여부를 확인할 것.

---

## 5. 권장 조치 순서

1. [ ] 팀원과 히스토리 재작성 일정 조율
2. [ ] **방법 B** 적용 (BFG 또는 filter-repo)
3. [ ] `git push --force` 후 팀원 전체 re-clone
4. [ ] 이후 `Data/` 폴더는 로컬에서만 유지하고 절대 `git add` 하지 않을 것

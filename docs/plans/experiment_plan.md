# 보이스피싱 탐지 모델 실험 계획서

> 최종 업데이트: 2026-05-13  
> 작성자: 캡스톤 팀

---

## 0. 데이터셋 현황

| 구분 | 샘플 수 | 비고 |
|------|---------|------|
| Train | 2,978건 | 학습에 사용 |
| Validation | - | 에폭별 성능 기준 |
| Test | 257건 | 최종 평가 기준 |

### 대화(텍스트) 길이 통계 (train.csv 기준)

| 항목 | 평균 | 중간값 | 최소 | 최대 |
|------|------|--------|------|------|
| 문장 수 | 21.4 | 16 | 2 | 429 |
| 어절 수 | 177.6 | 138 | 12 | 4,941 |
| 글자 수 | 734.8 | 576 | 46 | 19,857 |
| 토큰 수 (추정) | 213~266 | 166~207 | 14~18 | 5,929~7,412 |

> **분포 특성**: 평균 > 중간값으로 오른쪽으로 치우친 분포.  
> 중간값 기준 ~166~207 토큰으로 WINDOW_SIZE=128 적용 시 대부분 1~2 세그먼트 처리됨.

---

## 1. 원인 진단: Mamba 결합 모델 성능 저하 분석

단순 평균 풀링(KoELECTRA Only)보다 KoELECTRA+Mamba 조합이 성능이 낮았던 원인.

### 1-1. 시퀀스 부족
- 데이터 중간값(~166 토큰)을 WINDOW_SIZE=128로 자르면 **윈도우가 1~2개**에 불과
- Mamba는 긴 시퀀스(수십 개 이상)에서 상태 누적 효과가 발휘되는 모델
- 2개의 세그먼트로는 Mamba가 맥락을 누적할 기회 자체가 없음

### 1-2. Catastrophic Forgetting (가중치 붕괴)
- 학습이 완료된 KoELECTRA(수억 파라미터)와 무작위 초기화된 Mamba를 **동시에 같은 LR로** 학습
- 초기 Mamba의 큰 오차(loss)가 역전파를 통해 KoELECTRA의 사전학습 지식을 덮어씀
- 결과적으로 두 모델 모두 불안정한 상태에서 수렴

### 1-3. 키워드 희석
- 피싱 대화의 핵심 키워드("대포통장", "공증", "공무원 사칭" 등)가
  Mamba의 선형 상태 압축 과정에서 일반 대화 맥락과 혼합
- 특히 세그먼트가 1~2개일 때 Mamba 처리의 이점이 사실상 없음

---

## 2. 완료된 실험 결과

### 실험 결과 종합 (Test Set, n=257)

| 실험명 | Accuracy | Macro F1 | Weighted F1 | 상태 |
|--------|----------|----------|-------------|------|
| `koelectra_phishing5` (KoELECTRA Only) | **0.7860** | **0.8132** | **0.7881** | ✅ 완료 |
| `koelectra_mamba_phishing5` (KoELECTRA+Mamba, CE) | 0.7121 | 0.7438 | 0.7174 | ✅ 완료 |
| `koelectra_mamba_hce_ordinal` (KoELECTRA+Mamba, HCE+Ordinal) | 0.7121 | 0.7468 | 0.7151 | ✅ 완료 |

> **핵심 관찰**: Mamba 추가가 오히려 성능을 약 7% 저하시킴.  
> HCE+Ordinal 손실 추가는 Mamba 기본 CE 대비 미미한 개선(+0.003 Macro F1).

### 클래스별 F1 상세 (최신 완료 실험 비교)

| 클래스 | KoELECTRA Only | KoELECTRA+Mamba (HCE) | 차이 |
|--------|---------------|----------------------|------|
| 상품 가입 및 해지 | 0.770 | 0.707 | **-0.063** |
| 이체 출금 대출서비스 | 0.651 | 0.535 | **-0.116** |
| 잔고 및 거래내역 | 0.820 | 0.739 | **-0.081** |
| 수사기관 사칭형 | 0.928 | 0.901 | -0.027 |
| 대출 사기형 | 0.898 | 0.851 | -0.047 |

> 일반 클래스(0~2)에서 Mamba 결합 시 성능 하락이 두드러짐.  
> 피싱 클래스(3~4)는 상대적으로 영향 적음 → 짧은 시퀀스도 피싱은 키워드가 뚜렷함.

---

## 3. 진행 예정 실험 — 학습 전략 최적화

> 목표: Catastrophic Forgetting 방지 및 Mamba 초기 안정화를 통한 성능 회복

### 실험 A: `koelectra_mamba_freeze_init` — 동결 & 초기화

**폴더**: `models/classifier/experiments/koelectra_mamba_freeze_init/`

#### 전략
```
Phase 1 (Epoch 1~3): KoELECTRA 완전 동결
  → Mamba + Head만 학습 (LR = 5e-4)
  → Mamba 오차가 KoELECTRA에 역전파 차단

Phase 2 (Epoch 4~8): KoELECTRA 동결 해제
  → KoELECTRA: LR = 1e-5 (낮게)
  → Mamba + Head: LR = 5e-4 (유지)
```

#### 핵심 파라미터

| 파라미터 | 값 | 기존 대비 |
|---|---|---|
| `FREEZE_EPOCHS` | **3** | 신규 |
| `ENCODER_LR` (2단계) | **1e-5** | `2e-5` → 절반 |
| `UPPER_LR` | **5e-4** | 동일 |
| `EPOCHS` | 8 | 동일 |

#### 실행 방법
```bash
cd models\classifier\experiments\koelectra_mamba_freeze_init
python train.py
python evaluate.py --split test
```

#### 기대 효과
- Phase 1에서 Mamba가 KoELECTRA 표현 공간에 적응
- Phase 2에서 KoELECTRA가 안전하게 미세조정
- Catastrophic Forgetting 완전 차단 후 단계적 해제

---

### 실험 B: `koelectra_mamba_diff_lr` — 차등 학습률

**폴더**: `models/classifier/experiments/koelectra_mamba_diff_lr/`

#### 전략
```
Epoch 1~8 (단일 단계, 동결 없음):
  KoELECTRA: LR = 1e-5 (사전학습 지식 보호)
  Mamba + Head: LR = 1e-3 (빠른 수렴 유도)
  → LR 격차 100배로 암묵적 보호 효과
```

#### 핵심 파라미터

| 파라미터 | 값 | 기존 대비 |
|---|---|---|
| `ENCODER_LR` | **1e-5** | `2e-5` → 절반 |
| `UPPER_LR` | **1e-3** | `5e-4` → **2배** |
| LR 비율 (Mamba/ENC) | **100배** | 25배 → 4배 증가 |
| `FREEZE_EPOCHS` | 없음 | 없음 |

#### 실행 방법
```bash
cd models\classifier\experiments\koelectra_mamba_diff_lr
python train.py
python evaluate.py --split test
```

#### 기대 효과
- 동결 없이 처음부터 KoELECTRA 데이터 적응 가능
- LR 격차 100배로 Mamba가 초기에 주로 학습 → 암묵적 보호
- 구현 단순, 실험 A와 병렬 비교 가능

---

### 실험 A vs B 비교

| 관점 | 실험 A (freeze_init) | 실험 B (diff_lr) |
|---|---|---|
| **KoELECTRA 보호** | 명시적 동결 (gradient 완전 차단) | 암묵적 보호 (LR 격차) |
| **초기 KoELECTRA 업데이트** | epoch 1~3: 전혀 없음 | epoch 1부터 `1e-5`로 조금씩 |
| **Mamba 수렴 속도** | 보통 (`5e-4`) | 빠름 (`1e-3`) |
| **구현 복잡도** | 높음 (단계 전환 로직) | 낮음 (옵티마이저 그룹 분리) |
| **예상 장점** | Forgetting 방지 확실 | KoELECTRA도 처음부터 적응 |
| **예상 위험** | 단계 전환 시 일시적 불안정 | 초기 epoch KoELECTRA 일부 손상 가능 |

---

## 4. 향후 실험 계획 — 데이터 전처리 개선

> 목표: Mamba가 충분한 시퀀스(5개 이상 세그먼트)를 처리할 수 있도록 입력 파이프라인 재설계

### 실험 C: `koelectra_mamba_short_window` — 짧은 고정 윈도우 (대안 A)

**폴더 (예정)**: `models/classifier/experiments/koelectra_mamba_short_window/`

#### 전략
```
WINDOW_SIZE: 128 → 64 토큰
STRIDE:      100 → 32 토큰

기대 세그먼트 수 변화:
  현재:  중간값 166토큰 / (128-28) = ~1.7개
  변경:  중간값 166토큰 / (64-32) = ~5.2개
```

#### 핵심 파라미터

| 파라미터 | 현재 | 변경 | 기대 효과 |
|---|---|---|---|
| `WINDOW_SIZE` | 128 | **64** | 세그먼트 수 약 3배 증가 |
| `STRIDE` | 100 | **32** | 세그먼트 간 50% 중첩 |
| 평균 세그먼트 수 | ~2개 | **~5~8개** | Mamba 상태 누적 기회 확보 |

#### 구현 포인트
- `config.py`의 `WINDOW_SIZE`, `STRIDE` 값만 변경
- `dataset.py`는 그대로 사용 가능 (동적으로 슬라이딩 윈도우 처리)
- 세그먼트 수 증가로 배치당 메모리 사용량 증가 → `BATCH_SIZE` 조정 필요할 수 있음

---

### 실험 D: `koelectra_mamba_sent_window` — 문장 단위 동적 윈도우 (대안 B, 강력 추천)

**폴더 (예정)**: `models/classifier/experiments/koelectra_mamba_sent_window/`

#### 전략
```
토큰 수 기준 고정 분할 → 문장(마침표/화자교대) 기준 의미 단위 분할

예시:
  현재: "[안녕하세요 저희는 대출] [팀입니다 고객님 혹시]"
         → 단어가 중간에 잘림

  변경: ["안녕하세요 저희는 대출팀입니다."] ["고객님 혹시 대출이 필요하신가요?"]
         → 의미 완결 단위 유지
```

#### 구현 포인트
- `dataset.py`의 `build_segments()` 함수를 문장 분리 기반으로 교체
- 분리 기준: `。`, `.`, `?`, `!`, 화자 교대(`\n` 등)
- 고정 길이 초과 시: 문장을 유지하면서 그룹핑하여 128토큰 이하로 묶음
- 중요한 피싱 키워드가 세그먼트 경계에서 잘리는 현상 원천 차단

#### 전처리 로직 예시
```python
def build_segments_sentence(tokenizer, text: str) -> list[dict]:
    # 1. 문장 분리
    sentences = re.split(r'[.?!]+', text)
    # 2. 빈 문장 제거
    sentences = [s.strip() for s in sentences if s.strip()]
    # 3. 각 문장 토크나이즈
    # 4. 최대 WINDOW_SIZE 토큰 이하가 되도록 문장 그룹화
    # 5. 각 그룹 → segment dict 반환
```

---

## 5. 전체 실험 로드맵

```
[완료] ─────────────────────────────────────────────────────────────────
  ① koelectra_phishing5          Macro F1 = 0.8132 (기준선 - Best)
  ② koelectra_mamba_phishing5    Macro F1 = 0.7438 (Mamba 추가 시 하락)
  ③ koelectra_mamba_hce_ordinal  Macro F1 = 0.7468 (HCE+Ordinal 손실)

[진행 예정: 학습 전략 최적화] ────────────────────────────────────────
  ④ koelectra_mamba_freeze_init  KoELECTRA 동결 3에폭 후 단계적 해제
  ⑤ koelectra_mamba_diff_lr      차등 LR: ENC=1e-5, Mamba=1e-3

[향후 계획: 전처리 개선] ─────────────────────────────────────────────
  ⑥ koelectra_mamba_short_window  WINDOW=64, STRIDE=32 (세그먼트 수 증가)
  ⑦ koelectra_mamba_sent_window   문장 단위 의미 보존 분할 (강력 추천)

[최종 목표] ──────────────────────────────────────────────────────────
  Mamba 결합 모델이 KoELECTRA Only(0.8132)를 초과하는 성능 달성
```

---

## 6. 성능 목표 및 평가 기준

| 지표 | 현재 최고 (기준선) | 목표 |
|------|-------------------|------|
| Macro F1 | 0.8132 (KoELECTRA Only) | **0.85 이상** |
| 피싱 클래스 F1 평균 | (0.928 + 0.898) / 2 = 0.913 | 0.93 이상 |
| 일반 클래스 F1 평균 | (0.770 + 0.651 + 0.820) / 3 = 0.747 | 0.80 이상 |

- **주요 평가 지표**: Macro F1 (클래스 불균형 고려)
- **비교 기준**: Test Set (n=257)
- **체크포인트 선정**: Validation Macro F1 기준 best 모델 저장

---

## 7. 실험별 코드 위치 요약

```
models/classifier/experiments/
├── koelectra_phishing5/              ✅ 완료 (기준선)
├── koelectra_mamba_phishing5/        ✅ 완료
├── koelectra_mamba_hce_ordinal/      ✅ 완료 (HCE+Ordinal)
├── koelectra_mamba_freeze_init/      🔄 진행 예정 (실험 A)
├── koelectra_mamba_diff_lr/          🔄 진행 예정 (실험 B)
├── koelectra_mamba_short_window/     📋 계획 (실험 C)
└── koelectra_mamba_sent_window/      📋 계획 (실험 D)
```

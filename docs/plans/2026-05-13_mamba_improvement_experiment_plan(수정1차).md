# Mamba 성능 개선 실험 계획서

> 작성일: 2026-05-13  
> 관련 실험 폴더: `models/classifier/experiments/`  
> 작성 배경: KoELECTRA+Mamba 결합 모델이 KoELECTRA 단독 모델 대비 성능이 낮아진 원인을 분석하고, 개선 실험을 체계적으로 수행하기 위한 계획서

---

## 0. 데이터셋 현황

| 구분 | 샘플 수 |
|------|---------|
| Train | 2,978건 |
| Validation | - |
| Test | 257건 |

### 대화 길이 통계 (train.csv 기준)

| 항목 | 평균 | 중간값 | 최소 | 최대 |
|------|------|--------|------|------|
| 문장 수 | 21.4 | 16 | 2 | 429 |
| 어절 수 | 177.6 | 138 | 12 | 4,941 |
| 글자 수 | 734.8 | 576 | 46 | 19,857 |
| 토큰 수 (추정) | 213~266 | 166~207 | 14~18 | 5,929~7,412 |

> **핵심**: 중간값 기준 ~166~207 토큰 → WINDOW_SIZE=128 적용 시 대부분 **1~2 세그먼트**만 생성됨.

---

## 1. 원인 진단: Mamba 결합 모델 성능 저하 분석

### 1-1. 시퀀스 부족 (주요 원인)
- 데이터 중간값(~166 토큰)을 WINDOW_SIZE=128로 자르면 윈도우가 **1~2개**에 불과
- Mamba는 긴 시퀀스(수십 개 이상 세그먼트)에서 상태 누적 효과가 발휘되는 SSM
- 2개 세그먼트로는 Mamba가 맥락을 누적할 기회가 없어 단순 MLP와 다름없음

### 1-2. Catastrophic Forgetting (가중치 붕괴)
- 학습이 완료된 KoELECTRA(수억 파라미터)와 무작위 초기화된 Mamba를 **동일 LR로 동시 학습**
- 초기 Mamba의 큰 오차(loss)가 역전파 시 KoELECTRA의 사전학습 지식을 덮어씀
- 결과적으로 두 모델 모두 불안정한 상태에서 수렴

### 1-3. 키워드 희석
- 피싱 대화의 핵심 키워드("대포통장", "공증", "공무원 사칭" 등)가
  Mamba의 선형 상태 압축 과정에서 일반 대화 맥락과 혼합
- 세그먼트가 1~2개일 때 Mamba 처리 이점이 사실상 없음

---

## 2. 완료된 실험 결과

### 실험 결과 종합 (Test Set, n=257)

| 실험명 | Accuracy | Macro F1 | Weighted F1 | 상태 |
|--------|----------|----------|-------------|------|
| `koelectra_phishing5` (KoELECTRA Baseline) | 0.7860 | 0.8146 | 0.7894 | ✅ 완료 |
| `hce_ordinal` (KoELECTRA+Mamba L1) | **0.8366** | **0.8429** | **0.8378** | ✅ 완료 |
| `hce_ordinal` (KoELECTRA+Mamba L2) | 0.7121 | 0.7470 | 0.7098 | ✅ 완료 |

> **핵심 관찰**: **Mamba L1 레이어 추가 시 성능이 약 2.8%p(Macro F1) 향상**됨.  
> 단, L2 이상 레이어 깊이 증가 시 성능이 급격히 저하(71.2%)되거나 NaN 폭발 발생. HCE+Ordinal 손실은 L1 단계에서 강력한 시너지를 냄.

### 클래스별 F1 상세 (Baseline vs Mamba L1)

| 클래스 | KoELECTRA Only | KoELECTRA+Mamba L1 | 차이 |
|--------|---------------|----------------------|------|
| 상품 가입 및 해지 | 0.774 | 0.855 | **+0.081** |
| 이체 출금 대출서비스 | 0.691 | 0.756 | **+0.065** |
| 잔고 및 거래내역 | 0.783 | 0.855 | **+0.072** |
| 수사기관 사칭형 | 0.928 | 0.904 | -0.024 |
| 대출 사기형 | 0.898 | 0.844 | -0.054 |

> 금융 정보형 클래스(0~2)에서 Mamba L1 결합 시 성능 향상이 뚜렷함 (+7~8%p).  
> 피심 핵심 클래스(3~4)는 소폭 하락하는 trade-off 발생 → Mamba의 시퀀스 요약이 미세한 키워드 특징을 일부 희석시켰을 가능성.

### 혼동 행렬 분석 (KoELECTRA+Mamba L1 기준)

```
실제\예측  가입해지  이체출금  잔고거래  수사기관  대출사기
가입해지 [  62      10       2        0        0   ]
이체출금 [   7      48       5        0        0   ]
잔고거래 [   2       9      53        0        0   ]
수사기관 [   0       0       0       33        3   ]
대출사기 [   0       0       0        4       19   ]
```

- 일반 3개 클래스 간 혼동이 심각 (특히 "이체출금" ↔ "가입해지")
- 피싱 클래스 간 혼동은 비교적 적음

---

## 3. 개선 방향 및 실험 계획

개선 방향은 두 축으로 나뉨:
1. **학습 전략 최적화** — Catastrophic Forgetting 방지 (단기 실험)
2. **데이터 전처리 개선** — 세그먼트 수 확보 (중기 실험)

---

## 4. 실험 A: `koelectra_mamba_freeze_init` — 동결 & 초기화

**폴더**: `models/classifier/experiments/koelectra_mamba_freeze_init/`  
**상태**: 🔄 진행 예정

### 핵심 전략: 2단계 학습

```
Phase 1 — Epoch 1~3: KoELECTRA 완전 동결
  - KoELECTRA: requires_grad = False (gradient 완전 차단)
  - Mamba + Head만 학습: LR = 5e-4
  - 목적: Mamba가 KoELECTRA 표현 공간에 적응하도록 시간 부여

Phase 2 — Epoch 4~8: KoELECTRA 동결 해제
  - KoELECTRA: LR = 1e-5 (매우 낮게, 기존 2e-5의 절반)
  - Mamba + Head: LR = 5e-4 (유지)
  - 목적: 안정화된 Mamba와 함께 KoELECTRA 조심스럽게 미세조정
```

### 파라미터 설정

| 파라미터 | 값 | 기존(`hce_ordinal`) 대비 |
|---|---|---|
| `FREEZE_EPOCHS` | **3** | 신규 추가 |
| `ENCODER_LR` (2단계) | **1e-5** | `2e-5` → 절반 |
| `UPPER_LR` | **5e-4** | 동일 |
| `EPOCHS` | 8 | 동일 |
| `BATCH_SIZE` | 8 | 동일 |

### FREEZE_EPOCHS=3 근거
- 전체 8 에폭의 약 37%를 동결 구간으로 할당
- 너무 짧으면(1~2 에폭): Mamba 불안정 상태에서 해제 → 효과 미미
- 너무 길면(5+ 에폭): 2단계 fine-tuning 기회가 부족

### 실행 방법
```bash
cd models\classifier\experiments\koelectra_mamba_freeze_init
python train.py
python evaluate.py --split test
```

### 로그 특이사항
- `phase` 필드가 스텝/에폭 로그에 추가됨 (`"freeze"` / `"unfreeze"`)
- 단계 전환 시 `phase_change` 타입 로그 기록

---

## 5. 실험 B: `koelectra_mamba_diff_lr` — 차등 학습률

**폴더**: `models/classifier/experiments/koelectra_mamba_diff_lr/`  
**상태**: 🔄 진행 예정

### 핵심 전략: LR 100배 격차

```
Epoch 1~8 (단일 단계, 동결 없음):
  KoELECTRA: LR = 1e-5  ← 사전학습 지식 보호
  Mamba + Head: LR = 1e-3  ← 무작위 초기화, 빠른 수렴
  격차: 100배 → Mamba가 주도적으로 학습, KoELECTRA는 조금씩만 업데이트
```

### 파라미터 설정

| 파라미터 | 값 | 기존(`hce_ordinal`) 대비 |
|---|---|---|
| `ENCODER_LR` | **1e-5** | `2e-5` → 절반 |
| `UPPER_LR` | **1e-3** | `5e-4` → **2배** |
| LR 비율 (Mamba/ENC) | **100배** | 25배 → 4배 증가 |
| 동결 여부 | 없음 | 없음 |

### LR 비율 100배의 의미
- 한 스텝에서 KoELECTRA는 `1e-5`만큼, Mamba는 `1e-3`만큼 파라미터 이동
- 초기 수 에폭: 실질적으로 "Mamba만 주로 학습"되는 효과 자연 발생
- 동결 없이도 실험 A와 유사한 보호 효과 기대

### 실행 방법
```bash
cd models\classifier\experiments\koelectra_mamba_diff_lr
python train.py
python evaluate.py --split test
```

### 로그 특이사항
- `enc_lr`, `upper_lr` 두 파라미터 그룹의 LR을 스텝 로그에 분리 기록

---

## 6. 실험 A vs B 비교

| 관점 | 실험 A (freeze_init) | 실험 B (diff_lr) |
|---|---|---|
| **KoELECTRA 보호** | 명시적 동결 (gradient 완전 차단) | 암묵적 보호 (LR 격차) |
| **초기 KoELECTRA 업데이트** | epoch 1~3: 전혀 없음 | epoch 1부터 `1e-5`로 조금씩 |
| **Mamba LR** | `5e-4` | `1e-3` (더 빠름) |
| **KoELECTRA LR** | epoch 4~부터 `1e-5` | 처음부터 `1e-5` |
| **구현 복잡도** | 높음 (단계 전환 로직) | 낮음 (그룹 분리만) |
| **예상 장점** | Forgetting 방지 확실 | KoELECTRA도 처음부터 데이터 적응 |
| **예상 위험** | 단계 전환 시 일시적 불안정 | 초기 epoch KoELECTRA 일부 손상 가능 |

---

## 7. 실험 C: `koelectra_mamba_short_window` — 짧은 고정 윈도우 (향후)

**폴더 (예정)**: `models/classifier/experiments/koelectra_mamba_short_window/`  
**상태**: 📋 계획

### 핵심 전략

```
WINDOW_SIZE: 128 → 64 토큰
STRIDE:      100 → 32 토큰 (50% 중첩)

세그먼트 수 변화 (중간값 ~166 토큰 기준):
  현재:  (166 - 2) / (128 - 100) ≈ 약 2개
  변경:  (166 - 2) / (64 - 32)   ≈ 약 5개
```

### 파라미터 설정

| 파라미터 | 현재 | 변경 | 기대 효과 |
|---|---|---|---|
| `WINDOW_SIZE` | 128 | **64** | 세그먼트 수 약 3배 증가 |
| `STRIDE` | 100 | **32** | 세그먼트 간 50% 중첩 |
| 평균 세그먼트 수 | ~2개 | **~5~8개** | Mamba 상태 누적 기회 확보 |
| `BATCH_SIZE` | 8 | **4** (조정 필요) | 메모리 증가 대응 |

### 구현 포인트
- `config.py`의 `WINDOW_SIZE`, `STRIDE` 값만 변경
- `dataset.py`는 그대로 사용 가능 (슬라이딩 윈도우 자동 처리)
- 세그먼트 수 증가로 배치당 GPU 메모리 사용량 증가 → `BATCH_SIZE` 조정 필요

---

## 8. 실험 D: `koelectra_mamba_sent_window` — 문장 단위 동적 윈도우 (향후, 강력 추천)

**폴더 (예정)**: `models/classifier/experiments/koelectra_mamba_sent_window/`  
**상태**: 📋 계획

### 핵심 전략

```
현재: 토큰 수 기준 고정 분할 → 단어/의미 단위가 경계에서 잘림
변경: 문장(마침표/화자교대) 기준 의미 단위 분할 → 의미 완결 유지

예시:
  현재: ["안녕하세요 저희는 대출"] ["팀입니다 고객님 혹시"]  ← 문장이 잘림
  변경: ["안녕하세요 저희는 대출팀입니다."] ["고객님 혹시 대포통장이..."]
```

### 분리 기준
- 마침표(`.`), 물음표(`?`), 느낌표(`!`)
- 화자 교대 (`\n`, 화자 레이블 등)
- 한 세그먼트가 WINDOW_SIZE(128토큰) 초과 시: 문장 단위를 유지하면서 그룹핑

### 구현 포인트
- `dataset.py`의 `build_segments()` 함수를 문장 분리 기반으로 교체
- 피싱 핵심 키워드("대포통장", "공증", "수사기관" 등)가 세그먼트 경계에서 잘리는 현상 원천 차단
- 세그먼트 수가 실험 C보다 불규칙하므로 `collate_fn` 패딩 처리 중요

```python
# 변경될 build_segments 로직 개요
def build_segments_sentence(tokenizer, text: str) -> list[dict]:
    sentences = re.split(r'[.?!]+', text)          # 1. 문장 분리
    sentences = [s.strip() for s in sentences if s.strip()]
    groups = []                                      # 2. WINDOW_SIZE 이하로 그룹화
    current_group = []
    current_len = 0
    for sent in sentences:
        tokens = tokenizer(sent, add_special_tokens=False)["input_ids"]
        if current_len + len(tokens) > WINDOW_SIZE - 2:
            if current_group:
                groups.append(current_group)
            current_group = [tokens]
            current_len = len(tokens)
        else:
            current_group.append(tokens)
            current_len += len(tokens)
    if current_group:
        groups.append(current_group)
    # 3. 각 그룹 → [CLS] + tokens + [SEP] + padding → segment dict 반환
```

---

## 9. 전체 실험 로드맵

```
[완료] ─────────────────────────────────────────────────────────────────
  ① koelectra_phishing5          Macro F1 = 0.8146  ← 현재 기준선
  ② hce_ordinal (Mamba L1)       Macro F1 = 0.8429  ← Mamba L1 성공 (최고성능)
  ③ hce_ordinal (Mamba L2)       Macro F1 = 0.7470  ← 깊어질수록 급격한 하락
  ④ hce_ordinal (Mamba L4/L6)    NaN 폭발           ← 모델 붕괴

[진행 예정: 학습 전략 최적화] ────────────────────────────────────────
  ⑤ koelectra_mamba_freeze_init  KoELECTRA 동결 3에폭 → 단계적 해제 (L2+ 안정성 확보)
  ⑥ koelectra_mamba_diff_lr      차등 LR: ENC=1e-5, Mamba=1e-3 (NaN 방지 및 최적화)

[향후 계획: 전처리 개선] ─────────────────────────────────────────────
  ⑦ koelectra_mamba_short_window  WINDOW=64, STRIDE=32 (세그먼트 수 증가로 Mamba 강화)
  ⑧ koelectra_mamba_sent_window   문장 단위 의미 보존 분할 (★ 강력 추천)

[최종 목표] ──────────────────────────────────────────────────────────
  Mamba 결합 모델이 KoELECTRA Baseline(0.8146)을 넘어 **Macro F1 0.87 이상** 달성
```

---

## 10. 성능 목표 및 평가 기준

| 지표 | 현재 최고 (Baseline) | 목표 |
|------|-------------------|------|
| Macro F1 | 0.8146 (KoELECTRA) | **0.87 이상** |
| 피싱 클래스 F1 평균 | 0.913 | 0.94 이상 |
| 일반 클래스 F1 평균 | 0.749 | 0.82 이상 |

- **주요 평가 지표**: Macro F1 (클래스 불균형 고려)
- **비교 기준**: Test Set (n=257, 고정)
- **체크포인트 선정**: Validation Macro F1 기준 best 모델 저장

---

## 11. 파일 위치 요약

```
models/classifier/experiments/
├── koelectra_phishing5/              ✅ 완료 (기준선)
├── koelectra_mamba_phishing5/        ✅ 완료
├── koelectra_mamba_hce_ordinal/      ✅ 완료 (HCE+Ordinal)
├── koelectra_mamba_freeze_init/      🔄 진행 예정 (실험 A)
│   ├── config.py   FREEZE_EPOCHS=3, ENCODER_LR=1e-5, UPPER_LR=5e-4
│   ├── train.py    2단계 학습 (phase_change 로직 포함)
│   ├── evaluate.py
│   ├── dataset.py  (hce_ordinal과 동일)
│   ├── losses.py   (hce_ordinal과 동일)
│   └── model.py    (hce_ordinal과 동일)
├── koelectra_mamba_diff_lr/          🔄 진행 예정 (실험 B)
│   ├── config.py   ENCODER_LR=1e-5, UPPER_LR=1e-3
│   ├── train.py    enc_lr/upper_lr 분리 로깅
│   ├── evaluate.py
│   ├── dataset.py  (hce_ordinal과 동일)
│   ├── losses.py   (hce_ordinal과 동일)
│   └── model.py    (hce_ordinal과 동일)
├── koelectra_mamba_short_window/     📋 계획 (실험 C)
└── koelectra_mamba_sent_window/      📋 계획 (실험 D)
```

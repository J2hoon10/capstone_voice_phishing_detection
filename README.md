# 텍스트 기반 실시간 보이스피싱 탐지 시스템
**Text-based Streaming Voice Phishing Detection System**

서울과학기술대학교 인공지능융합학과 | 김나현, 정지훈, 박종열

---

## 개요

보이스피싱 피해 증가로 인해 통화 중 위험 신호를 조기에 탐지하는 기술이 필요하다. 본 프로젝트는 음성 통화를 텍스트로 변환한 뒤, 대화의 문맥과 시간적 흐름을 분석하여 보이스피싱 위험을 탐지하는 **텍스트 기반 스트리밍 보이스피싱 탐지 시스템**을 개발한다.

- AI Hub 대화 데이터와 금융감독원 피싱 녹취를 기반으로 Whisper STT, LLM 증강, 3단계 위험도 라벨링을 적용해 학습 데이터를 구성
- **KLUE-RoBERTa** 기반 인코더와 **Mamba** 기반 전역 문맥 모델링 구조를 결합해 세그먼트 단위 의미 표현과 대화 흐름 기반 위험 누적을 함께 모델링
- 베이스 모델 기준 **Accuracy 99.38% / Macro F1 99.37%** 달성
- 웹 데모 시스템을 통해 위험 접수·피싱 유형·대응 가이드를 실시간으로 제공

---

## 모델 아키텍처

```
Conversation Script
       │
   Sliding Window
       │
┌──────▼───────┐
│ Local Encoder │  KLUE-RoBERTa + Attention Pooling
│  (×T segments)│  → 세그먼트 내 주요 발화에 가중치 부여 후 고정 길이 벡터 생성
└──────┬───────┘
       │
┌──────▼──────────────┐
│ Global Context Model │  Mamba SSM Block (×2 Layers)
│                      │  → 대화 전반의 흐름·순서 정보·장기 문맥 반영
└──────┬──────────────┘
       │
  Fusion & Log
       │
  Final Output (4-class)
```

**Auxiliary Head** — 학습 단계에서 보조 분류 손실을 적용해 Local Encoder가 개별 세그먼트 수준의 보이스피싱 단서를 더 잘 학습하도록 유도

### 손실 함수 (Hierarchical CrossEntropy Loss)

```
L_HCE = L_super + λ · (L_normal + L_phishing) + β_t · L_aux

λ = 0.5,  β_t = 0.5 − 0.4 · (t / T)
```

| 손실 항 | 설명 |
|---|---|
| `L_super` | 4-class 전체 분류 손실 (Normal vs Phishing Superclass) |
| `L_phishing` | 피싱 클래스 내 세분류 손실 |
| `L_normal` | 일반 클래스 내 세분류 손실 |
| `L_aux` | Ordinal Regression 보조 손실 (단계별 경계 학습) |

---

## 데이터셋

**4-class 분류** — 상담 대화 / 일상 대화 / 대출 사기형 / 수사기관 사칭형

| 클래스 | Label | Train | Val | Test | Original+Noise | LLM+Noise | LLM+Clean | 합계 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 상담 대화 | 0 | 600 | 200 | 200 | 100 | 600 | 300 | 1,000 |
| 일상 대화 | 1 | 600 | 200 | 200 | 100 | 600 | 300 | 1,000 |
| 대출 사기형 | 2 | 600 | 200 | 200 | 190 | 540 | 270 | 1,000 |
| 수사기관 사칭형 | 3 | 600 | 200 | 200 | 323 | 451 | 226 | 1,000 |
| **합계** | — | **2,400** | **800** | **800** | 713 | 2,191 | 1,096 | **4,000** |

**데이터 출처**
- **AI Hub** — 민간 민원 상담 대화 (학습·Instruction Tuning), SNS 일상 대화
- **금융감독원** — 보이스피싱 음성 녹취 (대출 사기형 323건, 수사기관 사칭형 190건)

**데이터 파이프라인**

```
STEP 1            STEP 2              STEP 3                STEP 4
원본 데이터 통합  →  LLM 기반 대화 생성  →  Segment Risk Labeling  →  ASR 노이즈 주입
```

- Whisper small 기반 STT → Levenshtein 편집 거리로 오류율 측정
- `gpt-4o-mini`를 사용하여 문장별 3단계 피싱 위험도 라벨링
- 규칙적 변환(랜덤 샘플링)으로 텍스트에 노이즈 적용

---

## 실험 결과

### 베이스 모델 성능 (RoBERTa + Mamba L2, d_state=16, w64)

| 클래스 | F1 | Precision | Recall |
|---|:---:|:---:|:---:|
| 상담 대화 | 1.0000 | 1.0000 | 1.0000 |
| 일상 대화 | 1.0000 | 1.0000 | 1.0000 |
| 대출 사기형 | 0.9876 | 0.9803 | 0.9950 |
| 수사기관 사칭형 | 0.9874 | 0.9949 | 0.9800 |
| **Macro F1** | **0.9937** | — | — |

### Ablation Study

| 실험 | 변수 | 설정 | Accuracy | Macro F1 |
|---|---|---|:---:|:---:|
| Mamba Layer | Layer | L1 | 0.9938 | 0.9938 |
| | | **L2 (base)** | **0.9938** | **0.9937** |
| Local Encoder | Model | **RoBERTa (base)** | **0.9938** | **0.9937** |
| | | BERT | 0.9900 | 0.9900 |
| | | KoBERT | 0.9875 | 0.9875 |
| | | KoELECTRA | 0.9762 | 0.9764 |
| Global Context Model | Model | **Mamba (base)** | **0.9938** | **0.9937** |
| | | GRU | 0.9925 | 0.9925 |
| | | LSTM | 0.9925 | 0.9925 |
| d_state | Mamba state dim | **16 (base)** | **0.9938** | **0.9938** |
| | | 32 | 0.9925 | 0.9925 |
| Window Size | | 64 | 0.9950 | 0.9950 |
| | | **W64 (base)** | **0.9938** | **0.9937** |
| | | W32 | 0.9889 | 0.9889 |

---

## 실행 방법

> 🚧 추후 작성 예정

---

## 프로젝트 구조

→ [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) 참고

---

## 데이터 정책

### ⚠️ 이 저장소에는 학습 데이터가 포함되지 않습니다

본 프로젝트는 실제 보이스피싱 통화 및 일반 통화 음성·전사 데이터를 사용합니다.
해당 데이터는 **개인정보 및 저작권 보호**를 위해 Git 저장소에 포함하지 않습니다.

| 경로 | 설명 | 제외 이유 |
|---|---|---|
| `Data/` | 원본 음성 데이터셋 (`.wav`, `.mp3`) | 음성 개인정보 |
| `normal_data_reference/` | 일반 대화 JSON 원본 (AI Hub) | 라이선스·개인정보 |
| `models/main/model_architecture/data/` | 전처리된 학습 데이터 | 실제 통화 내용 포함 |
| `models/*/checkpoints/` | 모델 체크포인트 | 용량 |
| `models/*/logs/` | 학습 로그 | 재현 가능 |
| `models_analysis/output/` | 분석 산출물 | 전사 내용 파생 |

> **⚠️ 이 파일들을 절대 `git add` 하지 마세요.**
> 실수로 push되면 공개 저장소에서 개인정보가 노출될 수 있습니다.

---

## 참고 문헌

1. R. Pappagari et al., "Hierarchical transformers for long document classification," in *Proc. 2019 IEEE ASRU*, pp. 838–844.
2. Y. Liu et al., "RoBERTa: A robustly optimized BERT pretraining approach," *arXiv:1907.11692*, 2019.
3. J. Rae et al., "Compressive transformers for long range sequence modelling," *arXiv:1911.05507*, 2019.
4. A. Gu and T. Dao, "Mamba: Linear-time sequence modelling with selective state spaces," *arXiv:2312.00752*, 2023.
5. M. K. Moussavou Boussougou and D. J. Park, "Attention-based 1D CNN-BiLSTM hybrid model enhanced with FastText word embedding for Korean voice phishing detection," *Mathematics*, vol. 11, no. 14, Art. no. 3217, 2023.
6. J. Y. Sim et al., "Voice phishing detection scheme using a GPT-3.5 based large language model," *Journal of KIISE*, vol. 51, no. 1, 2024.
7. S. Kim and S. Noh, "딥러닝 기반 NLP 및 작성기법을 활용한 보이스피싱 의심발언 탐지," *한국전자거래학회지*, vol. 29, no. 4, pp. 139–148, 2024.
8. H. Park et al., "Enhanced voice phishing detection using an LLM-based framework for data augmentation and classification," *IEEE Access*, 2025. doi: 10.1109/ACCESS.2025.3603007.
9. 금융감독원, "보이스피싱피해 바로고" [Online]. Available: https://www.fss.or.kr/fss/bbs/B0000203/list.do?menuNo=200686. [Accessed: May 19, 2026].

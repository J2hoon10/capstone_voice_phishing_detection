# 🛡️ 보이스피싱 탐지 모델 파이프라인 설계서

## 1. 데이터 준비 (Data Preparation)
보이스피싱 탐지 모델 학습을 위한 고품질 데이터셋 구축 프로세스입니다. **오디오 전처리**, **STT 변환**, **데이터 병합**의 3단계 파이프라인으로 구성됩니다.

### 1-1. 오디오 처리 (Audio Processing Pipeline)
통화 녹음 환경의 특성을 고려하여 음성 품질을 향상시키고, Whisper 모델의 인식률을 높이기 위한 전처리 과정입니다.

* **Core Model:** `faster-whisper-large-v3-turbo`
* **Audio Enhancer (음질 개선):**
    1.  **Bandpass Filter:** 사람 목소리 대역(300~3400Hz)만 통과시켜 불필요한 고주파/저주파 노이즈 제거.
    2.  **Noise Reduction:** 배경 잡음을 제거하되, 목소리 왜곡을 방지하기 위해 강도 조절 (Stationary Noise Reduction).
    3.  **VAD (Voice Activity Detection):** `Silero VAD`를 사용하여 무음 구간을 정밀하게 잘라내어 Whisper의 환각(Hallucination) 방지.
    4.  **Normalization:** 오디오 볼륨을 일정 수준으로 평준화.
* **Text Post-processing (텍스트 후처리):**
    * **PII Masking:** 개인정보(주민등록번호, 전화번호, 계좌번호 등)를 `<RRN>`, `<PHONE>`, `<ACCOUNT>` 토큰으로 치환하여 모델의 일반화 성능 확보.
    * **Repetition Removal:** STT 모델 특유의 단어/구 반복 현상 제거.

### 1-2. 데이터셋 구축 (Dataset Construction)
유지보수 용이성을 위해 **생성(Builder)**과 **병합(Merger)** 과정을 분리하여 설계했습니다.

* **Phase 1: 보이스피싱 데이터 생성 (`DatasetBuilder`)**
    * 오디오 파일을 STT로 변환하여 중간 산출물(`dataset_voice_phishing.csv`) 생성.
    * GPU 자원이 많이 소모되는 과정이므로 독립적으로 수행.
* **Phase 2: 데이터 병합 (`CsvMerger`)**
    * 보이스피싱 데이터와 외부 일반 대화 데이터(`KorCCVi_v2.csv`)를 병합.
    * 일반 대화 데이터 포맷(Label 0, Class normal)을 통일하여 최종 마스터 데이터셋(`dataset_master.csv`) 생성.

| 컬럼명 | 설명 | 예시 |
| :--- | :--- | :--- |
| **ID** | 데이터 고유 식별자 (접두사 구분) | `P_0001` (피싱), `N_0001` (일반) |
| **script** | 전처리 및 마스킹된 대화 내용 | "서울중앙지검 김민수 검사입니다." |
| **label** | 이진 분류 라벨 | `1` (보이스피싱), `0` (일반) |
| **class** | 데이터 세부 유형 | `loan_fraud`, `impersonation`, `normal` |
| **filename**| 원본 파일 추적용 메타데이터 | `001_voice.wav` |

---

## 2. NLP Classification Model
변환된 텍스트를 분석하여 보이스피싱 여부를 실시간으로 판별하는 핵심 모델부입니다.

### 🧠 Model Architecture
* **Baseline:** `KLUE-RoBERTa` (기본 성능 확보)
* **Challenger:** `ModernBERT` (한국어 버전, 최신 아키텍처 적용)
* **Structure:**
    1.  **Backbone:** Transformer Encoder (RoBERTa / ModernBERT)
    2.  **Task Head:** Binary Classifier (Phishing vs Normal)
    3.  **Response Generator (Optional):** 문맥 파악 후 상황별 대처 방안 출력 (Multi-task Learning 또는 별도 모듈 고려).

### ⚖️ 누적 점수 시스템 (Leaky Bucket Algorithm)
단건 추론의 불안정성을 보완하기 위해, 게임의 HP 회복/감소 매커니즘을 차용한 점수 누적 시스템을 도입합니다.
* **초기 상태:** 위험 점수 0점 (Max 100점)
* **갱신 주기:** 매 5초마다 (Sliding Window 방식)

| 판별 결과 | 확률 조건 | 점수 변동 | 비고 |
| :--- | :--- | :--- | :--- |
| **🚨 피싱 (High)** | $Prob > 0.8$ | **+20점** | 강력한 위험 신호 |
| **⚠️ 의심 (Medium)**| $0.5 < Prob \le 0.8$ | **+10점** | 주의 필요 |
| **✅ 정상 (Normal)** | $Prob \le 0.5$ | **-10점** | 점수는 0점 미만으로 내려가지 않음 |

* **사용자 경고 시나리오:**
    * **Level 1 [주의]:** 점수 **30점** 이상 → 진동 1회, 화면 황색 경고등 (연속 탐지 약 2~3회 시)
    * **Level 2 [경고]:** 점수 **60점** 이상 → 연속 진동, 적색 점멸, "통화 종료" 알림 (강력 탐지 누적 시)

---

## 3. 학습 데이터 전처리 전략
모델의 강건함(Robustness)을 높이기 위한 데이터 증강 및 분할 전략입니다.

### 🛠️ 텍스트 데이터 증강 (Augmentation)
* **구조적 증강 (Sliding Window):**
    * 전체 대본을 문장 단위로 분리 후 `Window=5`, `Stride=2`로 겹쳐서 생성.
    * 긴 통화 내용을 실시간 입력과 유사한 짧은 시퀀스로 변환.
* **내용적 증강 (KoEDA - On-the-fly):**
    * Whisper의 인식 오류(오타, 누락)를 시뮬레이션하여 모델이 문맥 자체에 집중하도록 유도.
    * `SR` (유의어 교체), `RI` (단어 삽입), `RS` (위치 교환), `RD` (단어 삭제) 기법을 학습 시 확률적으로 적용.

### ✂️ 데이터 분할 (Data Splitting)
* **Group-based Split:** 슬라이딩 윈도우 적용 전, **파일 ID**를 기준으로 Train/Test 셋을 분리하여 데이터 유출(Data Leakage) 원천 차단.

---

## 4. 모델 경량화 (Knowledge Distillation)
실시간 처리를 위해 무거운 모델의 지식을 가벼운 모델로 전이하는 최적화 과정입니다.

* **Teacher:** 12-Layer Transformer (고성능)
* **Student:** 6-Layer Transformer (경량화)
* **Loss Function Strategy:**
    1.  **Hard Loss (Cross Entropy):** 정답 라벨(0/1)을 맞추는 기본 학습.
    2.  **Soft Loss (KL Divergence):** Teacher의 확률 분포(Soft Label)를 모방 ($Temperature=2.0$). Teacher가 느끼는 '애매함'까지 학습.
    3.  **Cosine Embedding Loss:** Teacher와 Student의 `[CLS]` 토큰 벡터 방향성을 일치시켜 문맥 표현력 전수.

---

## ✅ To-Do List & Next Steps

### Experiment & Modeling
- [ ] `KLUE-RoBERTa` 베이스라인 모델 학습 및 평가 **[지훈]**
- [ ] `ModernBERT` (한국어) 챌린저 모델 실험 및 성능 비교 **[나현]**
- [ ] 대처 방안 생성 모듈 아키텍처 확정 (통합 모델 vs 별도 모델)

### Engineering & Deployment
- [ ] 실시간 탐지 결과 시각화 웹 대시보드 개발 **[수효]**
- [ ] End-to-End 파이프라인 통합 (Audio Input → Preprocessing → Model → Alert)
- [ ] 오디오 전처리/데이터셋 빌더 코드 리팩토링 및 문서화 **[완료]**
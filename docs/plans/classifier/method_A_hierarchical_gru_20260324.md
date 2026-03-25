# 방법 A: Hierarchical Encoder (KoELECTRA + GRU)

## 논문 출처

| 논문 | 연도 | 링크 | 핵심 아이디어 |
|------|------|------|-------------|
| Hierarchical Transformers for Long Document Classification | 2019 | https://arxiv.org/abs/1910.10781 | 문단 단위 BERT 인코딩 → [CLS] 시퀀스를 Transformer/LSTM으로 집계 |
| RoBERT / ToBERT (Hierarchical BERT) | 2020 | https://github.com/Joran1101/Hierarchical-BERT-based-Text-Classification | BERT + Transformer 계층 구조로 긴 문서 분류 |
| Recurrent BERT 계열 | 다수 | document-level sentiment analysis 문헌 | BERT 출력을 RNN으로 시퀀스 모델링 |

---

## 아키텍처 상세

### 전체 구조

```
실시간 통화 스트림

[음성] → Whisper STT → 문장 분할 (kss) → 슬라이딩 윈도우 생성
                                            ↓
윈도우₁: "네 안녕하세요 고객님 OO은행입니다 무엇을 도와드릴까요"
  → KoELECTRA 인코딩 → [CLS]₁ (768차원 벡터)

윈도우₂: "고객님 현재 특별 대출 상품이 있는데요 금리가 매우 좋습니다"
  → KoELECTRA 인코딩 → [CLS]₂ (768차원 벡터)

윈도우₃: "본인 확인을 위해서 계좌번호와 비밀번호를 알려주셔야 합니다"
  → KoELECTRA 인코딩 → [CLS]₃ (768차원 벡터)

         ↓ [CLS] 벡터 시퀀스를 GRU에 입력

[CLS]₁ → GRU → h₁ → Linear → pred₁ (정상 0.85)
                ↓
[CLS]₂ → GRU(h₁) → h₂ → Linear → pred₂ (피싱 0.42)
                     ↓
[CLS]₃ → GRU(h₂) → h₃ → Linear → pred₃ (피싱 0.91)
                                           ↑
                              h₃에 "은행사칭 + 대출권유 + 개인정보요구"
                              문맥이 압축되어 있음
```

### 모듈 구성

```
┌─────────────────────────────────────────────────────────┐
│ HierarchicalClassifier                                   │
│                                                          │
│  ┌──────────────────────────────────────┐                │
│  │ KoELECTRA Encoder (frozen or ft)     │                │
│  │ - monologg/koelectra-base-v3         │                │
│  │ - 입력: 윈도우 텍스트 (≤512 tokens)    │                │
│  │ - 출력: [CLS] 벡터 (768d)             │                │
│  └──────────┬───────────────────────────┘                │
│             ↓                                             │
│  ┌──────────────────────────────────────┐                │
│  │ Context GRU                           │                │
│  │ - input_size: 768                     │                │
│  │ - hidden_size: 768                    │                │
│  │ - num_layers: 1                       │                │
│  │ - bidirectional: False (실시간이므로)   │                │
│  │ - 파라미터: ~1.77M                     │                │
│  └──────────┬───────────────────────────┘                │
│             ↓                                             │
│  ┌──────────────────────────────────────┐                │
│  │ Classifier Head                       │                │
│  │ - Linear(768, 2)                      │                │
│  │ - 파라미터: 1,538                      │                │
│  └──────────────────────────────────────┘                │
│                                                          │
│  총 추가 파라미터: ~1.77M (KoELECTRA 110M 대비 1.6%)      │
└─────────────────────────────────────────────────────────┘
```

### 우선 적용할 구조 개선

- 윈도우 표현은 `[CLS]` 단일값으로 고정하지 않고 아래 3가지를 같은 조건에서 비교
  - Baseline: `[CLS]`
  - Pooling 1: attention mask를 반영한 mean pooling
  - Pooling 2: 작은 attention pooling (`Linear(768, 1)`)
- GRU hidden size는 `768` 고정이 아니라 `256 / 384 / 768`을 탐색
- 가변 길이 윈도우 시퀀스는 padding만 하지 말고 실제 길이 정보를 함께 사용
  - `window_mask.sum(dim=1)`으로 실제 길이 계산
  - 가능하면 `pack_padded_sequence` / `pad_packed_sequence` 사용
  - 최소한 loss 계산 시에는 padding window를 완전히 제외

### 현재 서비스 기준 비교 원칙

현재 저장소의 분류기는 긴 윈도우 기반 단일 모델 추론을 사용 중이므로, 새 구조의 효과를 보려면 비교 축을 분리해야 한다.

- backbone 고정 비교: 같은 backbone에서 `single-window vs hierarchical-GRU`
- window 설정 고정 비교: 같은 `window_size / stride / max_length` 유지
- 집계 방식 고정 비교: `max window score`와 `sequential hidden state`를 분리 비교

즉, 첫 실험부터 KoELECTRA, 짧은 윈도우, GRU를 한 번에 동시에 바꾸지 않는다.

### PyTorch 모델 코드 (개념)

```python
import torch
import torch.nn as nn
from transformers import AutoModel

class HierarchicalPhishingClassifier(nn.Module):
    def __init__(self, encoder_name, hidden_size=768, num_labels=2, freeze_encoder=False):
        super().__init__()
        # 1. KoELECTRA 인코더
        self.encoder = AutoModel.from_pretrained(encoder_name)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # 2. 문맥 GRU
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,  # 실시간 → 단방향만
        )

        # 3. 분류기
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, window_input_ids, window_attention_masks, hidden=None):
        """
        window_input_ids: (batch, num_windows, seq_len)
        window_attention_masks: (batch, num_windows, seq_len)
        hidden: GRU의 이전 hidden state (추론 시 전달)

        returns: logits (batch, num_windows, num_labels), hidden
        """
        batch_size, num_windows, seq_len = window_input_ids.shape

        # 모든 윈도우를 한번에 인코딩 (효율성)
        flat_input_ids = window_input_ids.view(-1, seq_len)
        flat_attention_mask = window_attention_masks.view(-1, seq_len)

        encoder_output = self.encoder(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
        )

        # [CLS] 토큰 추출 (position 0)
        cls_embeddings = encoder_output.last_hidden_state[:, 0, :]  # (batch*num_windows, 768)
        cls_embeddings = cls_embeddings.view(batch_size, num_windows, -1)  # (batch, num_windows, 768)

        # GRU로 시퀀스 모델링
        gru_output, hidden = self.gru(cls_embeddings, hidden)  # (batch, num_windows, 768)

        # 각 time step에서 분류
        logits = self.classifier(gru_output)  # (batch, num_windows, 2)

        return logits, hidden

    def predict_streaming(self, window_input_ids, window_attention_mask, hidden):
        """실시간 추론: 윈도우 1개씩 처리"""
        # 인코딩
        encoder_output = self.encoder(
            input_ids=window_input_ids,       # (1, seq_len)
            attention_mask=window_attention_mask,
        )
        cls_embedding = encoder_output.last_hidden_state[:, 0, :]  # (1, 768)
        cls_embedding = cls_embedding.unsqueeze(1)  # (1, 1, 768)

        # GRU (이전 hidden 전달)
        gru_output, new_hidden = self.gru(cls_embedding, hidden)

        # 분류
        logits = self.classifier(gru_output.squeeze(1))  # (1, 2)

        return logits, new_hidden
```

---

## 학습 데이터 준비

### 전처리 파이프라인

```
[원본 데이터]
  all.csv (2,515건) + augmented (llm_fewshot.csv, asr_noised.csv)
      ↓
[Step 1] 문장 분할
  각 text → kss.split_sentences() → ["문장1", "문장2", ...]
  + audio_processor.py의 clean/filter 로직 적용
      ↓
[Step 2] 슬라이딩 윈도우 시퀀스 생성
  문장 리스트 → create_window_sequence(sentences, window=5, stride=3)
  → ["윈도우1_텍스트", "윈도우2_텍스트", ...]
      ↓
[Step 3] 데이터 분할
  파일명 기반 train/val/test = 80/10/10
  (같은 통화의 윈도우가 다른 split에 걸치지 않도록)
      ↓
[Step 4] 토큰화
  각 윈도우 텍스트 → KoELECTRA tokenizer
  max_length=512, padding='max_length', truncation=True
      ↓
[Step 5] 시퀀스 DataLoader
  한 통화 = 1 sample = 윈도우 시퀀스 + label
```

### 데이터 형태 예시

```python
# train.json 또는 train.pkl 형태
[
    {
        "id": "phishing_대출사기_001",
        "filename": "1.mp3",
        "label": 1,
        "sentences": ["네 안녕하세요", "고객님 대출 상품이...", ...],
        "windows": [
            "네 안녕하세요 고객님 무엇을 도와드릴까요 저는 OO은행 직원입니다 특별 대출 상품을 안내드리려고요",
            "저는 OO은행 직원입니다 특별 대출 상품을 안내드리려고요 금리가 연 2%로 매우 좋습니다",
            ...
        ],
        "num_windows": 8
    },
    {
        "id": "normal_상품가입_042",
        "filename": "45.mp3",
        "label": 0,
        "sentences": ["여보세요", "주문하신 상품 관련해서...", ...],
        "windows": [
            "여보세요 주문하신 상품 관련해서 연락드렸습니다 배송이 오늘 출발했습니다 문 앞에 놓아드리면 될까요",
            ...
        ],
        "num_windows": 3
    },
]
```

### Collate 함수 (가변 길이 시퀀스 처리)

```python
def collate_fn(batch):
    """통화마다 윈도우 수가 다르므로 padding 필요"""
    max_windows = max(item['num_windows'] for item in batch)

    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    all_window_masks = []  # 실제 윈도우 vs padding 구분

    for item in batch:
        windows = item['windows']
        label = item['label']
        num_windows = item['num_windows']

        # 토큰화
        tokenized = tokenizer(
            windows, max_length=512, padding='max_length',
            truncation=True, return_tensors='pt'
        )

        # 윈도우 수 padding
        pad_count = max_windows - num_windows
        if pad_count > 0:
            pad_ids = torch.full(
                (pad_count, 512),
                tokenizer.pad_token_id,
                dtype=torch.long,
            )
            pad_masks = torch.zeros(pad_count, 512, dtype=torch.long)
            tokenized['input_ids'] = torch.cat([tokenized['input_ids'], pad_ids])
            tokenized['attention_mask'] = torch.cat([tokenized['attention_mask'], pad_masks])

        all_input_ids.append(tokenized['input_ids'])
        all_attention_masks.append(tokenized['attention_mask'])
        all_labels.append(label)

        # 윈도우 마스크: 실제 윈도우=1, padding=0
        window_mask = [1] * num_windows + [0] * pad_count
        all_window_masks.append(window_mask)

    return {
        'input_ids': torch.stack(all_input_ids),          # (batch, max_windows, 512)
        'attention_mask': torch.stack(all_attention_masks), # (batch, max_windows, 512)
        'labels': torch.tensor(all_labels),                 # (batch,)
        'window_mask': torch.tensor(all_window_masks),      # (batch, max_windows)
    }
```

추가 권장:

- `window_lengths = window_mask.sum(dim=1)`를 같이 반환
- GRU 입력 전에 `cls_embeddings`를 실제 길이 기준으로 pack 처리
- encoder 단계에서는 padding window도 계산되므로, 비슷한 `num_windows`끼리 bucket batching을 고려

---

## 학습 전략

### 2단계 학습

**Stage 1: GRU만 학습 (KoELECTRA frozen)**
- KoELECTRA 가중치 고정 → [CLS] 벡터는 사전학습된 표현 그대로
- GRU + Linear만 학습 → 빠른 수렴 (학습 파라미터 ~1.77M)
- Epoch: 10~20, LR: 1e-3 ~ 5e-4

**Stage 2: 전체 fine-tune**
- KoELECTRA + GRU + Linear 전부 학습
- KoELECTRA LR: 2e-5 (작게), GRU LR: 1e-4 (상대적으로 크게)
- Epoch: 3~5

### 추천 실험 순서

**Exp A0: 공정 baseline 재현**
- 기존 서비스와 가장 가까운 `single-window` baseline 확보
- 이후 비교용 기준 지표 생성

**Exp A1: Hierarchical-GRU + [CLS]**
- 문서 원안 그대로의 가장 단순한 계층형 baseline

**Exp A2: Hierarchical-GRU + mean pooling**
- frozen encoder 단계에서 가장 먼저 비교할 후보

**Exp A3: Hierarchical-GRU + attention pooling**
- 파라미터는 소량 증가하지만, 윈도우 내 핵심 표현 추출에 유리할 수 있음

**Exp A4: hidden size sweep**
- `256 / 384 / 768`
- 데이터 규모상 작은 hidden이 더 안정적일 가능성이 있음

### Loss 함수 옵션

**옵션 1: 마지막 윈도우만** (통화 전체 판단)
```python
# 마지막 실제 윈도우의 logits로만 loss 계산
last_logits = logits[range(batch_size), last_window_indices]  # (batch, 2)
loss = CrossEntropyLoss(last_logits, labels)
```

**옵션 2: 모든 윈도우 동일 label** (간단한 baseline)
```python
# 모든 실제 윈도우에서 동일 label로 loss 계산
# 장점: 학습 신호가 더 많음
# 단점: 피싱 통화 초반의 정상 발화까지 양성으로 밀 수 있음
for t in range(num_windows):
    loss += CrossEntropyLoss(logits[:, t, :], labels) * window_mask[:, t]
loss /= window_mask.sum()
```

**옵션 3: 뒤쪽 윈도우 가중치 증가** (추천)
```python
# 정상 통화는 모든 윈도우를 동일 가중치
# 피싱 통화는 뒤로 갈수록 loss 가중치를 증가
weights = build_temporal_weights(window_mask)  # 예: 선형 또는 지수 증가

for t in range(num_windows):
    step_loss = CrossEntropyLoss(logits[:, t, :], labels, reduction='none')
    loss += (step_loss * weights[:, t] * window_mask[:, t]).sum()

loss /= (weights * window_mask).sum()
```

권장 메모:

- 초반에는 `옵션 1`과 `옵션 3`을 우선 비교
- 실제 서비스 목적이 조기 탐지라면 `옵션 2`를 바로 추천하지 않음
- 가능하면 장기적으로는 윈도우별 위험 annotation 또는 pseudo-onset label을 추가

### 하이퍼파라미터

| 항목 | 값 | 비고 |
|------|-----|------|
| 윈도우 크기 | 5문장 | 512 토큰 이내 확인 필요 |
| 스트라이드 | 3문장 | 2문장 겹침 |
| 윈도우 표현 | CLS / mean / attention | 비교 실험 권장 |
| GRU hidden | 256 / 384 / 768 | 768 고정 금지 |
| GRU layers | 1 | 데이터 양 고려 시 충분 |
| Batch size | 4~8 | 통화 단위 (윈도우 수 가변) |
| Optimizer | AdamW | |
| Scheduler | Linear warmup + decay | |

---

## 추론 (실시간)

```python
# 실시간 추론 루프 (의사 코드)
model.eval()
hidden = None  # GRU 초기 상태

while call_is_active:
    new_sentences = stt_engine.get_new_sentences()
    window = create_current_window(sentence_buffer, window_size=5)

    if window:
        input_ids, attention_mask = tokenize(window)

        with torch.no_grad():
            logits, hidden = model.predict_streaming(input_ids, attention_mask, hidden)
            prob = softmax(logits)[0, 1]  # 피싱 확률

        if prob > threshold:
            alert("피싱 의심!")

        # hidden은 유지 → 다음 윈도우에 문맥 전달
```

---

## 장단점 요약

### 장점
- 구현이 간단하고 직관적
- KoELECTRA를 수정 없이 사용 가능
- 2단계 학습으로 안정적 수렴
- GRU의 hidden state가 대화 문맥을 자연스럽게 압축

### 단점
- [CLS] 인코딩 시 과거 문맥 미반영 (인코딩 후 결합)
- "계좌번호" 같은 토큰이, 이전에 "금감원 사칭"이 있었는지와 무관하게 동일하게 인코딩됨
- GRU bottleneck: 768차원 벡터 하나에 전체 대화 이력을 압축해야 함

---

## 평가 설계

### 오프라인 분류 지표

- 통화 단위: AUROC, F1, recall@high-precision
- 윈도우 단위: prefix AUROC (`25% / 50% / 75%` 시점)
- 조기 탐지: `first alert latency`, `decisive cue 이전 false alert 비율`

### 데이터 분리 평가

- `original / llm_fewshot / asr_noise` 별 분리 평가
- 피싱 카테고리별 macro 평균
- 가능하면 category hold-out 실험 추가
  - 예: 특정 사기 유형 전체를 test로 두고 일반화 확인

### 서비스 집계 비교

아래 3가지를 같이 본다.

- 마지막 윈도우 점수
- 최대 윈도우 점수
- GRU hidden 기반 누적 점수

순차 모델은 전체 F1만 보면 장점이 잘 안 보일 수 있으므로, 반드시 조기 탐지 지표와 같이 평가한다.

---

## 최종 권장안

이 문서 기준 우선순위는 다음과 같다.

1. `Hierarchical-GRU + mean pooling`
2. `옵션 1 / 옵션 3 loss 비교`
3. hidden size `256 / 384 / 768` 탐색
4. source별 및 prefix 기준 평가

즉, 방법 A는 가장 먼저 구현/검증할 1순위 구조로 유지하되, `[CLS] 단일표현 + 모든 윈도우 동일 label`에 묶이지 않도록 실험 계획을 확장한다.

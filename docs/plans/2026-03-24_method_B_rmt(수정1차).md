# 방법 B: Recurrent Memory Transformer (RMT)

## 논문 출처

### 핵심 논문
| 논문 | 연도 / 학회 | 링크 | 핵심 아이디어 |
|------|------------|------|-------------|
| Recurrent Memory Transformer | NeurIPS 2022 | https://arxiv.org/abs/2207.06881 | BERT에 메모리 토큰 추가, 세그먼트 간 recurrence로 장기 문맥 유지 |
| Scaling RMT to 1M+ tokens | AAAI 2024 | https://arxiv.org/abs/2304.11062 | RMT를 100만 토큰까지 확장, 합성/실제 task 검증 |

### 공식 구현
| 이름 | 링크 | 비고 |
|------|------|------|
| 공식 RMT (Bulatov et al.) | https://github.com/booydar/recurrent-memory-transformer | BERT/RoBERTa/DeBERTa 호환 |
| lucidrains PyTorch 구현 | https://github.com/lucidrains/recurrent-memory-transformer-pytorch | 간결한 커뮤니티 구현 |

### 관련 연구
| 논문 | 연도 / 학회 | 링크 | 관련성 |
|------|------------|------|--------|
| MovieChat | CVPR 2024 | https://arxiv.org/abs/2307.16449 | 영상 도메인에서 dense→sparse 메모리 압축 (동일 문제) |
| Compact Recurrent Transformer | 2025 | https://arxiv.org/html/2505.00929 | Transformer + GRU, hidden state가 세그먼트 간 메모리 |
| HMT (Hierarchical Memory Transformer) | NAACL 2025 | https://aclanthology.org/2025.naacl-long.410.pdf | 세그먼트 요약 + cross-attention 메모리 |
| Memformer | 2020 | https://arxiv.org/abs/2010.06891 | 고정 크기 외부 메모리 + read/write attention |
| Memory-Augmented Transformers Survey | 2025 | https://arxiv.org/html/2508.10824v1 | 메모리 증강 트랜스포머 종합 서베이 |

---

## 아키텍처 상세

### RMT 핵심 아이디어

일반 BERT 입력:
```
[CLS] 오늘 날씨가 좋습니다 [SEP]
```

RMT 입력 (메모리 토큰 추가):
```
[MEM₁][MEM₂]...[MEM₁₀][CLS] 오늘 날씨가 좋습니다 [SEP]
  ↑ 이전 세그먼트의 기억    ↑ 현재 세그먼트 텍스트
```

- MEM 토큰은 다른 모든 토큰과 self-attention으로 상호작용
- 현재 텍스트의 각 토큰이 MEM에 저장된 과거 정보를 직접 참조
- 처리 후 MEM 토큰의 hidden state가 업데이트 → 다음 세그먼트로 전달

### 전체 구조

```
실시간 통화 스트림

[초기 메모리: MEM₁..MEM₁₀ = learnable parameters]

━━━ 윈도우 1 ━━━
입력: [MEM₁..MEM₁₀] + [CLS] 네 안녕하세요 고객님 OO은행입니다 [SEP]

KoELECTRA Self-Attention:
  "안녕하세요" ←→ MEM₁ (아직 정보 없음)
  "OO은행" ←→ MEM₂ (아직 정보 없음)
  ...

출력: pred₁ = 정상 (0.12)
메모리 업데이트: MEM'₁..MEM'₁₀ ← hidden_state[:, :10, :]
  (MEM'에 "OO은행 직원 자칭" 정보 저장됨)

━━━ 윈도우 2 ━━━
입력: [MEM'₁..MEM'₁₀] + [CLS] 특별 대출 상품이 있는데요 금리가 좋습니다 [SEP]

KoELECTRA Self-Attention:
  "대출" ←→ MEM'₁ ("OO은행 직원 자칭" 정보와 상호작용)
  "금리" ←→ MEM'₂
  ...

출력: pred₂ = 피싱 의심 (0.45)
메모리 업데이트: MEM''₁..MEM''₁₀
  (MEM''에 "은행 사칭 + 대출 권유" 정보 누적됨)

━━━ 윈도우 3 ━━━
입력: [MEM''₁..MEM''₁₀] + [CLS] 계좌번호와 비밀번호를 알려주세요 [SEP]

KoELECTRA Self-Attention:
  "계좌번호" ←→ MEM''₁ ("은행 사칭 + 대출 권유" 정보와 상호작용!)
  "비밀번호" ←→ MEM''₂
  → "계좌번호"의 representation이 일반 맥락과 다르게 인코딩됨

출력: pred₃ = 피싱 (0.94)  ← 문맥 반영으로 확신도 높음
메모리 업데이트: MEM'''₁..MEM'''₁₀
```

### 방법 A (GRU)와의 핵심 차이

```
방법 A (GRU):
  윈도우3 "계좌번호" → KoELECTRA → (이전 문맥 없이 인코딩) → [CLS]₃
  [CLS]₃ + h₂(GRU) → 판단

방법 B (RMT):
  [MEM'' = 이전 문맥] + 윈도우3 "계좌번호" → KoELECTRA
  → "계좌번호" 토큰이 MEM''과 attention하면서 인코딩 자체가 달라짐
  → 판단
```

### 모듈 구성

```
┌─────────────────────────────────────────────────────────┐
│ RMTPhishingClassifier                                    │
│                                                          │
│  ┌──────────────────────────────────────┐                │
│  │ Memory Token Embeddings (학습 가능)    │                │
│  │ - num_mem_tokens: 10                  │                │
│  │ - dim: 768                            │                │
│  │ - 파라미터: 7,680                      │                │
│  └──────────┬───────────────────────────┘                │
│             ↓ (입력 앞에 prepend)                         │
│  ┌──────────────────────────────────────┐                │
│  │ KoELECTRA Encoder (fine-tune 필수)    │                │
│  │ - monologg/koelectra-base-v3         │                │
│  │ - 입력: MEM + 윈도우 텍스트 (≤512)     │                │
│  │ - 출력: hidden states (all positions)  │                │
│  │ - MEM과 텍스트 간 self-attention 수행   │                │
│  └──────────┬───────────────────────────┘                │
│             ↓                                             │
│  ┌──────────────────────────────────────┐                │
│  │ Memory Extractor                      │                │
│  │ - output[:, :10, :] → 다음 MEM으로    │                │
│  └──────────┬───────────────────────────┘                │
│             ↓                                             │
│  ┌──────────────────────────────────────┐                │
│  │ Classifier Head                       │                │
│  │ - output[:, 10, :] → [CLS] 위치      │                │
│  │ - Linear(768, 2)                      │                │
│  │ - 파라미터: 1,538                      │                │
│  └──────────────────────────────────────┘                │
│                                                          │
│  총 추가 파라미터: ~9,218 (KoELECTRA 110M 대비 0.008%)    │
└─────────────────────────────────────────────────────────┘
```

### 적용 전제

방법 B는 이론적으로는 가장 매력적이지만, 현재 프로젝트에서는 바로 1순위로 구현하기보다 방법 A를 기준선으로 확보한 뒤 2차 실험으로 검증하는 것이 안전하다.

이유:

- backbone, 윈도우 길이, 메모리 구조를 한 번에 바꾸면 개선 원인을 분리하기 어려움
- 데이터 규모가 크지 않아 RMT의 fine-tuning 안정성이 낮을 수 있음
- 구현 디테일 하나만 틀려도 "RMT가 안 좋다"는 잘못된 결론에 도달하기 쉬움

### 현재 문서 기준에서 바로 고쳐야 할 구현 원칙

- 메모리 gradient를 매 세그먼트에서 항상 detach하지 않음
  - 학습 시에는 최근 `bptt_depth` 구간까지는 gradient 유지
  - `detach()`는 `forward()` 루프에서 시점 제어
- 가능하면 special token 기반 구현을 우선
  - `[MEM_0] ... [MEM_n]`를 tokenizer에 추가
  - `input_ids` 경로로 encoder를 타게 해서 embedding / position / token type 처리 일관성 확보
- RMT의 비교 대상은 Hierarchical-GRU와 같은 조건으로 둠
  - 같은 backbone
  - 같은 window_size / stride
  - 같은 데이터 split

### PyTorch 모델 코드 (개념)

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class RMTPhishingClassifier(nn.Module):
    def __init__(self, encoder_name, num_mem_tokens=10, hidden_size=768, num_labels=2):
        super().__init__()
        self.num_mem_tokens = num_mem_tokens
        self.hidden_size = hidden_size

        # 1. KoELECTRA 인코더
        self.encoder = AutoModel.from_pretrained(encoder_name)

        # 2. 학습 가능한 메모리 토큰 임베딩
        # 토크나이저에 [MEM_0]~[MEM_9] special token 추가 후
        # 해당 토큰의 embedding을 학습
        self.mem_token_ids = None  # 초기화 시 설정

        # 또는: embedding을 직접 관리하는 방식
        self.memory_embeddings = nn.Parameter(
            torch.randn(1, num_mem_tokens, hidden_size) * 0.02
        )

        # 3. 분류기
        self.classifier = nn.Linear(hidden_size, num_labels)

    def _prepend_memory(self, input_embeds, memory):
        """
        input_embeds: (batch, seq_len, 768) - 토큰 임베딩
        memory: (batch, num_mem, 768) - 메모리 hidden states
        returns: (batch, num_mem + seq_len, 768)
        """
        return torch.cat([memory, input_embeds], dim=1)

    def _extend_attention_mask(self, attention_mask):
        """attention mask에 메모리 토큰 위치 추가 (모두 1)"""
        batch_size = attention_mask.shape[0]
        mem_mask = torch.ones(batch_size, self.num_mem_tokens,
                             device=attention_mask.device, dtype=attention_mask.dtype)
        return torch.cat([mem_mask, attention_mask], dim=1)

    def forward_segment(self, input_ids, attention_mask, memory=None):
        """
        한 세그먼트(윈도우) 처리

        input_ids: (batch, seq_len) - 현재 윈도우 토큰
        attention_mask: (batch, seq_len)
        memory: (batch, num_mem, 768) - 이전 세그먼트의 메모리
                None이면 초기 메모리 사용

        returns: logits, new_memory
        """
        batch_size = input_ids.shape[0]

        # 초기 메모리 설정
        if memory is None:
            memory = self.memory_embeddings.expand(batch_size, -1, -1)

        # 권장 구현:
        # 1) tokenizer에 [MEM_i] special token 추가
        # 2) memory token id + text input id를 concat
        # 3) input_ids 경로로 encoder 호출
        #
        # 아래 inputs_embeds 방식은 프로토타입으로는 가능하지만,
        # embedding / position 처리 실수를 만들기 쉬우므로 우선순위는 낮다.
        input_embeds = self.encoder.embeddings(input_ids)
        combined_embeds = self._prepend_memory(input_embeds, memory)
        extended_mask = self._extend_attention_mask(attention_mask)
        outputs = self.encoder(
            inputs_embeds=combined_embeds,
            attention_mask=extended_mask,
        )

        hidden_states = outputs.last_hidden_state  # (batch, num_mem + seq_len, 768)

        # 메모리 추출 (처음 num_mem 위치)
        new_memory = hidden_states[:, :self.num_mem_tokens, :]  # (batch, num_mem, 768)

        # [CLS] 위치에서 분류 (num_mem 번째 위치)
        cls_hidden = hidden_states[:, self.num_mem_tokens, :]  # (batch, 768)
        logits = self.classifier(cls_hidden)  # (batch, 2)

        return logits, new_memory

    def forward(self, window_sequences, attention_masks, bptt_depth=2):
        """
        학습용: 전체 윈도우 시퀀스 처리

        window_sequences: (batch, num_windows, seq_len)
        attention_masks: (batch, num_windows, seq_len)
        bptt_depth: 역전파할 세그먼트 수 (truncated BPTT)
        """
        batch_size, num_windows, seq_len = window_sequences.shape
        memory = None
        all_logits = []

        for t in range(num_windows):
            input_ids = window_sequences[:, t, :]
            attention_mask = attention_masks[:, t, :]

            # Truncated BPTT: bptt_depth 이전의 메모리는 detach
            if memory is not None and t >= bptt_depth:
                memory = memory.detach()

            logits, memory = self.forward_segment(input_ids, attention_mask, memory)
            all_logits.append(logits)

        return torch.stack(all_logits, dim=1)  # (batch, num_windows, 2)

    def predict_streaming(self, input_ids, attention_mask, memory):
        """실시간 추론: 윈도우 1개씩 처리"""
        with torch.no_grad():
            logits, new_memory = self.forward_segment(input_ids, attention_mask, memory)
        return logits, new_memory
```

추가 권장:

- direct `inputs_embeds` 방식은 실험용으로만 유지
- 실제 구현은 special token + `input_ids` 방식으로 시작
- memory token에도 dropout / layernorm / gating을 붙일 수 있도록 확장 여지 확보

---

## 학습 데이터 준비

### 전처리 파이프라인 (방법 A와 동일)

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
      ↓
[Step 4] 토큰화 (RMT 특화: max_length = 512 - num_mem_tokens)
  각 윈도우 텍스트 → KoELECTRA tokenizer
  max_length = 502 (512 - 10 MEM tokens), padding, truncation
      ↓
[Step 5] 시퀀스 DataLoader
  방법 A와 동일한 collate_fn (윈도우 수 가변 padding)
```

### 방법 A와의 차이점

| 항목 | 방법 A (GRU) | 방법 B (RMT) |
|------|-------------|-------------|
| 텍스트 max_length | 512 | 502 (메모리 10토큰 예약) |
| 메모리 전달 | GRU hidden state | MEM hidden states |
| 토크나이저 | 변경 없음 | [MEM_0]~[MEM_9] special token 추가 (선택) |
| 전처리 | 동일 | 동일 |

### 토크나이저 확장 (권장)

```python
# 방법 1: special token 추가 (권장, 공식 RMT 방식)
tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
mem_tokens = [f"[MEM_{i}]" for i in range(10)]
tokenizer.add_special_tokens({"additional_special_tokens": mem_tokens})
model.encoder.resize_token_embeddings(len(tokenizer))

# 방법 2: embedding 직접 관리 (프로토타입 전용)
# special token 추가 없이, forward에서 직접 memory_embeddings를 prepend
# → 토크나이저 수정 불필요
# → 대신 position / segment / embedding stack 실수 가능성이 높음
```

---

## 학습 전략

### End-to-end 학습 (KoELECTRA + 메모리 + 분류기)

```python
# 학습 루프 (의사 코드)
optimizer = AdamW([
    {'params': model.encoder.parameters(), 'lr': 2e-5},        # KoELECTRA
    {'params': model.memory_embeddings, 'lr': 1e-3},           # 메모리 토큰
    {'params': model.classifier.parameters(), 'lr': 1e-3},     # 분류기
])

for batch in dataloader:
    all_logits = model(batch['input_ids'], batch['attention_mask'], bptt_depth=3)

    # 간단 baseline: 모든 윈도우 동일 label
    loss = 0
    for t in range(num_windows):
        mask = batch['window_mask'][:, t]
        step_loss = CrossEntropyLoss(
            all_logits[:, t, :], batch['labels'], reduction='none'
        )
        loss += (step_loss * mask).sum()
    loss = loss / batch['window_mask'].sum()

    loss.backward()
    optimizer.step()
```

### 추천 loss 변경

방법 A와 동일하게, 피싱 통화 초반까지 전부 양성으로 미는 방식은 피하는 것이 좋다.

우선 비교할 loss:

- L1: 마지막 윈도우만
- L2: 모든 윈도우 동일 label
- L3: 뒤쪽 윈도우 가중치 증가 (추천)

```python
weights = build_temporal_weights(batch['window_mask'])

for t in range(num_windows):
    step_loss = CrossEntropyLoss(
        all_logits[:, t, :], batch['labels'], reduction='none'
    )
    loss += (step_loss * weights[:, t] * batch['window_mask'][:, t]).sum()

loss /= (weights * batch['window_mask']).sum()
```

### 추천 실험 순서

**Exp B0: 비교 기준선 확보**
- 같은 backbone / 같은 window 설정에서 방법 A 성능 확보

**Exp B1: 최소 RMT**
- memory token만 추가
- classifier는 `[CLS]` 하나 사용
- mem tokens = `5 / 10 / 20`

**Exp B2: memory 안정화**
- memory dropout
- memory decay / reset
- segment boundary에서 gating 추가 여부 비교

**Exp B3: longer-context 검증**
- 통화 앞부분 prefix만 봤을 때 방법 A보다 빨리 반응하는지 확인

즉, 방법 B의 성공 기준은 단순 F1 상승이 아니라 조기 탐지 성능 개선이다.

### Truncated BPTT 설명

```
전체 시퀀스: [w₁] → [w₂] → [w₃] → [w₄] → [w₅] → [w₆]

Full BPTT (메모리 폭발):
  w₆의 gradient가 w₁까지 전파 → GPU 메모리 6배 필요

Truncated BPTT (depth=3):
  w₆의 gradient는 w₄까지만 전파
  w₃의 gradient는 w₁까지만 전파

  실제로:
  [w₁] → [w₂] → [w₃] → detach → [w₄] → [w₅] → [w₆]
                                   ↑ 여기서 gradient 차단

  메모리 자체(값)는 w₁부터 누적되지만,
  gradient는 최근 3 세그먼트에서만 계산
```

### 하이퍼파라미터

| 항목 | 값 | 비고 |
|------|-----|------|
| 메모리 토큰 수 | 10 | 5~20 범위에서 탐색 |
| 윈도우 크기 | 5문장 | 502 토큰 이내 확인 필요 |
| 스트라이드 | 3문장 | |
| Truncated BPTT depth | 2~3 | GPU 메모리에 따라 조절 |
| Memory regularization | dropout / decay / reset | contamination 방지 |
| Batch size | 2~4 | BPTT로 메모리 사용 증가 |
| KoELECTRA LR | 2e-5 | |
| Memory/Classifier LR | 1e-3 | |
| Optimizer | AdamW | |
| Scheduler | Linear warmup + cosine decay | |
| Epochs | 10~20 | |

---

## 추론 (실시간)

```python
# 실시간 추론 루프 (의사 코드)
model.eval()
memory = None  # 초기 메모리 (학습된 memory_embeddings 사용)

while call_is_active:
    new_sentences = stt_engine.get_new_sentences()
    window = create_current_window(sentence_buffer, window_size=5)

    if window:
        input_ids, attention_mask = tokenize(window, max_length=502)

        logits, memory = model.predict_streaming(input_ids, attention_mask, memory)
        prob = softmax(logits)[0, 1]  # 피싱 확률

        if prob > threshold:
            alert("피싱 의심!")

        # memory는 유지 → 다음 윈도우에 문맥 직접 전달
        # 다음 윈도우의 인코딩에 이 memory가 self-attention으로 참여
```

### 추론 속도

- 메모리 10토큰 추가 = 전체 512 토큰 중 2% → 속도 영향 미미
- Self-attention 계산량: O((502+10)²) vs O(512²) → 거의 동일
- 추가 연산: 메모리 추출 (인덱싱) → 무시 가능

---

## 장단점 요약

### 장점
- **과거 문맥이 현재 인코딩에 직접 참여**: "계좌번호" 토큰이 이전 "금감원 사칭" 맥락을 attention으로 참조
- **추가 파라미터 극소**: ~9K (GRU의 ~1.77M 대비 200배 적음)
- **추론 구조가 단순**: KoELECTRA forward 1회로 분류 + 메모리 업데이트 동시 수행
- **NeurIPS/AAAI 검증**: BERT 계열과의 호환성 확인됨

### 단점
- **KoELECTRA fine-tune 필수**: MEM 토큰과의 attention 패턴을 학습해야 함 (freeze 불가)
- **Truncated BPTT 필요**: 학습 시 메모리 사용량 증가 (BPTT depth에 비례)
- **메모리 토큰 수 튜닝**: 너무 적으면 정보 부족, 너무 많으면 텍스트 공간 축소
- **inputs_embeds 사용**: input_ids 대신 embedding을 직접 다루는 코드가 필요 (약간 복잡)

---

## 구현 시 주의사항

1. **Position Embedding**: 메모리 토큰에도 position embedding이 적용됨. KoELECTRA의 position 0~9가 항상 메모리에 할당되므로, 텍스트의 position은 10부터 시작. 이것이 학습 초기에 불안정할 수 있음 → warmup을 길게 설정

2. **메모리 초기화**: 첫 윈도우에서 memory=None일 때 self.memory_embeddings 사용. 이 초기값이 잘 학습되어야 cold start 문제 완화

3. **detach 시점**: memory.detach()를 너무 자주 하면 장기 문맥 학습 불가, 너무 적게 하면 메모리 폭발. BPTT depth 2~3이 실용적 균형점

4. **Gradient Accumulation**: 배치 사이즈가 작을 수밖에 없으므로 (BPTT 메모리 사용), gradient accumulation으로 effective batch size를 키우는 것을 권장

5. **Memory contamination**: STT 오류나 초반 오탐 정보가 memory에 누적되면 이후 윈도우를 계속 왜곡할 수 있음
   - memory dropout
   - decay (`new_memory = alpha * old + (1-alpha) * current`)
   - 긴 무의미 발화 구간에서 reset 정책

6. **공정 비교**: 방법 A보다 backbone이나 max_length까지 동시에 바꾸면 해석이 불가능해짐
   - 같은 split
   - 같은 window 설정
   - 같은 평가 지표로 비교

---

## 평가 설계

### 핵심 지표

- 통화 단위: AUROC, F1, recall@high-precision
- prefix 지표: `25% / 50% / 75%` 시점 AUROC
- 조기 탐지: `first alert latency`
- 안정성: `decisive cue 이전 false alert 비율`

### 데이터 축별 평가

- `original / llm_fewshot / asr_noise`
- 피싱 카테고리별 성능
- 가능하면 category hold-out

### 성공 기준

방법 B는 아래 중 하나를 명확히 만족할 때 채택 가치가 있다.

- 방법 A보다 prefix 구간에서 더 높은 recall
- 같은 recall에서 더 낮은 false alert
- 긴 통화에서 후반부 성능 저하가 덜함

반대로 전체 F1만 소폭 개선하거나, 구현 복잡도 대비 이득이 작으면 방법 A를 유지하는 편이 낫다.

---

## 최종 권장안

이 문서 기준 우선순위는 다음과 같다.

1. 방법 A를 먼저 구현해 기준선 확보
2. 방법 B는 special token 기반 최소 구현으로 시작
3. `mem tokens / bptt depth / memory decay`만 우선 탐색
4. 조기 탐지 지표에서 이득이 확인될 때만 확장

즉, 방법 B는 고위험-고보상 실험으로 유지하되, 구현 난이도와 평가 기준을 더 엄격하게 관리한다.

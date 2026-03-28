"""
Compressive Memory 기반 스트리밍 분류 모델

전체 구조:
  세그먼트 → RoBERTa Encoder → Projection (768→128)
  → Compressive Memory (FM + CM) → Attention → Classification Head → multi-hot 출력

주요 특징:
  - 2계층 메모리: Fine Memory (최근 r개 상세 보존) + Compressed Memory (1D Conv 압축)
  - 학습 가능한 초기 메모리 슬롯
  - Baseline 모드 지원 (메모리 없이 마지막 세그먼트만 분류)
"""

import io
from contextlib import redirect_stderr, redirect_stdout

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


IGNORED_LOAD_REPORT_PREFIXES = (
    "lm_head.",
    "pooler.",
    "roberta.embeddings.position_ids",
)


def load_encoder_backbone(encoder_name: str):
    """
    HuggingFace 모델 로드 출력을 캡처한 뒤,
    실제 backbone fine-tuning에 의미 있는 항목만 다시 출력한다.
    """
    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        encoder = AutoModel.from_pretrained(encoder_name)

    noteworthy_rows = []
    for line in buffer.getvalue().splitlines():
        if "|" not in line:
            continue

        key = line.split("|", 1)[0].strip()
        if (
            not key
            or key == "Key"
            or key.startswith("-")
            or any(key.startswith(prefix) for prefix in IGNORED_LOAD_REPORT_PREFIXES)
        ):
            continue
        noteworthy_rows.append(line.rstrip())

    if noteworthy_rows:
        print(f"[모델] {encoder_name} 로드 리포트 (backbone 관련 항목만 표시)")
        for row in noteworthy_rows:
            print(row)
    else:
        print(
            f"[모델] {encoder_name} 로드 완료 "
            "(lm_head/pooler/position_ids 관련 리포트는 숨김)"
        )

    return encoder


class LinearProjection(nn.Module):
    """
    인코더 출력을 메모리 슬롯 차원으로 축소하는 선형 변환.
    768차원 → slot_dim(128)차원
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class CompressiveMemory(nn.Module):
    """
    2계층 Compressive Memory 모듈.

    Fine Memory (FM): 최근 r개의 세그먼트 표현을 원본 그대로 보존
    Compressed Memory (CM): 오래된 세그먼트를 1D Conv로 압축하여 k개 슬롯에 보존

    메모리 업데이트 알고리즘:
      1. FM 끝에 새 세그먼트 추가
      2. FM 크기 > r이면:
         - FM에서 가장 오래된 슬롯(fm₁)을 제거
         - [cm_last, fm₁]을 1D Conv로 압축 → c_new
         - CM에서 가장 오래된 슬롯(cm₁)을 제거, c_new 추가
    """

    def __init__(
        self,
        slot_dim: int,
        fm_size: int,
        cm_size: int,
        conv_kernel_size: int = 2,
        use_learnable_init: bool = True,
        compress_fn: str = "conv",
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.fm_size = fm_size      # r: Fine Memory 최대 크기
        self.cm_size = cm_size      # k: Compressed Memory 최대 크기
        self.compress_fn = compress_fn

        # 1D Conv 압축 함수: [cm_last, fm_oldest] → c_new
        if compress_fn == "conv" and cm_size > 0:
            self.compressor = nn.Conv1d(
                in_channels=slot_dim,
                out_channels=slot_dim,
                kernel_size=conv_kernel_size,
            )
        else:
            self.compressor = None

        # 학습 가능한 초기 메모리 (t=1에서 FM/CM이 비어있을 때 사용)
        if use_learnable_init:
            if fm_size > 0:
                self.fm_init = nn.Parameter(torch.randn(fm_size, slot_dim) * 0.02)
            if cm_size > 0:
                self.cm_init = nn.Parameter(torch.randn(cm_size, slot_dim) * 0.02)
        else:
            self.fm_init = None
            self.cm_init = None

    def init_memory(self, batch_size: int, device: torch.device) -> dict:
        """
        배치 크기에 맞는 초기 메모리 상태를 생성한다.

        Returns:
            dict:
                - fine_memory: (B, r, slot_dim) 또는 None
                - compressed_memory: (B, k, slot_dim) 또는 None
                - fm_count: (B,) 각 샘플의 현재 FM 슬롯 사용 수
        """
        state = {"fm_count": torch.zeros(batch_size, dtype=torch.long, device=device)}

        # Fine Memory 초기화
        if self.fm_size > 0:
            if self.fm_init is not None:
                # 학습 가능한 초기값을 배치 크기만큼 복제
                state["fine_memory"] = self.fm_init.unsqueeze(0).expand(batch_size, -1, -1).clone()
            else:
                state["fine_memory"] = torch.zeros(batch_size, self.fm_size, self.slot_dim, device=device)
        else:
            state["fine_memory"] = None

        # Compressed Memory 초기화
        if self.cm_size > 0:
            if self.cm_init is not None:
                state["compressed_memory"] = self.cm_init.unsqueeze(0).expand(batch_size, -1, -1).clone()
            else:
                state["compressed_memory"] = torch.zeros(batch_size, self.cm_size, self.slot_dim, device=device)
        else:
            state["compressed_memory"] = None

        return state

    def _compress(self, cm_last: torch.Tensor, fm_oldest: torch.Tensor) -> torch.Tensor:
        """
        1D Conv 또는 Mean Pooling으로 두 슬롯을 하나로 압축한다.

        Args:
            cm_last: (B, slot_dim) CM의 가장 최근 슬롯
            fm_oldest: (B, slot_dim) FM에서 밀려난 가장 오래된 슬롯

        Returns:
            (B, slot_dim) 압축된 새 슬롯
        """
        if self.compress_fn == "mean" or self.compressor is None:
            # Mean Pooling: 두 벡터의 평균
            return (cm_last + fm_oldest) / 2.0

        # 1D Conv: [cm_last, fm_oldest]를 채널 차원으로 쌓아서 처리
        # 입력: (B, 2, slot_dim) → Conv1d 기대 형식: (B, C=slot_dim, L=2)
        stacked = torch.stack([cm_last, fm_oldest], dim=2)  # (B, slot_dim, 2)
        compressed = self.compressor(stacked)                # (B, slot_dim, 1)
        return compressed.squeeze(2)                         # (B, slot_dim)

    def update(self, state: dict, new_segment: torch.Tensor, valid_mask: torch.Tensor) -> dict:
        """
        새 세그먼트를 받아 메모리를 업데이트한다.

        Args:
            state: init_memory 또는 이전 update의 출력
            new_segment: (B, slot_dim) 현재 세그먼트의 projected 표현
            valid_mask: (B,) True인 샘플만 업데이트 (패딩 세그먼트 skip)

        Returns:
            업데이트된 state dict
        """
        if self.fm_size == 0:
            # Baseline 모드: 메모리 없음
            return state

        B = new_segment.shape[0]
        fm = state["fine_memory"].clone()       # (B, r, slot_dim)
        fm_count = state["fm_count"].clone()    # (B,)

        # CM이 있는 경우
        cm = state["compressed_memory"].clone() if state["compressed_memory"] is not None else None

        for b in range(B):
            if not valid_mask[b]:
                continue  # 패딩 세그먼트는 skip

            count = fm_count[b].item()

            if count < self.fm_size:
                # FM이 아직 가득 차지 않음 → 빈 슬롯에 추가
                fm[b, count] = new_segment[b]
                fm_count[b] = count + 1
            else:
                # FM이 가득 참 → 가장 오래된 슬롯을 CM으로 압축
                fm_oldest = fm[b, 0]  # 가장 오래된 슬롯

                if cm is not None and self.cm_size > 0:
                    # CM의 가장 최근 슬롯과 FM의 가장 오래된 슬롯을 압축
                    cm_last = cm[b, -1]
                    c_new = self._compress(
                        cm_last.unsqueeze(0),
                        fm_oldest.unsqueeze(0),
                    ).squeeze(0)

                    # CM 시프트: 가장 오래된 것 제거, 새것 추가
                    cm[b] = torch.cat([cm[b, 1:], c_new.unsqueeze(0)], dim=0)

                # FM 시프트: 가장 오래된 것 제거, 새것 추가
                fm[b] = torch.cat([fm[b, 1:], new_segment[b].unsqueeze(0)], dim=0)

        new_state = {
            "fine_memory": fm,
            "compressed_memory": cm,
            "fm_count": fm_count,
        }

        return new_state

    def get_memory_slots(self, state: dict) -> torch.Tensor:
        """
        Attention에 전달할 메모리 슬롯을 [CM; FM] 순서로 결합한다.

        Args:
            state: 현재 메모리 상태

        Returns:
            (B, k+r, slot_dim) 전체 메모리 슬롯
        """
        parts = []

        if state["compressed_memory"] is not None:
            parts.append(state["compressed_memory"])

        if state["fine_memory"] is not None:
            parts.append(state["fine_memory"])

        if not parts:
            return None

        return torch.cat(parts, dim=1)  # (B, k+r, slot_dim)


class MemoryAttention(nn.Module):
    """
    현재 세그먼트를 query로, 메모리 슬롯을 key/value로 하는 Attention.

    query: sₜ (현재 세그먼트 표현)
    key/value: [CM; FM] (전체 메모리 슬롯)
    """

    def __init__(self, slot_dim: int, num_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads

        if num_heads == 1:
            # Single-head Scaled Dot-Product Attention
            self.scale = slot_dim ** 0.5
            self.q_proj = nn.Linear(slot_dim, slot_dim)
            self.k_proj = nn.Linear(slot_dim, slot_dim)
            self.v_proj = nn.Linear(slot_dim, slot_dim)
            self.out_proj = nn.Linear(slot_dim, slot_dim)
        else:
            # Multi-Head Attention (PyTorch 내장)
            self.mha = nn.MultiheadAttention(
                embed_dim=slot_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (B, slot_dim) 현재 세그먼트 표현
            memory: (B, num_slots, slot_dim) 전체 메모리 [CM; FM]

        Returns:
            (B, slot_dim) 메모리로부터 추출된 context 벡터
        """
        # query를 (B, 1, slot_dim)으로 확장
        query = query.unsqueeze(1)

        if self.num_heads == 1:
            q = self.q_proj(query)       # (B, 1, D)
            k = self.k_proj(memory)      # (B, S, D)
            v = self.v_proj(memory)      # (B, S, D)

            # Scaled Dot-Product Attention
            scores = torch.bmm(q, k.transpose(1, 2)) / self.scale  # (B, 1, S)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            context = torch.bmm(attn_weights, v)  # (B, 1, D)
            context = self.out_proj(context)
        else:
            context, _ = self.mha(query, memory, memory)  # (B, 1, D)

        return context.squeeze(1)  # (B, D)


class ClassificationHead(nn.Module):
    """
    분류 헤드: slot_dim → hidden_dim → num_labels
    multi-hot 출력을 위해 Sigmoid 활성화 (학습 시에는 BCEWithLogitsLoss 사용)
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) context 벡터

        Returns:
            (B, num_labels) logits (Sigmoid 적용 전)
        """
        return self.classifier(x)


class CompressiveMemoryClassifier(nn.Module):
    """
    Compressive Memory 기반 스트리밍 분류 모델.

    구조:
      세그먼트 → RoBERTa → Projection → Memory Update → Attention → Head

    Baseline 모드:
      fm_size=0, cm_size=0이면 메모리 없이 마지막 세그먼트만 분류한다.

    Args:
        encoder_name: HuggingFace 모델명 (예: "roberta-base")
        encoder_hidden_size: 인코더 출력 차원 (768)
        slot_dim: 메모리 슬롯 차원 (128)
        fm_size: Fine Memory 크기 (r)
        cm_size: Compressed Memory 크기 (k)
        num_labels: 출력 레이블 수 (10)
        head_hidden_dim: Classification Head 중간 차원 (64)
        num_heads: Attention 헤드 수 (1)
        dropout: 드롭아웃 비율
        conv_kernel_size: 1D Conv 커널 크기
        use_learnable_init: 학습 가능한 초기 메모리 사용 여부
        freeze_encoder: 인코더 파라미터 고정 여부
        compress_fn: 압축 함수 종류 ("conv" 또는 "mean")
    """

    def __init__(
        self,
        encoder_name: str = "roberta-base",
        encoder_hidden_size: int = 768,
        slot_dim: int = 128,
        fm_size: int = 3,
        cm_size: int = 4,
        num_labels: int = 10,
        head_hidden_dim: int = 64,
        num_heads: int = 1,
        dropout: float = 0.1,
        conv_kernel_size: int = 2,
        use_learnable_init: bool = True,
        freeze_encoder: bool = False,
        compress_fn: str = "conv",
    ):
        super().__init__()

        self.slot_dim = slot_dim
        self.fm_size = fm_size
        self.cm_size = cm_size
        self.num_labels = num_labels
        self.is_baseline = (fm_size == 0 and cm_size == 0)

        # 1. RoBERTa 인코더
        self.encoder = load_encoder_backbone(encoder_name)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # 2. Linear Projection (768 → 128)
        self.projection = LinearProjection(encoder_hidden_size, slot_dim)

        # 3. Compressive Memory
        if not self.is_baseline:
            self.memory = CompressiveMemory(
                slot_dim=slot_dim,
                fm_size=fm_size,
                cm_size=cm_size,
                conv_kernel_size=conv_kernel_size,
                use_learnable_init=use_learnable_init,
                compress_fn=compress_fn,
            )

            # 4. Attention
            self.attention = MemoryAttention(
                slot_dim=slot_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
        else:
            self.memory = None
            self.attention = None

        # 5. Classification Head
        self.head = ClassificationHead(
            input_dim=slot_dim,
            hidden_dim=head_hidden_dim,
            num_labels=num_labels,
            dropout=dropout,
        )

    def encode_segment(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        하나의 세그먼트를 인코딩하여 projected 표현을 반환한다.

        Args:
            input_ids: (B, seq_len) 세그먼트 토큰 ID
            attention_mask: (B, seq_len) 어텐션 마스크

        Returns:
            (B, slot_dim) projected 세그먼트 표현
        """
        # RoBERTa 인코딩
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Fine-tuning 안정성을 위해 새로 초기화된 pooler는 사용하지 않는다.
        # RoBERTa의 첫 토큰(<s>) hidden state를 세그먼트 표현으로 사용한다.
        pooled = encoder_output.last_hidden_state[:, 0, :]

        # Projection (768 → 128)
        projected = self.projection(pooled)
        return projected

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_mask: torch.Tensor,
        num_segments: torch.Tensor,
        **kwargs,
    ) -> dict:
        """
        씬의 모든 세그먼트를 순차 처리하여 분류 결과를 반환한다.

        Args:
            input_ids: (B, max_S, L) 세그먼트별 토큰 ID
            attention_mask: (B, max_S, L) 세그먼트별 어텐션 마스크
            segment_mask: (B, max_S) 실제 세그먼트=True, 패딩=False
            num_segments: (B,) 각 씬의 실제 세그먼트 수

        Returns:
            dict:
                - logits: (B, num_labels) 최종 분류 logits
                - all_logits: list[(B, num_labels)] 각 세그먼트의 logits (스트리밍 분석용)
        """
        B, max_S, L = input_ids.shape
        device = input_ids.device

        # Baseline 모드: 마지막 세그먼트만 분류
        if self.is_baseline:
            return self._forward_baseline(input_ids, attention_mask, segment_mask, num_segments)

        # 메모리 초기화
        mem_state = self.memory.init_memory(B, device)

        all_logits = []

        # 세그먼트 순차 처리 루프
        for t in range(max_S):
            # 현재 세그먼트가 유효한 샘플만 마스킹
            valid = segment_mask[:, t]  # (B,) True/False

            if not valid.any():
                break  # 모든 샘플에서 더 이상 세그먼트 없음

            # 유효한 세그먼트만 인코딩하여 패딩 세그먼트의 불필요한 연산을 피한다.
            seg_input_ids = input_ids[:, t, :]        # (B, L)
            seg_attention_mask = attention_mask[:, t, :]  # (B, L)
            s_t = torch.zeros(B, self.slot_dim, device=device, dtype=self.projection.projection.weight.dtype)
            s_t_valid = self.encode_segment(seg_input_ids[valid], seg_attention_mask[valid])
            s_t[valid] = s_t_valid.to(s_t.dtype)

            # 메모리 업데이트
            mem_state = self.memory.update(mem_state, s_t, valid)

            # 메모리 슬롯 가져오기 [CM; FM]
            memory_slots = self.memory.get_memory_slots(mem_state)  # (B, k+r, slot_dim)

            # Attention: 현재 세그먼트 query로 메모리에서 context 추출
            context = self.attention(s_t, memory_slots)  # (B, slot_dim)

            # 분류
            logits = self.head(context)  # (B, num_labels)
            all_logits.append(logits)

        # 각 샘플의 마지막 유효 세그먼트의 logits를 최종 결과로 사용
        final_logits = self._gather_last_logits(all_logits, num_segments)

        return {
            "logits": final_logits,   # (B, num_labels) 최종 분류 결과
            "all_logits": all_logits,  # list[(B, num_labels)] 스트리밍 분석용
        }

    def _forward_baseline(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_mask: torch.Tensor,
        num_segments: torch.Tensor,
    ) -> dict:
        """
        Baseline 모드: 메모리 없이 마지막 세그먼트만 분류한다.
        """
        B = input_ids.shape[0]
        all_logits = []

        for t in range(input_ids.shape[1]):
            valid = segment_mask[:, t]
            if not valid.any():
                break

            seg_input_ids = input_ids[:, t, :]
            seg_attention_mask = attention_mask[:, t, :]
            s_t = torch.zeros(
                B,
                self.slot_dim,
                device=input_ids.device,
                dtype=self.projection.projection.weight.dtype,
            )
            s_t_valid = self.encode_segment(seg_input_ids[valid], seg_attention_mask[valid])
            s_t[valid] = s_t_valid.to(s_t.dtype)
            logits = self.head(s_t)
            all_logits.append(logits)

        final_logits = self._gather_last_logits(all_logits, num_segments)

        return {
            "logits": final_logits,
            "all_logits": all_logits,
        }

    def _gather_last_logits(
        self,
        all_logits: list[torch.Tensor],
        num_segments: torch.Tensor,
    ) -> torch.Tensor:
        """
        각 샘플의 마지막 유효 세그먼트에 해당하는 logits를 모은다.

        Args:
            all_logits: 각 타임스텝의 logits 리스트
            num_segments: (B,) 각 샘플의 세그먼트 수

        Returns:
            (B, num_labels) 각 샘플의 최종 logits
        """
        B = num_segments.shape[0]
        device = all_logits[0].device
        final = torch.zeros(B, self.num_labels, device=device)

        for b in range(B):
            last_t = num_segments[b].item() - 1  # 마지막 유효 세그먼트 인덱스
            last_t = min(last_t, len(all_logits) - 1)
            final[b] = all_logits[last_t][b]

        return final


def build_compressive_memory_model(config: dict) -> CompressiveMemoryClassifier:
    """
    config dict로부터 모델을 생성하는 팩토리 함수.

    Args:
        config: 설정 딕셔너리 (config.py의 각 설정 참조)

    Returns:
        CompressiveMemoryClassifier 인스턴스
    """
    from config import (
        ENCODER_CONFIG, MEMORY_CONFIG, ATTENTION_CONFIG,
        HEAD_CONFIG, MELD_CONFIG,
    )

    # Ablation 설정 오버라이드
    fm_size = config.get("FM_SIZE", MEMORY_CONFIG["FM_SIZE"])
    cm_size = config.get("CM_SIZE", MEMORY_CONFIG["CM_SIZE"])
    compress_fn = config.get("COMPRESS_FN", "conv")

    model = CompressiveMemoryClassifier(
        encoder_name=ENCODER_CONFIG["MODEL_NAME"],
        encoder_hidden_size=ENCODER_CONFIG["HIDDEN_SIZE"],
        slot_dim=MEMORY_CONFIG["SLOT_DIM"],
        fm_size=fm_size,
        cm_size=cm_size,
        num_labels=MELD_CONFIG["NUM_LABELS"],
        head_hidden_dim=HEAD_CONFIG["HIDDEN_DIM"],
        num_heads=ATTENTION_CONFIG["NUM_HEADS"],
        dropout=ATTENTION_CONFIG["DROPOUT"],
        conv_kernel_size=MEMORY_CONFIG["CONV_KERNEL_SIZE"],
        use_learnable_init=MEMORY_CONFIG["USE_LEARNABLE_INIT"],
        freeze_encoder=ENCODER_CONFIG["FREEZE_ENCODER"],
        compress_fn=compress_fn,
    )

    return model

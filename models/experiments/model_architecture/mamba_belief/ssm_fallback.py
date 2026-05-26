"""
Pure PyTorch Mamba SSM 구현 (Windows/CPU 환경 fallback)

mamba-ssm 패키지는 CUDA 확장을 요구하여 Windows에서 직접 설치가 불가능하다.
이 모듈은 동일한 인터페이스를 가진 순수 PyTorch 구현을 제공하여
WSL2/Docker 없이 로컬 디버깅을 가능하게 한다.

인터페이스:
  MambaPyTorch(d_model, d_state, d_conv, expand)
  forward(x: Tensor[B, L, d_model]) -> Tensor[B, L, d_model]

참고: Mamba - Linear-Time Sequence Modeling with Selective State Spaces
      Gu & Dao, 2023 (https://arxiv.org/abs/2312.00752)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaPyTorch(nn.Module):
    """
    Pure PyTorch Mamba SSM.

    핵심 수식:
      h_t = Ā_t · h_{t-1} + B̄_t · u_t     (selective state update)
      y_t = C_t · h_t                        (output)

    선택성(selectivity): Δ, B, C 가 입력 u_t 에 의존하여 동적으로 결정됨
      Ā_t = exp(Δ_t ⊗ A)    (ZOH discretization)
      B̄_t = Δ_t · B_t

    CUDA 가속 버전(mamba-ssm)과 동일한 forward 인터페이스를 사용하므로
    model.py에서 try/except로 투명하게 교체 가능하다.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)

        # 입력을 SSM 처리용(x)과 게이트(z)로 분리
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 인과적(causal) depthwise Conv1d — 로컬 컨텍스트 집계
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,  # 왼쪽 패딩으로 미래 토큰 차단
            groups=self.d_inner,
            bias=True,
        )

        # SSM 파라미터 투영: u → (Δ_raw, B, C)
        self.x_proj = nn.Linear(
            self.d_inner,
            self.d_inner + self.d_state + self.d_state,  # Δ + B + C
            bias=False,
        )

        # Δ 투영 (low-rank → full rank)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # A 행렬: (d_inner, d_state), log 스케일로 저장 (수치 안정성)
        # HiPPO 초기화: A[i,n] = n+1 (n=0..d_state-1)
        A = torch.arange(1, d_state + 1, dtype=torch.float)
        A = A.unsqueeze(0).expand(self.d_inner, -1).clone()
        self.A_log = nn.Parameter(torch.log(A))  # (d_inner, d_state)

        # D: skip connection 가중치
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 출력 투영
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)

        Returns:
            (B, L, d_model)
        """
        B, L, _ = x.shape

        # 1. 입력 분리: SSM 처리용 x_ssm + 게이트 z
        xz = self.in_proj(x)                                  # (B, L, d_inner*2)
        x_ssm, z = xz.chunk(2, dim=-1)                       # 각 (B, L, d_inner)

        # 2. Causal Conv1d (미래 토큰 차단을 위해 우측 패딩 제거)
        x_conv = self.conv1d(x_ssm.transpose(1, 2))          # (B, d_inner, L + d_conv-1)
        x_conv = x_conv[:, :, :L].transpose(1, 2)            # (B, L, d_inner)
        x_act = F.silu(x_conv)                               # (B, L, d_inner)

        # 3. SSM 파라미터 투영 (입력 의존적 = selective)
        dBC = self.x_proj(x_act)                              # (B, L, d_inner+d_state+d_state)
        delta_raw, B_ssm, C_ssm = dBC.split(
            [self.d_inner, self.d_state, self.d_state], dim=-1
        )

        # Δ: softplus로 양수 보장 (시간 스텝 크기)
        delta = F.softplus(self.dt_proj(delta_raw))           # (B, L, d_inner)

        # 4. A 복원 (음수 유지 → 안정적 상태 감쇠)
        A = -torch.exp(self.A_log.float())                    # (d_inner, d_state)

        # 5. Selective scan
        y = self._selective_scan(x_act, delta, A, B_ssm, C_ssm)  # (B, L, d_inner)

        # 6. Skip connection + Gating
        y = y + x_act * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)

        # 7. 출력 투영
        return self.out_proj(y)                               # (B, L, d_model)

    def _selective_scan(
        self,
        u: torch.Tensor,      # (B, L, d_inner)
        delta: torch.Tensor,  # (B, L, d_inner)
        A: torch.Tensor,      # (d_inner, d_state)
        B: torch.Tensor,      # (B, L, d_state)
        C: torch.Tensor,      # (B, L, d_state)
    ) -> torch.Tensor:
        """
        순차 selective scan.
          Ā_t = exp(Δ_t ⊗ A)     ZOH discretization
          B̄_t = Δ_t · B_t
          h_t = Ā_t * h_{t-1} + B̄_t * u_t
          y_t = C_t · h_t
        """
        B_batch, L, d_inner = u.shape

        # Discretize A: (B, L, d_inner, d_state)
        dA = torch.exp(
            delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )

        # Discretize B: (B, L, d_inner, d_state)
        dB = delta.unsqueeze(-1) * B.unsqueeze(2)

        # 초기 hidden state: 0으로 시작
        h = torch.zeros(B_batch, d_inner, self.d_state, device=u.device, dtype=u.dtype)
        ys = []

        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * u[:, t].unsqueeze(-1)
            # y_t = C_t · h_t: (B, d_inner)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
            ys.append(y_t)

        return torch.stack(ys, dim=1)  # (B, L, d_inner)

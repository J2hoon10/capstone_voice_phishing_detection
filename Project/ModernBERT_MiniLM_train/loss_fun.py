import torch
import torch.nn as nn
import torch.nn.functional as F

class LossFunction:
    def __init__(self):
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.mse_loss = nn.MSELoss()

    def hard_loss(self, logits, labels):
        return self.ce_loss(logits, labels)

    def minilm_loss(self, s_outputs, t_outputs, labels, alpha=0.5):
        """
        [MiniLM Strategy]
        1. Hard Loss (CrossEntropy)
        2. Attention Distribution Transfer (KL Divergence)
        3. (Optional) Value-Relation Transfer (여기서는 Attention Map Transfer로 대체)
        """
        # 1. Hard Loss
        loss_hard = self.ce_loss(s_outputs.logits, labels)
        
        # 2. Attention Transfer Loss
        # t_outputs.attentions: Tuple of [Batch, Num_Heads, Seq, Seq]
        # Student와 Teacher의 레이어 수가 다를 경우, 마지막 레이어(또는 매핑된 레이어)끼리 비교
        
        # 예: 마지막 레이어의 Attention Map 비교
        s_attn = s_outputs.attentions[-1] # [B, S_Heads, Seq, Seq]
        t_attn = t_outputs.attentions[-1] # [B, T_Heads, Seq, Seq]
        
        # 헤드 수가 다르므로(12 vs 6), 평균을 내거나 SVD 초기화 때 선택된 헤드 인덱스를 사용해야 함.
        # MiniLM 원문은 헤드 간의 관계(Relation)를 보지만, 구현의 편의를 위해
        # 여기서는 각 모델의 Attention Map을 Head 차원에서 평균내어 [B, Seq, Seq] 로 만든 뒤 비교하거나
        # Student Head 개수에 맞춰 Teacher Head를 선택/평균 해야 함.
        
        # 가장 간단한 방법: Head 차원 평균 (Global Attention Pattern Matching)
        s_attn_mean = s_attn.mean(dim=1) # [B, Seq, Seq]
        t_attn_mean = t_attn.mean(dim=1) # [B, Seq, Seq]
        
        # Log Softmax vs Softmax for KL Div
        # Attention 값은 이미 Softmax가 적용되어 있으므로 Log만 취함 (0 방지 위해 clamp)
        s_log_prob = torch.log(s_attn_mean + 1e-9)
        t_prob = t_attn_mean
        
        loss_attn = self.kl_loss(s_log_prob, t_prob)
        
        return alpha * loss_hard + (1 - alpha) * loss_attn
# loss_fun.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistilBERTLoss(nn.Module):
    """
    DistilBERT 논문의 Triple Loss 구현 + [추가] Weighted Loss 지원
    Loss = alpha * L_ce + beta * L_kd + gamma * L_cos
    """
    def __init__(self, alpha=5.0, beta=2.0, gamma=1.0, temperature=2.0, class_weights=None):
        """
        Args:
            class_weights (Tensor, optional): 클래스 불균형 해소용 가중치 (예: [1.0, 9.0])
        """
        super(DistilBERTLoss, self).__init__()
        
        self.alpha = alpha       # Student Loss (Hard Label) 가중치
        self.beta = beta         # Distillation Loss (Soft Label) 가중치
        self.gamma = gamma       # Cosine Embedding Loss (Hidden State) 가중치
        self.temperature = temperature 

        # [수정] 클래스 가중치가 있으면 CrossEntropyLoss에 적용
        if class_weights is not None:
            self.ce_loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce_loss_fct = nn.CrossEntropyLoss()
            
        self.kldiv_loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.cosine_loss_fct = nn.CosineEmbeddingLoss(margin=0.0)

    def forward(self, student_outputs, teacher_outputs, labels, attention_mask):
        """
        Args:
            student_outputs (dict): {'logits': ..., 'hidden_states': ...}
            teacher_outputs (dict): {'logits': ..., 'hidden_states': ...}
            labels (tensor): 정답 레이블
        """
        # 1. Student Loss (L_ce) [Weighted 적용됨]
        student_logits = student_outputs['logits']
        student_loss = self.ce_loss_fct(student_logits, labels)

        # 2. Distillation Loss (L_kd)
        teacher_logits = teacher_outputs['logits']
        p_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        log_p_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # KL Divergence (T^2 scaling)
        distillation_loss = self.kldiv_loss_fct(log_p_student, p_teacher) * (self.temperature ** 2)

        # 3. Cosine Loss (L_cos)
        s_hidden = student_outputs['hidden_states'] # (Embed + 6 Layers)
        t_hidden = teacher_outputs['hidden_states'] # (Embed + 12 Layers)
        
        total_cosine_loss = 0.0
        n_layers = len(s_hidden) - 1
        layer_ratio = (len(t_hidden) - 1) / n_layers 

        for i in range(1, len(s_hidden)):
            t_idx = int(i * layer_ratio)
            
            s_vec = s_hidden[i].view(-1, s_hidden[i].size(-1))
            t_vec = t_hidden[t_idx].view(-1, t_hidden[t_idx].size(-1))
            
            target = torch.ones(s_vec.size(0)).to(s_vec.device)
            loss_layer = self.cosine_loss_fct(s_vec, t_vec, target)
            total_cosine_loss += loss_layer

        cosine_loss = total_cosine_loss / n_layers

        # Final Total Loss
        total_loss = (self.alpha * student_loss) + \
                     (self.beta * distillation_loss) + \
                     (self.gamma * cosine_loss)

        return total_loss, {
            "loss": total_loss.item(),
            "ce_loss": student_loss.item(),
            "kd_loss": distillation_loss.item(),
            "cos_loss": cosine_loss.item()
        }
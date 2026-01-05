import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class RiskLevelTaskHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(RiskLevelTaskHead, self).__init__()
        
        # 구조: [Linear] -> [Tanh/GELU] -> [Dropout] -> [Linear(Output)]
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim), # 768 -> 768 (차원 유지하며 변환)
            nn.GELU(),                       # 활성화 함수
            nn.Dropout(0.1),                 # 과적합 방지 (보통 0.1 사용)
            nn.Linear(input_dim, num_classes) # 768 -> 2
        )

    def forward(self, x):
        return self.classifier(x)

class DistillableRoBERTaModel(nn.Module):
    """
    지식 증류(Knowledge Distillation)를 지원하는 RoBERTa 모델
    - 일반적인 학습/추론: Logits 반환
    - KD 학습 시: Logits + 모든 레이어의 Hidden States 반환
    """
    def __init__(self, model_name='klue/roberta-base', num_classes=2, is_student=False, student_layer_num=None):
        super(DistillableRoBERTaModel, self).__init__()
        
        # 1. Config 설정 (Hidden State 출력을 켜야 함)
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.output_hidden_states = True # [핵심] 중간 레이어 벡터 반환 설정
        
        # [옵션] 학생 모델일 경우 레이어 수를 줄여서 초기화 (DistilBERT 방식)
        if is_student and student_layer_num:
            self.config.num_hidden_layers = student_layer_num
            
        # 2. Backbone 로드
        # 학생 모델이라면 줄어든 config로 초기화, 아니면 pretrained 가중치 로드
        if is_student and student_layer_num:
             # 주의: 실제로는 Pretrained weights를 로드한 뒤 앞쪽 레이어만 잘라내는 방식 등을 씀
             # 여기서는 구조적 예시를 위해 Config기반 초기화로 작성
            self.backbone = AutoModel.from_config(self.config)
        else:
            self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        # 3. Task Head 초기화
        self.head = RiskLevelTaskHead(input_dim=self.config.hidden_size, num_classes=num_classes)

    def forward(self, input_ids, attention_mask):
        # 1. Backbone 통과
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # outputs.last_hidden_state: (Batch, Seq, Dim) -> 마지막 층
        # outputs.pooler_output: (Batch, Dim) -> [CLS] 토큰 + Tanh (RoBERTa는 없을 수도 있음)
        # outputs.hidden_states: (Layer+1, Batch, Seq, Dim) -> 튜플 형태의 모든 층 벡터
        
        # 2. Embedding 추출 (RoBERTa는 pooler가 없거나 안 쓰는 경우가 많아 [CLS] 직접 추출 추천)
        # last_hidden_state의 0번째 토큰([CLS]) 사용
        cls_token_vector = outputs.last_hidden_state[:, 0, :]

        # 3. Logit 계산
        logits = self.head(cls_token_vector)

        # 4. 결과 반환 (딕셔너리 형태)
        return {
            'logits': logits,                  # Soft-target Loss 계산용
            'hidden_states': outputs.hidden_states, # Feature-based Loss 계산용 (Tuple)
            'last_feature': cls_token_vector   # 필요 시 사용
        }
    
def initialize_student_weights(teacher_model, student_model):
    """
    [수정됨] Teacher의 레이어 수와 Student의 레이어 수 비율에 맞춰
    자동으로 등간격의 레이어를 선택하여 초기화하는 함수.
    """
    print(f"Initialize Student weights from Teacher (General Mode)...")
    
    t_backbone = teacher_model.backbone
    s_backbone = student_model.backbone

    # 1. Embeddings 복사 (필수: 무조건 동일해야 함)
    # ---------------------------------------------------------
    s_backbone.embeddings.load_state_dict(t_backbone.embeddings.state_dict())
    print(" - [Completed] Embeddings copied.")

    # 2. Encoder Layer 복사 (핵심: Dynamic Layer Mapping)
    # ---------------------------------------------------------
    teacher_layers = t_backbone.encoder.layer
    student_layers = s_backbone.encoder.layer
    
    n_teacher = len(teacher_layers)
    n_student = len(student_layers)
    
    if n_student > n_teacher:
        raise ValueError(f"Student layers ({n_student}) cannot be more than Teacher layers ({n_teacher}).")

    # 비율 계산 (예: 12 / 6 = 2.0, 12 / 4 = 3.0)
    layer_ratio = n_teacher / n_student

    print(f" - [Info] Layer Mapping Strategy (Ratio: {layer_ratio:.2f})")
    
    for i in range(n_student):
        # 학생의 i번째 레이어는 선생님의 몇 번째 레이어에 해당하는가?
        # int()를 사용하여 가장 가까운 앞쪽 인덱스를 선택 (0, 2.4->2, 4.8->4 ...)
        teacher_idx = int(i * layer_ratio)
        
        # 가중치 복사
        source_weights = teacher_layers[teacher_idx].state_dict()
        student_layers[i].load_state_dict(source_weights)
        
        print(f"   -> Student Layer {i}  <==  Teacher Layer {teacher_idx}")

    # 3. Task Head 초기화 (선택)
    # ---------------------------------------------------------
    # Head 구조가 같다면 복사 (Hidden dim이 768로 같다면 가능)
    try:
        if teacher_model.head.classifier[0].in_features == student_model.head.classifier[0].in_features:
            student_model.head.load_state_dict(teacher_model.head.state_dict())
            print(" - [Optional] Classifier Head copied.")
        else:
            print(" - [Skip] Classifier Head skipped (Dimension mismatch).")
    except Exception as e:
        print(" - [Skip] Classifier Head skipped (Structure mismatch).")

    print("Student initialization finished.\n")
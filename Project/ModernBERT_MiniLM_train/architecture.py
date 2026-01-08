import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig

class ModernBertTeacher(nn.Module):
    def __init__(self, model_name, num_labels):
        super(ModernBertTeacher, self).__init__()
        print(f"[Arch] Initializing Teacher: {model_name}")
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=self.config, trust_remote_code=True
        )

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels,
            output_attentions=True, output_hidden_states=True
        )
    
    # [추가] Teacher도 안전하게 사용하기 위해 추가
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

class ModernBertStudent(nn.Module):
    """
    Teacher Assistant(TA) 및 Student 공용 클래스
    """
    def __init__(self, teacher_model_name, target_config, num_labels):
        super(ModernBertStudent, self).__init__()
        
        print(f"[Arch] Initializing Student/TA with config: {target_config}")
        
        # 1. Teacher Config 로드
        base_config = AutoConfig.from_pretrained(teacher_model_name, num_labels=num_labels)
        
        # 2. Config 수정 (Hidden Size, Layers 등 다운사이징)
        for k, v in target_config.items():
            setattr(base_config, k, v)
            
        self.config = base_config
        
        # 3. 수정된 Config로 모델 생성
        self.model = AutoModelForSequenceClassification.from_config(self.config)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels,
            output_attentions=True, output_hidden_states=True
        )

    # ------------------------------------------------------------------------
    # [수정된 부분] 내부 모델(self.model)의 임베딩 함수를 외부로 연결
    # ------------------------------------------------------------------------
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)
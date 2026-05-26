import torch
import torch.nn as nn
from transformers import ModernBertModel, ModernBertPreTrainedModel, ModernBertConfig, AutoModelForSequenceClassification

def disable_hf_padding_warning(model):
    """Transformers 라이브러리의 패딩 검사 함수 무력화"""
    for module in model.modules():
        if hasattr(module, 'warn_if_padding_and_no_attention_mask'):
            module.warn_if_padding_and_no_attention_mask = lambda *args, **kwargs: None

class ModernBertForDistillation(ModernBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = ModernBertModel(config)
        disable_hf_padding_warning(self)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # [수정 1] 중복 인자 제거 (ModernBertModel은 labels를 받지 않음)
        kwargs.pop('labels', None)

        # [수정 2] 중복 인자 제거 ('output_hidden_states' 충돌 방지)
        kwargs.pop('output_hidden_states', None)

        # [수정 3] 중복 인자 제거 ('return_dict' 충돌 방지 - 여기가 에러 원인!)
        kwargs.pop('return_dict', None)
            
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True, # 명시적 지정
            return_dict=True,          # 명시적 지정 (kwargs에 또 있으면 충돌함)
            **kwargs
        )
        
        last_hidden_state = outputs.last_hidden_state
        
        if 'pooler_output' in outputs and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            if last_hidden_state.dim() == 2: 
                pooled_output = last_hidden_state 
            else:
                pooled_output = last_hidden_state[:, 0, :]

        logits = self.classifier(pooled_output)

        return {
            "logits": logits,
            "hidden_states": outputs.hidden_states,
            "last_hidden_state": last_hidden_state
        }

class DistillationWrapper(nn.Module):
    def __init__(self, teacher_model, student_model, config_distill, teacher_layer_map=None):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        self.teacher_layer_map = teacher_layer_map
        s_layers = self.student.config.num_hidden_layers
        t_layers = self.teacher.config.num_hidden_layers
        
        if self.teacher_layer_map is not None:
            if len(self.teacher_layer_map) != s_layers: pass
        elif s_layers != t_layers:
            raise ValueError("Layers mismatch")

        self.use_projection = config_distill.get('USE_PROJECTION', True)
        t_hidden = self.teacher.config.hidden_size
        s_hidden = self.student.config.hidden_size
        
        self.project_layer = None
        if self.use_projection and (t_hidden != s_hidden):
            self.project_layer = nn.Linear(s_hidden, t_hidden)
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Teacher: labels는 필요 없으므로 제거
        t_kwargs = kwargs.copy()
        t_kwargs.pop('labels', None)
        
        # [수정 4] Wrapper에서도 호출 시 중복 인자 사전 제거
        t_kwargs.pop('output_hidden_states', None)
        t_kwargs.pop('return_dict', None)  # <--- 여기도 추가하는 것이 안전함

        with torch.no_grad():
            t_out = self.teacher(
                input_ids, 
                attention_mask, 
                output_hidden_states=True, 
                return_dict=True, 
                **t_kwargs
            )
        
        # Student 호출
        s_out = self.student(input_ids, attention_mask, **kwargs)
        
        t_hidden = t_out['hidden_states']
        s_hidden = s_out['hidden_states']
        
        if self.teacher_layer_map:
            aligned_t = [t_hidden[i] for i in self.teacher_layer_map]
        else:
            aligned_t = t_hidden
            
        if self.project_layer:
            projected_s = [self.project_layer(h) for h in s_hidden]
        else:
            projected_s = s_hidden

        return {
            "t_hidden_states": aligned_t, "t_logits": t_out['logits'],
            "s_hidden_states": projected_s, "s_logits": s_out['logits']
        }

def build_model(mode, config, tokenizer=None, checkpoint_path=None, use_unpadding=False):
    """
    [모델 생성 팩토리]
    """
    print(f"[Architecture] Building model for mode: {mode} (Unpadding: {use_unpadding})")
    
    attn_impl = "flash_attention_2" if use_unpadding else "sdpa"
    
    if mode == 'teacher':
        model = AutoModelForSequenceClassification.from_pretrained(
            config['BASE_MODEL_NAME'],
            num_labels=config['NUM_LABELS'],
            trust_remote_code=True,
            attn_implementation=attn_impl
        )
        disable_hf_padding_warning(model)
        return model

    # Vocab Size 안전하게 설정
    if tokenizer:
        vocab_size = len(tokenizer)
        pad_token_id = tokenizer.pad_token_id
    else:
        vocab_size = config.get('VOCAB_SIZE_HINT', 50368)
        pad_token_id = config.get('PAD_TOKEN_ID', 50283)

    if pad_token_id is not None and pad_token_id >= vocab_size:
        vocab_size = pad_token_id + 1
        vocab_size = ((vocab_size + 63) // 64) * 64

    if mode == 'ta': target_config = config['TA_CONFIG']
    elif mode == 'student': target_config = config['STUDENT_CONFIG']
    else: raise ValueError(f"Unknown mode: {mode}")

    custom_config = ModernBertConfig(
        vocab_size=vocab_size,
        max_position_embeddings=config['MAX_LENGTH'],
        num_labels=config['NUM_LABELS'],
        hidden_size=target_config['hidden_size'],
        num_hidden_layers=target_config['num_hidden_layers'],
        num_attention_heads=target_config['num_attention_heads'],
        intermediate_size=target_config['intermediate_size'],
        pad_token_id=pad_token_id,
        attn_implementation=attn_impl
    )

    model = ModernBertForDistillation(custom_config)
    
    if checkpoint_path:
        print(f"[Architecture] Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        
    return model
# utils.py
import os
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import GroupShuffleSplit

def set_seed(seed):
    """재현성을 위한 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_data(config):
    """
    ID(통화 세션) 기준으로 학습/검증 데이터 분할 (Data Leakage 방지)
    """
    if not os.path.exists(config['PROCESSED_DATA_DIR']):
        os.makedirs(config['PROCESSED_DATA_DIR'])

    print(f"\n[Data Prep] Reading {config['MASTER_DATA_PATH']}...")
    try:
        df = pd.read_csv(config['MASTER_DATA_PATH'])
    except FileNotFoundError:
        # 파일이 없을 경우 빈 파일 생성 (테스트용) 또는 에러 발생
        raise FileNotFoundError(f"데이터 파일이 없습니다: {config['MASTER_DATA_PATH']}")

    # ID 컬럼 확인
    if 'ID' not in df.columns:
        print("[Warning] 'ID' 컬럼이 없어 인덱스를 기준으로 분할합니다.")
        df['ID'] = df.index

    # GroupShuffleSplit을 사용하여 ID 단위로 분할
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=config['SEED'])
    train_idx, val_idx = next(splitter.split(df, groups=df['ID']))
    
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    train_path = os.path.join(config['PROCESSED_DATA_DIR'], 'train_split.csv')
    val_path = os.path.join(config['PROCESSED_DATA_DIR'], 'val_split.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"   -> Train Samples: {len(train_df)}")
    print(f"   -> Val Samples:   {len(val_df)}")
    
    return train_path, val_path

def get_logger(output_dir):
    """로그 파일과 콘솔에 동시에 출력하는 로거 생성"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    logger = logging.getLogger('ModernBERT_Trainer')
    logger.setLevel(logging.INFO)
    logger.propagate = False # 중복 출력 방지

    # 포맷 설정
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # 파일 핸들러 (utf-8 인코딩)
    file_handler = logging.FileHandler(os.path.join(output_dir, 'training.log'), encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 콘솔 핸들러
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    """학습 중단 시 재개를 위한 체크포인트 저장"""
    # DataParallel로 감싸져 있을 경우 model.module을 저장
    if hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
        
    state = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }
    torch.save(state, path)

def calculate_class_weights(dataset):
    """클래스 불균형 해소를 위한 가중치 계산 (Clipping 적용)"""
    print("[Info] Calculating class weights...")
    # Dataset 샘플에서 라벨 추출
    labels = [sample['label'] for sample in dataset.samples]
    
    labels_np = np.array(labels)
    classes, counts = np.unique(labels_np, return_counts=True)
    
    total_samples = len(labels_np)
    n_classes = len(classes)
    
    # 가중치 계산 (Inverse Frequency)
    weights = total_samples / (n_classes * counts)
    
    # 가중치 제한 (너무 큰 가중치 방지)
    weights = np.clip(weights, a_min=1.0, a_max=5.0)
    
    # 텐서 변환 (CPU) -> 학습 시 device로 이동
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    
    print(f"   -> Class Counts: {dict(zip(classes, counts))}")
    print(f"   -> Calculated Weights: {weights_tensor.numpy()}")
    
    return weights_tensor

def get_projection_matrix(teacher_embeddings, student_hidden_size):
    """Teacher Embedding을 SVD하여 투영 행렬(V_proj) 생성"""
    print(f"[Init] Calculating SVD on Teacher Embeddings {teacher_embeddings.shape}...")
    
    # float32로 변환 후 연산 (안전성 확보)
    emb_float = teacher_embeddings.float()
    u, s, vh = torch.linalg.svd(emb_float, full_matrices=False)
    
    # Vh는 (N, N) 이므로 상위 k개 행을 가져와 전치 -> (768, 384)
    v_proj = vh[:student_hidden_size, :].t()
    return v_proj

def initialize_student_hybrid(teacher_model, student_model, v_proj):
    """
    [ModernBERT 맞춤형 초기화]
    Teacher(Raw AutoModel)와 Student(Wrapper)의 구조 차이를 자동으로 처리하여 초기화 수행
    """
    print("[Init] Applying Hybrid Initialization to Student...")
    device = v_proj.device

    # --- 1. Embedding Layer 초기화 ---
    # Wrapper든 Raw Model이든 get_input_embeddings() 메서드는 존재함 (architecture.py 수정 덕분)
    t_emb = teacher_model.get_input_embeddings().weight.data
    s_emb_layer = student_model.get_input_embeddings()
    
    # [Vocab, 768] @ [768, 384] -> [Vocab, 384]
    s_emb_layer.weight.data.copy_(torch.matmul(t_emb, v_proj))
    print("   -> Embeddings initialized.")

    # --- [핵심 수정] 레이어 접근 헬퍼 함수 ---
    def get_layers_safely(model_obj):
        # 1. Wrapper Class인 경우 내부 모델 꺼내기
        if hasattr(model_obj, 'model') and isinstance(model_obj.model, nn.Module):
            model_obj = model_obj.model
            
        # 2. Base Model Prefix 확인 (AutoModel 구조 대응)
        if hasattr(model_obj, 'base_model_prefix'):
            base_model = getattr(model_obj, model_obj.base_model_prefix, model_obj)
        else:
            base_model = model_obj

        # 3. 레이어 찾기 (ModernBERT vs BERT 구조 차이 대응)
        if hasattr(base_model, 'layers'): 
            return base_model.layers # ModernBERT
        elif hasattr(base_model, 'encoder') and hasattr(base_model.encoder, 'layer'):
            return base_model.encoder.layer # BERT/RoBERTa
        elif hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
            return base_model.model.layers # Nested 구조 대비
            
        raise AttributeError(f"레이어를 찾을 수 없습니다. 구조: {type(model_obj)}")

    # 레이어 추출 (Teacher: Raw, Student: Wrapper)
    t_layers = get_layers_safely(teacher_model)
    s_layers = get_layers_safely(student_model)
    
    # Config 정보 추출
    num_t_heads = teacher_model.config.num_attention_heads      # 12
    num_s_heads = student_model.config.num_attention_heads      # 6
    t_head_dim = teacher_model.config.hidden_size // num_t_heads # 64
    
    # Layer Mapping (Skip Init: 0, 2, 4...)
    layer_step = len(t_layers) // len(s_layers)

    for s_idx, s_layer in enumerate(s_layers):
        t_idx = s_idx * layer_step
        t_layer = t_layers[t_idx]
        
        # ---------------------------------------------------------
        # A. Attention: Wqkv (Fused Q,K,V)
        # ---------------------------------------------------------
        t_wqkv = t_layer.attn.Wqkv.weight.data 
        
        # Reshape & Select Heads
        t_wqkv_view = t_wqkv.view(3, num_t_heads, t_head_dim, -1)
        head_importance = t_wqkv_view.abs().sum(dim=(0, 2, 3))
        
        _, top_heads = torch.topk(head_importance, k=num_s_heads)
        top_heads, _ = torch.sort(top_heads) 
        
        # Project
        s_wqkv_selected = t_wqkv_view.index_select(1, top_heads.to(device))
        s_wqkv_proj = torch.matmul(s_wqkv_selected, v_proj)
        s_layer.attn.Wqkv.weight.data.copy_(s_wqkv_proj.reshape(-1, v_proj.shape[1]))

        # ---------------------------------------------------------
        # B. Attention: Wo (Output Projection)
        # ---------------------------------------------------------
        t_wo = t_layer.attn.Wo.weight.data 
        t_wo_view = t_wo.view(t_wo.shape[0], num_t_heads, t_head_dim)
        s_wo_cols = t_wo_view.index_select(1, top_heads.to(device))
        s_wo_cols = s_wo_cols.reshape(t_wo.shape[0], -1) 
        
        s_wo_final = torch.matmul(v_proj.t(), s_wo_cols)
        s_layer.attn.Wo.weight.data.copy_(s_wo_final)

        # ---------------------------------------------------------
        # C. MLP: Wi (Input)
        # ---------------------------------------------------------
        t_wi = t_layer.mlp.Wi.weight.data
        s_wi_shape = s_layer.mlp.Wi.weight.shape 
        t_wi_proj = torch.matmul(t_wi, v_proj)
        
        # Slicing (Pruning)
        if t_wi_proj.shape[0] >= s_wi_shape[0]:
             s_layer.mlp.Wi.weight.data.copy_(t_wi_proj[:s_wi_shape[0], :])
        else:
            repeat_times = (s_wi_shape[0] // t_wi_proj.shape[0]) + 1
            temp = t_wi_proj.repeat(repeat_times, 1)
            s_layer.mlp.Wi.weight.data.copy_(temp[:s_wi_shape[0], :])

        # ---------------------------------------------------------
        # D. MLP: Wo (Output)
        # ---------------------------------------------------------
        t_mlp_wo = t_layer.mlp.Wo.weight.data 
        s_mlp_wo_shape = s_layer.mlp.Wo.weight.shape 
        
        t_mlp_wo_proj = torch.matmul(v_proj.t(), t_mlp_wo)
        
        current_in_dim = t_mlp_wo_proj.shape[1]
        target_in_dim = s_mlp_wo_shape[1]
        
        if current_in_dim >= target_in_dim:
            s_layer.mlp.Wo.weight.data.copy_(t_mlp_wo_proj[:, :target_in_dim])
        
    print(f"   -> {len(s_layers)} Layers initialized with Attention Head Selection & SVD.")
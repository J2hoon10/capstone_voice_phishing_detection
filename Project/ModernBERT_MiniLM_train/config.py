import torch

CONFIG = {
    # [기본 환경 설정]
    'PROJECT_NAME': "ModernBERT_VoicePhishing",
    'SEED': 42,
    'DEVICE': "cuda" if torch.cuda.is_available() else "cpu",
    'NUM_WORKERS': 0,
    
    # [경로 설정]
    'MASTER_DATA_PATH': './dataset_master.csv',
    'PROCESSED_DATA_DIR': './processed_data',
    'OUTPUT_DIR_TEACHER': './ModernBERT_Result/Teacher',
    'OUTPUT_DIR_TA': './ModernBERT_Result/TA',       # TA 모델 저장 경로
    'OUTPUT_DIR_STUDENT': './ModernBERT_Result/Student', # Student 모델 저장 경로
    
    # [모델 설정 - Teacher]
    'MODEL_NAME': "neavo/modern_bert_multilingual",
    'NUM_LABELS': 2,
    'MAX_LENGTH': 1024,
    
    # [모델 설정 - TA (Teacher Assistant)]
    # Teacher와 Layer 수는 같지만(12), Hidden Size는 절반(384)
    'TA_CONFIG': {
        'hidden_size': 384,
        'num_hidden_layers': 12, # Teacher와 동일
        'num_attention_heads': 6, # 768->12헤드 였으므로 384->6헤드로 조정
        'intermediate_size': 1536 # FFN 크기도 줄임 (보통 hidden * 4)
    },

    # [모델 설정 - Student]
    # TA보다 Layer 수 절반(6), Hidden Size 유지(384)
    'STUDENT_CONFIG': {
        'hidden_size': 384,
        'num_hidden_layers': 6,  # TA의 절반
        'num_attention_heads': 6,
        'intermediate_size': 1536
    },
    
    # [데이터 처리 설정]
    'WINDOW_SIZE': 15,
    'STRIDE': 5,
    'AUG_PROB': 0.5,
    
    # [학습 파라미터]
    'EPOCHS': 5,
    'BATCH_SIZE': 8,
    'GRAD_ACCUM_STEPS': 4,
    'LEARNING_RATE': 5e-5, # Student는 조금 더 높은 LR 사용 권장
    'WEIGHT_DECAY': 0.01,
    'WARMUP_RATIO': 0.1,
    'USE_AMP': True
}
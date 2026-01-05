# config.py

# =============================================================================
# [설정] 프로젝트 하이퍼파라미터 및 경로 설정
# =============================================================================

CONFIG = {
    # 1. 기본 환경 설정
    'SEED': 42,
    'DEVICE': 'cuda',          # 'cuda' or 'cpu' (train.py에서 자동 감지하도록 짤 수도 있음)
    'NUM_WORKERS': 0,          # 데이터 로더 워커 수 (Windows는 0, Linux는 4 추천)
    
    # 2. 경로 설정 (Path)
    'MASTER_DATA_PATH': './dataset_master.csv',  # 전처리된 원본 데이터
    'PROCESSED_DATA_DIR': './processed_data',    # 학습/테스트 분할 파일 저장소
    'OUTPUT_DIR': './checkpoints',               # 학습된 모델 저장소
    
    # 3. 모델 설정 (Model)
    'MODEL_NAME': 'klue/roberta-base',
    'MAX_LENGTH': 512,         # RoBERTa 입력 최대 길이
    'STUDENT_LAYER': 6,        # 학생 모델 레이어 수 (12 -> 6 압축)
    'NUM_CLASSES': 2,          # 분류 클래스 수 (0: 정상, 1: 피싱)
    
    # 4. 데이터 전처리 설정 (Preprocessing)
    'WINDOW_SIZE': 5,          # 몇 문장을 묶을 것인가
    'STRIDE': 2,               # 몇 문장씩 이동할 것인가 (데이터 증강 효과)
    
    # 5. 학습 파라미터 (Training)
    'EPOCHS': 10,
    'BATCH_SIZE': 16,          # OOM(메모리 부족) 발생 시 8로 줄이세요
    'LEARNING_RATE': 1e-5,     # 학습률
    'WEIGHT_DECAY': 0.01,      # 가중치 규제 (옵션)
    
    # 6. 증강 설정 (Augmentation - KoEDA)
    'AUG_PROB': 0.5            # 텍스트 증강 적용 확률
}
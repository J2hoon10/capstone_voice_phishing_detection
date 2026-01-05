# utils.py
import os
import torch
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
    ID(통화 건) 기준으로 학습/테스트 셋 분할
    - 데이터 누수(Data Leakage) 방지
    """
    if not os.path.exists(config['PROCESSED_DATA_DIR']):
        os.makedirs(config['PROCESSED_DATA_DIR'])

    print(f"\n[Data Prep] Reading {config['MASTER_DATA_PATH']}...")
    try:
        df = pd.read_csv(config['MASTER_DATA_PATH'])
    except FileNotFoundError:
        raise FileNotFoundError(f"데이터 파일이 없습니다: {config['MASTER_DATA_PATH']}")
    
    # ID를 기준으로 8:2 분할
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=config['SEED'])
    train_idx, test_idx = next(splitter.split(df, groups=df['ID']))
    
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    
    train_path = os.path.join(config['PROCESSED_DATA_DIR'], 'train_split.csv')
    test_path = os.path.join(config['PROCESSED_DATA_DIR'], 'test_split.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"   -> Train Samples: {len(train_df)}")
    print(f"   -> Test Samples:  {len(test_df)}")
    
    return train_path, test_path

def get_logger(output_dir):
    """
    콘솔과 파일(train.log)에 동시에 로그를 남기는 객체 생성
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    logger = logging.getLogger('VoicePhishing')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # -------------------------------------------------------------------------
    # [수정] 파일 핸들러에 encoding='utf-8' 옵션 추가
    # 윈도우(cp949) 환경에서도 이모지나 특수문자를 안전하게 파일에 저장하도록 설정
    # -------------------------------------------------------------------------
    file_handler = logging.FileHandler(os.path.join(output_dir, 'train.log'), encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 스트림 핸들러 (터미널 출력)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

def save_checkpoint(model, optimizer, epoch, loss, path, scheduler=None):
    """학습 중단 대비용 체크포인트 저장"""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    # 스케줄러가 들어오면 함께 저장
    if scheduler is not None:
        state['scheduler_state_dict'] = scheduler.state_dict()
        
    torch.save(state, path)
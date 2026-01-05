# config.py
import os

# [수정] 현재 config.py 파일이 있는 폴더(test_inference) 경로 구하기
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    'MODEL_NAME': 'klue/roberta-base',
    'NUM_CLASSES': 2,
    'STUDENT_LAYER': 6,
    'MAX_LENGTH': 512,
    'WINDOW_SIZE': 5,
    'STRIDE': 2,
}

CONFIG['WHISPER_MODEL_SIZE'] = "deepdml/faster-whisper-large-v3-turbo-ct2"

# [수정] 시스템 경로 설정 (상대 경로 대신 절대 경로 결합 사용)
# 로그나 모델이 저장될 폴더를 'test_inference/model_weight' 로 지정
CONFIG['OUTPUT_DIR'] = os.path.join(BASE_DIR, "model_weight")

import torch
CONFIG['DEVICE'] = "cuda" if torch.cuda.is_available() else "cpu"
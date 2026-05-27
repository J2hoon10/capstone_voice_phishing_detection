# pipeline_config.py
# models/main/data_augmentation/ 전처리 파이프라인 경로 설정

import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # models/main/data_augmentation/
_MODEL_ARCH_DIR = os.path.join(os.path.dirname(_THIS_DIR), "model_architecture")  # models/main/model_architecture/

# === 오디오 데이터 경로 ===
DATA_DIR     = os.path.join(_MODEL_ARCH_DIR, "data")
PHISHING_DIR = os.path.join(DATA_DIR, "phishing")
NORMAL_DIR   = os.path.join(DATA_DIR, "normal")

# === 전처리 출력 경로 ===
TRANSCRIPTION_DIR = os.path.join(_THIS_DIR, "transcriptions")
ERROR_ANALYSIS_DIR = os.path.join(_THIS_DIR, "error_analysis")
AUGMENTED_DIR     = os.path.join(_THIS_DIR, "augmented")
FINAL_DIR         = os.path.join(_THIS_DIR, "output")

# === Whisper 모델 변형 ===
WHISPER_VARIANTS = {
    "gpu_small": {
        "model": "Systran/faster-whisper-small",
        "device": "cuda",
        "compute_type": "float16",
    },
    "cpu_base": {
        "model": "Systran/faster-whisper-base",
        "device": "cpu",
        "compute_type": "int8",
    },
}

DEFAULT_VARIANT = "gpu_small"

# === 오디오 확장자 ===
AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".m4a", ".ogg")

# === 오류 분석 설정 ===
GROUND_TRUTH_SAMPLE_COUNT = 10
GROUND_TRUTH_SEED = 42

# === 증강 설정 ===
LLM_AUGMENT_RATIO = 2
MIN_TEXT_LENGTH = 5
MIN_KOREAN_RATIO = 0.5

# === 데이터셋 분할 ===
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
SPLIT_SEED = 42

# === CSV 컬럼 ===
CSV_COLUMNS = ["id", "text", "label", "category", "source", "filename"]

# === GPU 배치 처리 설정 ===
BATCH_SIZE = 8
PREFETCH_WORKERS = 2

# pipeline_config.py
# P1/P2 데이터 전처리 파이프라인 설정

import os

# === 경로 설정 ===
# CLASSIFIER_DIR: models/classifier/  (이 파일의 부모 디렉터리)
CLASSIFIER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = CLASSIFIER_DIR  # 하위 호환용 별칭

DATA_DIR = os.path.join(CLASSIFIER_DIR, "data")
OUTPUT_DIR = os.path.join(CLASSIFIER_DIR, "preprocessing")

PHISHING_DIR = os.path.join(DATA_DIR, "phishing")
NORMAL_DIR = os.path.join(DATA_DIR, "normal")

TRANSCRIPTION_DIR = os.path.join(OUTPUT_DIR, "transcriptions")
ERROR_ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "error_analysis")
AUGMENTED_DIR = os.path.join(OUTPUT_DIR, "augmented")
FINAL_DIR = os.path.join(OUTPUT_DIR, "final")

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
GROUND_TRUTH_SAMPLE_COUNT = 10  # 카테고리당 샘플 수
GROUND_TRUTH_SEED = 42

# === 증강 설정 ===
LLM_AUGMENT_RATIO = 2  # 피싱 원본 1개당 LLM 생성 수
MIN_TEXT_LENGTH = 5     # 최소 텍스트 길이 (문자)
MIN_KOREAN_RATIO = 0.5  # 최소 한국어 비율

# === 데이터셋 분할 ===
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
SPLIT_SEED = 42

# === CSV 컬럼 ===
CSV_COLUMNS = ["id", "text", "label", "category", "source", "filename"]

# === GPU 배치 처리 설정 (RTX 4070 Ti Super 16GB VRAM 기준) ===
BATCH_SIZE = 8              # BatchedInferencePipeline batch_size (여유 있게)
PREFETCH_WORKERS = 2        # I/O 프리페치 스레드 수

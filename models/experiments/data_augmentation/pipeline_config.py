# pipeline_config.py
# models/experiments/data_augmentation/ 실험용 증강 파이프라인 경로 설정

import os

_THIS_DIR      = os.path.dirname(os.path.abspath(__file__))   # models/experiments/data_augmentation/
_MAIN_AUGM_DIR = os.path.join(
    os.path.dirname(os.path.dirname(_THIS_DIR)),              # models/
    "main", "data_augmentation"
)

# === 전처리 입력 경로 (main 파이프라인 산출물 공유) ===
TRANSCRIPTION_DIR  = os.path.join(_MAIN_AUGM_DIR, "transcriptions")
ERROR_ANALYSIS_DIR = os.path.join(_MAIN_AUGM_DIR, "error_analysis")

# === 실험 출력 경로 ===
AUGMENTED_DIR = os.path.join(_THIS_DIR, "output")
FINAL_DIR     = os.path.join(_THIS_DIR, "output", "final")

# === 증강 설정 ===
DEFAULT_VARIANT  = "gpu_small"
LLM_AUGMENT_RATIO = 2
MIN_TEXT_LENGTH   = 5
MIN_KOREAN_RATIO  = 0.5

# === CSV 컬럼 ===
CSV_COLUMNS = ["id", "text", "label", "category", "source", "filename"]

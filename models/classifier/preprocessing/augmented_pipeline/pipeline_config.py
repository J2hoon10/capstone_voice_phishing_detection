import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSING_DIR = os.path.dirname(PACKAGE_DIR)
TRANSCRIPTION_DIR = os.path.join(PREPROCESSING_DIR, "transcriptions")
ERROR_ANALYSIS_DIR = os.path.join(PREPROCESSING_DIR, "error_analysis")

# Keep outputs compatible with existing preprocessing structure.
LEGACY_AUGMENTED_DIR = os.path.join(PREPROCESSING_DIR, "augmented")
LEGACY_FINAL_DIR = os.path.join(PREPROCESSING_DIR, "final")

PROMPTS_DIR = os.path.join(PACKAGE_DIR, "prompts")
DEFAULT_VARIANT = "gpu_small"
CSV_COLUMNS = ["id", "text", "label", "category", "source", "filename"]
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
SPLIT_SEED = 42


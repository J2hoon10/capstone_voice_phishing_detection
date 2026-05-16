import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSING_DIR = os.path.dirname(PACKAGE_DIR)
TRANSCRIPTION_DIR = os.path.join(PREPROCESSING_DIR, "transcriptions")
ERROR_ANALYSIS_DIR = os.path.join(PREPROCESSING_DIR, "error_analysis")

AUGMENTED_DIR = os.path.join(PACKAGE_DIR, "output")
FINAL_DIR = os.path.join(PACKAGE_DIR, "output", "final")
STN_LABELING_OUTPUT_DIR = os.path.join(PREPROCESSING_DIR, "stn_labeling", "output")

PROMPTS_DIR = os.path.join(PACKAGE_DIR, "prompts")
DEFAULT_VARIANT = "gpu_small"
CSV_COLUMNS = ["id", "text", "label", "category", "source", "filename", "segment_risks"]
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
SPLIT_SEED = 42


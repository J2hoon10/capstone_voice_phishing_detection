import os

import torch


CONFIG = {
    "DEVICE": os.getenv("CLASSIFIER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
    "DEFAULT_THRESHOLD": float(os.getenv("DEFAULT_THRESHOLD", "0.5")),

}

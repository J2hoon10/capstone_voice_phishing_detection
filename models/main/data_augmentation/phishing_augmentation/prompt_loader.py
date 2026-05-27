import json
import os
from typing import Dict

from pipeline_config import PROMPTS_DIR


def load_system_prompt() -> str:
    path = os.path.join(PROMPTS_DIR, "system_prompt.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_category_prompts() -> Dict[str, str]:
    path = os.path.join(PROMPTS_DIR, "category_prompts.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


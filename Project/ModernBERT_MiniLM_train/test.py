import torch
from transformers import AutoModelForSequenceClassification, AutoConfig

model_name = "neavo/modern_bert_multilingual"
num_labels = 2

print(f"Loading {model_name} structure check...")

# 1. 모델 로드
config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, trust_remote_code=True)

# 2. 모델 전체 구조 출력 (가장 중요)
print("\n=== [1] Model Architecture (print(model)) ===")
print(model)

# 3. 실제 파라미터 키 이름 출력 (앞부분 20개만)
print("\n=== [2] State Dict Keys (Sample) ===")
for i, key in enumerate(model.state_dict().keys()):
    if i > 20: break
    print(key)

# 4. 특정 속성 존재 여부 확인
print("\n=== [3] Attribute Check ===")
print(f"Has 'bert'? {hasattr(model, 'bert')}")
print(f"Has 'model'? {hasattr(model, 'model')}")
print(f"Has 'encoder'? {hasattr(model, 'encoder')}")
print(f"Has 'layers'? {hasattr(model, 'layers')}")
import csv
import json
from pathlib import Path

data_dir = Path(r"c:\Users\myhom\Jihoon\capstone_project\capstone_voice_phishing_detection\models\classifier\preprocessing\final")
files = ["train.csv", "val.csv", "test.csv"]

stats = {"total_phishing": 0, "total_normal": 0, "categories": {}}

for f in files:
    with open(data_dir / f, "r", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            cat = row.get("category", "").strip()
            cat = "수사기관 사칭형" if cat == "바로 이 목소리" else cat
            if cat not in stats["categories"]:
                stats["categories"][cat] = 0
            stats["categories"][cat] += 1
            
            if cat in ["수사기관 사칭형", "대출 사기형"]:
                stats["total_phishing"] += 1
            elif cat in ["상품 가입 및 해지", "이체 출금 대출서비스", "잔고 및 거래내역"]:
                stats["total_normal"] += 1

print(json.dumps(stats, ensure_ascii=False, indent=2))

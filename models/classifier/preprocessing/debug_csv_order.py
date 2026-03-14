import csv

csv_path = r"c:\Users\myhom\Jihoon\capstone_project\capstone_voice_phishing_detection\models\classifier\preprocessing\transcriptions\gpu_small\phishing.csv"

count = 0
print("Filenames in phishing.csv for '수사기관 사칭형' in exact order:")
with open(csv_path, 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        if row['category'] == '수사기관 사칭형':
            print(f"  {count+1}. {row['filename']}")
            count += 1

print(f"\nTotal: {count}")

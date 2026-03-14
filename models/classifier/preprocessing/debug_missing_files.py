import os
import csv
import re

csv_path = r"c:\Users\myhom\Jihoon\capstone_project\capstone_voice_phishing_detection\models\classifier\preprocessing\transcriptions\gpu_small\phishing.csv"
audio_dir = r"c:\Users\myhom\Jihoon\capstone_project\capstone_voice_phishing_detection\models\classifier\data\phishing\수사기관 사칭형"

# 1. Read CSV to find which files actually exist for the category
csv_files = set()
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['category'] == '수사기관 사칭형':
            csv_files.add(row['filename'])

print(f"Files in CSV for '수사기관 사칭형': {len(csv_files)}")
for f in sorted(list(csv_files))[:5]:
    print("  ", f)

# 2. Read Original Directory
audio_files = set([f for f in os.listdir(audio_dir) if f.endswith(('.mp3', '.wav', '.flac', '.m4a', '.ogg'))])
print(f"\nFiles in Audio Dir: {len(audio_files)}")

missing_in_csv = audio_files - csv_files
print(f"\nFiles missing in CSV (Count: {len(missing_in_csv)}):")

def extract_num(f):
    m = re.search(r'(\d+)', f)
    return int(m.group(1)) if m else 9999

sorted_missing = sorted(list(missing_in_csv), key=extract_num)
print("First few missing files:")
for f in sorted_missing[:20]:
    print("  ", f)

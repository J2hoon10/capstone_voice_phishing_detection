# STT 테스트 스크립트
# 피싱 30개 + 일반 30개 음성 파일을 STT 처리하여 결과를 저장

import os
import sys
import random
import csv
from datetime import datetime

# classifier 모듈 임포트를 위해 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from audio_processor import AudioProcessor

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.dirname(__file__)
SAMPLE_COUNT = 30


def collect_audio_files(base_dir):
    """하위 폴더를 재귀 탐색하여 오디오 파일 목록 수집"""
    audio_extensions = ('.mp3', '.wav', '.flac', '.m4a', '.ogg')
    files = []
    for root, _, filenames in os.walk(base_dir):
        for f in filenames:
            if f.lower().endswith(audio_extensions):
                files.append(os.path.join(root, f))
    return files


def run_test():
    # 1. 파일 수집
    phishing_dir = os.path.join(DATA_DIR, 'phishing')
    normal_dir = os.path.join(DATA_DIR, 'normal')

    phishing_files = collect_audio_files(phishing_dir)
    normal_files = collect_audio_files(normal_dir)

    print(f"피싱 파일: {len(phishing_files)}개, 일반 파일: {len(normal_files)}개")

    # 2. 랜덤 샘플링
    random.seed(42)
    phishing_sample = random.sample(phishing_files, min(SAMPLE_COUNT, len(phishing_files)))
    normal_sample = random.sample(normal_files, min(SAMPLE_COUNT, len(normal_files)))

    # 3. AudioProcessor 초기화
    print("AudioProcessor 로딩 중...")
    processor = AudioProcessor()
    print("로딩 완료!\n")

    # 4. 결과 저장용
    results = []

    # 5. 피싱 데이터 처리
    print("=" * 60)
    print(f"[피싱 데이터] {len(phishing_sample)}개 처리 시작")
    print("=" * 60)
    for i, filepath in enumerate(phishing_sample, 1):
        rel_path = os.path.relpath(filepath, DATA_DIR)
        print(f"\n[{i}/{len(phishing_sample)}] {rel_path}")

        sentences = processor.process_file(filepath)
        script = ' '.join(sentences) if sentences else "(STT 결과 없음)"

        results.append({
            'label': 'phishing',
            'category': os.path.basename(os.path.dirname(filepath)),
            'filename': os.path.basename(filepath),
            'sentence_count': len(sentences),
            'script': script
        })
        print(f"   → {len(sentences)}개 문장 추출")

    # 6. 일반 데이터 처리
    print("\n" + "=" * 60)
    print(f"[일반 데이터] {len(normal_sample)}개 처리 시작")
    print("=" * 60)
    for i, filepath in enumerate(normal_sample, 1):
        rel_path = os.path.relpath(filepath, DATA_DIR)
        print(f"\n[{i}/{len(normal_sample)}] {rel_path}")

        sentences = processor.process_file(filepath)
        script = ' '.join(sentences) if sentences else "(STT 결과 없음)"

        results.append({
            'label': 'normal',
            'category': os.path.basename(os.path.dirname(filepath)),
            'filename': os.path.basename(filepath),
            'sentence_count': len(sentences),
            'script': script
        })
        print(f"   → {len(sentences)}개 문장 추출")

    # 7. CSV 저장
    csv_path = os.path.join(OUTPUT_DIR, 'stt_test_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['label', 'category', 'filename', 'sentence_count', 'script'])
        writer.writeheader()
        writer.writerows(results)

    # 8. 읽기 쉬운 텍스트 리포트 저장
    report_path = os.path.join(OUTPUT_DIR, 'stt_test_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"STT 테스트 결과 리포트\n")
        f.write(f"생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"테스트 수: 피싱 {len(phishing_sample)}개 + 일반 {len(normal_sample)}개\n")
        f.write("=" * 80 + "\n\n")

        # 피싱 결과
        f.write("[ 피싱 데이터 STT 결과 ]\n")
        f.write("-" * 80 + "\n\n")
        for r in results:
            if r['label'] != 'phishing':
                continue
            f.write(f"파일: {r['category']}/{r['filename']}\n")
            f.write(f"문장 수: {r['sentence_count']}\n")
            f.write(f"내용:\n{r['script']}\n")
            f.write("-" * 80 + "\n\n")

        # 일반 결과
        f.write("\n[ 일반 데이터 STT 결과 ]\n")
        f.write("-" * 80 + "\n\n")
        for r in results:
            if r['label'] != 'normal':
                continue
            f.write(f"파일: {r['category']}/{r['filename']}\n")
            f.write(f"문장 수: {r['sentence_count']}\n")
            f.write(f"내용:\n{r['script']}\n")
            f.write("-" * 80 + "\n\n")

    # 9. 요약
    phishing_results = [r for r in results if r['label'] == 'phishing']
    normal_results = [r for r in results if r['label'] == 'normal']

    print("\n" + "=" * 60)
    print("테스트 완료!")
    print(f"  피싱: {len(phishing_results)}개 처리, 평균 {sum(r['sentence_count'] for r in phishing_results) / max(len(phishing_results), 1):.1f}문장")
    print(f"  일반: {len(normal_results)}개 처리, 평균 {sum(r['sentence_count'] for r in normal_results) / max(len(normal_results), 1):.1f}문장")
    print(f"\n결과 파일:")
    print(f"  CSV:    {csv_path}")
    print(f"  리포트: {report_path}")


if __name__ == '__main__':
    run_test()

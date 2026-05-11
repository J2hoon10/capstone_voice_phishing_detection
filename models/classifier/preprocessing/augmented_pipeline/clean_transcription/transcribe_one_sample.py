import argparse
import csv
import os
from pathlib import Path

from openai import OpenAI


def find_first_audio(audio_root: str) -> str:
    root = Path(audio_root)
    for ext in ("*.mp3", "*.wav", "*.m4a", "*.flac", "*.ogg", "*.webm"):
        found = sorted(root.rglob(ext))
        if found:
            return str(found[0])
    return ""


def transcribe(client: OpenAI, audio_path: str, model: str, language: str) -> str:
    with open(audio_path, "rb") as af:
        out = client.audio.transcriptions.create(
            model=model,
            file=af,
            language=language,
            response_format="text",
        )
    if isinstance(out, str):
        return out.strip()
    return (getattr(out, "text", "") or "").strip()


def main():
    parser = argparse.ArgumentParser(description="오디오 1개 샘플 전사 테스트")
    parser.add_argument("--audio-root", default="models/classifier/data/phishing")
    parser.add_argument("--audio-path", default="", help="직접 테스트할 오디오 파일 경로")
    parser.add_argument("--model", default="gpt-4o-mini-transcribe")
    parser.add_argument("--language", default="ko")
    parser.add_argument(
        "--output-csv",
        default="models/classifier/preprocessing/augmented/smoke_clean_transcription.csv",
    )
    parser.add_argument("--dry-run", action="store_true", help="파일 선택만 확인하고 API 호출 생략")
    args = parser.parse_args()

    audio_path = args.audio_path.strip() or find_first_audio(args.audio_root)
    if not audio_path:
        raise SystemExit("[ERROR] 테스트할 오디오 파일을 찾지 못했습니다.")
    if not os.path.exists(audio_path):
        raise SystemExit(f"[ERROR] 오디오 파일이 없습니다: {audio_path}")

    print(f"[INFO] sample audio: {audio_path}")
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    if args.dry_run:
        print("[DONE] dry-run 완료 (API 호출 없음)")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("[ERROR] OPENAI_API_KEY 환경변수가 필요합니다.")

    client = OpenAI(api_key=api_key)
    text = transcribe(client, audio_path, args.model, args.language)

    with open(args.output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "model", "language", "text_clean"])
        writer.writeheader()
        writer.writerow(
            {
                "audio_path": audio_path,
                "model": args.model,
                "language": args.language,
                "text_clean": text,
            }
        )
    print(f"[DONE] {args.output_csv}")


if __name__ == "__main__":
    main()


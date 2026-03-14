import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline_config import WHISPER_VARIANTS, DEFAULT_VARIANT
from faster_whisper import WhisperModel, BatchedInferencePipeline

def test_transcribe(audio_path):
    print(f"Testing {os.path.basename(audio_path)} ...")
    if not os.path.exists(audio_path):
        print("  -> File not found!")
        return
    file_size = os.path.getsize(audio_path)
    print(f"  -> Size: {file_size} bytes")
    
    variant = WHISPER_VARIANTS[DEFAULT_VARIANT]
    whisper_model = WhisperModel(variant["model"], device=variant["device"], compute_type=variant["compute_type"])
    pipeline = BatchedInferencePipeline(model=whisper_model)
    
    try:
        segs, info = pipeline.transcribe(
            audio_path,
            language="ko",
            batch_size=8,
            beam_size=1,
            temperature=0.0,
            vad_filter=True,
            no_speech_threshold=0.6,
        )
        text = " ".join([seg.text for seg in segs])
        print(f"  -> SUCCESS! Text length: {len(text)}")
        print(f"  -> Snippet: {text[:100]}...")
    except Exception as e:
        print(f"  -> FAILED! Error: {e}")

if __name__ == "__main__":
    base_dir = r"c:\Users\myhom\Jihoon\capstone_project\capstone_voice_phishing_detection\models\classifier\data\phishing\수사기관 사칭형"
    test_transcribe(os.path.join(base_dir, "1.mp3"))
    test_transcribe(os.path.join(base_dir, "186.mp3"))

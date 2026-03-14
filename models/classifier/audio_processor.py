# audio_processor.py
import os
import re
import time
import torch
import numpy as np
from faster_whisper import WhisperModel
from audio_enhancer import AudioEnhancer

class AudioProcessor:
    def __init__(self, whisper_model_size="deepdml/faster-whisper-large-v3-turbo-ct2",
                 device=None, compute_type=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        compute_type = compute_type or ("float16" if self.device == "cuda" else "int8")

        self.enhancer = AudioEnhancer()
        print(f"[AudioProcessor] Loading Whisper '{whisper_model_size}'...")
        self.whisper = WhisperModel(whisper_model_size, device=self.device, compute_type=compute_type)

    def clean_text_basic(self, text: str) -> str:
        """기본 정제 + 구두점 정리 (원본 전사 내용 최대한 보존)"""
        text = re.sub(r'[^0-9a-zA-Z가-힣\s.,?!]', '', text)
        text = re.sub(r'([?!])\s*\.+', r'\1', text)
        text = re.sub(r'\.+\s*([?!])', r'\1', text)
        text = re.sub(r'([?!.])\1+', r'\1', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def is_valid_sentence(self, text: str) -> bool:
        if len(text) < 2:
            return False
        no_space = text.replace(' ', '')
        if len(no_space) > 10:
            if len(set(no_space)) / len(no_space) < 0.1:
                return False
        return True

    def remove_duplicates(self, sentences: list) -> list:
        unique = []
        prev = None
        for s in sentences:
            s = s.strip()
            if not s: continue
            if s != prev: unique.append(s)
            prev = s
        return unique

    def process_file(self, input_data) -> list:
        try:
            # 1. 오디오 향상 (Bandpass → Noise Reduction → VAD → Normalize)
            enhanced_audio = self.enhancer.enhance(input_data)

            if enhanced_audio is None:
                target_input = input_data
            else:
                target_input = enhanced_audio

            if isinstance(target_input, np.ndarray) and len(target_input) == 0:
                return []

            t_whisper_start = time.time()

            # 2. Whisper STT
            segments, _ = self.whisper.transcribe(
                target_input,
                beam_size=1,
                language="ko",
                condition_on_previous_text=False,
                repetition_penalty=1.2,
                vad_filter=True,
                temperature=0.0,
                no_speech_threshold=0.6
            )

            full_text = " ".join([seg.text for seg in segments])

            t_whisper_end = time.time()
            print(f"   [Profile] Whisper STT: {(t_whisper_end - t_whisper_start):.4f}s")

        except Exception as e:
            return []

        # 3. 텍스트 후처리
        cleaned = self.clean_text_basic(full_text)

        try:
            import kss
            sentences = kss.split_sentences(cleaned)
        except ImportError:
            sentences = re.split(r'(?<=[.?!])\s+', cleaned)

        sentences = [s for s in sentences if self.is_valid_sentence(s)]
        return self.remove_duplicates(sentences)

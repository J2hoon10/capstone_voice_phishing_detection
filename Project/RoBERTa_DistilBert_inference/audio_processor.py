# audio_processor.py
import os
import re
import time
import torch
import numpy as np
from faster_whisper import WhisperModel
from audio_enhancer import AudioEnhancer

class AudioProcessor:
    def __init__(self, whisper_model_size="deepdml/faster-whisper-large-v3-turbo-ct2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # cuda일 때 float16 사용 (속도 향상)
        compute_type = "float16" if self.device == "cuda" else "int8"

        self.enhancer = AudioEnhancer()
        print(f"[AudioProcessor] Loading Whisper '{whisper_model_size}'...")
        self.whisper = WhisperModel(whisper_model_size, device=self.device, compute_type=compute_type)

        self.blacklist = [
            "correct", "hate-speech", "generated", "silence", "subtitles", "transcript", 
            "MBC", "SBS", "KBS", "YTN", "뉴스", "자막", "배달의 민족", "저작권", 
            "시청해 주셔서", "구독과 좋아요", "알림 설정", "Provided by", "Auto-generated", 
            "오디오 자막", "허허허", "하하하"
        ]

    def mask_pii(self, text: str) -> str:
        text = re.sub(r'\d{6}[-\s]*[1-4]\d{6}', '<RRN>', text)
        text = re.sub(r'01[016789][-\s]*\d{3,4}[-\s]*\d{4}', '<PHONE>', text)
        text = re.sub(r'0(2|3[1-3]|4[1-4]|5[1-5]|6[1-4])[-\s]*\d{3,4}[-\s]*\d{4}', '<PHONE>', text)
        text = re.sub(r'\d{10,}', '<ACCOUNT>', text)
        return text

    def remove_phrase_repetition(self, text: str) -> str:
        return re.sub(r'(\b\w+\b)( \1){2,}', r'\1', text)

    def clean_text_basic(self, text: str) -> str:
        text = re.sub(r'[^0-9a-zA-Z가-힣\s.,?!]', '', text)
        text = re.sub(r'([?!])\s*\.+', r'\1', text)
        text = re.sub(r'\.+\s*([?!])', r'\1', text)
        text = re.sub(r'([?!.])\1+', r'\1', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = self.mask_pii(text)
        text = self.remove_phrase_repetition(text)
        return text

    def is_valid_sentence(self, text: str) -> bool:
        if len(text) < 2: return False
        for keyword in self.blacklist:
            if keyword.lower() in text.lower(): return False
        
        no_space = text.replace(' ', '')
        if len(no_space) > 10:
            if len(set(no_space)) / len(no_space) < 0.1: return False
        
        if text.strip() in ["<PHONE>", "<RRN>", "<ACCOUNT>"]: return False
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

    # [디버깅] 시간 측정 로직 추가된 처리 함수
    def process_file(self, input_data) -> list:
        try:
            # 1. Enhance (AudioEnhancer 내부에서 시간 로그 출력됨)
            enhanced_audio = self.enhancer.enhance(input_data)
            
            # 2. Whisper 입력 준비
            if enhanced_audio is None:
                target_input = input_data
            else:
                target_input = enhanced_audio
            
            if isinstance(target_input, np.ndarray) and len(target_input) == 0:
                return []

            # [측정 시작] Whisper STT
            t_whisper_start = time.time()

            # 3. Whisper 추론
            segments, _ = self.whisper.transcribe(
                target_input,
                beam_size=1,            # [Speed] Greedy decoding
                language="ko",
                condition_on_previous_text=False,
                repetition_penalty=1.2, 
                vad_filter=True,       
                temperature=0.0,
                no_speech_threshold=0.6
            )
            
            full_text = " ".join([seg.text for seg in segments])
            
            t_whisper_end = time.time()
            # [로그 출력] Whisper 시간
            print(f"   [Profile] Whisper STT: {(t_whisper_end - t_whisper_start):.4f}s")

        except Exception as e:
            # print(f"❌ STT Error: {e}") 
            return []

        # 4. 텍스트 후처리 (매우 빠름)
        cleaned = self.clean_text_basic(full_text)
        
        try:
            import kss
            sentences = kss.split_sentences(cleaned)
        except ImportError:
            sentences = re.split(r'(?<=[.?!])\s+', cleaned)

        sentences = [s for s in sentences if self.is_valid_sentence(s)]
        return self.remove_duplicates(sentences)
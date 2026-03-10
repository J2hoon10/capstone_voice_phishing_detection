# audio_enhancer.py
import os
import time
import torch
import numpy as np
import librosa
import noisereduce as nr
from scipy.signal import butter, sosfiltfilt

class AudioEnhancer:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # [준비] 나중에 VAD를 쓸 수 있도록 모델은 미리 로드합니다.
        self.vad_model = None
        self.get_speech_timestamps = None
        try:
            print(f"[AudioEnhancer] Loading Silero VAD on {self.device}...")
            self.vad_model, utils = torch.hub.load(
                "snakers4/silero-vad", "silero_vad", onnx=False, trust_repo=True
            )
            (self.get_speech_timestamps, _, _, _, _) = utils
            self.vad_model.to(self.device)
        except Exception as e:
            print(f"[AudioEnhancer] Silero VAD unavailable: {e}")

    def bandpass_filter(self, audio):
        """사람 목소리 대역(300~3400Hz)만 남기고 나머지 제거"""
        lowcut = 300.0
        highcut = 3400.0 
        if len(audio) <= self.target_sr * 0.01: return audio
        sos = butter(10, [lowcut, highcut], btype='band', fs=self.target_sr, output='sos')
        return sosfiltfilt(sos, audio).astype(np.float32)

    def reduce_noise(self, audio):
        """[주의] CPU 연산량이 많아 속도가 느려질 수 있음"""
        try:
            if len(audio) < self.target_sr * 0.1: return audio
            return nr.reduce_noise(
                y=audio, sr=self.target_sr, prop_decrease=0.75,
                n_fft=1024, stationary=True, use_tqdm=False, n_jobs=1
            )
        except:
            return audio

    def vad_trim(self, audio):
        """무음 구간(Silence)을 잘라내고 목소리만 남김"""
        if self.vad_model is None or self.get_speech_timestamps is None:
            return audio

        wav_torch = torch.from_numpy(audio).float().to(self.device)
        if wav_torch.ndim == 1: wav_torch = wav_torch.unsqueeze(0)
        
        speech_timestamps = self.get_speech_timestamps(
            wav_torch, self.vad_model, threshold=0.4, 
            min_speech_duration_ms=250, min_silence_duration_ms=500,
            sampling_rate=self.target_sr
        )
        if not speech_timestamps: return audio

        pad_samples = int(self.target_sr * 0.3) 
        segments = []
        for ts in speech_timestamps:
            start = max(0, ts["start"] - pad_samples)
            end = min(len(audio), ts["end"] + pad_samples)
            if end > start: segments.append(audio[start:end])

        return np.concatenate(segments).astype(np.float32) if segments else audio

    def normalize(self, audio, target_rms=0.1):
        """볼륨 크기 일정하게 맞추기"""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-6: return audio
        normalized = audio * (target_rms / rms)
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def enhance(self, input_data):
        try:
            t_start = time.time()

            # 1. 입력 로드 (파일 경로 or Numpy 배열)
            if isinstance(input_data, str):
                if not os.path.exists(input_data): return None
                audio, _ = librosa.load(input_data, sr=self.target_sr, mono=True)
            elif isinstance(input_data, np.ndarray):
                audio = input_data.astype(np.float32)
            else:
                print("❌ Invalid input type for enhancer")
                return None

            if len(audio) == 0: return None
            
            t_load = time.time() 

            # =========================================================
            # [전처리 파이프라인] 
            # * 속도를 위해 기본적으로 주석 처리 해두었습니다.
            # * 기능이 필요하면 주석(#)을 지우세요.
            # =========================================================
            
            # 1. 주파수 필터 (목소리 대역만 남기기)
            audio = self.bandpass_filter(audio)
            
            # 2. 노이즈 제거 (가장 느림, 실시간 처리 시 비추천)
            audio = self.reduce_noise(audio)
            
            # 3. VAD (무음 제거, 필요 시 주석 해제)
            # audio = self.vad_trim(audio)
            
            # 4. 정규화 (빠름, 켜두는 것을 추천)
            audio = self.normalize(audio)

            # =========================================================

            t_end = time.time()
            
            # 시간 측정 로그
            print(f"   [Profile] Enhance: {(t_end - t_start):.4f}s (Load: {(t_load - t_start):.4f}s, Process: {(t_end - t_load):.4f}s)")

            return audio 
        except Exception as e:
            print(f"❌ Enhancement Error: {e}")
            return None

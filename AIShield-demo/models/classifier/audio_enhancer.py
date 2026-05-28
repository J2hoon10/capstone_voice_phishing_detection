# audio_enhancer.py
import os
import time
import torch
import numpy as np
import librosa
import noisereduce as nr
from scipy.signal import butter, sosfiltfilt

class AudioEnhancer:
    def __init__(self, target_sr=16000, device=None, vad_device=None,
                 enable_bandpass=True, enable_denoise=True, enable_vad=False, enable_normalize=True):
        self.target_sr = target_sr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vad_device = vad_device or self.device
        self.enable_bandpass = enable_bandpass
        self.enable_denoise = enable_denoise
        self.enable_vad = enable_vad
        self.enable_normalize = enable_normalize

        # [준비] 나중에 VAD를 쓸 수 있도록 모델은 미리 로드합니다.
        self.vad_model = None
        self.get_speech_timestamps = None
        if self.enable_vad:
            try:
                print(f"[AudioEnhancer] Loading Silero VAD on {self.vad_device}...")
                self.vad_model, utils = torch.hub.load(
                    "snakers4/silero-vad", "silero_vad", onnx=False, trust_repo=True
                )
                (self.get_speech_timestamps, _, _, _, _) = utils
                self.vad_model.to(self.vad_device)
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
        """노이즈 제거 (GPU 가속)"""
        try:
            if len(audio) < self.target_sr * 0.1: return audio
            return nr.reduce_noise(
                y=audio, sr=self.target_sr, prop_decrease=0.75,
                n_fft=1024, stationary=True, use_tqdm=False, n_jobs=1,
                device=self.device
            )
        except:
            return audio

    def vad_trim(self, audio):
        """무음 구간(Silence)을 잘라내고 목소리만 남김"""
        if self.vad_model is None or self.get_speech_timestamps is None:
            return audio

        wav_torch = torch.from_numpy(audio).float().to(self.vad_device)
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

    def normalize(self, audio, target_rms=0.1, min_rms=1e-4, peak=0.98):
        """볼륨 크기 일정하게 맞추기 (저음량 증폭 제한 + 간단 리미터)"""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < min_rms:
            return audio
        normalized = audio * (target_rms / rms)
        peak_val = np.max(np.abs(normalized))
        if peak_val > peak:
            normalized = normalized * (peak / peak_val)
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
            # =========================================================

            if self.enable_bandpass:
                audio = self.bandpass_filter(audio)

            if self.enable_denoise:
                audio = self.reduce_noise(audio)

            if self.enable_vad:
                audio = self.vad_trim(audio)

            if self.enable_normalize:
                audio = self.normalize(audio)

            # =========================================================

            t_end = time.time()

            # 시간 측정 로그
            print(f"   [Profile] Enhance: {(t_end - t_start):.4f}s (Load: {(t_load - t_start):.4f}s, Process: {(t_end - t_load):.4f}s)")

            return audio
        except Exception as e:
            print(f"❌ Enhancement Error: {e}")
            return None

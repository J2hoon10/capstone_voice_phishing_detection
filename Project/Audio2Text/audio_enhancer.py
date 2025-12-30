# Whisper 모델이 목소리를 더 잘 알아들을 수 있도록, 통화 녹음의 품질을 개선하는 오디오 향상 도구
# 1. 대역 통과 필터 (Bandpass Filter)
# 2. 잡음 제거 (Noise Reduction)
# 3. 무음 구간 제거 (Voice Activity Detection)
# 4. 볼륨 평준화 (Normalization)

import os
import torch
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from scipy.signal import butter, sosfiltfilt

class AudioEnhancer:
    """
    [STT 최적화 버전]
    전화망 음성(8kHz 대역)을 16kHz Whisper 모델에 맞게 업샘플링 및 정제하는 전처리 클래스
    """
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[AudioEnhancer] Loading Silero VAD on {self.device}...")

        # Silero VAD (Voice Activity Detection) 모델 로드
        self.vad_model, utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", onnx=False, trust_repo=True
        )
        (self.get_speech_timestamps, _, _, _, _) = utils
        self.vad_model.to(self.device)

    def bandpass_filter(self, audio):
        """
        사람 목소리 대역(300~3400Hz)만 남기고 노이즈 제거
        """
        lowcut = 300.0
        highcut = 3400.0 
        
        if len(audio) <= self.target_sr * 0.01: # 너무 짧으면 패스
            return audio

        sos = butter(10, [lowcut, highcut], btype='band', fs=self.target_sr, output='sos')
        filtered = sosfiltfilt(sos, audio)
        return filtered.astype(np.float32)

    def reduce_noise(self, audio):
        """
        배경 잡음 제거 (목소리 왜곡 방지를 위해 강도 조절)
        """
        try:
            if len(audio) < self.target_sr * 0.1:
                return audio

            # [수정 1] n_jobs=-1 은 Windows에서 에러 발생 -> 1로 원상복구
            # [수정 2] n_fft를 2048 -> 1024 (또는 512)로 줄여서 연산 속도 확보
            reduced_audio = nr.reduce_noise(
                y=audio, 
                sr=self.target_sr, 
                prop_decrease=0.75,
                n_fft=1024,      # [속도 개선] 해상도를 약간 낮춰 속도 향상
                stationary=True,
                use_tqdm=False,
                n_jobs=1         # [안정성] Windows 에러 방지를 위해 1로 설정
            )
            return reduced_audio
        except Exception as e:
            print(f"⚠️ Noise reduction skipped due to error: {e}")
            return audio

    def vad_trim(self, audio):
        """
        무음 구간 제거 (Whisper 환각 방지 핵심)
        """
        wav_torch = torch.from_numpy(audio).float().to(self.device)
        if wav_torch.ndim == 1:
            wav_torch = wav_torch.unsqueeze(0)

        # threshold=0.4: 작은 목소리도 놓치지 않도록 설정
        speech_timestamps = self.get_speech_timestamps(
            wav_torch, 
            self.vad_model, 
            threshold=0.4, 
            min_speech_duration_ms=250, 
            min_silence_duration_ms=500,
            sampling_rate=self.target_sr
        )

        if not speech_timestamps:
            return audio

        # 앞뒤 300ms 여유를 두고 자름 (Padding)
        pad_samples = int(self.target_sr * 0.3) 
        segments = []
        for ts in speech_timestamps:
            start = max(0, ts["start"] - pad_samples)
            end = min(len(audio), ts["end"] + pad_samples)
            if end > start:
                segments.append(audio[start:end])

        if not segments:
            return audio
            
        return np.concatenate(segments).astype(np.float32)

    def normalize(self, audio, target_rms=0.1):
        """볼륨 평준화"""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-6:
            return audio
        normalized = audio * (target_rms / rms)
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    # [수정] 파일 경로 대신 처리된 오디오 데이터(numpy array)를 직접 반환하도록 변경
    # [속도 개선] Disk Write 작업을 제거하여 I/O 병목 해소
    def enhance(self, input_path):
        """전처리 파이프라인 실행: Load -> Bandpass -> NR -> VAD -> Normalize -> Return Array"""
        try:
            # [참고] librosa.load는 느린 편입니다. 더 빠른 속도가 필요하면 soundfile.read 사용을 권장합니다.
            # (단, soundfile은 리샘플링을 직접 구현해야 함)
            audio, _ = librosa.load(input_path, sr=self.target_sr, mono=True)
            if len(audio) == 0: return None

            audio = self.bandpass_filter(audio)
            audio = self.reduce_noise(audio)
            audio = self.vad_trim(audio)
            audio = self.normalize(audio)

            # sf.write(output_path, audio, self.target_sr) # [제거] 디스크 쓰기 생략
            return audio # [추가] 메모리상의 데이터 반환
        except Exception as e:
            print(f"❌ Enhancement Error processing {input_path}: {e}")
            return None
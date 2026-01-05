# inference.py
import torch
import torch.nn.functional as F
import numpy as np
import os
import time

from config import CONFIG
from architecture import DistillableRoBERTaModel
from audio_processor import AudioProcessor
from transformers import AutoTokenizer

# ==========================================
# 1. 누적 점수 관리 클래스 (Leaky Bucket)
# ==========================================
class PhishingRiskScorer:
    def __init__(self):
        self.current_score = 0.0
        self.max_score = 100.0
        self.min_score = 0.0
        self.threshold_level1 = 30 # [주의]
        self.threshold_level2 = 60 # [경고]

    def update_score(self, prob: float):
        score_change = 0
        if prob > 0.8:
            score_change = 20    # 🚨 확실한 위험
        elif 0.5 < prob <= 0.8:
            score_change = 10    # ⚠️ 의심
        else:
            score_change = -10   # ✅ 정상 (점수 차감)

        self.current_score += score_change
        self.current_score = max(self.min_score, min(self.current_score, self.max_score))

        return self.current_score, self._get_warning_level()

    def _get_warning_level(self):
        if self.current_score >= self.threshold_level2:
            return "LEVEL_2_WARNING"
        elif self.current_score >= self.threshold_level1:
            return "LEVEL_1_CAUTION"
        else:
            return "NORMAL"

# ==========================================
# 2. 보이스피싱 탐지기 클래스
# ==========================================
class VoicePhishingDetector:
    def __init__(self, model_path, device=None):
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device(CONFIG['DEVICE'])
        
        print(f"--- [Inference] Initializing Detector on {self.device} ---")

        self.model_name = CONFIG['MODEL_NAME']
        self.max_length = CONFIG['MAX_LENGTH']
        self.window_size = CONFIG['WINDOW_SIZE']
        self.stride = CONFIG['STRIDE']

        print(f"[Init] Loading Audio Processor ({CONFIG['WHISPER_MODEL_SIZE']})...")
        self.processor = AudioProcessor(whisper_model_size=CONFIG['WHISPER_MODEL_SIZE'])
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        print("[Init] Loading Student Model Architecture...")
        self.model = DistillableRoBERTaModel(
            model_name=self.model_name,
            num_classes=CONFIG['NUM_CLASSES'],
            is_student=True,       
            student_layer_num=CONFIG['STUDENT_LAYER']
        ).to(self.device)

        self._load_weights(model_path)
        self.model.eval()
        print("[Init] System Ready.")

    def _load_weights(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
        try:
            checkpoint = torch.load(path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"✅ Model weights loaded from {os.path.basename(path)}")
        except Exception as e:
            raise RuntimeError(f"가중치 로드 실패: {e}")

    def _create_windows(self, sentences):
        if not sentences: return []
        windows = []     
        if len(sentences) <= self.window_size:
            chunk = " ".join(sentences)
            windows.append(chunk)
        else:
            for i in range(0, len(sentences) - self.window_size + 1, self.stride):
                chunk_list = sentences[i : i + self.window_size]
                chunk_text = " ".join(chunk_list)
                windows.append(chunk_text)
        return windows

    def predict(self, audio_file_path, threshold=0.5):
        """
        오디오 파일 경로 -> 보이스피싱 여부 및 위험도 반환
        """
        # [Step 1] 전처리
        cleaned_sentences = self.processor.process_file(audio_file_path)
        
        if not cleaned_sentences:
            return {"status": "fail", "message": "No text detected"}

        # [Step 2] 윈도우 생성
        windows = self._create_windows(cleaned_sentences)
        if not windows:
             return {"status": "fail", "message": "Window creation failed"}

        # [Step 3] 토큰화
        inputs = self.tokenizer(
            windows,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        # [측정 시작] RoBERTa NLP
        t_nlp_start = time.time()

        # [Step 4] 모델 추론
        with torch.no_grad():
            # [수정] 경고 해결을 위해 최신 문법 적용 (torch.amp.autocast)
            with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                outputs = self.model(input_ids, attention_mask)
                logits = outputs['logits']
                probs = F.softmax(logits.float(), dim=-1) 
                
            phishing_scores = probs[:, 1].cpu().numpy()

        t_nlp_end = time.time()
        print(f"   [Profile] RoBERTa NLP: {(t_nlp_end - t_nlp_start):.4f}s")

        # [Step 5] 결과 집계
        max_score = float(np.max(phishing_scores))
        max_idx = np.argmax(phishing_scores)
        dangerous_segment = windows[max_idx]

        is_phishing = max_score >= threshold

        return {
            "status": "success",
            "is_phishing": is_phishing,
            "max_risk_score": max_score,
            "dangerous_segment": dangerous_segment,
        }
# inference.py
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import soundfile as sf # [추가] 웜업용 파일 생성을 위해 필요

# Hugging Face 관련 모듈
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

# 사용자 정의 모듈
from config import CONFIG
from audio_processor import AudioProcessor

class PhishingRiskScorer:
    def __init__(self):
        self.current_score = 0.0
        self.max_score = 100.0
        self.min_score = 0.0
        self.threshold_level1 = 30 
        self.threshold_level2 = 60 

    def update_score(self, prob: float):
        score_change = 0
        if prob > 0.8:
            score_change = 20    
        elif 0.5 < prob <= 0.8:
            score_change = 10    
        else:
            score_change = -10   
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

class VoicePhishingDetector:
    def __init__(self, model_path, device=None):
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device(CONFIG['DEVICE'])
        
        print(f"--- [Inference] Initializing Detector on {self.device} ---")

        self.base_model_name = CONFIG['BASE_MODEL_NAME']
        self.max_length = CONFIG['MAX_LENGTH']
        self.window_size = CONFIG['WINDOW_SIZE']
        self.stride = CONFIG['STRIDE']

        print(f"[Init] Loading Audio Processor ({CONFIG.get('WHISPER_MODEL_SIZE', 'base')})...")
        self.processor = AudioProcessor(whisper_model_size=CONFIG.get('WHISPER_MODEL_SIZE', "base"))
        
        print(f"[Init] Loading Tokenizer from {self.base_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

        print("[Init] Building Student Architecture (Custom ModernBERT)...")
        self._build_student_model()

        self._load_weights(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        # [수정] 파일 기반 실전 웜업 실행
        self._warmup()
        
        print("[Init] System Ready.")

    def _warmup(self):
        """
        [수정] VAD 필터를 끄고(False) 강제로 추론을 실행하여,
        노이즈 데이터라도 무조건 텍스트 변환(Decoding) 연산을 수행하게 만듭니다.
        """
        print("🔥 [Warm-up] Running full-path simulation (Forcing Decode)...")
        warmup_filename = "warmup_temp.wav"
        try:
            # 1. 15초 길이의 랜덤 노이즈 파일 생성
            sr = 16000
            duration = self.window_size # 15초
            
            # -1.0 ~ 1.0 범위의 랜덤 노이즈
            audio_data = np.random.uniform(-0.5, 0.5, int(sr * duration)).astype(np.float32)
            sf.write(warmup_filename, audio_data, sr)
            
            # 2. 실제 추론 실행 (가장 중요한 부분!)
            print("   - Forcing Whisper to transcode (VAD Disabled)...")
            
            # [핵심] Processor를 통하지 않고 Whisper 모델 직접 호출
            # vad_filter=False로 설정하면 소음이라도 억지로 번역을 시도하며 GPU를 풀가동시킵니다.
            if hasattr(self.processor, 'whisper'):
                self.processor.whisper.transcribe(
                    warmup_filename,
                    language="ko",
                    beam_size=1,       
                    temperature=0.0,
                    vad_filter=False,  # <--- [치트키] 무조건 디코딩 수행 (Skip 방지)
                )
            
            # 3. NLP 모델 웜업 (긴 문장으로 최대 길이 메모리 확보)
            long_dummy_text = ["보이스피싱 탐지 시스템 웜업을 위한 긴 문장입니다. " * 5]
            inputs = self.tokenizer(
                long_dummy_text, return_tensors='pt', 
                max_length=self.max_length, padding='max_length', truncation=True
            )
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            with torch.no_grad():
                self.model(input_ids, attention_mask=attention_mask)
            
            # 4. KSS 웜업
            try:
                import kss
                kss.split_sentences("KSS 로딩용 문장입니다.")
            except:
                pass

            print("✅ [Warm-up] Completed. Latency is optimized.")
            
        except Exception as e:
            print(f"⚠️ [Warm-up] Failed: {e}")
        finally:
            if os.path.exists(warmup_filename):
                try:
                    os.remove(warmup_filename)
                except:
                    pass

    def _build_student_model(self):
        try:
            config = AutoConfig.from_pretrained(self.base_model_name, num_labels=CONFIG['NUM_LABELS'])
            student_cfg = CONFIG['STUDENT']['config']
            config.hidden_size = student_cfg['hidden_size']
            config.num_hidden_layers = student_cfg['num_hidden_layers']
            config.num_attention_heads = student_cfg['num_attention_heads']
            config.intermediate_size = student_cfg['intermediate_size']
            self.model = AutoModelForSequenceClassification.from_config(config)
        except Exception as e:
            raise RuntimeError(f"모델 아키텍처 생성 실패: {e}")

    def _load_weights(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
        try:
            print(f" -> Loading weights from {os.path.basename(path)}...")
            checkpoint = torch.load(path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k
                if new_key.startswith("student."): new_key = new_key.replace("student.", "")
                if new_key.startswith("module."): new_key = new_key.replace("module.", "")
                new_state_dict[new_key] = v

            keys = self.model.load_state_dict(new_state_dict, strict=False)
            if len(keys.missing_keys) > 0: print(f"⚠️ [Warning] Missing keys: {len(keys.missing_keys)}")
            print("✅ Weights loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"가중치 로드 중 치명적 오류: {e}")

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
            windows, padding='max_length', truncation=True,
            max_length=self.max_length, return_tensors='pt'
        )
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        # [Step 4] 모델 추론
        t_nlp_start = time.time()
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = F.softmax(logits.float(), dim=-1) 
            phishing_scores = probs[:, 1].cpu().numpy()
        t_nlp_end = time.time()
        
        # 로그 출력
        print(f"   [Profile] NLP Inference: {(t_nlp_end - t_nlp_start):.4f}s")

        # [Step 5] 결과 집계
        max_score = float(np.max(phishing_scores)) * 100
        max_idx = np.argmax(phishing_scores)
        dangerous_segment = windows[max_idx]
        is_phishing = max_score >= (threshold * 100)

        return {
            "status": "success",
            "is_phishing": is_phishing,
            "max_risk_score": max_score,
            "dangerous_segment": dangerous_segment,
        }
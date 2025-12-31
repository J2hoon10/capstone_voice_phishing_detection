import torch
from torch.utils.data import Dataset
import pandas as pd
import random
import kss
from koeda import EDA 

class VoicePhishingDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512, window_size=5, stride=2, inference_mode=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.window_size = window_size
        self.stride = stride
        self.inference_mode = inference_mode
        
        # 1. 원본 데이터 로드
        print(f"[Dataset] Loading data from {file_path} (Inference Mode: {inference_mode})...")
        try:
            self.raw_df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"파일을 읽을 수 없습니다: {e}")

        # 2. 증강 모듈 초기화 (KoEDA 0.0.4 버전 맞춤 수정)
        if not self.inference_mode:
            print("[Dataset] Initializing KoEDA for text augmentation...")
            try:
                # [수정 1] 초기화 시에는 'morpheme_analyzer'만 설정합니다.
                # (alpha 값들을 여기서 설정하면 에러가 발생합니다)
                self.eda = EDA(morpheme_analyzer="Okt")
            except Exception as e:
                print(f"[Warning] KoEDA 초기화 실패 (Augmentation Off): {e}")
                self.eda = None
        else:
            self.eda = None
        
        # 3. 데이터 전처리
        print(f"[Dataset] Processing with KSS split (Window: {window_size}, Stride: {stride})...")
        self.samples = self._prepare_samples(self.raw_df, inference_mode)
        print(f"[Dataset] Ready. Total samples: {len(self.samples)}")

    def _split_sentences(self, text):
        if not isinstance(text, str) or not text.strip():
            return []
        try:
            sentences = kss.split_sentences(text)
            return sentences
        except Exception:
            return text.strip().split('\n')

    def _prepare_samples(self, df, inference_mode):
        processed_data = []
        for idx, row in df.iterrows():
            script = row['script']
            
            if 'label' in row and not pd.isna(row['label']):
                label = int(row['label'])
            else:
                label = 0 if not inference_mode else -1
            
            sentences = self._split_sentences(script)
            if not sentences: continue

            if len(sentences) <= self.window_size:
                combined_text = " ".join(sentences)
                processed_data.append({'input_text': combined_text, 'label': label})
                continue
            
            for i in range(0, len(sentences) - self.window_size + 1, self.stride):
                window = sentences[i : i + self.window_size]
                combined_text = " ".join(window)
                processed_data.append({'input_text': combined_text, 'label': label})
                
        return processed_data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        text = data['input_text']
        label = data['label']
        
        # [수정 2] 실제 증강을 수행하는 시점에 alpha 값과 개수를 전달합니다.
        if not self.inference_mode and self.eda is not None and random.random() < 0.5:
            try:
                # num_aug=1: 증강된 문장을 1개만 생성 (기본값은 9개라 리스트가 길어짐)
                aug_text = self.eda(
                    text,
                    alpha_sr=0.1, 
                    alpha_ri=0.1, 
                    alpha_rs=0.1, 
                    alpha_rd=0.1,
                    num_aug=1 
                )
                
                # KoEDA는 결과를 리스트로 반환하므로 문자열로 꺼내줍니다.
                if isinstance(aug_text, list) or isinstance(aug_text, tuple):
                    text = aug_text[0]
                else:
                    text = str(aug_text)
                    
            except Exception:
                pass # 증강 실패 시 원본 사용

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
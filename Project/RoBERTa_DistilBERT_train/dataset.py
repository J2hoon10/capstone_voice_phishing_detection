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
        
        # [수정 3] ID(대화) 단위 랜덤 믹스 구현
        # 학습 모드일 때만 데이터프레임의 행(Row, 즉 대화 1개 단위)을 무작위로 섞습니다.
        # 이렇게 하면 대화의 순서는 섞이지만, 아래 for 문에서 처리되는 대화 내부의 문장 순서는 유지됩니다.
        if not inference_mode:
            print("[Dataset] Shuffling conversation order (ID-level mixing)...")
            # frac=1: 전체 데이터, random_state: 재현성을 위한 시드값(필요시 변경 가능)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

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
                aug_text = self.eda(
                    text,
                    alpha_sr=0.1, 
                    alpha_ri=0.1, 
                    alpha_rs=0.1, 
                    alpha_rd=0.1,
                    num_aug=1 
                )
                
                if isinstance(aug_text, list) or isinstance(aug_text, tuple):
                    text = aug_text[0]
                else:
                    text = str(aug_text)
                    
            except Exception:
                pass 

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
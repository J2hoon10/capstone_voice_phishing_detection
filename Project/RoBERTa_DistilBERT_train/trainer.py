# trainer.py
import torch
import torch.nn as nn
from torch.optim import AdamW
from loss_fun import DistilBERTLoss
from tqdm import tqdm  # [★추가] 진행률 표시 라이브러리

class DistillationTrainer:
    def __init__(self, teacher_model, student_model, train_dataloader, device, lr=5e-5):
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.dataloader = train_dataloader
        self.device = device
        
        self.loss_fn = DistilBERTLoss(alpha=5.0, beta=2.0, gamma=1.0, temperature=2.0)
        self.optimizer = AdamW(self.student.parameters(), lr=lr)

    def train_epoch(self, epoch_idx):
        self.teacher.eval()
        self.student.train()
        
        total_loss = 0
        # tqdm으로 감싸서 진행률 바 생성
        progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch_idx+1} Train", leave=False)
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            with torch.no_grad():
                t_outputs = self.teacher(input_ids, attention_mask)

            s_outputs = self.student(input_ids, attention_mask)

            loss, loss_dict = self.loss_fn(
                student_outputs=s_outputs,
                teacher_outputs=t_outputs,
                labels=labels,
                attention_mask=attention_mask
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            
            # 진행률 바 옆에 실시간 Loss 표시
            progress_bar.set_postfix(loss=loss.item(), kd=loss_dict['kd_loss'])

        avg_loss = total_loss / len(self.dataloader)
        return avg_loss  # 평균 Loss 반환

    def evaluate(self, test_dataloader):
        self.student.eval()
        total_loss = 0
        correct = 0
        total_samples = 0
        criterion = nn.CrossEntropyLoss()
        
        # 검증 과정도 진행률 표시
        progress_bar = tqdm(test_dataloader, desc="Validation", leave=False)
        
        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.student(input_ids, attention_mask)
                logits = outputs['logits']
                
                loss = criterion(logits, labels)
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
        
        avg_loss = total_loss / len(test_dataloader)
        accuracy = (correct / total_samples) * 100
        
        return avg_loss, accuracy
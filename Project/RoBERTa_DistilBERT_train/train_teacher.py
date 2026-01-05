import os
import torch
import torch.nn as nn
import csv
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.amp import GradScaler # [수정] 최신 PyTorch 권장 Import
from tqdm import tqdm

from config import CONFIG
from architecture import DistillableRoBERTaModel
from dataset import VoicePhishingDataset
from utils import set_seed, prepare_data, get_logger, save_checkpoint

def calculate_class_weights(dataset, device):
    """
    [수정됨] 클래스 가중치 계산 및 Clipping(제한) 적용
    """
    print("[Info] Calculating class weights...")
    labels = [sample['label'] for sample in dataset.samples]
    
    labels_np = np.array(labels)
    classes, counts = np.unique(labels_np, return_counts=True)
    
    total_samples = len(labels_np)
    n_classes = len(classes)
    
    # 1. 기본 가중치 계산 (Inverse Frequency)
    weights = total_samples / (n_classes * counts)
    
    # -------------------------------------------------------------------------
    # [수정 1] 가중치 제한 (Weight Clipping) 적용
    # 가중치가 너무 커지면(예: 100배) 학습이 불안정해지므로 최대 10배까지만 허용
    # -------------------------------------------------------------------------
    weights = np.clip(weights, a_min=1.0, a_max=5.0)
    
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    
    print(f"   -> Class Counts: {dict(zip(classes, counts))}")
    print(f"   -> Final Weights (Clipped): {weights_tensor.cpu().numpy()}")
    
    return weights_tensor

def evaluate_teacher(model, dataloader, device):
    """선생님 모델 평가"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Val] Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            logits = outputs['logits']
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = (correct / total_samples) * 100
    
    return avg_loss, accuracy

def main():
    # 1. 초기 설정
    set_seed(CONFIG['SEED'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger = get_logger(CONFIG['OUTPUT_DIR'])
    logger.info(f"--- [Teacher Training] Device: {device} ---")
    
    use_amp = True if CONFIG['DEVICE'] == 'cuda' and torch.cuda.is_available() else False
    if use_amp:
        logger.info("⚡ AMP (Automatic Mixed Precision) Enabled.")

    if not os.path.exists(CONFIG['OUTPUT_DIR']):
        os.makedirs(CONFIG['OUTPUT_DIR'])
        
    log_csv_path = os.path.join(CONFIG['OUTPUT_DIR'], 'training_log_teacher.csv')
    if not os.path.exists(log_csv_path):
        with open(log_csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'val_acc'])

    # 2. 데이터 준비
    train_path, test_path = prepare_data(CONFIG)
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['MODEL_NAME'])
    
    train_dataset = VoicePhishingDataset(train_path, tokenizer, inference_mode=False)
    test_dataset = VoicePhishingDataset(test_path, tokenizer, inference_mode=True)
    
    kwargs = {'num_workers': CONFIG['NUM_WORKERS'], 'pin_memory': True} if use_amp else {}
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, **kwargs)
    
    class_weights = calculate_class_weights(train_dataset, device)

    # 3. 모델 초기화
    logger.info("Initializing Teacher Model (12 Layers)...")
    model = DistillableRoBERTaModel(
        model_name=CONFIG['MODEL_NAME'],
        num_classes=CONFIG['NUM_CLASSES'],
        is_student=False
    ).to(device)
    
    # 4. 학습 설정
    optimizer = AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'], weight_decay=CONFIG['WEIGHT_DECAY'])
    criterion = nn.CrossEntropyLoss(weight=class_weights) # Weighted Loss 적용
    
    # [AMP] Scaler 초기화
    scaler = GradScaler(enabled=use_amp)
    
    total_steps = len(train_loader) * CONFIG['EPOCHS']
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps*0.1), total_steps)
    
    best_acc = 0.0
    start_epoch = 0 

    # Resume Checkpoint
    last_ckpt_path = os.path.join(CONFIG['OUTPUT_DIR'], "teacher_last.pt")
    if os.path.exists(last_ckpt_path):
        logger.info(f"🔄 Resuming from {last_ckpt_path}...")
        ckpt = torch.load(last_ckpt_path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1

    logger.info("Start Teacher Fine-tuning...")
    
    # 5. 학습 루프
    for epoch in range(start_epoch, CONFIG['EPOCHS']):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # [수정] AMP Autocast 문법 변경 (warning 해결)
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                outputs = model(input_ids, attention_mask)
                logits = outputs['logits']
                loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            
            # [AMP] Backward
            scaler.scale(loss).backward()
            
            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
        
        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_acc = evaluate_teacher(model, test_loader, device)
        
        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        with open(log_csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, avg_train_loss, val_loss, val_acc])
        
        save_checkpoint(model, optimizer, epoch, avg_train_loss, os.path.join(CONFIG['OUTPUT_DIR'], "teacher_last.pt"))
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(CONFIG['OUTPUT_DIR'], "teacher_best.pt")
            torch.save(model.state_dict(), best_path)
            logger.info(f"★ New Best Teacher! Saved to {best_path}")

    logger.info(f"Teacher Training Finished. Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
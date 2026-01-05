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
from architecture import DistillableRoBERTaModel, initialize_student_weights
from dataset import VoicePhishingDataset
from loss_fun import DistilBERTLoss
from utils import set_seed, prepare_data, get_logger, save_checkpoint 

def calculate_class_weights(dataset, device):
    """클래스 불균형 해소를 위한 가중치 계산"""
    print("[Info] Calculating class weights for Weighted Loss...")
    labels = [sample['label'] for sample in dataset.samples]
    
    labels_np = np.array(labels)
    classes, counts = np.unique(labels_np, return_counts=True)
    
    total_samples = len(labels_np)
    n_classes = len(classes)
    
    weights = total_samples / (n_classes * counts)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    
    print(f"   -> Class Counts: {dict(zip(classes, counts))}")
    print(f"   -> Calculated Weights: {weights_tensor.cpu().numpy()}")
    
    return weights_tensor

def evaluate_student(teacher_model, student_model, dataloader, device):
    student_model.eval()
    teacher_model.eval()
    
    total_loss = 0
    correct = 0
    total_samples = 0
    
    criterion = nn.CrossEntropyLoss() 
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Val] Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = student_model(input_ids, attention_mask)
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
    set_seed(CONFIG['SEED'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger = get_logger(CONFIG['OUTPUT_DIR'])
    logger.info(f"--- [Student Training] Device: {device} ---")
    
    use_amp = True if CONFIG['DEVICE'] == 'cuda' and torch.cuda.is_available() else False
    if use_amp:
        logger.info("⚡ AMP (Automatic Mixed Precision) Enabled.")
    
    log_csv_path = os.path.join(CONFIG['OUTPUT_DIR'], 'training_log_student.csv')
    if not os.path.exists(log_csv_path):
        with open(log_csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'val_acc'])
    
    # 데이터 준비
    train_path, test_path = prepare_data(CONFIG)
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['MODEL_NAME'])
    
    train_dataset = VoicePhishingDataset(train_path, tokenizer, inference_mode=False)
    test_dataset = VoicePhishingDataset(test_path, tokenizer, inference_mode=True)
    
    kwargs = {'num_workers': CONFIG['NUM_WORKERS'], 'pin_memory': True} if use_amp else {}
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, **kwargs)
    
    # Weighted Loss 가중치 계산
    class_weights = calculate_class_weights(train_dataset, device)
    
    # 모델 초기화
    logger.info("Initializing Models...")
    teacher_model = DistillableRoBERTaModel(CONFIG['MODEL_NAME'], CONFIG['NUM_CLASSES'], is_student=False).to(device)
    teacher_path = os.path.join(CONFIG['OUTPUT_DIR'], "teacher_best.pt")
    
    if os.path.exists(teacher_path):
        teacher_model.load_state_dict(torch.load(teacher_path, map_location=device))
        logger.info("✅ Teacher model loaded.")
    else:
        logger.error("❌ Teacher weights not found!")
        return
    
    teacher_model.eval()
    for param in teacher_model.parameters(): param.requires_grad = False

    student_model = DistillableRoBERTaModel(CONFIG['MODEL_NAME'], CONFIG['NUM_CLASSES'], is_student=True, student_layer_num=CONFIG['STUDENT_LAYER']).to(device)
    initialize_student_weights(teacher_model, student_model)
    
    # Optimizer & Scaler
    optimizer = AdamW(student_model.parameters(), lr=CONFIG['LEARNING_RATE'], weight_decay=CONFIG['WEIGHT_DECAY'])
    scaler = GradScaler(enabled=use_amp)
    
    # Loss 함수에 가중치 전달
    loss_fn = DistilBERTLoss(
        alpha=5.0, beta=2.0, gamma=1.0, temperature=2.0,
        class_weights=class_weights
    )
    
    total_steps = len(train_loader) * CONFIG['EPOCHS']
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps*0.1), total_steps)

    best_acc = 0.0
    start_epoch = 0

    # Resume Logic
    last_ckpt_path = os.path.join(CONFIG['OUTPUT_DIR'], "student_last.pt")
    if os.path.exists(last_ckpt_path):
        ckpt = torch.load(last_ckpt_path)
        student_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        logger.info(f"Resumed from Epoch {start_epoch + 1}")

    logger.info("Start Training with Weighted Loss & AMP...")
    
    for epoch in range(start_epoch, CONFIG['EPOCHS']):
        student_model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.no_grad():
                t_outputs = teacher_model(input_ids, attention_mask)
            
            # [수정] AMP Autocast 문법 변경 (warning 해결)
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                s_outputs = student_model(input_ids, attention_mask)
                loss, loss_dict = loss_fn(s_outputs, t_outputs, labels, attention_mask)

            optimizer.zero_grad()
            
            # [AMP] Backward
            scaler.scale(loss).backward()
            
            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_acc = evaluate_student(teacher_model, student_model, test_loader, device)
        
        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        with open(log_csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, avg_train_loss, val_loss, val_acc])

        save_checkpoint(student_model, optimizer, epoch, avg_train_loss, os.path.join(CONFIG['OUTPUT_DIR'], "student_last.pt"))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student_model.state_dict(), os.path.join(CONFIG['OUTPUT_DIR'], "student_best.pt"))
            logger.info(f"★ New Best Accuracy: {best_acc:.2f}%")

    logger.info("Training Finished.")

if __name__ == "__main__":
    main()
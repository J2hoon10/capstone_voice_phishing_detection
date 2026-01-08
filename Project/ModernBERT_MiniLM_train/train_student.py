import os
import csv
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.amp import GradScaler
from tqdm import tqdm

# 사용자 정의 모듈
from config import CONFIG
from architecture import ModernBertStudent
from dataset import VoicePhishingDataset
from loss_fun import LossFunction
from utils import (
    set_seed, prepare_data, get_logger, save_checkpoint, 
    calculate_class_weights, 
)

def initialize_student_from_ta(ta_model, student_model):
    """TA(12L) -> Student(6L) Skip-Layer Copy"""
    print("[Init] Copying weights from TA to Student (Skip-Layer Strategy)...")
    
    # Embedding 복사
    student_model.model.get_input_embeddings().weight.data.copy_(
        ta_model.model.get_input_embeddings().weight.data
    )
    
    # Layer 복사 (짝수 층)
    ta_layers = ta_model.model.layers if hasattr(ta_model.model, 'layers') else ta_model.model.encoder.layer
    s_layers = student_model.model.layers if hasattr(student_model.model, 'layers') else student_model.model.encoder.layer

    for i in range(len(s_layers)):
        ta_idx = i * 2 
        s_layers[i].load_state_dict(ta_layers[ta_idx].state_dict())
        
    # Classifier 복사
    if hasattr(ta_model.model, 'classifier'):
        student_model.model.classifier.load_state_dict(ta_model.model.classifier.state_dict())
    elif hasattr(ta_model.model, 'score'):
        student_model.model.score.load_state_dict(ta_model.model.score.state_dict())
    print("[Init] Done.")

def main():
    # 1. 초기 설정
    set_seed(CONFIG['SEED'])
    device = torch.device(CONFIG['DEVICE'])
    output_dir = CONFIG['OUTPUT_DIR_STUDENT']
    
    logger = get_logger(output_dir)
    logger.info(f"--- [Student Training] Device: {device} ---")

    use_amp = CONFIG['USE_AMP'] and torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)

    log_csv_path = os.path.join(output_dir, 'training_log.csv')
    if not os.path.exists(log_csv_path):
        with open(log_csv_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['epoch', 'loss', 'val_acc', 'val_loss'])

    # 2. 데이터 준비
    train_path, val_path = prepare_data(CONFIG)
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['MODEL_NAME'])
    
    train_dataset = VoicePhishingDataset(train_path, tokenizer, max_length=CONFIG['MAX_LENGTH'], aug_prob=CONFIG['AUG_PROB'])
    val_dataset = VoicePhishingDataset(val_path, tokenizer, max_length=CONFIG['MAX_LENGTH'], inference_mode=True)
    
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, 
        num_workers=CONFIG['NUM_WORKERS'], pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=CONFIG['NUM_WORKERS'])

    # 3. 모델 준비
    # (1) TA Model (Teacher 역할)
    logger.info("Loading Trained TA Model...")
    ta_teacher = ModernBertStudent(CONFIG['MODEL_NAME'], CONFIG['TA_CONFIG'], CONFIG['NUM_LABELS']).to(device)
    
    # [중요] 학습된 TA 모델 경로 (Best Model 권장)
    ta_ckpt_path = os.path.join(CONFIG['OUTPUT_DIR_TA'], "best_model.pt")
    if os.path.exists(ta_ckpt_path):
        ta_teacher.load_state_dict(torch.load(ta_ckpt_path))
    else:
        logger.warning(f"Checkpoint not found at {ta_ckpt_path}. initializing random TA (Not recommended)")
        
    ta_teacher.eval()
    for param in ta_teacher.parameters():
        param.requires_grad = False

    # (2) Student Model
    logger.info("Initializing Student Model...")
    student = ModernBertStudent(CONFIG['MODEL_NAME'], CONFIG['STUDENT_CONFIG'], CONFIG['NUM_LABELS']).to(device)
    
    # (3) Init
    initialize_student_from_ta(ta_teacher, student)

    # 4. 학습 설정
    class_weights = calculate_class_weights(train_dataset).to(device)
    criterion = LossFunction()
    criterion.ce_loss.weight = class_weights

    optimizer = AdamW(student.parameters(), lr=CONFIG['LEARNING_RATE'], weight_decay=CONFIG['WEIGHT_DECAY'])
    
    total_steps = (len(train_loader) // CONFIG['GRAD_ACCUM_STEPS']) * CONFIG['EPOCHS']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * CONFIG['WARMUP_RATIO']), num_training_steps=total_steps)

    # 5. 체크포인트 로드 (Resume)
    start_epoch = 0
    best_acc = 0.0
    last_ckpt_path = os.path.join(output_dir, "last_checkpoint.pt")
    
    if os.path.exists(last_ckpt_path):
        logger.info(f"🔄 Resuming from {last_ckpt_path}...")
        ckpt = torch.load(last_ckpt_path)
        student.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1

    # 6. 학습 루프
    logger.info(">>> Start Training Student")
    
    for epoch in range(start_epoch, CONFIG['EPOCHS']):
        # --- Train ---
        student.train()
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.no_grad():
                ta_outputs = ta_teacher(input_ids, mask)
            
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                s_outputs = student(input_ids, mask)
                # Distillation Loss
                loss = criterion.minilm_loss(s_outputs, ta_outputs, labels, alpha=0.5)
                loss = loss / CONFIG['GRAD_ACCUM_STEPS']
            
            scaler.scale(loss).backward()
            
            if (step + 1) % CONFIG['GRAD_ACCUM_STEPS'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * CONFIG['GRAD_ACCUM_STEPS']
            progress_bar.set_postfix({'loss': f"{total_loss / (step + 1):.4f}"})

        avg_train_loss = total_loss / len(train_loader)

        # --- Validation ---
        student.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="[Val]", leave=False):
                input_ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = student(input_ids, mask)
                logits = outputs.logits
                
                v_loss = criterion.hard_loss(logits, labels)
                val_loss += v_loss.item()
                
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = (correct / total) * 100
        
        logger.info(f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # CSV Logging
        with open(log_csv_path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([epoch+1, avg_train_loss, val_acc, avg_val_loss])

        # Save Last Checkpoint
        save_checkpoint(student, optimizer, scheduler, epoch, avg_train_loss, last_ckpt_path)

        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(output_dir, "best_model.pt")
            torch.save(student.state_dict(), best_path)
            logger.info(f"★ New Best Student Saved! (Acc: {val_acc:.2f}%)")

    logger.info("Student Training Finished.")

if __name__ == '__main__':
    main()
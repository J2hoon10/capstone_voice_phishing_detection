import os
import csv
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.amp import GradScaler # PyTorch 최신 버전 AMP
from tqdm import tqdm

# 사용자 정의 모듈
from config import CONFIG
from dataset import VoicePhishingDataset
from utils import set_seed, prepare_data, get_logger, save_checkpoint, calculate_class_weights

def main():
    # 1. 초기 설정
    set_seed(CONFIG['SEED'])
    device = torch.device(CONFIG['DEVICE'])
    
    logger = get_logger(CONFIG['OUTPUT_DIR'])
    logger.info(f"--- [ModernBERT Training] Device: {device} ---")
    
    # AMP 설정
    use_amp = CONFIG['USE_AMP'] and torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)
    if use_amp:
        logger.info("⚡ AMP (Automatic Mixed Precision) Enabled.")

    # CSV 로그 초기화
    log_csv_path = os.path.join(CONFIG['OUTPUT_DIR'], 'training_log.csv')
    if not os.path.exists(log_csv_path):
        with open(log_csv_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'val_f1'])

    # 2. 데이터 준비
    # utils.py의 prepare_data를 통해 ID 기준 분할 수행
    train_path, val_path = prepare_data(CONFIG)
    
    logger.info(f"Loading Tokenizer: {CONFIG['MODEL_NAME']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['MODEL_NAME'], trust_remote_code=True)

    # Dataset 로드 (dataset.py는 수정하지 않음)
    train_dataset = VoicePhishingDataset(
        file_path=train_path,
        tokenizer=tokenizer,
        max_length=CONFIG['MAX_LENGTH'],
        window_size=CONFIG['WINDOW_SIZE'],
        stride=CONFIG['STRIDE'],
        inference_mode=False
    )
    
    val_dataset = VoicePhishingDataset(
        file_path=val_path,
        tokenizer=tokenizer,
        max_length=CONFIG['MAX_LENGTH'],
        window_size=CONFIG['WINDOW_SIZE'],
        stride=CONFIG['STRIDE'],
        inference_mode=True
    )
    
    # DataLoader 설정
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['BATCH_SIZE'], 
        shuffle=True, 
        num_workers=CONFIG['NUM_WORKERS'],
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['BATCH_SIZE'], 
        shuffle=False, 
        num_workers=CONFIG['NUM_WORKERS']
    )
    
    # 3. 모델 및 학습 설정
    logger.info("Initializing ModernBERT Model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG['MODEL_NAME'],
        num_labels=CONFIG['NUM_LABELS'],
        trust_remote_code=True
    )
    model.to(device)

    # 가중치 손실 계산 (클래스 불균형 해결)
    class_weights = calculate_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'], weight_decay=CONFIG['WEIGHT_DECAY'])
    
    # 스케줄러 설정
    total_steps = (len(train_loader) // CONFIG['GRAD_ACCUM_STEPS']) * CONFIG['EPOCHS']
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * CONFIG['WARMUP_RATIO']), 
        num_training_steps=total_steps
    )

    # 4. 체크포인트 로드 (Resume)
    start_epoch = 0
    best_acc = 0.0
    last_ckpt_path = os.path.join(CONFIG['OUTPUT_DIR'], "last_checkpoint.pt")
    
    if os.path.exists(last_ckpt_path):
        logger.info(f"🔄 Resuming from {last_ckpt_path}...")
        ckpt = torch.load(last_ckpt_path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1

    # 5. 학습 루프
    logger.info("Start Training...")
    
    for epoch in range(start_epoch, CONFIG['EPOCHS']):
        # --- Train ---
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        
        for step, batch in enumerate(progress_bar):
            # ModernBERT 호환: token_type_ids 제거 및 device 이동
            inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            labels = batch['labels'].to(device)
            
            # AMP Forward
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                outputs = model(**inputs)
                logits = outputs.logits
                loss = criterion(logits, labels)
                
                # Gradient Accumulation (메모리 절약)
                loss = loss / CONFIG['GRAD_ACCUM_STEPS']

            # AMP Backward
            scaler.scale(loss).backward()
            
            if (step + 1) % CONFIG['GRAD_ACCUM_STEPS'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * CONFIG['GRAD_ACCUM_STEPS']
            progress_bar.set_postfix(loss=f"{loss.item() * CONFIG['GRAD_ACCUM_STEPS']:.4f}")

        avg_train_loss = total_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="[Val]", leave=False):
                inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
                labels = batch['labels'].to(device)
                
                outputs = model(**inputs)
                logits = outputs.logits
                
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = (correct / total_samples) * 100
        
        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {val_acc:.2f}%")
        
        # CSV 기록
        with open(log_csv_path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([epoch+1, avg_train_loss, avg_val_loss, val_acc])
        
        # 체크포인트 저장 (Last)
        save_checkpoint(model, optimizer, scheduler, epoch, avg_train_loss, last_ckpt_path)
        
        # 최고 성능 모델 저장 (Best)
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(CONFIG['OUTPUT_DIR'], "best_model.pt")
            
            # 모델 가중치만 저장 (용량 절약)
            if hasattr(model, 'module'):
                torch.save(model.module.state_dict(), best_path)
            else:
                torch.save(model.state_dict(), best_path)
                
            logger.info(f"★ New Best Model Saved! (Acc: {val_acc:.2f}%)")

    logger.info("Training Finished.")

if __name__ == "__main__":
    main()
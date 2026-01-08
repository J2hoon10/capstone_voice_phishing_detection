import os
import csv
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.amp import GradScaler 
from tqdm import tqdm

# 사용자 정의 모듈
from config import CONFIG
from architecture import ModernBertStudent # TA는 Wrapper 클래스 사용
from dataset import VoicePhishingDataset
from loss_fun import LossFunction
from utils import (
    set_seed, prepare_data, get_logger, save_checkpoint, 
    calculate_class_weights, get_projection_matrix, initialize_student_hybrid
)

def main():
    # 1. 초기 설정
    set_seed(CONFIG['SEED'])
    device = torch.device(CONFIG['DEVICE'])
    output_dir = CONFIG['OUTPUT_DIR_TA'] # TA 모델 저장 경로
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logger = get_logger(output_dir)
    logger.info(f"--- [TA Training] Device: {device} ---")

    # AMP 설정
    use_amp = CONFIG['USE_AMP'] and torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)

    # CSV 로그 초기화
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
    
    # (1) Teacher (Freeze)
    # [수정] Config 경로 사용 및 안전한 로딩
    logger.info("Loading Teacher Model...")
    
    teacher_ckpt_path = os.path.join(CONFIG['OUTPUT_DIR_TEACHER'], 'best_model.pt')
    
    # 만약 Teacher 폴더에 파일이 없다면, 사용자가 이전에 저장했을 법한 경로(./ModernBERT_Result)도 확인 (유연성)
    if not os.path.exists(teacher_ckpt_path):
        fallback_path = os.path.join('./ModernBERT_Result', 'best_model.pt')
        if os.path.exists(fallback_path):
            logger.warning(f"Config path empty. Found checkpoint at fallback: {fallback_path}")
            teacher_ckpt_path = fallback_path
        else:
            raise FileNotFoundError(f"Teacher Checkpoint not found at: {teacher_ckpt_path}. Please train the teacher first.")

    # 모델 구조 초기화 (Raw AutoModel 사용)
    teacher = AutoModelForSequenceClassification.from_pretrained(
        CONFIG['MODEL_NAME'], 
        num_labels=CONFIG['NUM_LABELS'],
        trust_remote_code=True,
        output_attentions=True, 
        output_hidden_states=True
    ).to(device)
    
    # 가중치 로드
    logger.info(f"Loading Teacher weights from: {teacher_ckpt_path}")
    checkpoint = torch.load(teacher_ckpt_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        teacher.load_state_dict(checkpoint['model_state_dict'])
    else:
        teacher.load_state_dict(checkpoint)

    # Teacher 모델 Freeze
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # (2) TA (Init)
    logger.info("Initializing Teacher Assistant...")
    ta_model = ModernBertStudent(CONFIG['MODEL_NAME'], CONFIG['TA_CONFIG'], CONFIG['NUM_LABELS']).to(device)
    
    # (3) SVD + Hybrid Init
    # Teacher의 Embedding을 SVD하여 초기화
    v_proj = get_projection_matrix(teacher.get_input_embeddings().weight.data, CONFIG['TA_CONFIG']['hidden_size'])
    v_proj = v_proj.to(device)
    
    # utils.py의 수정된 initialize_student_hybrid 호출 (Wrapper/Raw 자동 처리)
    initialize_student_hybrid(teacher, ta_model, v_proj)

    # 4. 학습 설정
    class_weights = calculate_class_weights(train_dataset).to(device)
    criterion = LossFunction()
    criterion.ce_loss.weight = class_weights 

    optimizer = AdamW(ta_model.parameters(), lr=CONFIG['LEARNING_RATE'], weight_decay=CONFIG['WEIGHT_DECAY'])
    
    total_steps = (len(train_loader) // CONFIG['GRAD_ACCUM_STEPS']) * CONFIG['EPOCHS']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * CONFIG['WARMUP_RATIO']), num_training_steps=total_steps)

    # 5. 체크포인트 로드 (Resume for TA)
    start_epoch = 0
    best_acc = 0.0
    last_ckpt_path = os.path.join(output_dir, "last_checkpoint.pt")
    
    if os.path.exists(last_ckpt_path):
        logger.info(f"🔄 Resuming TA training from {last_ckpt_path}...")
        ckpt = torch.load(last_ckpt_path)
        ta_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1

    # 6. 학습 루프
    logger.info(">>> Start Training TA")
    
    for epoch in range(start_epoch, CONFIG['EPOCHS']):
        # --- Train ---
        ta_model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Teacher Forward (No Grad)
            with torch.no_grad():
                t_outputs = teacher(input_ids=input_ids, attention_mask=mask)
            
            # TA Forward
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                s_outputs = ta_model(input_ids, mask)
                # MiniLM Loss
                loss = criterion.minilm_loss(s_outputs, t_outputs, labels, alpha=0.5)
                loss = loss / CONFIG['GRAD_ACCUM_STEPS']
            
            scaler.scale(loss).backward()
            
            if (step + 1) % CONFIG['GRAD_ACCUM_STEPS'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ta_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * CONFIG['GRAD_ACCUM_STEPS']
            progress_bar.set_postfix({'loss': f"{total_loss / (step + 1):.4f}"})

        avg_train_loss = total_loss / len(train_loader)

        # --- Validation ---
        ta_model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="[Val]", leave=False):
                input_ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = ta_model(input_ids, mask)
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

        # Save Checkpoint
        save_checkpoint(ta_model, optimizer, scheduler, epoch, avg_train_loss, last_ckpt_path)

        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(output_dir, "best_model.pt")
            torch.save(ta_model.state_dict(), best_path)
            logger.info(f"★ New Best TA Saved! (Acc: {val_acc:.2f}%)")

    logger.info("TA Training Finished.")

if __name__ == '__main__':
    main()
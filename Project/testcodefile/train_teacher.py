import os
import torch
import torch.nn as nn
import csv
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm  # 진행률 표시 바

# 우리가 만든 모듈들
from config import CONFIG
from architecture import DistillableRoBERTaModel
from dataset import VoicePhishingDataset
from utils import set_seed, prepare_data, get_logger, save_checkpoint

def evaluate_teacher(model, dataloader, device):
    """선생님 모델 성능 평가 함수"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    correct = 0
    total_samples = 0
    
    # 검증 진행률 바 생성
    progress_bar = tqdm(dataloader, desc="[Val] Evaluating", leave=False)
    
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 선생님 모델 추론
            outputs = model(input_ids, attention_mask)
            logits = outputs['logits']
            
            # Loss 및 정확도 계산
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
    
    # 로거 설정
    logger = get_logger(CONFIG['OUTPUT_DIR'])
    logger.info(f"--- [Teacher Training] Device: {device} ---")
    
    if not os.path.exists(CONFIG['OUTPUT_DIR']):
        os.makedirs(CONFIG['OUTPUT_DIR'])
        
    # 로그 CSV 생성 (이어하기 시에는 append 모드로 열어야 하지만, 
    # 간단하게 헤더 중복 방지 처리를 하거나 파일이 없을 때만 생성하게 함)
    log_csv_path = os.path.join(CONFIG['OUTPUT_DIR'], 'training_log_teacher.csv')
    if not os.path.exists(log_csv_path):
        with open(log_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc'])

    # 2. 데이터 준비
    train_path, test_path = prepare_data(CONFIG)
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['MODEL_NAME'])
    
    # 선생님은 증강된 데이터(KoEDA)로 강하게 키웁니다.
    train_dataset = VoicePhishingDataset(train_path, tokenizer, inference_mode=False)
    test_dataset = VoicePhishingDataset(test_path, tokenizer, inference_mode=True)
    
    kwargs = {'num_workers': CONFIG['NUM_WORKERS'], 'pin_memory': True} if torch.cuda.is_available() else {}

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, **kwargs)
    
    # 3. 모델 초기화
    logger.info("Initializing Teacher Model (12 Layers)...")
    model = DistillableRoBERTaModel(
        model_name=CONFIG['MODEL_NAME'],
        num_classes=CONFIG['NUM_CLASSES'],
        is_student=False
    ).to(device)
    
    # 4. 학습 설정
    optimizer = AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'])
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    start_epoch = 0 # 기본 시작점 (처음부터)

    # -------------------------------------------------------------------------
    # [Resume Logic] 체크포인트가 있으면 불러오기
    # -------------------------------------------------------------------------
    last_ckpt_path = os.path.join(CONFIG['OUTPUT_DIR'], "teacher_last.pt")
    if os.path.exists(last_ckpt_path):
        logger.info(f"🔄 Found checkpoint at '{last_ckpt_path}'. Loading...")
        try:
            checkpoint = torch.load(last_ckpt_path)
            
            # 1) 가중치 복원
            model.load_state_dict(checkpoint['model_state_dict'])
            # 2) 옵티마이저 복원
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # 3) 시작 에포크 복원 (저장된 에포크가 끝난 상태이므로 +1)
            start_epoch = checkpoint['epoch'] + 1
            
            logger.info(f"✅ Successfully resumed training from Epoch {start_epoch + 1}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load checkpoint. Error: {e}")
            logger.warning("   -> Starting from scratch (Epoch 1).")
    else:
        logger.info("🆕 No checkpoint found. Starting from scratch.")

    logger.info("Start Teacher Fine-tuning...")
    
    # 5. 학습 루프 (range 시작점 변경)
    for epoch in range(start_epoch, CONFIG['EPOCHS']):
        model.train()
        total_loss = 0
        
        # [Train] 진행률 바 생성
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']} [Train]", leave=False)
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            logits = outputs['logits']
            
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 실시간 Loss 표시
            progress_bar.set_postfix(loss=loss.item())
        
        avg_train_loss = total_loss / len(train_loader)
        
        # [Validation] 성능 평가
        val_loss, val_acc = evaluate_teacher(model, test_loader, device)
        
        # [Log] 결과 기록
        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        with open(log_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_train_loss, val_loss, val_acc])
        
        # [Save] 체크포인트 저장
        # 1. Last Checkpoint (재개용 - 모델 + 옵티마이저)
        save_checkpoint(
            model, optimizer, epoch, avg_train_loss,
            os.path.join(CONFIG['OUTPUT_DIR'], "teacher_last.pt")
        )
        
        # 2. Best Model (최고 성능용 - 가중치만)
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(CONFIG['OUTPUT_DIR'], "teacher_best.pt")
            torch.save(model.state_dict(), best_path)
            logger.info(f"★ New Best Teacher! Saved to {best_path}")

    logger.info(f"Teacher Training Finished. Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
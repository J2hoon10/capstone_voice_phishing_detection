import os
import torch
import csv
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import CONFIG
from architecture import DistillableRoBERTaModel, initialize_student_weights
from dataset import VoicePhishingDataset
from trainer import DistillationTrainer
from utils import set_seed, prepare_data, get_logger, save_checkpoint 

def main():
    set_seed(CONFIG['SEED'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 로거 초기화
    logger = get_logger(CONFIG['OUTPUT_DIR'])
    logger.info(f"--- [Student Training] Device: {device} ---")
    
    # CSV 로그 파일 설정 (이어하기 고려: 파일이 없을 때만 헤더 작성)
    log_csv_path = os.path.join(CONFIG['OUTPUT_DIR'], 'training_log_student.csv')
    if not os.path.exists(log_csv_path):
        with open(log_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc'])
    
    # 2. 데이터 준비
    train_path, test_path = prepare_data(CONFIG)
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['MODEL_NAME'])
    
    train_dataset = VoicePhishingDataset(train_path, tokenizer, inference_mode=False)
    test_dataset = VoicePhishingDataset(test_path, tokenizer, inference_mode=True)
    
    kwargs = {'num_workers': CONFIG['NUM_WORKERS'], 'pin_memory': True} if torch.cuda.is_available() else {}

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, **kwargs)
    
    # 3. 모델 초기화
    logger.info("Initializing Models...")
    
    # (1) 선생님 모델 로드 (Teacher는 항상 Best 모델을 불러옵니다)
    teacher_model = DistillableRoBERTaModel(
        model_name=CONFIG['MODEL_NAME'], 
        num_classes=CONFIG['NUM_CLASSES'], 
        is_student=False
    )
    teacher_path = os.path.join(CONFIG['OUTPUT_DIR'], "teacher_best.pt")
    
    if os.path.exists(teacher_path):
        teacher_model.load_state_dict(torch.load(teacher_path, map_location=device))
        logger.info("✅ Teacher model loaded successfully.")
    else:
        raise FileNotFoundError(f"Teacher weights not found at {teacher_path}!")
    
    # (2) 학생 모델 초기화
    student_model = DistillableRoBERTaModel(
        model_name=CONFIG['MODEL_NAME'], 
        num_classes=CONFIG['NUM_CLASSES'], 
        is_student=True, 
        student_layer_num=CONFIG['STUDENT_LAYER']
    )
    
    # 기본적으로 선생님 가중치로 초기화 (Resume 시에는 아래에서 덮어씌워짐)
    initialize_student_weights(teacher_model, student_model)
    
    # 4. 트레이너 및 옵티마이저 설정
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        train_dataloader=train_loader,
        device=device,
        lr=CONFIG['LEARNING_RATE']
    )
    
    best_acc = 0.0
    start_epoch = 0 # 기본 시작점

    # =================================================================
    # [Resume Logic] 학생 모델 이어하기 기능
    # =================================================================
    last_ckpt_path = os.path.join(CONFIG['OUTPUT_DIR'], "student_last.pt")
    
    if os.path.exists(last_ckpt_path):
        logger.info(f"🔄 Found checkpoint at '{last_ckpt_path}'. Loading...")
        try:
            checkpoint = torch.load(last_ckpt_path)
            
            # 1) 학생 모델 가중치 복구
            student_model.load_state_dict(checkpoint['model_state_dict'])
            
            # 2) 옵티마이저 상태 복구 (Trainer 안에 있는 optimizer 접근)
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 3) 시작 에포크 복구
            start_epoch = checkpoint['epoch'] + 1
            
            logger.info(f"✅ Successfully resumed Student training from Epoch {start_epoch + 1}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load checkpoint: {e}. Starting from scratch.")
    else:
        logger.info("🆕 No checkpoint found. Starting Student training from scratch.")

    logger.info("Start Training...")
    
    # 5. 학습 루프 (start_epoch 부터 시작)
    for epoch in range(start_epoch, CONFIG['EPOCHS']):
        # 1) 학습
        train_loss = trainer.train_epoch(epoch)
        
        # 2) 검증
        val_loss, val_acc = trainer.evaluate(test_loader)
        
        # 3) 로그 기록
        logger.info(f"Epoch {epoch+1}/{CONFIG['EPOCHS']} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        with open(log_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, val_loss, val_acc])

        # 4) 체크포인트 저장
        # [Last] 재개용 저장 (Optimizer 포함)
        save_checkpoint(
            student_model, trainer.optimizer, epoch, train_loss,
            os.path.join(CONFIG['OUTPUT_DIR'], "student_last.pt")
        )

        # [Best] 최고 성능용 저장 (모델 가중치만)
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(CONFIG['OUTPUT_DIR'], "student_best.pt")
            torch.save(student_model.state_dict(), best_path)
            logger.info(f"★ New Best Accuracy! Model saved to {best_path}")

    logger.info(f"All Finished. Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
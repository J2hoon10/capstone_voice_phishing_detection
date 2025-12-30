# build_main.py
import torch
import config as CONFIG
from dataset_builder import DatasetBuilder
from csv_merger import CsvMerger  # [신규] 병합 모듈 임포트

def main():
    # GPU 체크 및 정보 출력
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- [System] Running on {device.upper()} ---")

    print(f"--- [Config] Settings ---")
    print(f"    BASE_DIR: {CONFIG.BASE_DIR}")
    print(f"    PHISHING_CSV: {CONFIG.CSV_PATH}") # 수정된 변수명 확인
    print(f"    MASTER_CSV: {CONFIG.MASTER_CSV_PATH}") # 수정된 변수명 확인
    
    # ---------------------------------------------------------
    # Phase 1: 오디오 처리 (보이스피싱 데이터 구축)
    # ---------------------------------------------------------
    builder = DatasetBuilder(CONFIG)
    
    try:
        # [Step 1] 오디오 -> 텍스트 변환 (dataset_voice_phishing.csv 생성)
        builder.build_dataset()
        
        # -----------------------------------------------------
        # Phase 2: 데이터 병합 (보이스피싱 + KorCCVi)
        # -----------------------------------------------------
        print("\n--- [System] Starting Data Merge Phase ---")
        
        # [Step 2] 별도 모듈을 통해 병합 수행 (dataset_master.csv 생성)
        merger = CsvMerger()
        merger.merge()
        
        print("\n🎉 [Main] All processes (Build + Merge) completed successfully.")
        
    except KeyboardInterrupt:
        print("\n🛑 [Main] Process interrupted by user.")
    except Exception as e:
        print(f"\n❌ [Main] Critical Error: {e}")

if __name__ == "__main__":
    main()
import os
import re
import csv
import pandas as pd
from tqdm import tqdm
from audio_processor import AudioProcessor

class DatasetBuilder:
    def __init__(self, config):
        self.base_dir = config.BASE_DIR
        self.input_folders = config.INPUT_FOLDERS
        self.output_dir = config.OUTPUT_DIR
        self.csv_path = config.CSV_PATH
        self.folder_label_map = config.FOLDER_LABEL_MAP
        self.folder_class_map = config.FOLDER_CLASS_MAP
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.processor = AudioProcessor(whisper_model_size=config.WHISPER_MODEL_SIZE)

    def get_existing_progress(self):
        """
        CSV를 읽어 다음 ID, 처리된 파일 목록, 그리고 [마지막 작업 폴더]를 반환합니다.
        """
        if not os.path.exists(self.csv_path):
            return 1, set(), None # 마지막 폴더 없음

        try:
            df = pd.read_csv(self.csv_path)
            if df.empty: return 1, set(), None

            last_idx = df.index[-1]
            last_row = df.iloc[-1]
            
            # --- 무결성 검사 (기존과 동일) ---
            required_cols = ['ID', 'script', 'label'] 
            is_complete = True
            for col in required_cols:
                if pd.isna(last_row[col]):
                    is_complete = False
                    break
            
            if is_complete and str(last_row['script']).strip() == "":
                is_complete = False

            # 불완전한 마지막 행 삭제 로직
            if not is_complete:
                print(f"🔄 마지막 행(ID: {last_row['ID']}) 삭제 후 재시작")
                df = df.drop(last_idx)
                df.to_csv(self.csv_path, index=False, encoding='utf-8-sig')
                
                if df.empty:
                    return 1, set(), None
                
                # 삭제 후 다시 마지막 행 정보 갱신
                last_row = df.iloc[-1]

            # --- [핵심] 마지막 작업 폴더 추출 ---
            last_id_str = last_row['ID']
            next_index = int(last_id_str.split('_')[1]) + 1
            
            # filename 예시: "수사기관 사칭형/1.wav"
            last_filename = str(last_row['filename'])
            
            # 경로 구분자(/ 또는 \)를 기준으로 폴더명 추출
            # os.path.dirname을 쓰거나 split을 사용
            last_folder = os.path.dirname(last_filename)
            if not last_folder: # 혹시 폴더 경로 없이 파일명만 있는 경우 대비
                last_folder = last_filename.split('/')[0] if '/' in last_filename else None

            print(f"✅ 이어하기: ID P_{next_index:04d} 부터 시작 (마지막 폴더: {last_folder})")

            processed_files = set(df['filename'].astype(str).values)
            
            return next_index, processed_files, last_folder

        except Exception as e:
            print(f"⚠️ CSV 읽기 오류 (새로 시작): {e}")
            return 1, set(), None

    def build_dataset(self):
        print(f"🚀 데이터셋 구축 시작 (Output: {self.csv_path})")
        
        # 1. 이어하기 정보 가져오기 (마지막 폴더 포함)
        global_index, processed_files, last_processed_folder = self.get_existing_progress()

        file_exists = os.path.exists(self.csv_path)
        
        with open(self.csv_path, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            
            if not file_exists or os.path.getsize(self.csv_path) == 0:
                writer.writerow(['ID', 'script', 'label', 'class', 'filename'])

            # [핵심] 폴더 건너뛰기 로직을 위한 플래그
            # last_processed_folder가 있으면 True(건너뛰기 모드), 없으면 False(처음부터 시작)
            skip_mode = True if last_processed_folder else False

            for folder_name in self.input_folders:
                
                # --- [Folder Skip Logic] ---
                if skip_mode:
                    if folder_name == last_processed_folder:
                        print(f"📍 마지막 작업 폴더 발견: {folder_name} (여기서부터 재개합니다)")
                        skip_mode = False # 건너뛰기 해제, 이 폴더부터 처리 시작
                    else:
                        # 아직 마지막 폴더에 도달하지 못했으므로 통과
                        # print(f"⏩ 건너뜀: {folder_name} (이미 완료됨)") 
                        continue 
                # ---------------------------

                folder_path = os.path.join(self.base_dir, folder_name)
                
                if not os.path.exists(folder_path):
                    print(f"⚠️ 폴더 없음: {folder_name}")
                    continue

                file_list = [f for f in os.listdir(folder_path) if f.lower().endswith((".mp3", ".wav"))]
                # 파일 정렬 (1.wav, 2.wav ...)
                file_list.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

                current_label = self.folder_label_map.get(folder_name, 1)
                current_class = self.folder_class_map.get(folder_name, None)

                print(f"\n📂 Processing {folder_name} (Label: {current_label})")

                for filename in tqdm(file_list, desc=folder_name):
                    
                    # 파일명 생성: "폴더명/파일명.wav"
                    relative_filename = os.path.join(folder_name, filename).replace("\\", "/")

                    # 이미 처리된 파일이면 건너뜀
                    # (마지막 폴더 내부에서도, 이미 한 파일은 건너뛰어야 하므로 필요)
                    if relative_filename in processed_files:
                        continue

                    audio_path = os.path.join(folder_path, filename)
                    
                    # --- STT 변환 ---
                    sentences = self.processor.process_file(audio_path)
                    full_script = " ".join(sentences).strip()
                    
                    if full_script:
                        file_id = f"P_{global_index:04d}"
                        
                        writer.writerow([
                            file_id,
                            full_script,
                            current_label,
                            current_class,
                            relative_filename
                        ])
                        
                        f.flush()
                        global_index += 1
                        processed_files.add(relative_filename)

        print(f"\n✅ [완료] 모든 작업이 끝났습니다.")
        print(f"   - 저장 경로: {os.path.abspath(self.csv_path)}")
# csv_merger.py
import pandas as pd
import os
import config as CONFIG

class CsvMerger:
    """
    [역할]
    1. STT로 생성된 'dataset_voice_phishing.csv' 로드
    2. 외부 'KorCCVi_v2.csv' 로드 및 전처리 (label=0 필터링)
    3. 두 데이터를 병합하여 'dataset_master.csv' 생성
    """
    def __init__(self):
        self.phishing_path = CONFIG.CSV_PATH         # 보이스피싱 데이터
        self.normal_path = CONFIG.NORMAL_CSV_PATH    # KorCCVi_v2 데이터
        self.output_path = CONFIG.MASTER_CSV_PATH    # 최종 저장 경로
        self.mapping = CONFIG.NORMAL_COLUMN_MAPPING  # 컬럼 매핑 정보

    def merge(self):
        print(f"\n🧩 [Merger] 데이터 병합 시작...")

        # 1. 보이스피싱 데이터 로드
        if not os.path.exists(self.phishing_path):
            print(f"❌ [Merger] 보이스피싱 데이터가 없습니다: {self.phishing_path}")
            return
        
        try:
            df_phishing = pd.read_csv(self.phishing_path, encoding='utf-8-sig')
            print(f"    - 보이스피싱 데이터: {len(df_phishing)} 건 로드")
        except Exception as e:
            print(f"❌ [Merger] 보이스피싱 CSV 로드 실패: {e}")
            return

        # 2. 일반 대화 데이터(KorCCVi) 로드
        if not os.path.exists(self.normal_path):
            print(f"⚠️ [Merger] 일반 대화 데이터 파일을 찾을 수 없습니다: {self.normal_path}")
            print("    -> 병합 없이 보이스피싱 데이터만 최종 저장합니다.")
            df_phishing.to_csv(self.output_path, index=False, encoding='utf-8-sig')
            return

        try:
            # KorCCVi_v2는 보통 utf-8 혹은 cp949 인코딩을 사용함
            try:
                df_normal_raw = pd.read_csv(self.normal_path, encoding='utf-8-sig')
            except UnicodeDecodeError:
                df_normal_raw = pd.read_csv(self.normal_path, encoding='cp949')
            
            print(f"    - KorCCVi 원본 데이터: {len(df_normal_raw)} 건 로드")
            
        except Exception as e:
            print(f"❌ [Merger] 일반 대화 CSV 로드 실패: {e}")
            return

        # 3. 일반 대화 데이터 전처리 (포맷 통일)
        df_normal = self._preprocess_korccvi(df_normal_raw)
        
        # 4. 병합 (위: 보이스피싱, 아래: 일반)
        if not df_normal.empty:
            df_final = pd.concat([df_phishing, df_normal], ignore_index=True)
            print(f"    - 병합 완료: 일반 대화 {len(df_normal)} 건 추가됨")
        else:
            df_final = df_phishing
            print("    ⚠️ 추가될 일반 대화 데이터가 없습니다.")

        # 5. 최종 저장
        df_final.to_csv(self.output_path, index=False, encoding='utf-8-sig')
        print(f"🎉 [Merger] 최종 데이터셋 저장 완료: {os.path.abspath(self.output_path)}")
        print(f"    - 총 데이터 개수: {len(df_final)} (Phishing: {len(df_phishing)} + Normal: {len(df_normal)})")

    def _preprocess_korccvi(self, df):
        """
        KorCCVi_v2 데이터를 우리 데이터셋 스키마(ID, script, label, class)에 맞춤
        """
        # (1) Label 0 (일반 대화) 필터링
        if 'label' in df.columns:
            # 안전하게 문자열/숫자 모두 처리하기 위해 astype 사용 가능하나, 일단 조건 검색
            df_filtered = df[df['label'] == 0].copy()
        else:
            print("⚠️ [Merger] KorCCVi 데이터에 'label' 컬럼이 없어 필터링을 건너뜁니다.")
            df_filtered = df.copy()

        new_df = pd.DataFrame()

        # (2) Script 컬럼 매핑 (transcript -> script)
        source_text_col = self.mapping.get("text", "transcript") # config에서 설정한 이름
        if source_text_col in df_filtered.columns:
            new_df['script'] = df_filtered[source_text_col]
        else:
            print(f"⚠️ [Merger] '{source_text_col}' 컬럼을 찾을 수 없습니다.")
            return pd.DataFrame() # 빈 데이터프레임 반환

        # (3) 필수 컬럼 생성
        new_df['label'] = 0                # 일반 대화는 무조건 0
        new_df['class'] = "normal"         # 클래스 명시
        
        # (4) Filename (KorCCVi에 파일명이 없다면 원본 ID나 빈값 사용)
        source_id_col = self.mapping.get("id", "id")
        if source_id_col in df_filtered.columns:
            new_df['filename'] = df_filtered[source_id_col] # 추적을 위해 원본 ID를 filename에 저장
        else:
            new_df['filename'] = None

        # (5) ID 생성 (N_0001, N_0002 ...)
        # 보이스피싱(P_)과 구분하기 위해 N_ 접두사 사용
        ids = [f"N_{i+1:04d}" for i in range(len(new_df))]
        new_df.insert(0, 'ID', ids)

        return new_df
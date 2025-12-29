# csv_merger.py (신규 생성)
import pandas as pd
import os
import config as CONFIG

class CsvMerger:
    """
    [역할]
    1. STT로 생성된 '보이스피싱 데이터셋' 로드
    2. 외부에 존재하는 '일반 대화 데이터셋' 로드
    3. 두 데이터를 규격(ID, script, label, class)에 맞춰 병합
    """
    def __init__(self):
        self.phishing_path = CONFIG.CSV_PATH
        self.normal_path = CONFIG.NORMAL_CSV_PATH
        self.output_path = CONFIG.FINAL_DATASET_PATH
        self.mapping = CONFIG.NORMAL_COLUMN_MAPPING

    def merge(self):
        print(f"\n🧩 [Merger] 데이터 병합 시작...")

        # 1. 보이스피싱 데이터 로드
        if not os.path.exists(self.phishing_path):
            print(f"❌ [Merger] 보이스피싱 데이터가 없습니다: {self.phishing_path}")
            return
        
        df_phishing = pd.read_csv(self.phishing_path, encoding='utf-8-sig')
        print(f"    - 보이스피싱 데이터: {len(df_phishing)} 건 로드 완료")

        # 2. 일반 대화 데이터 로드
        if not os.path.exists(self.normal_path):
            print(f"❌ [Merger] 일반 대화 데이터가 없습니다: {self.normal_path}")
            # 일반 데이터가 없으면 보이스피싱 데이터만이라도 저장할지 결정 (여기선 중단)
            return

        # 인코딩은 파일에 따라 cp949 혹은 utf-8 일 수 있음. 에러 시 try-except 처리 권장
        try:
            df_normal_raw = pd.read_csv(self.normal_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df_normal_raw = pd.read_csv(self.normal_path, encoding='cp949')
            
        print(f"    - 일반 대화 데이터(Raw): {len(df_normal_raw)} 건 로드 완료")

        # 3. 일반 대화 데이터 전처리 (포맷 통일)
        df_normal = self._preprocess_normal_data(df_normal_raw)
        
        # 4. 병합 (위: 보이스피싱, 아래: 일반)
        df_final = pd.concat([df_phishing, df_normal], ignore_index=True)
        
        # 5. 최종 저장
        df_final.to_csv(self.output_path, index=False, encoding='utf-8-sig')
        print(f"🎉 [Merger] 최종 데이터셋 저장 완료: {self.output_path}")
        print(f"    - 총 데이터: {len(df_final)} 건 (피싱: {len(df_phishing)} + 일반: {len(df_normal)})")

    def _preprocess_normal_data(self, df):
        """일반 데이터를 우리 데이터셋 스키마(ID, script, label, class)에 맞춤"""
        new_df = pd.DataFrame()

        # (1) 스크립트 복사 (컬럼 매핑 이용)
        source_col = self.mapping.get("text", "script") # config에 설정된 이름 혹은 기본값 'script'
        if source_col in df.columns:
            new_df['script'] = df[source_col]
        else:
            print(f"⚠️ [Merger] 일반 데이터에 '{source_col}' 컬럼이 없습니다. 확인 필요!")
            new_df['script'] = ""

        # (2) 라벨링 (일반 대화 = 0)
        new_df['label'] = 0

        # (3) 클래스 (Normal)
        new_df['class'] = "normal"

        # (4) 파일명 (원본 파일명이 있다면 유지, 없으면 None)
        new_df['filename'] = df['filename'] if 'filename' in df.columns else None

        # (5) ID 생성 (N_0001, N_0002 ...)
        # 보이스피싱은 P_xxxx 였으므로 구분하기 쉽게 N_xxxx로 설정
        ids = [f"N_{i+1:04d}" for i in range(len(new_df))]
        new_df.insert(0, 'ID', ids)

        return new_df
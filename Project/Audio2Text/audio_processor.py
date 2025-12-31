# 오디오 데이터를 텍스트 데이터로 변환하는 Audio2Text 모듈
# 민감 정보(PII) 마스킹 및 문장 내 반복 제거 기능 추가
# 1. <RRN> (Resident Registration Number)
# 2. <PHONE> (Phone Number)
# 3. <ACCOUNT> (Bank Account Number)

import os
import re
# import tempfile # [제거] 임시 파일 생성 불필요
import torch
from faster_whisper import WhisperModel
from audio_enhancer import AudioEnhancer

class AudioProcessor:
    def __init__(self, whisper_model_size="deepdml/faster-whisper-large-v3-turbo-ct2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if self.device == "cuda" else "int8"

        self.enhancer = AudioEnhancer()
        print(f"[AudioProcessor] Loading Whisper '{whisper_model_size}'...")
        self.whisper = WhisperModel(whisper_model_size, device=self.device, compute_type=compute_type)

        # 기존 블랙리스트 유지
        self.blacklist = [
            "correct", "hate-speech", "generated", "silence", "subtitles", "transcript", 
            "MBC", "SBS", "KBS", "YTN", "뉴스", 
            "자막", "배달의 민족", "저작권", "시청해 주셔서", "구독과 좋아요", "알림 설정", 
            "Provided by", "Auto-generated", "공인이 진행 의사", "오디오 자막",
            "허허허", "하하하"
        ]

    # [★신규 기능] 민감 정보(PII) 마스킹 -> 모델 일반화 성능 향상
    def mask_pii(self, text: str) -> str:
        # 1. 주민등록번호 (6자리-7자리)
        text = re.sub(r'\d{6}[-\s]*[1-4]\d{6}', '<RRN>', text)
        
        # 2. 전화번호 (010-XXXX-XXXX, 02-XXX-XXXX 등)
        text = re.sub(r'01[016789][-\s]*\d{3,4}[-\s]*\d{4}', '<PHONE>', text)
        text = re.sub(r'0(2|3[1-3]|4[1-4]|5[1-5]|6[1-4])[-\s]*\d{3,4}[-\s]*\d{4}', '<PHONE>', text)

        # 3. 계좌번호 (숫자가 10자리 이상 연속되면 계좌로 의심)
        text = re.sub(r'\d{10,}', '<ACCOUNT>', text)
        
        # 4. 금액 (숫자 + 만원/천원/억) -> 단순화
        # 예: "삼천만 원" -> "3000만 원" 변환은 어렵지만, 패턴을 <MONEY>로 바꿀 수 있음
        # 여기서는 모델이 금액 단위를 배우는 게 중요할 수 있으므로 마스킹보다는 유지 추천
        # 대신 너무 긴 숫자는 <NUMBER>로 치환 가능
        return text

    # [★개선 기능] 문장 내 반복 문구 제거 (N-gram Repetition)
    def remove_phrase_repetition(self, text: str) -> str:
        """
        "여보세요 여보세요 여보세요" 처럼 같은 단어가 3회 이상 반복되면 1회로 줄임
        """
        # 단어 단위로 쪼개서 3번 이상 연속 반복되는 패턴 찾기
        # (regex: (\b\w+\b)( \1){2,}) -> 단어(공백+단어)가 2번 이상 더 반복됨
        text = re.sub(r'(\b\w+\b)( \1){2,}', r'\1', text)
        return text

    def clean_text_basic(self, text: str) -> str:
        """기본 정제 + 구두점 정리 + PII 마스킹 + 반복 제거"""
        # 1. 기본적인 특수문자 제거
        text = re.sub(r'[^0-9a-zA-Z가-힣\s.,?!]', '', text)
        
        # 2. 구두점 정리
        text = re.sub(r'([?!])\s*\.+', r'\1', text)
        text = re.sub(r'\.+\s*([?!])', r'\1', text)
        text = re.sub(r'([?!.])\1+', r'\1', text)
        
        # 3. [추가] 과도한 공백 줄이기
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 4. [추가] PII 마스킹 적용
        text = self.mask_pii(text)
        
        # 5. [추가] 문장 내 단어 반복 제거
        text = self.remove_phrase_repetition(text)

        return text

    def is_valid_sentence(self, text: str) -> bool:
        if len(text) < 2: return False
        
        for keyword in self.blacklist:
            if keyword.lower() in text.lower():
                return False
        
        # 무의미한 반복 문자 제거 (예: "네네네네네")
        no_space = text.replace(' ', '')
        if len(no_space) > 10:
            if len(set(no_space)) / len(no_space) < 0.1: return False
        
        # [추가] 마스킹 토큰만 남은 경우 제거 (예: "<PHONE>")
        if text.strip() in ["<PHONE>", "<RRN>", "<ACCOUNT>"]:
            return False

        return True

    def remove_duplicates(self, sentences: list) -> list:
        unique = []
        prev = None
        for s in sentences:
            s = s.strip()
            if not s: continue
            if s != prev: unique.append(s)
            prev = s
        return unique

    def process_file(self, audio_path: str) -> list:
        if not os.path.exists(audio_path):
            return []

        # [수정] 임시 파일 생성 로직(tempfile) 제거 -> 메모리 처리로 변경
        try:
            # enhance 함수가 이제 경로가 아닌 numpy array를 반환함
            enhanced_audio = self.enhancer.enhance(audio_path)
            
            # 전처리에 실패했거나 결과가 없으면 원본 사용을 위해 로드 시도 (혹은 건너뛰기)
            if enhanced_audio is None:
                # 원본 사용 시 로드 필요 (whisper는 경로도 받지만, 일관성을 위해 여기선 경로 전달)
                target_input = audio_path
            else:
                target_input = enhanced_audio # Numpy Array 전달
            
            segments, _ = self.whisper.transcribe(
                target_input,
                beam_size=3,    # [속도 개선] 5 -> 1 (Greedy Search). 속도 3~5배 향상, 정확도 저하 미미
                language="ko",
                condition_on_previous_text=False,
                repetition_penalty=1.2, 
                vad_filter=False, 
                temperature=0.0,
                no_speech_threshold=0.6
            )
            full_text = " ".join([seg.text for seg in segments])

        except Exception as e:
            print(f"❌ STT Error {audio_path}: {e}")
            return []
            
        # [제거] finally 블록의 임시 파일 삭제 로직 불필요

        cleaned = self.clean_text_basic(full_text)
        
        try:
            import kss
            sentences = kss.split_sentences(cleaned)
        except ImportError:
            sentences = re.split(r'(?<=[.?!])\s+', cleaned)

        # 유효성 검사 및 중복 제거
        sentences = [s for s in sentences if self.is_valid_sentence(s)]
        return self.remove_duplicates(sentences)
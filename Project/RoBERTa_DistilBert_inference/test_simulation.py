import os
import time
import librosa
import soundfile as sf
import numpy as np

# [수정] inference.py에서 탐지기(Detector)와 점수 관리자(Scorer) 모두 가져오기
from inference import VoicePhishingDetector, PhishingRiskScorer

# [설정]
SAMPLE_RATE = 16000

# [수정] 텍스트 중복을 피하기 위해 윈도우 크기와 이동 간격을 10초로 동일하게 설정
WINDOW_SECONDS = 15    # 한 번에 15초 분석
STRIDE_SECONDS = 5     # 10초씩 이동 (겹치는 구간 없음)

def run_simulation(long_audio_path, model_path):
    print(f"--- [Simulation] File: {long_audio_path} ---")
    
    # 1. 모델 초기화
    detector = VoicePhishingDetector(model_path=model_path)
    
    # 2. 누적 점수 관리 객체 생성
    risk_scorer = PhishingRiskScorer()
    
    # 3. 오디오 로드
    print(f"📂 Loading audio file...")
    full_audio, _ = librosa.load(long_audio_path, sr=SAMPLE_RATE)
    total_duration = len(full_audio) / SAMPLE_RATE
    print(f"⏱️  Total Duration: {total_duration:.2f} seconds")
    print(f"⚙️  Settings: Window={WINDOW_SECONDS}s, Stride={STRIDE_SECONDS}s")
    
    # 4. 윈도우 샘플 수 계산
    window_samples = int(WINDOW_SECONDS * SAMPLE_RATE)
    stride_samples = int(STRIDE_SECONDS * SAMPLE_RATE)
    
    # 5. 시뮬레이션 루프
    step = 0
    temp_filename = "temp_sim_chunk.wav"
    
    print("\n[Start Simulation Loop]")
    print("="*100)
    
    # range(시작, 끝, 간격)
    for start_idx in range(0, len(full_audio) - window_samples + 1, stride_samples):
        step += 1
        end_idx = start_idx + window_samples
        
        # (1) 오디오 자르기 & 저장
        current_chunk = full_audio[start_idx : end_idx]
        sf.write(temp_filename, current_chunk, SAMPLE_RATE)
        
        # 시간 정보 문자열 생성 (00:00 형식)
        start_time_str = time.strftime('%M:%S', time.gmtime(start_idx / SAMPLE_RATE))
        end_time_str = time.strftime('%M:%S', time.gmtime(end_idx / SAMPLE_RATE))
        
        # ==========================================
        # [측정 시작] 속도 측정 (Latency & FPS)
        # ==========================================
        t0 = time.time()
        
        # (2) 추론 실행
        result = detector.predict(temp_filename)
        
        t1 = time.time()
        # ==========================================
        # [측정 종료]
        # ==========================================
        
        inference_time = t1 - t0  # Latency (초)
        fps = 1.0 / inference_time if inference_time > 0 else 0.0
        
        # (3) 누적 점수 업데이트
        current_prob = 0.0
        text_segment = ""
        
        if result.get('status') == 'success':
            current_prob = result['max_risk_score']
            text_segment = result['dangerous_segment']
        
        accumulated_score, warning_level = risk_scorer.update_score(current_prob)
        
        # (4) 결과 출력 (속도 정보 포함)
        print_simulation_result(
            step, start_time_str, end_time_str, 
            current_prob, accumulated_score, warning_level, 
            text_segment, inference_time, fps
        )
        
    print("="*100)
    print("✅ Simulation Completed.")
    
    # 임시 파일 삭제
    if os.path.exists(temp_filename):
        try:
            os.remove(temp_filename)
        except:
            pass

def print_simulation_result(step, start, end, prob, score, level, text, latency, fps):
    """
    출력 형식: [Step] 시간 | 상태 | 점수 | 텍스트 | 속도
    """
    icon = "✅"
    status_msg = "NORMAL"
    color_reset = "\033[0m"
    color_code = color_reset

    if level == "LEVEL_2_WARNING":
        icon = "🚨"
        status_msg = "WARNING"
        color_code = "\033[91m" # Red
    elif level == "LEVEL_1_CAUTION":
        icon = "⚠️ "
        status_msg = "CAUTION"
        color_code = "\033[93m" # Yellow
    
    # 텍스트 길이 제한
    short_text = text[:30] + "..." if text else "(No Text)"

    # 포맷팅
    # 1. 상태 및 점수
    info_part = f"[{step:03d}] {start}~{end} | {icon} {color_code}{status_msg:<7}{color_reset} | Score: {score:3.0f} ({prob:.2f})"
    
    # 2. 속도 정보 (⚡ 표시)
    speed_part = f"⚡ {latency:.3f}s ({fps:4.1f} FPS)"
    
    # 3. 전체 출력
    print(f"{info_part} | {speed_part} | 🗣️ \"{short_text}\"")

if __name__ == "__main__":
    current_folder = os.path.dirname(os.path.abspath(__file__))
    
    # [주의] 파일명과 경로가 실제와 맞는지 꼭 확인하세요
    TEST_FILE = os.path.join(current_folder, "test_audio_data", "54.mp3")
    MODEL_PATH = os.path.join(current_folder, "model_weight", "student_best.pt")

    print(f"📍 Base Dir: {current_folder}")
    
    if not os.path.exists(TEST_FILE):
        print(f"❌ Audio file not found: {TEST_FILE}")
    elif not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
    else:
        run_simulation(TEST_FILE, MODEL_PATH)
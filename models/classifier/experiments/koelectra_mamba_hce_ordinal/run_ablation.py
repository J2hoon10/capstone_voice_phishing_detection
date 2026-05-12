import subprocess
import sys
from pathlib import Path

# Mamba Layer 개수를 변화시키며 Ablation Study 수행
MAMBA_LAYERS = [1, 2, 4, 6]

def main():
    base_dir = Path(__file__).parent.resolve()
    train_script = base_dir / "train.py"

    print("=== Mamba Depth Ablation Study 시작 ===")
    
    for layers in MAMBA_LAYERS:
        print(f"\n>> Mamba Layers = {layers} 학습 시작...")
        cmd = [sys.executable, str(train_script), "--mamba-layers", str(layers)]
        
        # 모델 학습 실행
        try:
            subprocess.run(cmd, check=True)
            print(f">> Mamba Layers = {layers} 학습 완료.")
        except subprocess.CalledProcessError as e:
            print(f"[오류] Mamba Layers = {layers} 학습 중 오류 발생: {e}")
            print("다음 설정으로 넘어갑니다...")
            continue

    print("\n=== 모든 Ablation Study 완료 ===")
    print("logs 폴더와 checkpoints 폴더를 확인하여 결과를 비교하세요.")

if __name__ == "__main__":
    main()

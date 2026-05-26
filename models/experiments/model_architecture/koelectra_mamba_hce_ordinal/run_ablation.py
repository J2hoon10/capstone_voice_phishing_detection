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
            print(f"  [1/2] 학습(Train) 진행 중...")
            subprocess.run(cmd, check=True)
            print(f"  [1/2] Mamba Layers = {layers} 학습 완료.")
            
            # 테스트 세트 평가 실행
            print(f"  [2/2] 평가(Test) 진행 중...")
            eval_cmd = [sys.executable, str(base_dir / "evaluate.py"), "--mamba-layers", str(layers), "--split", "test"]
            subprocess.run(eval_cmd, check=True)
            print(f"  [2/2] Mamba Layers = {layers} 평가 완료.\n")
            
        except subprocess.CalledProcessError as e:
            print(f"[오류] Mamba Layers = {layers} 단계 중 오류 발생: {e}")
            print("다음 설정으로 넘어갑니다...\n")
            continue

    print("\n=== 모든 Ablation Study 완료 ===")
    print("logs 폴더와 checkpoints 폴더를 확인하여 결과를 비교하세요.")

if __name__ == "__main__":
    main()

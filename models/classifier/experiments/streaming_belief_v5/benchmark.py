"""
Streaming Belief v5 추론 속도 벤치마크.

FPS = 총 처리 세그먼트 수 / 총 추론 시간 (세그먼트/초)

두 가지 모드:
  batch     - 배치 단위 추론 (DataLoader 그대로 사용, 기본)
  streaming - 샘플 1개씩 세그먼트를 순차적으로 추가해 실시간 스트리밍 시뮬레이션
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast

from config import (
    ABLATION_CONFIGS,
    CHECKPOINT_DIR,
    DATA_CONFIG,
    DATA_DIR,
    DEVICE,
    GPU_CONFIG,
    LOG_DIR,
    TRAIN_CONFIG,
)
from dataset import StreamingBeliefDataset, create_dataloaders
from model import build_streaming_belief_model


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------

def _sync():
    """GPU 커널이 모두 끝날 때까지 대기 (정확한 타이밍)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def resolve_checkpoint_path(experiment_name: str, run_id: str | None = None) -> tuple[Path, str]:
    if run_id is not None:
        return CHECKPOINT_DIR / f"{experiment_name}_{run_id}_best.pt", run_id
    latest_meta = CHECKPOINT_DIR / f"{experiment_name}_latest.json"
    if latest_meta.exists():
        with latest_meta.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        return Path(meta["checkpoint_path"]), meta.get("run_id", "latest")
    return CHECKPOINT_DIR / f"{experiment_name}_best.pt", "legacy"


# ---------------------------------------------------------------------------
# 배치 모드 벤치마크
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_batch_benchmark(
    model: torch.nn.Module,
    loader,
    warmup_batches: int = 3,
) -> dict:
    """배치 DataLoader 전체를 순회하며 FPS 측정."""
    model.eval()
    amp_enabled = GPU_CONFIG["ENABLED"] and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if GPU_CONFIG["DTYPE"] == "bf16" else torch.float16
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    batch_times: list[float] = []
    batch_segments: list[int] = []
    total_samples = 0

    print(f"[bench] warm-up {warmup_batches} batches...")
    for i, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        segment_mask = batch["segment_mask"].to(DEVICE)
        num_segments = batch["num_segments"].to(DEVICE)
        n_segs = int(batch["num_segments"].sum().item())

        with autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                segment_mask=segment_mask,
                num_segments=num_segments,
            )

        if i + 1 >= warmup_batches:
            break

    print("[bench] measuring...")
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        segment_mask = batch["segment_mask"].to(DEVICE)
        num_segments = batch["num_segments"].to(DEVICE)
        n_segs = int(batch["num_segments"].sum().item())
        bs = input_ids.shape[0]

        _sync()
        t0 = time.perf_counter()
        with autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                segment_mask=segment_mask,
                num_segments=num_segments,
            )
        _sync()
        elapsed = time.perf_counter() - t0

        batch_times.append(elapsed)
        batch_segments.append(n_segs)
        total_samples += bs

    times = np.array(batch_times)          # 배치별 경과 시간 (초)
    segs = np.array(batch_segments)        # 배치별 세그먼트 수

    total_time = float(times.sum())
    total_segs = int(segs.sum())
    avg_fps = total_segs / total_time if total_time > 0 else 0.0

    # 배치별 FPS 분포
    batch_fps = segs / times
    return {
        "mode": "batch",
        "total_time_sec": round(total_time, 4),
        "total_segments": total_segs,
        "total_samples": total_samples,
        "avg_fps_segments_per_sec": round(avg_fps, 2),
        "avg_samples_per_sec": round(total_samples / total_time, 2) if total_time > 0 else 0.0,
        "batch_fps_mean": round(float(batch_fps.mean()), 2),
        "batch_fps_std": round(float(batch_fps.std()), 2),
        "batch_fps_min": round(float(batch_fps.min()), 2),
        "batch_fps_max": round(float(batch_fps.max()), 2),
        "batch_fps_p50": round(float(np.percentile(batch_fps, 50)), 2),
        "batch_fps_p95": round(float(np.percentile(batch_fps, 95)), 2),
        "num_batches_measured": len(batch_times),
        "warmup_batches": warmup_batches,
    }


# ---------------------------------------------------------------------------
# 스트리밍 시뮬레이션 모드
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_streaming_benchmark(
    model: torch.nn.Module,
    dataset: StreamingBeliefDataset,
    warmup_samples: int = 5,
    max_samples: int | None = None,
) -> dict:
    """
    샘플 1개씩, 세그먼트를 1→2→…→N 순으로 추가하며 forward 수행.
    실시간 스트리밍에서 새 세그먼트가 도착할 때마다 업데이트하는 상황을 모사.
    """
    model.eval()
    amp_enabled = GPU_CONFIG["ENABLED"] and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if GPU_CONFIG["DTYPE"] == "bf16" else torch.float16
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    indices = list(range(len(dataset)))
    if max_samples is not None:
        indices = indices[:max_samples]

    seg_latencies: list[float] = []   # 세그먼트 1개 추가 시 latency (초)
    sample_latencies: list[float] = []  # 샘플 전체 처리 시간 (초)

    print(f"[bench/streaming] warm-up {warmup_samples} samples...")
    for idx in indices[:warmup_samples]:
        item = dataset[idx]
        n = item["num_segments"]
        ids = item["input_ids"].unsqueeze(0).to(DEVICE)      # (1, S, L)
        amask = item["attention_mask"].unsqueeze(0).to(DEVICE)

        for t in range(1, n + 1):
            seg_mask = torch.ones(1, t, dtype=torch.bool, device=DEVICE)
            num_seg = torch.tensor([t], dtype=torch.long, device=DEVICE)
            with autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                _ = model(
                    input_ids=ids[:, :t, :],
                    attention_mask=amask[:, :t, :],
                    segment_mask=seg_mask,
                    num_segments=num_seg,
                )

    print(f"[bench/streaming] measuring {len(indices) - warmup_samples} samples...")
    for idx in indices[warmup_samples:]:
        item = dataset[idx]
        n = item["num_segments"]
        ids = item["input_ids"].unsqueeze(0).to(DEVICE)
        amask = item["attention_mask"].unsqueeze(0).to(DEVICE)

        sample_start = time.perf_counter()
        _sync()

        for t in range(1, n + 1):
            seg_mask = torch.ones(1, t, dtype=torch.bool, device=DEVICE)
            num_seg = torch.tensor([t], dtype=torch.long, device=DEVICE)

            _sync()
            t0 = time.perf_counter()
            with autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                _ = model(
                    input_ids=ids[:, :t, :],
                    attention_mask=amask[:, :t, :],
                    segment_mask=seg_mask,
                    num_segments=num_seg,
                )
            _sync()
            seg_latencies.append(time.perf_counter() - t0)

        sample_latencies.append(time.perf_counter() - sample_start)

    seg_lat = np.array(seg_latencies)
    samp_lat = np.array(sample_latencies)

    avg_seg_latency = float(seg_lat.mean()) if len(seg_lat) > 0 else 0.0
    avg_fps = 1.0 / avg_seg_latency if avg_seg_latency > 0 else 0.0

    return {
        "mode": "streaming",
        "total_segments_measured": len(seg_lat),
        "total_samples_measured": len(sample_latencies),
        "avg_fps_segments_per_sec": round(avg_fps, 2),
        "seg_latency_mean_ms": round(avg_seg_latency * 1000, 3),
        "seg_latency_std_ms": round(float(seg_lat.std()) * 1000, 3),
        "seg_latency_min_ms": round(float(seg_lat.min()) * 1000, 3),
        "seg_latency_max_ms": round(float(seg_lat.max()) * 1000, 3),
        "seg_latency_p50_ms": round(float(np.percentile(seg_lat, 50)) * 1000, 3),
        "seg_latency_p95_ms": round(float(np.percentile(seg_lat, 95)) * 1000, 3),
        "seg_latency_p99_ms": round(float(np.percentile(seg_lat, 99)) * 1000, 3),
        "sample_latency_mean_ms": round(float(samp_lat.mean()) * 1000, 3),
        "sample_latency_std_ms": round(float(samp_lat.std()) * 1000, 3),
        "warmup_samples": warmup_samples,
    }


# ---------------------------------------------------------------------------
# 출력
# ---------------------------------------------------------------------------

def print_results(results: dict) -> None:
    mode = results.get("mode", "?")
    print("\n" + "=" * 60)
    print(f"  Benchmark Results (mode={mode})")
    print("=" * 60)

    if mode == "batch":
        print(f"  Batches measured   : {results['num_batches_measured']}  (warm-up={results['warmup_batches']})")
        print(f"  Total samples      : {results['total_samples']}")
        print(f"  Total segments     : {results['total_segments']}")
        print(f"  Total time         : {results['total_time_sec']:.3f} s")
        print(f"")
        print(f"  Avg FPS (seg/s)    : {results['avg_fps_segments_per_sec']:.2f}")
        print(f"  Avg throughput     : {results['avg_samples_per_sec']:.2f} samples/s")
        print(f"")
        print(f"  Per-batch FPS")
        print(f"    mean  : {results['batch_fps_mean']:.2f}")
        print(f"    std   : {results['batch_fps_std']:.2f}")
        print(f"    min   : {results['batch_fps_min']:.2f}")
        print(f"    p50   : {results['batch_fps_p50']:.2f}")
        print(f"    p95   : {results['batch_fps_p95']:.2f}")
        print(f"    max   : {results['batch_fps_max']:.2f}")

    elif mode == "streaming":
        print(f"  Samples measured   : {results['total_samples_measured']}  (warm-up={results['warmup_samples']})")
        print(f"  Segments measured  : {results['total_segments_measured']}")
        print(f"")
        print(f"  Avg FPS (seg/s)    : {results['avg_fps_segments_per_sec']:.2f}")
        print(f"")
        print(f"  Per-segment latency")
        print(f"    mean  : {results['seg_latency_mean_ms']:.1f} ms")
        print(f"    std   : {results['seg_latency_std_ms']:.1f} ms")
        print(f"    min   : {results['seg_latency_min_ms']:.1f} ms")
        print(f"    p50   : {results['seg_latency_p50_ms']:.1f} ms")
        print(f"    p95   : {results['seg_latency_p95_ms']:.1f} ms")
        print(f"    p99   : {results['seg_latency_p99_ms']:.1f} ms")
        print(f"    max   : {results['seg_latency_max_ms']:.1f} ms")
        print(f"")
        print(f"  Per-sample latency")
        print(f"    mean  : {results['sample_latency_mean_ms']:.1f} ms")
        print(f"    std   : {results['sample_latency_std_ms']:.1f} ms")

    print("=" * 60)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming Belief v5 추론 속도 벤치마크")
    parser.add_argument(
        "--experiment",
        type=str,
        default="v5_lora_r8",
        choices=list(ABLATION_CONFIGS.keys()),
        help="측정할 ablation 실험 (기본: v5_lora_r8)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="사용할 데이터 split (기본: test)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="batch",
        choices=["batch", "streaming"],
        help="batch=배치 추론 FPS / streaming=실시간 시뮬레이션 FPS (기본: batch)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="배치 크기 (batch 모드, 기본: config BATCH_SIZE)",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=3,
        help="warm-up 배치 수 (batch 모드, 기본: 3)",
    )
    parser.add_argument(
        "--warmup-samples",
        type=int,
        default=5,
        help="warm-up 샘플 수 (streaming 모드, 기본: 5)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="streaming 모드에서 측정할 최대 샘플 수 (기본: 전체)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="특정 run-id 체크포인트 지정",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader 워커 수",
    )
    args = parser.parse_args()

    batch_size = args.batch_size or TRAIN_CONFIG["BATCH_SIZE"]

    print(f"\n[bench] experiment={args.experiment}  split={args.split}  mode={args.mode}  device={DEVICE}")

    # 모델 로드
    ablation_config = ABLATION_CONFIGS.get(args.experiment, {})
    model = build_streaming_belief_model(ablation_config)
    ckpt_path, model_run_id = resolve_checkpoint_path(args.experiment, args.run_id)

    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
        print(f"[model] loaded: {ckpt_path}  (run={model_run_id})")
    else:
        print(f"[warn] checkpoint not found: {ckpt_path}. benchmarking with random weights.")
        model_run_id = "init"

    model = model.to(DEVICE)
    model.eval()

    # GPU 세팅
    if torch.cuda.is_available():
        if GPU_CONFIG.get("TF32_MATMUL"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if GPU_CONFIG.get("TF32_CUDNN"):
            torch.backends.cudnn.allow_tf32 = True
        if GPU_CONFIG.get("CUDNN_BENCHMARK"):
            torch.backends.cudnn.benchmark = True
        print(f"[gpu] {torch.cuda.get_device_name(0)}")

    # 실행
    if args.mode == "batch":
        loaders = create_dataloaders(
            DATA_DIR,
            batch_size=batch_size,
            num_workers=args.num_workers,
            splits=(args.split,),
        )
        if args.split not in loaders:
            raise FileNotFoundError(f"{args.split}.json not found. run data_preprocessing.py first.")
        results = run_batch_benchmark(
            model,
            loaders[args.split],
            warmup_batches=args.warmup_batches,
        )

    else:  # streaming
        data_path = DATA_DIR / f"{args.split}.json"
        if not data_path.exists():
            raise FileNotFoundError(f"{data_path} not found. run data_preprocessing.py first.")
        dataset = StreamingBeliefDataset(data_path)
        results = run_streaming_benchmark(
            model,
            dataset,
            warmup_samples=args.warmup_samples,
            max_samples=args.max_samples,
        )

    print_results(results)

    # 저장
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    bench_run_id = time.strftime("%Y%m%d_%H%M%S")
    save_path = LOG_DIR / f"{args.experiment}_{model_run_id}_{args.split}_{args.mode}_bench_{bench_run_id}.json"
    meta = {
        "experiment": args.experiment,
        "split": args.split,
        "mode": args.mode,
        "batch_size": batch_size,
        "device": str(DEVICE),
        "model_run_id": model_run_id,
    }
    with save_path.open("w", encoding="utf-8") as f:
        json.dump({**meta, **results}, f, indent=2, ensure_ascii=False)
    print(f"\n[save] {save_path}")

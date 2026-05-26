# whisper_error_analysis.py
# 사람 정답 vs STT 결과 비교 → 오류 분포 및 요약 산출

import os
import csv
import json
import argparse

from pipeline_config import ERROR_ANALYSIS_DIR

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def levenshtein_ops(ref, hyp):
    n = len(ref)
    m = len(hyp)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "ins"

    for i in range(1, n + 1):
        r = ref[i - 1]
        for j in range(1, m + 1):
            h = hyp[j - 1]
            cost = 0 if r == h else 1

            del_cost = dp[i - 1][j] + 1
            ins_cost = dp[i][j - 1] + 1
            sub_cost = dp[i - 1][j - 1] + cost

            best = min(del_cost, ins_cost, sub_cost)
            dp[i][j] = best

            if best == sub_cost:
                back[i][j] = "ok" if cost == 0 else "sub"
            elif best == del_cost:
                back[i][j] = "del"
            else:
                back[i][j] = "ins"

    i, j = n, m
    subs = dels = ins = ws = 0
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "ok":
            i -= 1
            j -= 1
        elif op == "sub":
            subs += 1
            if ref[i - 1] == " " or hyp[j - 1] == " ":
                ws += 1
            i -= 1
            j -= 1
        elif op == "del":
            dels += 1
            if ref[i - 1] == " ":
                ws += 1
            i -= 1
        elif op == "ins":
            ins += 1
            if hyp[j - 1] == " ":
                ws += 1
            j -= 1
        else:
            break

    return subs, dels, ins, ws


def load_rows(csv_path):
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_convergence(cers, window_size=10, threshold=0.01):
    windows = []
    for i in range(0, len(cers) - window_size + 1):
        w = cers[i:i + window_size]
        windows.append(sum(w) / window_size)

    converged = False
    if len(windows) >= 3:
        last3 = windows[-3:]
        converged = (max(last3) - min(last3)) <= threshold

    return {
        "window_size": window_size,
        "threshold": threshold,
        "window_means": windows,
        "converged": converged,
    }


def main():
    parser = argparse.ArgumentParser(description="Whisper 오류 분석 (CER)")
    parser.add_argument("--input", default=os.path.join(ERROR_ANALYSIS_DIR, "ground_truth_annotated.csv"),
                        help="사람 정답이 포함된 CSV")
    parser.add_argument("--output-report", default=os.path.join(ERROR_ANALYSIS_DIR, "error_report.csv"),
                        help="샘플별 오류 리포트")
    parser.add_argument("--output-summary", default=os.path.join(ERROR_ANALYSIS_DIR, "error_summary.json"),
                        help="오류 요약 JSON")
    parser.add_argument("--plot", action="store_true", help="수렴 그래프 출력")
    parser.add_argument("--no-progress", action="store_true", help="진행률 표시 비활성화")
    args = parser.parse_args()

    rows = load_rows(args.input)
    if not rows:
        raise SystemExit(f"입력 CSV를 찾을 수 없거나 비어있음: {args.input}")

    report_rows = []
    total_ref = 0
    total_sub = total_del = total_ins = total_ws = 0
    cers = []

    iterator = rows
    if tqdm and not args.no_progress:
        iterator = tqdm(rows, total=len(rows), desc="error-analysis", unit="sample")

    for row in iterator:
        ref = (row.get("human_transcription") or "").strip()
        hyp = (row.get("stt_text") or row.get("text") or "").strip()

        if not ref:
            continue

        subs, dels, ins, ws = levenshtein_ops(ref, hyp)
        ref_len = max(len(ref), 1)
        cer = (subs + dels + ins) / ref_len

        total_ref += ref_len
        total_sub += subs
        total_del += dels
        total_ins += ins
        total_ws += ws
        cers.append(cer)

        report_rows.append({
            "id": row.get("id", ""),
            "filename": row.get("filename", ""),
            "label": row.get("label", ""),
            "category": row.get("category", ""),
            "stt_text": hyp,
            "human_transcription": ref,
            "ref_len": ref_len,
            "cer": f"{cer:.6f}",
            "sub": subs,
            "del": dels,
            "ins": ins,
            "whitespace": ws,
        })

    if total_ref == 0:
        raise SystemExit("정답 텍스트가 비어있어 분석할 수 없습니다.")

    os.makedirs(ERROR_ANALYSIS_DIR, exist_ok=True)

    with open(args.output_report, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "id", "filename", "label", "category", "stt_text", "human_transcription",
            "ref_len", "cer", "sub", "del", "ins", "whitespace",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    summary = {
        "sample_count": len(report_rows),
        "total_ref_len": total_ref,
        "cer": (total_sub + total_del + total_ins) / total_ref,
        "substitution": total_sub / total_ref,
        "deletion": total_del / total_ref,
        "insertion": total_ins / total_ref,
        "whitespace": total_ws / total_ref,
        "convergence": compute_convergence(cers),
    }

    with open(args.output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"오류 리포트 저장: {args.output_report}")
    print(f"오류 요약 저장: {args.output_summary}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            conv = summary["convergence"]
            if conv["window_means"]:
                plt.plot(conv["window_means"], marker="o")
                plt.title("CER Convergence (window mean)")
                plt.xlabel("Window Index")
                plt.ylabel("CER")
                plt.grid(True)
                plt.show()
        except Exception as e:
            print(f"[WARN] plot 실패: {e}")


if __name__ == "__main__":
    main()

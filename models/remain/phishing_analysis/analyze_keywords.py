"""
phishing_analysis/analyze_keywords.py

phishing_clean_fewshot.csv 를 대상으로:
  1. MeCab 형태소 분석 (명사/동사 추출)
  2. 전체 / 카테고리별 TF-IDF 분석
  3. 카테고리 변별 키워드 (카이제곱) 분석
  4. 상위 키워드 목록 CSV + 막대 그래프 저장

출력 파일 (phishing_analysis/ 내):
  - morphs_all.csv              : 문서별 추출 형태소 목록
  - tfidf_overall.csv           : 전체 TF-IDF 상위 키워드 (일반어 제거)
  - tfidf_대출_사기형.csv        : 카테고리별 TF-IDF (변별 단어 위주)
  - tfidf_수사기관_사칭형.csv
  - discriminative_keywords.csv : 카이제곱 기반 카테고리 변별 키워드
  - freq_*.png                  : 막대 그래프
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import mecab

# ── 경로 설정 ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR.parent / "augmented" / "phishing_clean_fewshot.csv"
OUT_DIR  = BASE_DIR
TOP_N    = 20

# ── 한글 폰트 ──────────────────────────────────────────────────────────────
def _set_korean_font():
    for p in [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]:
        if Path(p).exists():
            fm.fontManager.addfont(p)
            font_name = fm.FontProperties(fname=p).get_name()
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            return font_name
    for f in fm.findSystemFonts():
        if "Nanum" in f or "nanum" in f:
            fm.fontManager.addfont(f)
            font_name = fm.FontProperties(fname=f).get_name()
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            return font_name
    return None

# ── 불용어: 일반 대화어 + 금융·수사 공통어 ────────────────────────────────
STOPWORDS = {
    # 대화 상투어
    "네", "아", "예", "뭐", "저", "제", "그", "이", "것", "수", "거",
    "좀", "또", "더", "다", "안", "못", "왜", "어", "음", "그럼",
    "그리고", "하지만", "그래서", "근데", "그런데", "때문", "경우",
    "때", "곳", "분", "쪽", "중", "내", "우리", "저희", "고객",
    "말씀", "부분", "내용", "방법", "상황", "관련", "기준", "정도",
    "현재", "이제", "지금", "아까", "오늘", "어제", "잠깐", "바로",
    "보통", "일단", "우선", "다시", "계속", "이미", "아직", "물론",
    "혹시", "사실", "보니", "하여", "따라", "통해", "위해",
    # 일반 동사 (피싱 특화 아님)
    "하다", "되다", "있다", "없다", "말하다", "보다", "오다", "가다",
    "받다", "주다", "드리다", "알다", "모르다", "생각하다", "나오다",
    "들어가다", "나가다", "들어오다",
    # 일반 금융/통신 공통어 (양쪽 카테고리 모두 등장)
    "본인", "확인", "전화", "사용", "은행", "계좌", "통장", "처리",
    "연락", "번호", "신청", "등록", "발급", "완료", "진행", "안내",
    "신용", "금융", "고객님", "담당", "담당자", "서류",
    # 형식어
    "것이다", "수있다", "그렇다", "이렇다", "저렇다",
}

TARGET_POS = {"NNG", "NNP"}  # 명사만 (동사/형용사 제외 → 어미 노이즈 방지)

# ── 형태소 추출 ────────────────────────────────────────────────────────────
def extract_tokens(text: str, m: mecab.MeCab) -> list[str]:
    tokens = []
    for surface, pos in m.pos(text):
        base_pos = pos.split("+")[0]
        if base_pos not in TARGET_POS:
            continue
        word = surface.strip()
        if len(word) < 2:
            continue
        if word in STOPWORDS:
            continue
        tokens.append(word)
    return tokens


def build_corpus(df: pd.DataFrame, m: mecab.MeCab) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        tokens = extract_tokens(str(row["text_clean"]), m)
        records.append({
            "id": row["id"],
            "category": row["category"],
            "tokens": tokens,
            "doc": " ".join(tokens),
        })
    return pd.DataFrame(records)


# ── TF-IDF ─────────────────────────────────────────────────────────────────
def tfidf_top(docs: list[str], top_n: int = TOP_N) -> pd.DataFrame:
    if len(docs) < 2:
        counter = Counter(" ".join(docs).split())
        return pd.DataFrame([
            {"rank": i+1, "word": w, "tfidf_mean": float(c)}
            for i, (w, c) in enumerate(counter.most_common(top_n))
        ])
    vec = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[가-힣]{2,}",
        max_features=5000,
        sublinear_tf=True,
    )
    X = vec.fit_transform(docs)
    feature_names = vec.get_feature_names_out()
    mean_scores = np.asarray(X.mean(axis=0)).flatten()
    top_idx = mean_scores.argsort()[::-1][:top_n]
    return pd.DataFrame([
        {"rank": i+1, "word": feature_names[idx], "tfidf_mean": round(float(mean_scores[idx]), 5)}
        for i, idx in enumerate(top_idx)
    ])


# ── 카이제곱 기반 카테고리 변별 키워드 ────────────────────────────────────
def discriminative_keywords(corpus: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    """
    카테고리별로 다른 카테고리 대비 유독 많이 등장하는 단어를 카이제곱으로 추출.
    결과: category / word / chi2_score / freq_in_cat / freq_out_cat
    """
    categories = corpus["category"].unique()
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    labels = corpus["category"].map(cat_to_idx).values

    vec = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[가-힣]{2,}",
        max_features=5000,
        binary=True,          # 출현 여부만 (카이제곱에 적합)
    )
    X = vec.fit_transform(corpus["doc"])
    feature_names = vec.get_feature_names_out()

    rows = []
    for cat, cat_idx in cat_to_idx.items():
        binary_labels = (labels == cat_idx).astype(int)
        scores, _ = chi2(X, binary_labels)

        # 해당 카테고리에서의 평균 빈도
        cat_mask = labels == cat_idx
        freq_in  = np.asarray(X[cat_mask].mean(axis=0)).flatten()
        freq_out = np.asarray(X[~cat_mask].mean(axis=0)).flatten()

        top_idx = scores.argsort()[::-1][:top_n]
        for rank, idx in enumerate(top_idx, 1):
            # 해당 카테고리에서 더 많이 등장하는 단어만
            if freq_in[idx] > freq_out[idx]:
                rows.append({
                    "category": cat,
                    "rank": rank,
                    "word": feature_names[idx],
                    "chi2_score": round(float(scores[idx]), 3),
                    "freq_in_cat": round(float(freq_in[idx]), 3),
                    "freq_out_cat": round(float(freq_out[idx]), 3),
                })
    return pd.DataFrame(rows)


# ── 시각화 ─────────────────────────────────────────────────────────────────
def plot_top(df: pd.DataFrame, score_col: str, title: str, out_path: Path, color: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    words  = df["word"].tolist()[:TOP_N][::-1]
    scores = df[score_col].tolist()[:TOP_N][::-1]
    ax.barh(words, scores, color=color)
    ax.set_xlabel(score_col)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  그래프 저장: {out_path.name}")


# ── 메인 ───────────────────────────────────────────────────────────────────
def main():
    font = _set_korean_font()
    print(f"[font] {font}")

    df = pd.read_csv(CSV_PATH)
    print(f"[data] {len(df)}개 문서 | 카테고리: {df['category'].unique().tolist()}")

    m = mecab.MeCab()
    corpus = build_corpus(df, m)

    # 형태소 저장
    morph_out = corpus[["id", "category", "tokens"]].copy()
    morph_out["tokens"] = morph_out["tokens"].apply(" ".join)
    morph_out.to_csv(OUT_DIR / "morphs_all.csv", index=False, encoding="utf-8-sig")
    print("[morph] morphs_all.csv 저장")

    # ── 전체 TF-IDF ──────────────────────────────────────────────────────
    print("\n[tfidf] 전체")
    overall = tfidf_top(corpus["doc"].tolist())
    overall.to_csv(OUT_DIR / "tfidf_overall.csv", index=False, encoding="utf-8-sig")
    plot_top(overall, "tfidf_mean", "전체 피싱 TF-IDF Top20 (변별어)", OUT_DIR / "freq_overall.png", "#555555")
    print(f"  {'순위':>4}  {'단어':<12}  TF-IDF")
    for _, r in overall.iterrows():
        print(f"  {int(r['rank']):>4}  {r['word']:<12}  {r['tfidf_mean']:.5f}")

    # ── 카테고리별 TF-IDF ─────────────────────────────────────────────────
    colors = {"대출 사기형": "#E05C5C", "수사기관 사칭형": "#5C7AE0"}
    for cat in corpus["category"].unique():
        sub = corpus[corpus["category"] == cat]
        print(f"\n[tfidf] [{cat}] ({len(sub)}개)")
        cat_tf = tfidf_top(sub["doc"].tolist())
        safe   = cat.replace(" ", "_")
        cat_tf.to_csv(OUT_DIR / f"tfidf_{safe}.csv", index=False, encoding="utf-8-sig")
        plot_top(cat_tf, "tfidf_mean", f"[{cat}] TF-IDF Top{TOP_N}", OUT_DIR / f"freq_{safe}.png", colors.get(cat, "#888888"))
        print(f"  {'순위':>4}  {'단어':<12}  TF-IDF")
        for _, r in cat_tf.iterrows():
            print(f"  {int(r['rank']):>4}  {r['word']:<12}  {r['tfidf_mean']:.5f}")

    # ── 카이제곱 변별 키워드 ──────────────────────────────────────────────
    print("\n[chi2] 카테고리 변별 키워드 분석...")
    disc = discriminative_keywords(corpus)
    disc.to_csv(OUT_DIR / "discriminative_keywords.csv", index=False, encoding="utf-8-sig")

    for cat in corpus["category"].unique():
        sub_disc = disc[disc["category"] == cat].reset_index(drop=True)
        safe     = cat.replace(" ", "_")
        plot_top(sub_disc, "chi2_score", f"[{cat}] 변별 키워드 (χ²)", OUT_DIR / f"discriminative_{safe}.png", colors.get(cat, "#888888"))

    # ── 최종 요약 ─────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  카테고리 변별 Top10 키워드 (χ² 기준 — 해당 카테고리 특화)")
    print("="*65)
    for cat in corpus["category"].unique():
        words = disc[disc["category"] == cat]["word"].tolist()[:10]
        print(f"  [{cat}]")
        print(f"    {', '.join(words)}")

    print("\n" + "="*65)
    print("  전체 TF-IDF Top10 (공통 고빈도어 제거 후)")
    print("="*65)
    print(f"  {', '.join(overall['word'].tolist()[:10])}")

    print(f"\n[done] 결과 폴더: {OUT_DIR}")


if __name__ == "__main__":
    main()

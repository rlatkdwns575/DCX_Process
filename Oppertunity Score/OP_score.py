# ============================================================
# Actor-Action별 Opportunity Score 계산 + 그래프 시각화
# 기준:
# - Actor      : 기존 cluster 사용
# - Action     : lda_coherence_perplexity.py 선정 k 토픽
# - Satisfaction : 감성사전 기반 감성점수 → 0~10 정규화
# - Importance   : Actor-Action 빈도 비중 → 0~10 정규화
# - Opportunity  : Importance + max(Importance - Satisfaction, 0)
#
# 실행 순서:
#   1. python lda_coherence_perplexity.py  → out/opportunity_score/coherence_perplexity/
#   2. python OP_score.py                  → out/opportunity_score/opportunity/
# ============================================================

import pandas as pd
import numpy as np
import json
import re
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from organize_files import organize_project_files

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

# ------------------------------------------------------------
# 0. 파일 경로 설정
# ------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "opportunity_score")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "out", "opportunity_score")
LDA_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "coherence_perplexity")
OPPORTUNITY_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "opportunity")

INPUT_CSV = os.path.join(DATA_DIR, "형태소추출.csv")
SENTI_JSON = os.path.join(DATA_DIR, "SentiWord_info.json")
SENTI_JSON_FALLBACK = os.path.join(DATA_DIR, "SentiWord_info.json")

TFIDF_FILES = {
    1: os.path.join(DATA_DIR, "클러스터1_tf_idf.csv"),
    2: os.path.join(DATA_DIR, "클러스터2_tf_idf.csv"),
    3: os.path.join(DATA_DIR, "클러스터3_tf_idf.csv"),
    4: os.path.join(DATA_DIR, "클러스터4_tf_idf.csv"),
}

OUTPUT_SCORE_CSV = os.path.join(OPPORTUNITY_OUTPUT_DIR, "actor_action_opportunity_score.csv")
OUTPUT_SCATTER = os.path.join(OPPORTUNITY_OUTPUT_DIR, "actor_action_opportunity_map.png")
OUTPUT_BAR = os.path.join(OPPORTUNITY_OUTPUT_DIR, "actor_action_opportunity_top20.png")
OUTPUT_HEATMAP = os.path.join(OPPORTUNITY_OUTPUT_DIR, "actor_action_opportunity_heatmap.png")
OUTPUT_OPTIMAL_K = os.path.join(LDA_OUTPUT_DIR, "lda_optimal_k_summary.csv")


def resolve_lda_file(filename):
    """coherence_perplexity/ 에서 LDA 결과 로드"""
    path = os.path.join(LDA_OUTPUT_DIR, filename)
    if os.path.exists(path):
        return path

    raise FileNotFoundError(
        f"{path} 파일이 없습니다. 먼저 python lda_coherence_perplexity.py 를 실행하세요."
    )


def resolve_senti_json():
    if os.path.exists(SENTI_JSON):
        return SENTI_JSON
    if os.path.exists(SENTI_JSON_FALLBACK):
        return SENTI_JSON_FALLBACK
    raise FileNotFoundError(f"감성사전 파일 없음: {SENTI_JSON}")


def load_lda_optimal_k(scores_path):
    scores_df = pd.read_csv(scores_path, encoding="utf-8-sig")
    selected = scores_df[scores_df["is_selected"]].copy()
    if selected.empty:
        raise ValueError("LDA 선정 k(is_selected=True) 결과가 없습니다.")

    optimal_k_df = (
        selected.sort_values("cluster")
        .rename(columns={
            "num_topics": "optimal_k",
            "coherence": "lda_coherence",
            "perplexity": "lda_perplexity",
        })
        [["cluster", "actor", "optimal_k", "lda_coherence", "lda_perplexity"]]
        .reset_index(drop=True)
    )
    return optimal_k_df


def organize_output_dirs():
    organize_project_files(SCRIPT_DIR)


# ------------------------------------------------------------
# 1. 한글 폰트 설정
# ------------------------------------------------------------
def set_korean_font():
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]
    available = {f.name for f in fm.fontManager.ttflist}

    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            break

    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()
organize_output_dirs()


# ------------------------------------------------------------
# 2. 데이터 불러오기
# ------------------------------------------------------------
df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

required_cols = ["content", "cluster", "cleaned_content", "tokenized_content"]
missing_cols = [c for c in required_cols if c not in df.columns]

if missing_cols:
    raise ValueError(f"필수 컬럼이 없습니다: {missing_cols}")

for col in ["content", "cleaned_content", "tokenized_content"]:
    df[col] = df[col].fillna("").astype(str)

print("원본 데이터 shape:", df.shape)
print(df["cluster"].value_counts().sort_index())


# ------------------------------------------------------------
# 3. TF-IDF 상위 단어 확인용 로드
# ------------------------------------------------------------
tfidf_top_words = {}

for cluster_id, path in TFIDF_FILES.items():
    if os.path.exists(path):
        temp = pd.read_csv(path, encoding="utf-8-sig")
        top_words = temp["단어"].head(8).tolist()
        tfidf_top_words[cluster_id] = top_words

print("\n클러스터별 TF-IDF 상위 단어")
for k, v in tfidf_top_words.items():
    print(f"Cluster {k}: {v}")


# ------------------------------------------------------------
# 4. Actor 이름 설정
# 필요하면 여기만 직접 수정하면 됩니다.
# ------------------------------------------------------------
ACTOR_NAME_MAP = {
    1: "세정 루틴/외출 전 냄새 걱정",
    2: "두피·샴푸·탈모 관리",
    3: "드라이샴푸·향수/미스트 사용",
    4: "머리감기 귀찮음·말리기 부담",
}

df["actor"] = df["cluster"].apply(
    lambda x: f"Actor{x}_{ACTOR_NAME_MAP.get(x, '미정')}"
)


# ------------------------------------------------------------
# 5. LDA Action 분류 결과 불러오기
# lda_coherence_perplexity.py 에서 선정한 클러스터별 최적 k 토픽 사용
# ------------------------------------------------------------
lda_scores_path = resolve_lda_file("lda_coherence_perplexity_scores.csv")
lda_topics_path = resolve_lda_file("lda_action_topic_summary.csv")
lda_actions_path = resolve_lda_file("lda_document_actions.csv")

optimal_k_df = load_lda_optimal_k(lda_scores_path)
optimal_k_df.to_csv(OUTPUT_OPTIMAL_K, index=False, encoding="utf-8-sig")
print("\n클러스터별 LDA 선정 k (Coherence 최대 → Perplexity 최소)")
display(optimal_k_df)
print(f"저장 완료: {OUTPUT_OPTIMAL_K}")

lda_topics = pd.read_csv(lda_topics_path, encoding="utf-8-sig")
valid_actions = lda_topics[["cluster", "action", "topic_id", "top_words"]].drop_duplicates(
    subset=["cluster", "action"]
)

action_map = pd.read_csv(lda_actions_path, encoding="utf-8-sig")
action_map = action_map.set_index("row_id")["action"]
df["action"] = action_map.reindex(df.index, fill_value="A99_기타")

before_count = len(df)
df = df.merge(valid_actions[["cluster", "action"]], on=["cluster", "action"], how="inner")
after_count = len(df)
if after_count < before_count:
    print(f"\nLDA 최적 k 토픽 외 문서 {before_count - after_count}건 제외 (A99 등)")

for cluster_id, group in df.groupby("cluster"):
    expected_k = int(optimal_k_df.loc[optimal_k_df["cluster"] == cluster_id, "optimal_k"].iloc[0])
    actual_actions = group["action"].nunique()
    if actual_actions != expected_k:
        print(
            f"경고: Cluster {cluster_id} Action 수 {actual_actions} != 선정 k {expected_k}"
        )

print("\nAction 분류 예시 (LDA 최적 k 기준)")
display(df[["cluster", "actor", "action", "cleaned_content"]].head())


# ------------------------------------------------------------
# 6. 감성사전 로드 및 Satisfaction 계산
# ------------------------------------------------------------
with open(resolve_senti_json(), "r", encoding="utf-8") as f:
    sent_dicts = json.load(f)

sentiment_lookup = {}

for item in sent_dicts:
    word = str(item.get("word", "")).strip()
    word_root = str(item.get("word_root", "")).strip()

    try:
        polarity = int(item.get("polarity", 0))
    except:
        continue

    if len(word) >= 2:
        sentiment_lookup[word] = polarity

    if len(word_root) >= 2:
        sentiment_lookup[word_root] = polarity


# 도메인 특화 감성어 보강
DOMAIN_SENTIMENT = {
    # 추가하고 싶은 감성어 및 점수수 작성
}


def remove_space(text):
    return re.sub(r"\s+", "", str(text))


def calc_sentiment_raw(row):
    """
    tokenized_content 기준으로 감성어를 찾고,
    cleaned_content + tokenized_content에서 도메인 감성어를 추가로 찾습니다.

    반환값:
    - 감성어가 있으면 평균 polarity
    - 없으면 0, 즉 중립
    """
    token_text = str(row["tokenized_content"])
    clean_text = str(row["cleaned_content"])

    tokens = token_text.split()
    scores = []

    for token in tokens:
        if token in sentiment_lookup:
            scores.append(sentiment_lookup[token])

    full_text_no_space = remove_space(clean_text + " " + token_text)

    for kw, score in DOMAIN_SENTIMENT.items():
        if remove_space(kw) in full_text_no_space:
            scores.append(score)

    if len(scores) == 0:
        return 0.0

    return float(np.mean(scores))


df["sentiment_raw"] = df.apply(calc_sentiment_raw, axis=1)

# polarity 범위 -2 ~ +2를 Satisfaction 0 ~ 10으로 변환
# -2 = 매우 불만족, 0 = 중립, +2 = 매우 만족
df["satisfaction"] = ((df["sentiment_raw"].clip(-2, 2) + 2) / 4) * 10

print("\nSatisfaction 기초 통계")
display(df[["sentiment_raw", "satisfaction"]].describe())


# ------------------------------------------------------------
# 7. Actor-Action 단위 집계
# LDA는 문서당 1개 Action(토픽)을 부여합니다.
# ------------------------------------------------------------
action_df = df.copy()

print("\nAction 분포")
display(action_df["action"].value_counts())


# ------------------------------------------------------------
# 8. Importance / Satisfaction / Opportunity 계산
# ------------------------------------------------------------
summary = (
    action_df
    .groupby(["cluster", "actor", "action"], as_index=False)
    .agg(
        count=("content", "size"),
        satisfaction=("satisfaction", "mean"),
        sentiment_raw=("sentiment_raw", "mean")
    )
)

summary = summary.merge(
    lda_topics[["cluster", "action", "topic_id", "top_words"]],
    on=["cluster", "action"],
    how="left",
)
summary = summary.merge(
    optimal_k_df[["cluster", "optimal_k", "lda_coherence", "lda_perplexity"]],
    on="cluster",
    how="left",
)

# Importance 1: 전체 Actor-Action mention 중 비중
summary["importance_pct"] = summary["count"] / len(action_df) * 100

# Importance 2: 그래프용 0~10 정규화
# 가장 많이 등장한 Actor-Action을 10점으로 둡니다.
summary["importance"] = summary["count"] / summary["count"].max() * 10

# Opportunity Score
summary["opportunity"] = summary.apply(
    lambda x: x["importance"] + max(x["importance"] - x["satisfaction"], 0),
    axis=1
)

# 보기 좋게 반올림
score_cols = [
    "satisfaction", "sentiment_raw", "importance_pct", "importance",
    "opportunity", "lda_coherence", "lda_perplexity",
]
summary[score_cols] = summary[score_cols].round(3)

summary = summary.sort_values("opportunity", ascending=False).reset_index(drop=True)

print("\nActor-Action Opportunity Score 결과")
display(summary)

summary.to_csv(OUTPUT_SCORE_CSV, index=False, encoding="utf-8-sig")
print(f"\n저장 완료: {OUTPUT_SCORE_CSV}")


# ------------------------------------------------------------
# 9. 그래프 1: Opportunity Map
# x축 = Importance, y축 = Satisfaction
# 각 축 중앙(0) = 해당 점수 평균, 유도선은 평균 지점에서 출발해 우상단에서 만남
# ------------------------------------------------------------
TOP_N = 25
plot_df = summary.head(TOP_N).copy()

SCORE_MAX = 10.0
SCATTER_POINT_SIZE = 120

IMP_MEAN = summary["importance"].mean()
SAT_MEAN = summary["satisfaction"].mean()

plot_df["importance_plot"] = plot_df["importance"] - IMP_MEAN
plot_df["satisfaction_plot"] = plot_df["satisfaction"] - SAT_MEAN

MEET_X = SCORE_MAX - IMP_MEAN
MEET_Y = SCORE_MAX - SAT_MEAN
LINE1_START = (-IMP_MEAN, 0.0)
LINE2_START = (0.0, -SAT_MEAN)
LINE1_SLOPE = MEET_Y / (MEET_X - LINE1_START[0])
LINE2_SLOPE = (MEET_Y - LINE2_START[1]) / (MEET_X - LINE2_START[0])

ACTOR_COLORS = {
    1: "#4472C4",
    2: "#70AD47",
    3: "#ED7D31",
    4: "#7030A0",
}


def format_map_label(row, keyword_count=3):
    """Actor{번호}_Action{번호} + 토픽 키워드 N개"""
    actor_num = int(row["cluster"])
    action_num = int(row["topic_id"]) + 1

    parts = str(row["action"]).split("_")
    if parts and re.match(r"A\d+", parts[0], re.I):
        keywords = parts[1:1 + keyword_count]
    else:
        keywords = []

    if len(keywords) < keyword_count and pd.notna(row.get("top_words")):
        from_top = [
            token.strip().split("(")[0]
            for token in str(row["top_words"]).split(",")
            if token.strip()
        ]
        keywords = (keywords + from_top)[:keyword_count]

    keyword_line = " ".join(keywords[:keyword_count])
    return f"Actor{actor_num}_Action{action_num}\n{keyword_line}"


def guide_line1_y(x):
    """Overserved / Well-served 경계: (0, 감정평균) → (10, 10)"""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.where(
        x_arr <= LINE1_START[0],
        LINE1_START[1],
        LINE1_START[1] + LINE1_SLOPE * (x_arr - LINE1_START[0]),
    )
    return np.clip(y_arr, None, MEET_Y)


def guide_line2_y(x):
    """Well-served / Underserved 경계: (중요도평균, 0) → (10, 10)"""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.where(
        x_arr <= LINE2_START[0],
        LINE2_START[1],
        LINE2_START[1] + LINE2_SLOPE * (x_arr - LINE2_START[0]),
    )
    return np.clip(y_arr, None, MEET_Y)


def plot_axis_limits(values, anchor_min, anchor_max, pad=0.8):
    lo = min(values.min(), anchor_min) - pad
    hi = max(values.max(), anchor_max) + pad
    return lo, hi


x_lo, x_hi = plot_axis_limits(
    plot_df["importance_plot"],
    LINE1_START[0],
    MEET_X,
)
y_lo, y_hi = plot_axis_limits(
    plot_df["satisfaction_plot"],
    LINE2_START[1],
    MEET_Y,
)

fig, ax = plt.subplots(figsize=(13, 9))

xs = np.linspace(x_lo, x_hi, 400)
y_upper = guide_line1_y(xs)
y_lower = guide_line2_y(xs)

ax.fill_between(xs, y_upper, y_hi + 0.5, color="#E8DAEF", alpha=0.55, zorder=0)
ax.fill_between(xs, y_lo - 0.5, y_lower, color="#D6EAF8", alpha=0.55, zorder=0)
ax.fill_between(
    xs, y_lower, y_upper, where=(y_upper >= y_lower),
    color="#D5F5E3", alpha=0.55, zorder=0,
)

ax.plot(
    [LINE1_START[0], MEET_X], [LINE1_START[1], MEET_Y],
    color="#666666", linewidth=1.5, zorder=1,
)
ax.plot(
    [LINE2_START[0], MEET_X], [LINE2_START[1], MEET_Y],
    color="#666666", linewidth=1.5, zorder=1,
)
ax.axhline(0, color="#999999", linewidth=0.8, linestyle=":", zorder=1)
ax.axvline(0, color="#999999", linewidth=0.8, linestyle=":", zorder=1)


def area_label_position(kind):
    """각 영역 내부 중심부 좌표 계산"""
    if kind == "overserved":
        x = x_lo * 0.55 + LINE1_START[0] * 0.25 + MEET_X * 0.20 + 1
        y_line = float(guide_line1_y(x))
        y = (y_line + y_hi) / 2
        return x, y
    if kind == "well_served":
        x = LINE2_START[0] + 0.42 * (MEET_X - LINE2_START[0])
        y = (float(guide_line1_y(x)) + float(guide_line2_y(x))) / 2 +0.5
        return x, y
    x = LINE2_START[0] + 0.62 * (MEET_X - LINE2_START[0])
    y_line = float(guide_line2_y(x))
    y = (y_lo + y_line) / 2
    return x, y


def draw_area_label(text, position, color):
    ax.text(
        position[0],
        position[1],
        text,
        fontsize=11,
        color=color,
        ha="center",
        va="center",
        zorder=2,
        clip_on=False,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.65),
    )


draw_area_label("A : Overserved Area", area_label_position("overserved"), "#6C3483")
draw_area_label("B : Well-served Area", area_label_position("well_served"), "#1E8449")
draw_area_label("C : Underserved Area", area_label_position("underserved"), "#1F618D")

for cluster_id, subset in plot_df.groupby("cluster"):
    ax.scatter(
        subset["importance_plot"],
        subset["satisfaction_plot"],
        s=SCATTER_POINT_SIZE,
        c=ACTOR_COLORS.get(cluster_id, "#595959"),
        alpha=0.78,
        edgecolors="black",
        linewidths=0.8,
        label=f"Actor{cluster_id}",
        zorder=3,
    )

for _, row in plot_df.iterrows():
    ax.annotate(
        format_map_label(row, keyword_count=3),
        (row["importance_plot"], row["satisfaction_plot"]),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=6,
        zorder=4,
    )

ax.set_xlim(x_lo, x_hi)
ax.set_ylim(y_lo, y_hi)
ax.margins(x=0.06, y=0.08)


def centered_ticks(lo, hi, mean_value, step=2.0):
    actual_lo = lo + mean_value
    actual_hi = hi + mean_value
    start = np.floor(actual_lo / step) * step
    actual_ticks = np.arange(start, actual_hi + step * 0.01, step)
    plot_ticks = actual_ticks - mean_value
    mask = (plot_ticks >= lo - 0.01) & (plot_ticks <= hi + 0.01)
    plot_ticks = plot_ticks[mask]
    labels = [f"{(t + mean_value):g}" for t in plot_ticks]
    return plot_ticks, labels


x_ticks, x_tick_labels = centered_ticks(x_lo, x_hi, IMP_MEAN, step=2.0)
y_ticks, y_tick_labels = centered_ticks(y_lo, y_hi, SAT_MEAN, step=2.0)
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_tick_labels)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_tick_labels)

ax.set_xlabel("측정된 중요도\n(Importance Score, 중앙 = 평균)")
ax.set_ylabel("측정된 감정\n(Satisfaction Score, 중앙 = 평균)")
ax.set_title("Actor-Action Opportunity Map", fontsize=16)

ax.grid(True, alpha=0.25, zorder=0)
ax.legend(title="Actor", loc="upper left", fontsize=9)

plt.tight_layout(pad=1.4)
plt.savefig(OUTPUT_SCATTER, dpi=300, bbox_inches="tight", pad_inches=0.45)
plt.close()

print(f"저장 완료: {OUTPUT_SCATTER}")


# ------------------------------------------------------------
# 10. 그래프 2: Opportunity Top 20 Bar Chart
# ------------------------------------------------------------
top_bar = summary.head(20).copy()
top_bar["label"] = top_bar["actor"] + " / " + top_bar["action"]

plt.figure(figsize=(12, 9))
plt.barh(top_bar["label"][::-1], top_bar["opportunity"][::-1])
plt.title("Opportunity Score Top 20", fontsize=16)
plt.xlabel("Opportunity Score")
plt.ylabel("Actor / Action")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_BAR, dpi=300, bbox_inches="tight")
plt.close()

print(f"저장 완료: {OUTPUT_BAR}")


# ------------------------------------------------------------
# 11. 그래프 3: Actor x Action Opportunity Heatmap
# seaborn 없이 matplotlib만 사용
# ------------------------------------------------------------
heatmap_data = summary.pivot_table(
    index="actor",
    columns="action",
    values="opportunity",
    aggfunc="mean",
    fill_value=0
)

plt.figure(figsize=(16, 7))
plt.imshow(heatmap_data.values, aspect="auto")

plt.xticks(
    ticks=np.arange(len(heatmap_data.columns)),
    labels=heatmap_data.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    ticks=np.arange(len(heatmap_data.index)),
    labels=heatmap_data.index
)

plt.colorbar(label="Opportunity Score")
plt.title("Actor x Action Opportunity Heatmap", fontsize=16)

# 값 표시
for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        value = heatmap_data.iloc[i, j]
        if value > 0:
            plt.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_HEATMAP, dpi=300, bbox_inches="tight")
plt.close()

print(f"저장 완료: {OUTPUT_HEATMAP}")


# ------------------------------------------------------------
# 12. 최종 Top 결과만 다시 확인
# ------------------------------------------------------------
display(summary.head(20))
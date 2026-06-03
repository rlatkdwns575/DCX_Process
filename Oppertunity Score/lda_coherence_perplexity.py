# ============================================================
# 클러스터별 LDA 토픽 수 탐색
# - Coherence Score + Perplexity Score 계산
# - Coherence 최대 → 동률 시 Perplexity 최소 로 최적 k 선정
#
# 실행: python lda_coherence_perplexity.py
#
# 생성 파일 (out/opportunity_score/coherence_perplexity/):
#   lda_coherence_perplexity_scores.csv  (k별 점수)
#   lda_action_topic_summary.csv         (선정 k 토픽 요약)
#   lda_document_actions.csv             (문서별 Action 라벨)
#   lda_cluster{N}_perplexity.png
#   lda_cluster{N}_coherence.png
#   lda_cluster{N}_coherence_perplexity.png
#   lda_cluster{N}.html                  (pyLDAvis 인터랙티브 시각화)
# ============================================================

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

from organize_files import organize_project_files

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "opportunity_score")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "out", "opportunity_score")
LDA_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "coherence_perplexity")

INPUT_CSV = os.path.join(DATA_DIR, "형태소추출_수정본.csv")
OUTPUT_SCORES_CSV = os.path.join(LDA_OUTPUT_DIR, "lda_coherence_perplexity_scores.csv")
OUTPUT_TOPIC_SUMMARY = os.path.join(LDA_OUTPUT_DIR, "lda_action_topic_summary.csv")
OUTPUT_DOCUMENT_ACTIONS = os.path.join(LDA_OUTPUT_DIR, "lda_document_actions.csv")

LDA_TOPIC_MIN = 2
LDA_TOPIC_MAX = 10
LDA_PASSES_TUNING = 2
LDA_ITER_TUNING = 100
LDA_PASSES_FINAL = 20
LDA_ITER_FINAL = 200
LDA_RANDOM_STATE = 42
LDA_COHERENCE_TOPN = 5

ACTOR_NAME_MAP = {
    1: "세정 루틴/외출 전 냄새 걱정",
    2: "두피·샴푸·탈모 관리",
    3: "드라이샴푸·향수/미스트 사용",
    4: "머리감기 귀찮음·말리기 부담",
}


def organize_output_dirs():
    organize_project_files(SCRIPT_DIR)


def set_korean_font():
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False


def tokenize_for_lda(text):
    return [token for token in str(text).split() if len(token) >= 2]


def build_lda_corpus(texts):
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(doc) for doc in texts]
    return dictionary, corpus


def select_optimal_topic_count(k_list, coherence_scores, perplexity_scores):
    """1순위: Coherence 최대 / 2순위: 동률 시 Perplexity 최소"""
    if len(k_list) == 1:
        return k_list[0]

    c_arr = np.array(coherence_scores, dtype=float)
    p_arr = np.array(perplexity_scores, dtype=float)

    max_coherence = c_arr.max()
    best_indices = np.flatnonzero(np.isclose(c_arr, max_coherence, rtol=0, atol=1e-12))

    if len(best_indices) == 1:
        return k_list[int(best_indices[0])]

    tie_index = best_indices[np.argmin(p_arr[best_indices])]
    return k_list[int(tie_index)]


def evaluate_topic_range(texts, dictionary, corpus, topic_range):
    k_list = []
    coherence_scores = []
    perplexity_scores = []

    for num_topics in topic_range:
        if num_topics >= len(corpus):
            continue

        lda_model = LdaModel(
            corpus,
            num_topics=num_topics,
            id2word=dictionary,
            passes=LDA_PASSES_TUNING,
            iterations=LDA_ITER_TUNING,
            random_state=LDA_RANDOM_STATE,
        )

        coherence_model = CoherenceModel(
            model=lda_model,
            texts=texts,
            dictionary=dictionary,
            topn=LDA_COHERENCE_TOPN,
            processes=1,
        )

        k_list.append(num_topics)
        coherence_scores.append(coherence_model.get_coherence())
        perplexity_scores.append(float(np.exp(-lda_model.log_perplexity(corpus))))

    return k_list, coherence_scores, perplexity_scores


def plot_lda_scores(cluster_id, k_list, coherence_scores, perplexity_scores, optimal_k):
    plt.figure(figsize=(7, 4))
    plt.plot(k_list, perplexity_scores, marker="o")
    plt.axvline(optimal_k, color="#C0392B", linestyle="--", linewidth=1.2, label=f"선정 k={optimal_k}")
    plt.xlabel("Number of Topics")
    plt.ylabel("Perplexity Scores")
    plt.title(f"Cluster {cluster_id} Perplexity")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    perplexity_path = os.path.join(LDA_OUTPUT_DIR, f"lda_cluster{cluster_id}_perplexity.png")
    plt.savefig(perplexity_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"저장 완료: {perplexity_path}")

    plt.figure(figsize=(7, 4))
    plt.plot(k_list, coherence_scores, marker="o", color="#ED7D31")
    plt.axvline(optimal_k, color="#C0392B", linestyle="--", linewidth=1.2, label=f"선정 k={optimal_k}")
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Scores")
    plt.title(f"Cluster {cluster_id} Coherence")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    coherence_path = os.path.join(LDA_OUTPUT_DIR, f"lda_cluster{cluster_id}_coherence.png")
    plt.savefig(coherence_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"저장 완료: {coherence_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(k_list, perplexity_scores, marker="o")
    axes[0].axvline(optimal_k, color="#C0392B", linestyle="--", linewidth=1.2, label=f"선정 k={optimal_k}")
    axes[0].set_xlabel("Number of Topics")
    axes[0].set_ylabel("Perplexity Scores")
    axes[0].set_title(f"Cluster {cluster_id} Perplexity")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(k_list, coherence_scores, marker="o", color="#ED7D31")
    axes[1].axvline(optimal_k, color="#C0392B", linestyle="--", linewidth=1.2, label=f"선정 k={optimal_k}")
    axes[1].set_xlabel("Number of Topics")
    axes[1].set_ylabel("Coherence Scores")
    axes[1].set_title(f"Cluster {cluster_id} Coherence")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    combined_path = os.path.join(LDA_OUTPUT_DIR, f"lda_cluster{cluster_id}_coherence_perplexity.png")
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"저장 완료: {combined_path}")


def save_lda_html(cluster_id, lda_model, corpus, dictionary, max_docs=4000):
    """pyLDAvis HTML 저장 (대용량 corpus는 샘플링)"""
    html_path = os.path.join(LDA_OUTPUT_DIR, f"lda_cluster{cluster_id}.html")
    vis_corpus = corpus
    if len(corpus) > max_docs:
        rng = np.random.default_rng(LDA_RANDOM_STATE)
        sample_idx = rng.choice(len(corpus), size=max_docs, replace=False)
        vis_corpus = [corpus[i] for i in np.sort(sample_idx)]
        print(
            f"Cluster {cluster_id}: HTML용 corpus {len(corpus)} → {len(vis_corpus)} 샘플링"
        )

    try:
        # pyLDAvis 내부 joblib 병렬 처리가 Windows에서 OOM/크래시를 유발할 수 있음
        os.environ["LOKY_MAX_CPU_COUNT"] = "1"
        vis_data = gensimvis.prepare(
            lda_model,
            vis_corpus,
            dictionary,
            mds="tsne",
            sort_topics_by="coherence",
        )
        pyLDAvis.save_html(vis_data, html_path)
        print(f"저장 완료: {html_path}")
    except Exception as exc:
        print(f"경고: Cluster {cluster_id} HTML 생성 실패 → {exc}")


def assign_dominant_topics(lda_model, corpus):
    assigned_topics = []
    for doc_topics in lda_model.get_document_topics(corpus):
        if not doc_topics:
            assigned_topics.append(0)
            continue
        topic_probs = [prob for _, prob in doc_topics]
        topic_ids = [topic_id for topic_id, _ in doc_topics]
        assigned_topics.append(topic_ids[int(np.argmax(topic_probs))])
    return assigned_topics


def make_action_label(topic_id, lda_model, num_words=3):
    top_words = lda_model.show_topic(topic_id, topn=num_words)
    keyword_label = "_".join(word for word, _ in top_words)
    return f"A{topic_id + 1:02d}_{keyword_label}"


def load_data():
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    required_cols = ["cluster", "tokenized_content"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_cols}")

    df["tokenized_content"] = df["tokenized_content"].fillna("").astype(str)
    return df


def run_lda_evaluation(df):
    organize_output_dirs()
    set_korean_font()

    lda_metrics_rows = []
    lda_summary_rows = []
    document_action_rows = []

    df = df.copy()
    df["action"] = "A99_기타"

    print("원본 데이터 shape:", df.shape)
    print(df["cluster"].value_counts().sort_index())

    for cluster_id in sorted(df["cluster"].unique()):
        cluster_mask = df["cluster"] == cluster_id
        cluster_index_all = df.loc[cluster_mask].index
        cluster_texts = [
            tokenize_for_lda(text)
            for text in df.loc[cluster_mask, "tokenized_content"]
        ]

        valid_indices = [i for i, tokens in enumerate(cluster_texts) if len(tokens) > 0]
        if len(valid_indices) == 0:
            print(f"\nCluster {cluster_id}: 유효 토큰 없음 → A99_기타 유지")
            continue

        texts = [cluster_texts[i] for i in valid_indices]
        dictionary, corpus = build_lda_corpus(texts)

        if len(dictionary) == 0:
            print(f"\nCluster {cluster_id}: 사전 생성 실패 → A99_기타 유지")
            continue

        max_topics = min(LDA_TOPIC_MAX - 1, len(corpus) - 1)
        min_topics = LDA_TOPIC_MIN

        if max_topics < min_topics:
            optimal_k = 1
            k_list, coherence_scores, perplexity_scores = [1], [0.0], [0.0]
            lda_metrics_rows.append({
                "cluster": cluster_id,
                "actor": ACTOR_NAME_MAP.get(cluster_id, "미정"),
                "num_topics": 1,
                "coherence": 0.0,
                "perplexity": 0.0,
                "is_selected": True,
            })
        else:
            topic_range = range(min_topics, max_topics + 1)
            k_list, coherence_scores, perplexity_scores = evaluate_topic_range(
                texts, dictionary, corpus, topic_range
            )
            optimal_k = select_optimal_topic_count(
                k_list, coherence_scores, perplexity_scores
            )
            plot_lda_scores(
                cluster_id, k_list, coherence_scores, perplexity_scores, optimal_k
            )

            for num_topics, coherence, perplexity in zip(
                k_list, coherence_scores, perplexity_scores
            ):
                lda_metrics_rows.append({
                    "cluster": cluster_id,
                    "actor": ACTOR_NAME_MAP.get(cluster_id, "미정"),
                    "num_topics": num_topics,
                    "coherence": round(float(coherence), 6),
                    "perplexity": round(float(perplexity), 6),
                    "is_selected": num_topics == optimal_k,
                })

        lda_model = LdaModel(
            corpus,
            num_topics=optimal_k,
            id2word=dictionary,
            passes=LDA_PASSES_FINAL,
            iterations=LDA_ITER_FINAL,
            random_state=LDA_RANDOM_STATE,
        )

        save_lda_html(cluster_id, lda_model, corpus, dictionary)

        topic_ids = assign_dominant_topics(lda_model, corpus)
        topic_label_map = {
            topic_id: make_action_label(topic_id, lda_model)
            for topic_id in range(optimal_k)
        }

        cluster_index = cluster_index_all[valid_indices]
        assigned_actions = [topic_label_map[topic_id] for topic_id in topic_ids]
        df.loc[cluster_index, "action"] = assigned_actions

        for row_idx, action in zip(cluster_index, assigned_actions):
            document_action_rows.append({
                "row_id": row_idx,
                "cluster": cluster_id,
                "action": action,
            })

        print(f"\nCluster {cluster_id} LDA 결과")
        print(f"- 문서 수: {len(texts)}")
        print(f"- 후보 k: {k_list}")
        print(f"- Coherence: {[round(v, 4) for v in coherence_scores]}")
        print(f"- Perplexity: {[round(v, 4) for v in perplexity_scores]}")
        print(f"- 선정 k: {optimal_k}")

        for topic_id in range(optimal_k):
            top_words = lda_model.show_topic(topic_id, topn=8)
            top_word_text = ", ".join(f"{word}({prob:.3f})" for word, prob in top_words)
            print(f"  Topic {topic_id}: {top_word_text}")

            lda_summary_rows.append({
                "cluster": cluster_id,
                "actor": ACTOR_NAME_MAP.get(cluster_id, "미정"),
                "topic_id": topic_id,
                "action": topic_label_map[topic_id],
                "optimal_k": optimal_k,
                "coherence_at_k": coherence_scores[k_list.index(optimal_k)]
                if optimal_k in k_list else np.nan,
                "perplexity_at_k": perplexity_scores[k_list.index(optimal_k)]
                if optimal_k in k_list else np.nan,
                "top_words": top_word_text,
            })

    lda_metrics_df = pd.DataFrame(lda_metrics_rows)
    lda_metrics_df = lda_metrics_df.sort_values(["cluster", "num_topics"]).reset_index(drop=True)
    lda_metrics_df.to_csv(OUTPUT_SCORES_CSV, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {OUTPUT_SCORES_CSV}")
    print(lda_metrics_df.to_string(index=False))

    lda_summary = pd.DataFrame(lda_summary_rows)
    lda_summary.to_csv(OUTPUT_TOPIC_SUMMARY, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {OUTPUT_TOPIC_SUMMARY}")

    document_actions = pd.DataFrame(document_action_rows)
    document_actions.to_csv(OUTPUT_DOCUMENT_ACTIONS, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {OUTPUT_DOCUMENT_ACTIONS}")

    return lda_metrics_df, lda_summary, document_actions


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"입력 파일 없음: {INPUT_CSV}")

    df = load_data()
    run_lda_evaluation(df)
    print("\n완료: Coherence / Perplexity 계산 및 최적 k 선정")


if __name__ == "__main__":
    main()

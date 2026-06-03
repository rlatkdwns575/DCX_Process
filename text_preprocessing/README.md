# text_preprocessing

해외(영문) 텍스트 데이터 — 특히 **냄새(smell / fragrance / scent)** 관련 도메인 — 를 위한 전처리·토픽 선별·임베딩·클러스터링 비교 노트북입니다.

## 파이프라인

```mermaid
flowchart LR
    raw[원본 CSV<br/>data/text_preprocessing/raw_data.csv] --> nb1[text_preprocessing_1st_ENG.ipynb]
    nb1 --> csv1[out/text_preprocessing/ENG_1st_contents.csv]
    csv1 --> nb2[text_preprocessing_2nd_ENG.ipynb]
    nb2 --> csv2[out/text_preprocessing/ENG_2nd_contents.csv]
    csv2 --> nb3[embedding_clustering_comparison_ENG.ipynb]
    nb3 --> result[out/text_preprocessing/clustering_comparison_results.csv]
```

| 단계 | 노트북 | 입력 | 출력 |
| --- | --- | --- | --- |
| 1차 | `text_preprocessing_1st_ENG.ipynb` | `data/text_preprocessing/raw_data.csv` | `out/text_preprocessing/ENG_1st_contents.csv` |
| 2차 | `text_preprocessing_2nd_ENG.ipynb` | `out/text_preprocessing/ENG_1st_contents.csv` | `out/text_preprocessing/ENG_2nd_contents.csv` |
| 비교 | `embedding_clustering_comparison_ENG.ipynb` | `out/text_preprocessing/ENG_2nd_contents.csv` | `out/text_preprocessing/clustering_comparison_results.csv` |

> 경로는 DCX 프로젝트 루트 기준입니다. 노트북 설정 셀에서 `PROJECT_ROOT` · `DATA_DIR` · `OUT_DIR`을 자동으로 잡습니다.

## 요구사항

- Python 3.10+
- GPU(CUDA / Apple MPS) 권장 (1차 분류기·임베딩·클러스터링에서 속도 차이 큼)

### 패키지

**1차**

```bash
pip install pandas transformers torch tqdm
```

**2차**

```bash
pip install bertopic sentence-transformers umap-learn hdbscan scikit-learn pandas numpy tqdm plotly openpyxl
```

**임베딩 × 클러스터링 비교**

```bash
pip install sentence-transformers transformers torch scikit-learn hdbscan pandas numpy matplotlib tqdm
```

각 노트북 상단 `0. 의존성 설치` 셀에 동일 명령이 주석 처리되어 있습니다.

---

## `text_preprocessing_1st_ENG.ipynb` — 1차 전처리

**목적**: 원본 CSV(제목 + 본문 + 댓글 컬럼 분리)를 받아 단계별로 정제한 뒤 `ENG_1st_contents.csv`로 저장합니다.

### 실행 전 설정

| 변수 | 설명 | 기본값 |
| --- | --- | --- |
| `RAW_CSV_PATH` | 원본 CSV 경로 | `data/text_preprocessing/raw_data.csv` |
| `TITLE_COL` | 제목 컬럼명 | `title` |
| `CONTENT_COL` | 본문 컬럼명 | `content` |
| `COMMENT_COL` | 댓글 컬럼명 | `comment` |
| `OUTPUT_CSV` | 결과 파일 | `out/text_preprocessing/ENG_1st_contents.csv` |
| `MIN_CHAR_LEN` | 최소 글자 수 | `30` |
| `BATCH_SIZE` | 분류 배치 크기 | `64` |

> `RAW_CSV_PATH`, `TITLE_COL`, `CONTENT_COL`, `COMMENT_COL`만 원본 CSV에 맞게 바꾸면 다른 데이터셋에도 동일하게 적용할 수 있습니다.

### 처리 단계

| Step | 내용 |
| --- | --- |
| 1 | 원본 CSV 로드 |
| 2 | 제목 + 본문 + 댓글 → `contents` 단일 컬럼으로 결합 (`source`: `title` / `content` / `comment`) |
| 3 | URL 제거 (`http\S+`, `www.\S+`) |
| 4 | 30자 미만 텍스트 제거 |
| 5 | 중복 제거 |
| 5-1 | (선택) `ㅋㅎ` 2회 이상 반복 표현 제거 — 한국어 데이터용 |
| 6~8 | 스팸·광고 분류 (`blockenters/sms-spam-classifier`) → `LABEL_0`(ham)만 유지 |
| 9~11 | AI 생성 텍스트 감지 (`fakespot-ai/roberta-base-ai-text-detection-v1`) → `Human`만 유지 |
| 12 | `ENG_1st_contents.csv` 저장 |

### 사용 모델

| 용도 | 모델 |
| --- | --- |
| 스팸·광고 | `blockenters/sms-spam-classifier` (`LABEL_0` = ham, `LABEL_1` = spam) |
| AI 텍스트 | `fakespot-ai/roberta-base-ai-text-detection-v1` |

Step 6·9에서 `id2label`을 출력하므로, 라벨명이 다르면 필터 조건(`LABEL_0`, `HUMAN_LABEL`)을 그에 맞게 수정합니다. AI 필터는 데이터량이 크게 줄 수 있습니다.

### 출력: `ENG_1st_contents.csv`

| 컬럼 | 설명 |
| --- | --- |
| `contents` | 정제된 텍스트 (1행 = 1문장) |

메타 컬럼(`source`, `spam_label`, `ai_label` 등)을 함께 저장하려면 Step 12 셀의 주석을 해제합니다.

---

## `text_preprocessing_2nd_ENG.ipynb` — 2차 전처리 (토픽 클러스터링)

**목적**: 1차 결과(`ENG_1st_contents.csv`)를 BERTopic으로 토픽별 클러스터링한 뒤, 토픽 내용을 직접 확인해 **유효한 토픽만 남긴** `ENG_2nd_contents.csv`를 생성합니다.

### 입력·출력

| 항목 | 값 |
| --- | --- |
| 입력 | `out/text_preprocessing/ENG_1st_contents.csv` (`content` / `contents` 컬럼명 자동 보정) |
| 출력 | `out/text_preprocessing/ENG_2nd_contents.csv` |
| 임베딩 캐시 | `out/text_preprocessing/ENG_1st_embeddings.npy` |

### 임베딩 모델

| 모델 | 장점 | 단점 |
| --- | --- | --- |
| `sentence-transformers/all-MiniLM-L6-v2` | 가볍고 빠름 (384d) | mpnet 대비 품질 약간 낮음 |
| **`sentence-transformers/all-mpnet-base-v2`** | **768d, 영어 의미 유사도 SBERT 계열 상위권 — 채택** | MiniLM보다 느림 |
| pure RoBERTa | 문장 표현 가능 | 평균 풀링·클러스터링 품질이 SBERT 대비 낮음 |

`BERTopic`은 *임베딩 → UMAP → HDBSCAN → c-TF-IDF*를 한 번에 처리하므로, 클러스터·대표 키워드·시각화를 보며 유효 토픽을 고르기에 적합합니다.

### 처리 단계

| Step | 내용 |
| --- | --- |
| 1 | `ENG_1st_contents.csv` 로드·정리 |
| 2 | `all-mpnet-base-v2` 로드 |
| 3 | 임베딩 계산 (`ENG_1st_embeddings.npy` 캐시) |
| 4 | UMAP(5d, cosine) + HDBSCAN(`min_cluster_size=30`) + CountVectorizer(영문 불용어, 1~2-gram) |
| 5 | BERTopic 학습 |
| 6 | `get_topic_info()` — 토픽 ID, 문서 수, 대표 키워드 |
| 7-1 | `show_topic_examples(topic_id, n)` — 토픽별 샘플 문장 수동 확인 |
| 7-2 | `export_topic_examples_wide_excel()` — 토픽별 샘플 20개 → CSV/XLSX (`topic_examples_20_wide.csv` 등). AI에게 파일 전체를 읽고 도메인 관련 토픽 ID만 골라 달라고 요청하는 워크플로우 권장 |
| 8 | (선택) `visualize_topics()`, `visualize_barchart()` |
| 9 | `VALID_TOPIC_IDS` — 검토 후 유효 토픽 ID 입력 (노이즈 `-1`은 기본 제외) |
| 10 | 유효 토픽만 필터링 |
| 11 | `ENG_2nd_contents.csv` 저장 |

### 권장 워크플로우

1. Step 1~7까지 실행해 토픽 키워드·샘플 문장을 확인합니다.
2. Step 7-2로보낸 `topic_examples_20_wide.csv`(또는 XLSX)를 검토하거나 AI에 전달해, 냄새 도메인과 **관련 있는 토픽 ID**를 정합니다.
3. Step 9의 `VALID_TOPIC_IDS`에 반영한 뒤 Step 10~11을 실행해 `ENG_2nd_contents.csv`를 생성합니다.

### 출력: `ENG_2nd_contents.csv`

| 컬럼 | 설명 |
| --- | --- |
| `contents` | 유효 토픽에 속하는 정제 텍스트 |
| `topic` | BERTopic 토픽 ID (추적·분석용) |

`contents`만 필요하면 Step 11 셀 주석을 참고해 `topic` 컬럼 저장을 생략할 수 있습니다.

---

## `embedding_clustering_comparison_ENG.ipynb` — 임베딩 × 클러스터링 조합 비교

**목적**: 3개 임베딩(SBERT / E5 / SRoBERTa) × 4개 클러스터링(KMeans / HDBSCAN / DBSCAN / Hierarchical) **총 12개 조합**을 `ENG_2nd_contents.csv` 전체에 적용해 내부 지표로 비교하고, **Silhouette 점수가 가장 높은 조합**을 선별합니다.

### 비교 대상

| 임베딩 | 클러스터링 |
| --- | --- |
| SBERT | KMeans / HDBSCAN / DBSCAN / Hierarchical |
| E5 | KMeans / HDBSCAN / DBSCAN / Hierarchical |
| SRoBERTa | KMeans / HDBSCAN / DBSCAN / Hierarchical |

### 내부 지표

- **Silhouette Score** (제1 기준, 높을수록 좋음, -1 ~ 1)
- **Calinski–Harabasz Index** (높을수록 좋음)
- **Davies–Bouldin Index** (낮을수록 좋음)

### K 선정

- **KMeans**: `K ∈ [2, 50]` 전부 스윕 → Step 6a **Elbow(inertia)** 로 적정 K 구간 확인 → Step 6b **Silhouette** 로 대표 K 확정(Silhouette 최대). `K=1`은 제외.
- **Hierarchical**: `distance_threshold`로 **자동 병합** — 결과 클러스터 수는 데이터에 따라 결정.
- **HDBSCAN · DBSCAN**: K 없음, 파라미터로 단발 실행.

모든 임베딩은 **L2 정규화** 후 지표 계산. Silhouette은 `metric="cosine"`, CH·DB는 정규화 벡터의 유클리드 거리.

### 실행 전 설정

| 변수 | 설명 | 기본값 |
| --- | --- | --- |
| `INPUT_CSV` | 입력 CSV | `out/text_preprocessing/ENG_2nd_contents.csv` |
| `RESULT_CSV` | 요약 결과 | `out/text_preprocessing/clustering_comparison_results.csv` |
| `USE_FULL_DATA` | 전체 행 사용 | `True` |
| `SAMPLE_SIZE` | 샘플 모드 시 행 수 | `5000` |
| `K_MIN`, `K_MAX` | KMeans K 스윕 범위 | `2`, `50` |
| `HIER_DISTANCE_THRESHOLD` | 계층적 군집 병합 임계 | `0.5` |
| `HDBSCAN_MIN_CLUSTER_SIZE` | HDBSCAN | `30` |
| `DBSCAN_EPS` / `DBSCAN_MIN_SAMPLES` | DBSCAN | `0.4` / `10` |

빠른 시험만 필요하면 `USE_FULL_DATA = False`와 `SAMPLE_SIZE`를 설정합니다.

### 임베딩 모델 카탈로그

| 키 | 모델 | 프리픽스 |
| --- | --- | --- |
| SBERT | `sentence-transformers/all-mpnet-base-v2` | (없음) |
| E5 | `intfloat/e5-base-v2` | `passage: ` (문서 임베딩 시 필수) |
| SRoBERTa | `sentence-transformers/all-distilroberta-v1` | (없음) |

메모리 여유 시 SRoBERTa를 `sentence-transformers/all-roberta-large-v1`로 교체 가능. 캐시 파일 `compare_emb_<키>.npy`는 행 수가 바뀌면 자동 재계산됩니다.

### 처리 단계

| Step | 내용 |
| --- | --- |
| 1 | `ENG_2nd_contents.csv` 로드 (`USE_FULL_DATA` / 샘플링) |
| 2 | 임베딩 모델 카탈로그 정의 |
| 3 | 모델별 임베딩 계산·캐시 (`compare_emb_*.npy`) |
| 4 | L2 정규화 + Silhouette / CH / DB 계산 헬퍼 |
| 5 | KMeans / Hierarchical / HDBSCAN / DBSCAN 실행 함수 |
| 6 | 12개 조합 실행 (KMeans는 K=2..50 스윕) |
| 6a | KMeans Elbow (Inertia vs K) |
| 6b | KMeans Silhouette vs K |
| 7 | 결과 표 (Silhouette 제1 기준 정렬) |
| 8 | Silhouette 막대 차트 (노이즈 비율·클러스터 수 주석) |
| 9 | 최적 조합 출력 → `clustering_comparison_results.csv` 저장 |

### 해석 시 유의

- HDBSCAN / DBSCAN은 **노이즈 비율이 높으면** Silhouette은 높아도 유효 클러스터가 적을 수 있음 — **노이즈 비율·클러스터 수**를 함께 확인.
- DBSCAN 결과가 전부 노이즈면 `DBSCAN_EPS`를 0.5–0.6으로 올려 재실행.
- 전체 데이터 기준 Step 6(KMeans 49×3회)과 계층적 군집은 **수십 분~수 시간** 걸릴 수 있음.

Silhouette 1위 조합을 확인한 뒤, 2차 노트북의 임베딩·클러스터러 설정을 해당 조합으로 맞추면 됩니다.

---

## 실행 순서

1. `text_preprocessing_1st_ENG.ipynb` — `RAW_CSV_PATH`, `TITLE_COL`, `CONTENT_COL`, `COMMENT_COL` 수정 후 전체 실행 → `out/text_preprocessing/ENG_1st_contents.csv`
2. `text_preprocessing_2nd_ENG.ipynb` — Step 6~7(또는 7-2)로 토픽 검토 → `VALID_TOPIC_IDS` 작성 → Step 10~11 실행 → `out/text_preprocessing/ENG_2nd_contents.csv`
3. `embedding_clustering_comparison_ENG.ipynb` — Step 1~9 실행 → `out/text_preprocessing/clustering_comparison_results.csv` 및 최적 조합 확인

## 트러블슈팅

| 증상 | 조치 |
| --- | --- |
| 컬럼 `KeyError` | 1차 노트북 `TITLE_COL`, `CONTENT_COL`, `COMMENT_COL`이 원본 CSV와 일치하는지 확인 |
| 스팸·AI 라벨이 예상과 다름 | Step 6·9의 `id2label` 출력을 보고 `LABEL_0`, `HUMAN_LABEL` 수정 |
| 2차에서 대부분 토픽 `-1`(노이즈) | `min_cluster_size` 축소, UMAP `n_neighbors` / `min_dist` 조정 |
| GPU 메모리 부족 | `BATCH_SIZE` 또는 `encode`의 `batch_size` 축소 |
| `ENG_1st_embeddings.npy` 불일치 | `out/text_preprocessing/` 내 캐시 삭제 후 2차 Step 3 재실행 |
| `compare_emb_*.npy` 불일치 | `out/text_preprocessing/` 내 해당 `.npy` 삭제 후 비교 노트북 Step 3 재실행 |
| DBSCAN 전부 노이즈 | `DBSCAN_EPS` 상향(0.5–0.6) 후 재실행 |

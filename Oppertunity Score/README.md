# Opportunity Score

Actor(페르소나) × Action(LDA 토픽) 단위로 **중요도(Importance)**, **만족도(Satisfaction)**, **기회 점수(Opportunity Score)** 를 계산하고 시각화하는 프로젝트입니다.

클러스터별 LDA 최적 k 선정(Coherence / Perplexity) → Action 분류 → Opportunity Map 생성까지 한 번에 실행할 수 있습니다.

---

## 개요

| 개념 | 설명 |
|------|------|
| **Actor** | 기존 `cluster` (1~4). 각 Actor는 헤어케어 관련 페르소나를 나타냅니다. |
| **Action** | 클러스터별 LDA 토픽. 문서마다 1개의 Action 라벨이 부여됩니다. (예: `A03_머리_감다_냄새`) |
| **Importance** | Actor-Action 언급 빈도를 0~10으로 정규화한 중요도 |
| **Satisfaction** | 감성사전(`SentiWord_info.json`) 기반 감성 점수를 0~10으로 정규화한 만족도 |
| **Opportunity** | 개선 우선순위 지표. 중요도는 높은데 만족도가 낮을수록 점수가 커집니다. |

### Opportunity Score 공식

```
Opportunity = Importance + max(Importance - Satisfaction, 0)
```

- Importance가 높을수록 점수 ↑  
- Satisfaction이 낮을수록(중요도 대비 갭이 클수록) 점수 ↑  
- Underserved Area(미충족 영역)에 해당하는 Action을 우선 개선 후보로 볼 수 있습니다.

### Actor 매핑

| Cluster | Actor 설명 |
|---------|------------|
| 1 | 세정 루틴 / 외출 전 냄새 걱정 |
| 2 | 두피·샴푸·탈모 관리 |
| 3 | 드라이샴푸·향수/미스트 사용 |
| 4 | 머리감기 귀찮음·말리기 부담 |

---

## 프로젝트 구조

입·출력 경로는 **DCX 루트**의 `data/` · `out/`을 사용합니다. 전체 구조는 [`../README.md`](../README.md)를 참고하세요.

```
DCX/
├── data/opportunity_score/          # 입력 CSV·감성사전
├── out/opportunity_score/
│   ├── coherence_perplexity/        # LDA 결과
│   └── opportunity/                 # Opportunity Score 결과
└── Oppertunity Score/
    ├── lda_coherence_perplexity.py
    ├── OP_score.py
    ├── organize_files.py
    └── requirements.txt
```

---

## 설치

Python 3.10+ 권장 (Windows / macOS / Linux)

```bash
cd "Oppertunity Score"
pip install -r requirements.txt
```

### 주요 의존성

- `pandas`, `numpy`, `matplotlib`
- `gensim` — LDA, Coherence, Perplexity
- `pyLDAvis` — LDA 인터랙티브 HTML

---

## 실행 방법

**반드시 아래 순서대로** 실행합니다.

### 1단계 — LDA + Action 분류

```bash
python lda_coherence_perplexity.py
```

클러스터별로 k=2~9(또는 문서 수 한도)를 탐색하고, **Coherence 최대 → 동률 시 Perplexity 최소** 로 최적 k를 선정합니다.

**생성 파일** (`out/opportunity_score/coherence_perplexity/`)

| 파일 | 설명 |
|------|------|
| `lda_coherence_perplexity_scores.csv` | k별 Coherence / Perplexity 점수 |
| `lda_action_topic_summary.csv` | 선정 k 토픽 요약 (Action 라벨, top words) |
| `lda_document_actions.csv` | 문서(row_id)별 Action 라벨 |
| `lda_optimal_k_summary.csv` | 클러스터별 선정 k 요약 |
| `lda_cluster{N}_perplexity.png` | Perplexity 그래프 |
| `lda_cluster{N}_coherence.png` | Coherence 그래프 |
| `lda_cluster{N}_coherence_perplexity.png` | 통합 그래프 |
| `lda_cluster{N}.html` | pyLDAvis 인터랙티브 시각화 |

> LDA HTML 생성은 문서 수가 많을 경우 수 분 이상 걸릴 수 있습니다.  
> Windows에서는 메모리 부족을 방지하기 위해 HTML용 corpus를 최대 4,000건으로 샘플링합니다.

### 2단계 — Opportunity Score + 그래프

```bash
python OP_score.py
```

**생성 파일** (`out/opportunity_score/opportunity/`)

| 파일 | 설명 |
|------|------|
| `actor_action_opportunity_score.csv` | Actor-Action별 점수 테이블 |
| `actor_action_opportunity_map.png` | Opportunity Map (산점도) |
| `actor_action_opportunity_top20.png` | Opportunity Top 20 막대 그래프 |
| `actor_action_opportunity_heatmap.png` | Actor × Action 히트맵 |

### (선택) 산출물 폴더 정리

루트나 `out/` 에 흩어진 CSV·PNG·HTML을 하위 폴더로 이동합니다.

```bash
python organize_files.py
```

---

## Opportunity Map 설명

`actor_action_opportunity_map.png`는 Actor-Action을 **중요도(x)** × **만족도(y)** 평면에 배치합니다.

- **축 중앙(0)** = 해당 지표의 **전체 평균**
- **유도선 1**: 감정 평균 지점(y축) → 우상단 (10, 10)
- **유도선 2**: 중요도 평균 지점(x축) → 우상단 (10, 10)
- 두 유도선이 우상단에서 만나며 3개 영역을 구분합니다.

| 영역 | 의미 |
|------|------|
| **A: Overserved** | 만족도가 상대적으로 높음 (과잉 충족) |
| **B: Well-served** | 중요도·만족도 균형 |
| **C: Underserved** | 중요도 대비 만족도 낮음 → **개선 우선 후보** |

포인트 라벨 형식:

```
Actor{번호}_Action{번호}
키워드1 키워드2 키워드3
```

---

## 입력 데이터 주의사항

- `lda_coherence_perplexity.py` → `data/opportunity_score/형태소추출_수정본.csv`
- `OP_score.py` → `data/opportunity_score/형태소추출.csv`

두 파일의 **행 순서(row_id)가 일치**해야 `lda_document_actions.csv`와 정확히 매칭됩니다.  
데이터를 교체할 때는 두 CSV를 함께 갱신하거나, 동일 파일을 사용하도록 스크립트 경로를 맞춰 주세요.

### 필수 컬럼

**LDA 입력** (`형태소추출_수정본.csv`)

- `cluster`, `tokenized_content`

**Opportunity Score 입력** (`형태소추출.csv`)

- `content`, `cluster`, `cleaned_content`, `tokenized_content`

---

## 트러블슈팅

### `TerminatedWorkerError` (pyLDAvis / joblib)

```
A worker process managed by the executor was unexpectedly terminated...
```

**원인:** pyLDAvis가 내부적으로 병렬 worker를 사용하는데, 문서 수가 많으면 Windows에서 메모리 부족(OOM)으로 worker가 종료됩니다.

**대응:** `lda_coherence_perplexity.py`의 `save_lda_html()`에서 corpus 샘플링·worker 제한·예외 처리가 적용되어 있습니다. HTML만 실패해도 CSV/PNG는 정상 생성됩니다. 그래도 실패하면 `max_docs` 값을 더 줄여 보세요.

### `FileNotFoundError: lda_coherence_perplexity_scores.csv`

1단계 `python lda_coherence_perplexity.py`를 먼저 실행했는지 확인하세요.

### 한글 깨짐

Windows에서는 **맑은 고딕(Malgun Gothic)** 등 한글 폰트가 설치되어 있어야 그래프 한글이 정상 표시됩니다.

---

## 커스터마이징

| 파일 | 수정 포인트 |
|------|-------------|
| `lda_coherence_perplexity.py` | `LDA_TOPIC_MIN/MAX`, `ACTOR_NAME_MAP`, `INPUT_CSV` |
| `OP_score.py` | `ACTOR_NAME_MAP`, `DOMAIN_SENTIMENT`, `TOP_N`(맵 표시 개수) |
| `organize_files.py` | 산출물 이동 규칙 |

---

## 라이선스 / 참고

- LDA k 선정: Day2 실습 노트북 방식 (Coherence max → Perplexity min)
- 감성 분석: `SentiWord_info.json` 기반

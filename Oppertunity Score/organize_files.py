"""프로젝트 산출물을 out/opportunity_score/coherence_perplexity · out/opportunity_score/opportunity 로 정리"""

import glob
import os
import shutil

LDA_FILE_NAMES = (
    "lda_coherence_perplexity_scores.csv",
    "lda_action_topic_summary.csv",
    "lda_document_actions.csv",
    "lda_optimal_k_summary.csv",
)

LDA_GLOB_PATTERNS = ("lda_cluster*.png", "lda_cluster*.html")
OPPORTUNITY_GLOB_PATTERN = "actor_action_opportunity_*"


def _project_paths(script_dir=None):
    script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data", "opportunity_score")
    output_dir = os.path.join(project_root, "out", "opportunity_score")
    lda_dir = os.path.join(output_dir, "coherence_perplexity")
    opportunity_dir = os.path.join(output_dir, "opportunity")
    return project_root, data_dir, output_dir, lda_dir, opportunity_dir


def _relocate_file(src, dst):
    if not os.path.isfile(src):
        return False

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(src)
    else:
        shutil.move(src, dst)
    return True


def organize_project_files(script_dir=None):
    project_root, data_dir, output_dir, lda_dir, opportunity_dir = _project_paths(script_dir)
    module_dir = script_dir or os.path.dirname(os.path.abspath(__file__))

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(lda_dir, exist_ok=True)
    os.makedirs(opportunity_dir, exist_ok=True)

    moved = []

    senti_root = os.path.join(module_dir, "SentiWord_info.json")
    senti_data = os.path.join(data_dir, "SentiWord_info.json")
    if os.path.isfile(senti_root):
        if os.path.exists(senti_data):
            os.remove(senti_root)
        else:
            shutil.move(senti_root, senti_data)
        moved.append(senti_data)

    search_dirs = [output_dir, module_dir, project_root]
    for search_dir in search_dirs:
        for name in LDA_FILE_NAMES:
            src = os.path.join(search_dir, name)
            dst = os.path.join(lda_dir, name)
            if _relocate_file(src, dst):
                moved.append(dst)

        for pattern in LDA_GLOB_PATTERNS:
            for src in glob.glob(os.path.join(search_dir, pattern)):
                dst = os.path.join(lda_dir, os.path.basename(src))
                if _relocate_file(src, dst):
                    moved.append(dst)

        for src in glob.glob(os.path.join(search_dir, OPPORTUNITY_GLOB_PATTERN)):
            dst = os.path.join(opportunity_dir, os.path.basename(src))
            if _relocate_file(src, dst):
                moved.append(dst)

    legacy_dir = os.path.join(module_dir, "Coherence_Perplexcity")
    if os.path.isdir(legacy_dir):
        for name in os.listdir(legacy_dir):
            src = os.path.join(legacy_dir, name)
            if not os.path.isfile(src):
                continue
            if name.startswith("lda_") or name.startswith("lda_cluster"):
                dst = os.path.join(lda_dir, name)
            elif name.startswith("actor_action_opportunity"):
                dst = os.path.join(opportunity_dir, name)
            else:
                continue
            if _relocate_file(src, dst):
                moved.append(dst)
        if not os.listdir(legacy_dir):
            os.rmdir(legacy_dir)

    return moved


if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    files = organize_project_files(root)
    print(f"정리 완료: {len(files)}개 파일 이동/정리")
    for path in files:
        print(f"  - {path}")

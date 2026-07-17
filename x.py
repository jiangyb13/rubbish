#!/bin/bash
# ================================================================
# Stage 5: Generate Training Pairs（独立脚本）
# ================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/data/huanan/code/jwx1520881/HUAWEI_CrossPairDataset_v2/outputs/outputs_multiperson}"
LOG_DIR="${LOG_DIR:-/data/huanan/code/jwx1520881/HUAWEI_CrossPairDataset_v2/outputs/outputs_multiperson/stage5_logs}"

IDENTITY_OUTPUT_DIR="${IDENTITY_OUTPUT_DIR:-/data/huanan/code/jwx1520881/HUAWEI_CrossPairDataset_v2/outputs_multiperson}"
PERSON_CLUSTER_DIR="${PERSON_CLUSTER_DIR:-$IDENTITY_OUTPUT_DIR}"
STAGE3_MODE="${STAGE3_MODE:-per_video}"
INDEX_FILENAME="${INDEX_FILENAME:-after_pipeline_index.json}"

TRAINING_PAIRS_DIR="${TRAINING_PAIRS_DIR:-$OUTPUT_ROOT/training_pairs}"
TRAINING_PAIRS_JSONL="${TRAINING_PAIRS_JSONL:-$TRAINING_PAIRS_DIR/pairs.jsonl}"
TRAINING_PAIRS_STATS="${TRAINING_PAIRS_STATS:-$TRAINING_PAIRS_DIR/stats.json}"
TRAINING_FIRST_FRAME_DIR="${TRAINING_FIRST_FRAME_DIR:-$TRAINING_PAIRS_DIR/first_frames}"
TRAINING_PAIRS_UNIT_LIST="${TRAINING_PAIRS_UNIT_LIST:-$TRAINING_PAIRS_DIR/person_clusters_list.txt}"
ANGLE_REF_COUNT="${ANGLE_REF_COUNT:-5}"
EMO_REF_COUNT="${EMO_REF_COUNT:-5}"
BODY_POSE_REF_COUNT="${BODY_POSE_REF_COUNT:-5}"
BUCKET_CANDIDATE_TOPK="${BUCKET_CANDIDATE_TOPK:-8}"
MIN_SAME_PREFIX_SHOT_GAP="${MIN_SAME_PREFIX_SHOT_GAP:-0}"
OVERWRITE_SIMILARITY_MATRIX="${OVERWRITE_SIMILARITY_MATRIX:-False}"
OVERWRITE_FIRST_FRAMES="${OVERWRITE_FIRST_FRAMES:-False}"

mkdir -p "$LOG_DIR" "$TRAINING_PAIRS_DIR"

count_lines() {
    local path=$1
    if [ -f "$path" ]; then wc -l < "$path"; else echo 0; fi
}

count_jsonl_tree_by_name() {
    local root=$1
    local filename=$2
    local exclude_path=${3:-}
    "$PYTHON_BIN" -c '
import sys
from pathlib import Path

root = Path(sys.argv[1])
filename = sys.argv[2]
exclude = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 and sys.argv[3] else None
total = 0
if root.exists():
    for path in root.rglob(filename):
        if not path.is_file():
            continue
        if exclude and path.resolve() == exclude:
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as file:
            total += sum(1 for line in file if line.strip())
print(total)
' "$root" "$filename" "$exclude_path"
}

build_person_clusters_unit_list() {
    local identity_root=$1
    local output_list=$2
    mkdir -p "$(dirname "$output_list")"
    "$PYTHON_BIN" -c '
import sys
from pathlib import Path
from tqdm import tqdm

identity_root = Path(sys.argv[1]).resolve()
output_list = Path(sys.argv[2])
rows = []

# 直接获取第一层目录 (UUID 文件夹)
uuid_dirs = [d for d in identity_root.iterdir() if d.is_dir()]

# 使用 tqdm 遍历，速度极快
for uuid_dir in tqdm(uuid_dirs, desc="Processing clusters", unit="dir"):
    # 构造目标路径: identity_root / UUID / identity_matching / person_clusters
    path = uuid_dir / "identity_matching" / "person_clusters"
    
    # 校验目录是否存在及有效性 (保持原过滤逻辑)
    if not path.is_dir():
        continue
    if not any(child.is_dir() and child.name.startswith("person_") for child in path.iterdir()):
        continue
        
    # --- 核心：保持和原代码一致的 parts 逻辑 ---
    # 原代码中 parts = path.parent.relative_to(identity_root).parts
    # 在你的结构下，path.parent 就是 identity_root / UUID / identity_matching
    # 相对路径就是 UUID / identity_matching
    rel_parent = path.parent.resolve().relative_to(identity_root)
    parts = rel_parent.parts
    
    # 严格保持原赋值逻辑
    video = parts[0] if len(parts) > 0 else ""
    part = parts[1] if len(parts) > 1 else ""
    uuid = parts[2] if len(parts) > 2 else ""
    
    rows.append(f"{path.resolve()}|{video}|{part}|{uuid}")

output_list.parent.mkdir(parents=True, exist_ok=True)
output_list.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
print(f"[unit_list] {output_list}: {len(rows)} person_clusters roots")
' "$identity_root" "$output_list"
}

echo "========================================"
echo "  Stage 5: Generate Training Pairs"
echo "  identity: $IDENTITY_OUTPUT_DIR"
echo "  输出:     $TRAINING_PAIRS_DIR"
echo "  索引文件: $INDEX_FILENAME"
echo "  日志:     $LOG_DIR"
echo "========================================"

if [ "$STAGE3_MODE" = "per_video" ]; then
    echo "[Stage 5] per_video 模式：生成 person_clusters 列表并按视频镜像输出"
    build_person_clusters_unit_list "$IDENTITY_OUTPUT_DIR" "$TRAINING_PAIRS_UNIT_LIST"
    UNIT_COUNT=$(count_lines "$TRAINING_PAIRS_UNIT_LIST")
    if [ "$UNIT_COUNT" -eq 0 ]; then
        echo "[ERROR] Stage 5 未找到任何 person_clusters: $IDENTITY_OUTPUT_DIR"
        exit 1
    fi

    env TASK_NAME=generate_training_pairs "$PYTHON_BIN" main.py \
        --person_clusters_dir "$PERSON_CLUSTER_DIR" \
        --unit_list_file "$TRAINING_PAIRS_UNIT_LIST" \
        --unit_list_input_base_dir "$IDENTITY_OUTPUT_DIR" \
        --output_jsonl "$TRAINING_PAIRS_JSONL" \
        --stats_json "$TRAINING_PAIRS_STATS" \
        --first_frame_dir "$TRAINING_FIRST_FRAME_DIR" \
        --index_filename "$INDEX_FILENAME" \
        --angle_ref_count "$ANGLE_REF_COUNT" \
        --emo_ref_count "$EMO_REF_COUNT" \
        --body_pose_ref_count "$BODY_POSE_REF_COUNT" \
        --bucket_candidate_topk "$BUCKET_CANDIDATE_TOPK" \
        --min_same_prefix_shot_gap "$MIN_SAME_PREFIX_SHOT_GAP" \
        --overwrite_similarity_matrix "$OVERWRITE_SIMILARITY_MATRIX" \
        --overwrite_first_frames "$OVERWRITE_FIRST_FRAMES" \
        > "$LOG_DIR/stage5_generate_training_pairs.log" 2>&1

    PAIRS_BASENAME="$(basename "$TRAINING_PAIRS_JSONL")"
    PAIRS_TOTAL=$(count_jsonl_tree_by_name "$TRAINING_PAIRS_DIR" "$PAIRS_BASENAME" "$TRAINING_PAIRS_JSONL")
    echo "[Stage 5] 完成: $TRAINING_PAIRS_DIR/<video>/$PAIRS_BASENAME (${PAIRS_TOTAL} 条 pairs, ${UNIT_COUNT} 个 person_clusters)"
else
    env TASK_NAME=generate_training_pairs "$PYTHON_BIN" main.py \
        --person_clusters_dir "$PERSON_CLUSTER_DIR" \
        --output_jsonl "$TRAINING_PAIRS_JSONL" \
        --stats_json "$TRAINING_PAIRS_STATS" \
        --first_frame_dir "$TRAINING_FIRST_FRAME_DIR" \
        --index_filename "$INDEX_FILENAME" \
        --angle_ref_count "$ANGLE_REF_COUNT" \
        --emo_ref_count "$EMO_REF_COUNT" \
        --body_pose_ref_count "$BODY_POSE_REF_COUNT" \
        --bucket_candidate_topk "$BUCKET_CANDIDATE_TOPK" \
        --min_same_prefix_shot_gap "$MIN_SAME_PREFIX_SHOT_GAP" \
        --overwrite_similarity_matrix "$OVERWRITE_SIMILARITY_MATRIX" \
        --overwrite_first_frames "$OVERWRITE_FIRST_FRAMES" \
        > "$LOG_DIR/stage5_generate_training_pairs.log" 2>&1

    echo "[Stage 5] 完成: $TRAINING_PAIRS_JSONL ($(count_lines "$TRAINING_PAIRS_JSONL") 条 pairs)"
fi

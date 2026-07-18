#!/bin/bash
# ================================================================
# Stage 4: Index Add（独立脚本）
# ================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_P="${NUM_P:-32}"
DEVICE_RUNTIME="${DEVICE_RUNTIME:-npu}"
DEVICE_ARG="${DEVICE_ARG:-cuda:0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/data/huanan/code/jwx1520881/HUAWEI_CrossPairDataset_v2/outputs/outputs_multiperson}"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/logs}"

ONE_SHOT_OUTPUT_JSONL="${ONE_SHOT_OUTPUT_JSONL:-/data/huanan/code/jwx1520881/HUAWEI_CrossPairDataset_v2/outputs/merged_all.jsonl}"
IDENTITY_OUTPUT_DIR="${IDENTITY_OUTPUT_DIR:-/data/huanan/code/jwx1520881/HUAWEI_CrossPairDataset_v2/outputs_multiperson}"
INDEX_FILENAME="${INDEX_FILENAME:-after_pipeline_index.json}"
INDEX_ENABLE_EMOTION="${INDEX_ENABLE_EMOTION:-False}"
INDEX_ENABLE_EMOTION_VLM="${INDEX_ENABLE_EMOTION_VLM:-False}"
INDEX_ENABLE_BODY_POSE="${INDEX_ENABLE_BODY_POSE:-False}"
INDEX_BODY_POSE_DETECTOR="${INDEX_BODY_POSE_DETECTOR:-yolo}"
INDEX_OVERWRITE="${INDEX_OVERWRITE:-True}"

mkdir -p "$LOG_DIR"

run_on_device() {
    local device_id=$1
    shift
    if [ "$DEVICE_RUNTIME" = "npu" ]; then
        ASCEND_RT_VISIBLE_DEVICES=$device_id "$@"
    else
        CUDA_VISIBLE_DEVICES=$device_id "$@"
    fi
}

wait_workers() {
    local failed=0
    local pid
    for pid in "$@"; do
        if ! wait "$pid"; then failed=1; fi
    done
    if [ "$failed" -ne 0 ]; then
        echo "[ERROR] 至少一个 worker 失败，请查看日志目录: $LOG_DIR"
        exit 1
    fi
}

echo "========================================"
echo "  Stage 4: Index Add"
echo "  输入 identity: $IDENTITY_OUTPUT_DIR"
echo "  输入 member:   $ONE_SHOT_OUTPUT_JSONL"
echo "  索引文件:      $INDEX_FILENAME"
echo "  日志:          $LOG_DIR"
echo "========================================"

pids=()
for ((g=0; g<NUM_P; g++)); do
    gpu_id=$(( g % NUM_GPUS ))
    echo "[Stage 4] 启动 Worker $g/$NUM_P 运行在设备: $gpu_id"
    (
        run_on_device "$gpu_id" env TASK_NAME=index_add "$PYTHON_BIN" main.py \
            --path "$IDENTITY_OUTPUT_DIR" \
            --member_recovery_jsonl "$ONE_SHOT_OUTPUT_JSONL" \
            --output_filename "$INDEX_FILENAME" \
            --enable_emotion "$INDEX_ENABLE_EMOTION" \
            --enable_emotion_vlm "$INDEX_ENABLE_EMOTION_VLM" \
            --enable_body_pose "$INDEX_ENABLE_BODY_POSE" \
            --body_pose_detector "$INDEX_BODY_POSE_DETECTOR" \
            --body_pose_device "$DEVICE_ARG" \
            --emotion_vlm_device "$DEVICE_ARG" \
            --update_features mask_hole_quality face_boundary_quality \
            --mask_hole_threshold 5 \
            --enable_face_bbox_boundary_quality_check True \
            --enable_face_mask_coverage_quality_check True \
            --face_mask_min_foreground_ratio 0.90 \
            --face_quality_model_name buffalo_l \
            --face_quality_model_root /data/huanan/code/jwx1520881/HUAWEI_CrossPairDataset/pretrained_models/insightface \
            --face_quality_device cuda:0 \
            --face_mask_coverage_max_abs_yaw 30.0 \
            --overwrite "$INDEX_OVERWRITE" \
            --rank "$g" \
            --total_rank "$NUM_P"
    ) > "$LOG_DIR/stage4_worker_${g}.log" 2>&1 &
    pids+=("$!")
    sleep 1
done

echo "[Stage 4] 等待所有 worker 完成..."
wait_workers "${pids[@]}"
echo "[Stage 4] 完成"

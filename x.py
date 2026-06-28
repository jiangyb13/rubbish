#!/bin/bash
# ============================================================
# index_add (after-pipeline index builder, v2) 多卡多进程启动脚本
#
# 目录结构: <IDENTITY_ROOT>/<video>/person_clusters/<person_id>
#           每个 video 目录下各有一份 output.jsonl
#
# 用法:
#   bash scripts/index_add.sh [NUM_GPUS] [PROCS_PER_GPU] [IDENTITY_ROOT]
# 示例:
#   bash scripts/index_add.sh 8 2                 # 8 卡，每卡 2 进程 = 16 个分片
#   bash scripts/index_add.sh 4 1 outputs/identity_matching
#
# 分片逻辑: total_rank = NUM_GPUS * PROCS_PER_GPU，每个进程分到一个唯一 rank，
#           pipeline 内部按 rank 连续切分「待处理的 (video, person) 单元」。
#           每个进程通过 CUDA_VISIBLE_DEVICES 绑定到一张卡（卡内看到的设备即 cuda:0）。
# ============================================================
set -u

export TASK_NAME=index_add

# ─────────────── 入参（带默认值） ───────────────
NUM_GPUS=${1:-1}                                   # 一共几张 GPU
PROCS_PER_GPU=${2:-2}                              # 每张 GPU 起几个进程
IDENTITY_ROOT=${3:-"outputs_demo/identity_matching"}    # identity_matching 根目录

# ─────────────── 业务配置 ───────────────
OUTPUT_FILENAME="post_process_index.json"        # 每个 person 目录下写出的结果文件名
LOG_DIR="logs/index_add"

# 恢复 jsonl（上游 one_shot 输出，含全部成员）：identity 的 output.jsonl 只保留了部分被聚类的人，
# person_clusters 里其余「恢复簇」需靠它按 cluster_meta 补回成员；留空则这些人会得到空索引。
MEMBER_RECOVERY_JSONL="outputs/one_shot_process/output.jsonl"

# 特征开关（构建模式下生效）
DISABLE_POSE=False         # 是否禁用人脸姿态(pitch/yaw/roll)属性
ENABLE_EMOTION=True        # 是否抽取表情
ENABLE_EMOTION_VLM=False    # 是否用 Qwen3-VL 对表情做二次复筛
ENABLE_BODY_POSE=True      # 是否抽取体态/身体朝向

# 增量更新：留空 -> 构建模式；非空 -> 增量模式（对已有索引补加特征，不重建）
#   可选值: emotion / emotion_vlm / body_pose（可空格分隔多个），如 UPDATE_FEATURES="emotion_vlm"
UPDATE_FEATURES=""

# 是否覆盖已有结果（仅构建模式生效；增量模式总是改写已有索引）
OVERWRITE=True

# 模型 / 权重路径
EMOTION_VLM_MODEL_PATH="pretrained_models/Qwen3-VL-8B-Instruct"
EMOTION_VLM_MAX_NEW_TOKENS=512
BODY_POSE_CHECKPOINT="./pretrained_models/4D-Humans/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt"
BODY_POSE_YOLO_CHECKPOINT="./pretrained_models/yolo/yolo11n-pose.pt"
BODY_POSE_DETECTOR="vitdet"

# ─────────────── 启动 ───────────────
TOTAL_RANK=$((NUM_GPUS * PROCS_PER_GPU))
mkdir -p "$LOG_DIR"

UPDATE_ARG=""
if [ -n "$UPDATE_FEATURES" ]; then
    UPDATE_ARG="--update_features ${UPDATE_FEATURES}"
fi

echo "============================================================"
echo " index_add 多卡启动"
echo "   NUM_GPUS=${NUM_GPUS}, PROCS_PER_GPU=${PROCS_PER_GPU}, total_rank=${TOTAL_RANK}"
echo "   identity_root=${IDENTITY_ROOT}"
echo "   mode=$([ -n "$UPDATE_FEATURES" ] && echo "update(${UPDATE_FEATURES})" || echo build)"
echo "   日志目录=${LOG_DIR}"
echo "============================================================"

PIDS=()
rank=0
# 外层 proc、内层 gpu：前 NUM_GPUS 个 rank 正好一卡一个，进程均匀铺到各卡
for ((p=0; p<PROCS_PER_GPU; p++)); do
    for ((g=0; g<NUM_GPUS; g++)); do
        LOG_FILE="${LOG_DIR}/rank_${rank}.log"
        echo "  launch rank ${rank}/${TOTAL_RANK} on GPU ${g} (proc ${p}) -> ${LOG_FILE}"
        CUDA_VISIBLE_DEVICES=${g} python main.py \
            --path "${IDENTITY_ROOT}" \
            --output_filename "${OUTPUT_FILENAME}" \
            --member_recovery_jsonl "${MEMBER_RECOVERY_JSONL}" \
            --rank ${rank} \
            --total_rank ${TOTAL_RANK} \
            --disable_pose ${DISABLE_POSE} \
            --enable_emotion ${ENABLE_EMOTION} \
            --enable_emotion_vlm ${ENABLE_EMOTION_VLM} \
            --enable_body_pose ${ENABLE_BODY_POSE} \
            --overwrite ${OVERWRITE} \
            --emotion_vlm_model_path "${EMOTION_VLM_MODEL_PATH}" \
            --emotion_vlm_device "cuda:0" \
            --emotion_vlm_max_new_tokens ${EMOTION_VLM_MAX_NEW_TOKENS} \
            --body_pose_checkpoint "${BODY_POSE_CHECKPOINT}" \
            --body_pose_yolo_checkpoint "${BODY_POSE_YOLO_CHECKPOINT}" \
            --body_pose_detector "${BODY_POSE_DETECTOR}" \
            --body_pose_device "cuda:0" \
            ${UPDATE_ARG} \
            > "${LOG_FILE}" 2>&1 &
        PIDS+=($!)
        rank=$((rank + 1))
    done
done

echo "  已启动 ${#PIDS[@]} 个进程，等待全部完成..."
FAIL=0
for pid in "${PIDS[@]}"; do
    wait ${pid} || FAIL=$((FAIL + 1))
done

echo "============================================================"
if [ ${FAIL} -eq 0 ]; then
    echo " 全部 ${TOTAL_RANK} 个分片完成。日志: ${LOG_DIR}/rank_*.log"
else
    echo " 完成，但有 ${FAIL} 个进程异常退出，请检查 ${LOG_DIR}/rank_*.log"
fi
echo "============================================================"

#!/bin/bash



# ==============================================================================

# 批量运行 Stage 4 脚本的包装器（防重运行安全改良版）

# ==============================================================================



set -euo pipefail



# 1. 定义你要遍历的父目录

BASE_DIR="/data/huanan/code/jwx1520881/HUAWEI_CrossPairDataset_v2/outputs_singleperson_v2/identity_matching/video_720p_15min_0"

# Stage 4 脚本的路径

STAGE4_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/stage4_index_add_update_quality.sh"



if [ ! -f "$STAGE4_SCRIPT" ]; then

    echo "[ERROR] 找不到 Stage 4 脚本: $STAGE4_SCRIPT"

    exit 1

fi



if [ ! -d "$BASE_DIR" ]; then

    echo "[ERROR] 目标遍历目录不存在: $BASE_DIR"

    exit 1

fi



echo "=============================================================================="

echo " 开始遍历目录并运行 Stage 4"

echo " 目标父目录: $BASE_DIR"

echo "=============================================================================="



# 2. 改用进程替换驱动循环，阻断 stdin 管道污染

while read -r sub_dir; do

    # 略过空行

    [ -z "$sub_dir" ] && continue

   

    dir_name=$(basename "$sub_dir")

    echo ""

    echo "=============================================================================="

    echo " 👉 正在处理子目录: $dir_name"

    echo " 路径: $sub_dir"

    echo "=============================================================================="



    # # 【防重核心 1】在启动新任务前，确保前一个任务的 main.py 已经彻底退出

    # # 防止因异常崩溃导致前一个任务的僵尸进程影响当前任务

    # if pkill -0 -f main.py 2>/dev/null; then

    #     echo "[INFO] 检测到后台有残留的 main.py 进程，正在进行环境清理..."

    #     pkill -f main.py || true

    #     sleep 2

    # fi



    # 3. 注入环境变量

    export IDENTITY_OUTPUT_DIR="$sub_dir"

    export LOG_DIR="/data/huanan/code/jwx1520881/HUAWEI_CrossPairDataset_v2/outputs/logs/$dir_name"

    export ONE_SHOT_OUTPUT_JSONL="/data/huanan-4931/data/AIGC_VIDEO/TRAIN_DATA_I2V/video_720p_preprocessed_cross_shot_pair/one_shot_process/video_720p_15min_0/$dir_name/output.jsonl"



    ls "/data/huanan-4931/data/AIGC_VIDEO/TRAIN_DATA_I2V/video_720p_preprocessed_cross_shot_pair/one_shot_process/video_720p_15min_0/$dir_name/output.jsonl"



    # 执行 Stage 4 脚本（使用 </dev/null 断开标准输入流，双重保险）

    if bash "$STAGE4_SCRIPT" </dev/null; then

        echo "✅ 子目录 $dir_name 处理成功！"

    else

        echo "❌ [ERROR] 子目录 $dir_name 处理失败，触发熔断退出。"

        exit 1

    fi



# 通过 <<< 将 find 的结果安全喂给 while，不占用主进程的 stdin

done <<< "$(find "$BASE_DIR" -maxdepth 1 -mindepth 1 -type d | sort)"





echo ""

echo "🎉 所有子目录从 part_0000 到 part_0406 全部处理完毕！"



这个脚本好慢 


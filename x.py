# HUAWEI CrossPairDataset — 跨镜头人物配对数据集构建工具

本项目用于从原始长视频中构建跨镜头人物配对训练数据。主入口是 `main.py`，通过环境变量 `TASK_NAME` 切换五个阶段；所有命令行参数由 `utils/config.py` 中的 dataclass 定义，并通过 `HfArgumentParser` 解析。

推荐运行脚本是：

```bash
scripts/run_pipeline.sh
```

`scripts/run_8gpu_pipeline.sh` 和 `scripts/run_8npu_pipeline.sh` 目前都是该脚本的轻量封装，GPU/NPU 的实际设备可见性由 `run_pipeline.sh` 同时设置。

---

## 目录

- [整体流程](#整体流程)
- [任务输入与标准目录](#任务输入与标准目录)
- [多卡或多 NPU 完整运行](#多卡或多-npu-完整运行)
- [单阶段运行](#单阶段运行)
- [各阶段说明](#各阶段说明)
- [并行与任务切分](#并行与任务切分)
- [Manifest 与目录搜索控制](#manifest-与目录搜索控制)
- [核心配置](#核心配置)
- [代码结构](#代码结构)
- [依赖与模型](#依赖与模型)
- [Flask 可视化](#flask-可视化)
- [断点续跑与常见问题](#断点续跑与常见问题)

---

## 整体流程

```text
Stage-1 任务 JSONL（每行一个原始视频）
  │
  ▼
Stage 1: shot_detection
  TransNetV2 镜头边界检测，裁剪 shot 视频
  │
  ▼
Stage 2: one_shot_process
  YOLO11-pose 检人，SAM3 跟踪，保存全身/人脸图、mask、DINO 与人脸特征、头部姿态
  │
  ▼
Stage 3: identity_matching
  质量过滤、跨 shot 人脸相似度匹配、Union-Find 聚类，生成 person_clusters
  │
  ▼
Stage 4: index_add
  为每个 person 构建 post_process_index.json，补充表情、身体朝向与质量标签
  │
  ▼
Stage 5: generate_training_pairs
  按角度、表情、身体姿态选择参考图，匹配目标 shot，生成训练 pairs
```

五个阶段都使用同一份视频任务 JSONL。阶段之间根据 `outputs/<video_id>` 下的固定相对路径读取上游结果。

可用的 `TASK_NAME`：

```text
shot_detection
one_shot_process
identity_matching
index_add
generate_training_pairs
```

`main.py` 只负责解析当前阶段配置并分发到对应 Pipeline；批量任务解析、连续切片和同 Stage 内的多视频循环由各阶段代码完成。

---

## 任务输入与标准目录

### 唯一批量输入：Stage-1 任务 JSONL

推荐输入文件：

```text
input/video_paths.jsonl
```

每一行必须是一个独立 JSON 对象，而不是一个 JSON 数组：

```jsonl
{"video_id":"video_000001","video_path":"/data/raw/video_000001.mp4"}
{"video_id":"video_000002","video_path":"/data/raw/video_000002.mp4"}
```

字段说明：

| 字段 | 是否必需 | 说明 |
|---|---:|---|
| `video_path` | 是 | 原始视频路径；读取任务时会转为绝对路径 |
| `video_id` | 否 | workspace 名称；默认使用视频文件名去掉扩展名后的 stem，并清理不安全字符 |
| `video_dir` | 否 | 显式指定该视频的 workspace；默认是 `OUTPUT_ROOT/<video_id>` |

同一份 JSONL 中的 `video_id` 必须唯一。空行会被忽略；非法 JSON、缺失 `video_path` 或重复 `video_id` 会直接报错。

### 每视频 workspace

默认目录结构：

```text
outputs/
├── logs/
│   ├── shot_detection_worker_0.log
│   ├── one_shot_process_worker_0.log
│   ├── identity_matching_worker_0.log
│   ├── index_add_worker_0.log
│   └── generate_training_pairs_worker_0.log
├── _pipeline_runtime/
│   └── stage4_phase_<n>.txt
└── <video_id>/
    ├── pipeline_manifest.json
    ├── shot_detection/
    │   ├── output.jsonl
    │   ├── transnet_scores.npy
    │   └── shots/
    │       └── <video_id>_shot_<n>.mp4
    ├── one_shot_process/
    │   ├── output.jsonl
    │   ├── raw_tracking/
    │   └── <shot_key>/
    │       └── id_<obj_id>/
    │           ├── full/
    │           ├── face/
    │           └── features/
    ├── identity_matching/
    │   ├── output.jsonl
    │   ├── output_simple.jsonl
    │   ├── global_data.json
    │   ├── persons.jsonl
    │   ├── global_matrices/
    │   └── person_clusters/
    │       └── person_<n>/
    │           ├── cluster_meta.json
    │           ├── frame_manifest.jsonl
    │           ├── post_process_index.json
    │           ├── face_white/
    │           ├── face_orig/
    │           ├── full_white/
    │           ├── full_orig/
    │           ├── face_angle_library/
    │           ├── face_diversity_topk/
    │           └── dino_diversity_topk/
    └── training_pairs/
        ├── pairs.jsonl
        ├── stats.json
        ├── first_frames/
        └── person_<n>_candidate_similarity.npz
```


---

## 多卡或多 NPU 完整运行

### 推荐命令

```bash
INPUT_JSONL=input/video_paths.jsonl \
OUTPUT_ROOT=outputs \
NUM_DEVICES=8 \
PROCS_PER_DEVICE=1 \
bash scripts/run_pipeline.sh
```

GPU 封装：

```bash
INPUT_JSONL=input/video_paths.jsonl \
OUTPUT_ROOT=outputs \
NUM_DEVICES=8 \
bash scripts/run_8gpu_pipeline.sh
```

NPU 封装：

```bash
INPUT_JSONL=input/video_paths.jsonl \
OUTPUT_ROOT=outputs \
NUM_DEVICES=8 \
bash scripts/run_8npu_pipeline.sh
```

两个封装脚本当前都转发到同一个 `run_pipeline.sh`。`run_pipeline.sh` 对每个 worker 同时设置：

```text
CUDA_VISIBLE_DEVICES=<device_id>
ASCEND_RT_VISIBLE_DEVICES=<device_id>
```

模型参数中的逻辑设备默认仍使用 `cuda:0`；进程通过可见设备映射到物理卡。实际使用 GPU 还是 NPU 取决于当前 PyTorch/torch-npu 环境及模型实现。脚本名称和可见设备变量本身不保证所有第三方依赖都原生兼容 NPU，例如 InsightFace 的 ONNX Runtime provider 仍需单独确认。

### 自定义每阶段 worker 数量

worker 数量不要求是 8，也不要求五个阶段相同：

```bash
STAGE1_WORKERS=8 \
STAGE2_WORKERS=8 \
STAGE3_WORKERS=16 \
STAGE4_WORKERS=8 \
STAGE5_WORKERS=24 \
bash scripts/run_pipeline.sh
```

若没有单独设置，默认值是：

```text
NUM_DEVICES * PROCS_PER_DEVICE
```

worker 数大于任务数时，脚本会把该阶段 worker 数缩小到任务数。worker 数大于设备数时，使用：

```text
device_id = phase % NUM_DEVICES
```

此时同一张卡会同时运行多个 Python 进程，必须根据模型显存/显存占用谨慎设置。

每个 worker 的 stdout/stderr 都重定向到：

```text
OUTPUT_ROOT/logs/<stage>_worker_<phase>.log
```

因此脚本控制台主要显示阶段启动、失败提示和最终汇总，详细进度应查看对应 worker 日志。

### 只运行部分阶段

```bash
RUN_STAGE1=False \
RUN_STAGE2=False \
RUN_STAGE3=False \
RUN_STAGE4=True \
RUN_STAGE5=True \
bash scripts/run_pipeline.sh
```

只运行 Stage 4 也可使用：

```bash
INPUT_JSONL=input/video_paths.jsonl \
OUTPUT_ROOT=outputs \
bash scripts/run_stage4_only.sh
```

### 常用参数覆盖

```bash
INPUT_JSONL=input/video_paths.jsonl \
OUTPUT_ROOT=outputs \
NUM_DEVICES=8 \
FACE_MATCH_THRESHOLD=0.65 \
TOP_K_MATCHES=10 \
INDEX_ENABLE_EMOTION=True \
INDEX_ENABLE_BODY_POSE=True \
INDEX_ENABLE_MASK_HOLE_QUALITY_CHECK=True \
ANGLE_REF_COUNT=5 \
EMO_REF_COUNT=5 \
BODY_POSE_REF_COUNT=5 \
bash scripts/run_pipeline.sh
```

Stage 3 为避免大矩阵 JSON 带来的二次方内存和写盘开销，主脚本默认：

```bash
SAVE_GLOBAL_MATRIX_JSON=False
SAVE_PAIRWISE_SIMILARITIES=False
```

NPY 相似度矩阵仍由 `save_global_matrix_as_npy=True` 的代码默认值控制。

---

## 单阶段运行

每个阶段都支持两种入口：

- 批量模式：传 `--pipeline_input_jsonl` 和 `--output_root`，再用 `--phase/--total` 选择连续视频区间。
- 单 workspace 模式：传 `--video_dir outputs/<video_id>`；Stage 2–5 从 manifest 和固定相对路径推导输入。

### Stage 1: Shot Detection

批量运行：

```bash
TASK_NAME=shot_detection python3 main.py \
  --pipeline_input_jsonl input/video_paths.jsonl \
  --output_root outputs \
  --transnetv2_checkpoint pretrained_models/TransNetV2/transnetv2-pytorch-weights.pt \
  --scene_threshold 0.5 \
  --device cuda:0 \
  --phase 0 \
  --total 1
```

单视频初始化 workspace：

```bash
TASK_NAME=shot_detection python3 main.py \
  --video_dir outputs/video_000001 \
  --video_path /data/raw/video_000001.mp4 \
  --video_id video_000001 \
  --device cuda:0 \
  --phase 0 \
  --total 1
```

Stage 1 是唯一在新 workspace 上必须显式提供源视频的阶段。之后源视频路径会写入 `pipeline_manifest.json`。

### Stage 2: One-Shot Process

```bash
TASK_NAME=one_shot_process python3 main.py \
  --video_dir outputs/video_000001 \
  --sam3_checkpoint pretrained_models/sam3/sam3.pt \
  --dinov3_model pretrained_models/dinov3-vitl16-pretrain-lvd1689m \
  --yolon11_pose_checkpoint pretrained_models/yolo/yolo11n-pose.pt \
  --deca_model_path pretrained_models/deca_model.tar \
  --insightface_root pretrained_models/insightface \
  --device cuda:0 \
  --device_sam cuda:0
```

输入固定推导为：

```text
outputs/video_000001/shot_detection/output.jsonl
```

### Stage 3: Identity Matching

```bash
TASK_NAME=identity_matching python3 main.py \
  --video_dir outputs/video_000001 \
  --face_match_threshold 0.65 \
  --top_k_matches 10 \
  --keep_unmatched False \
  --save_global_matrix_as_json False \
  --save_pairwise_similarities False
```

输入固定推导为：

```text
outputs/video_000001/one_shot_process/output.jsonl
```

### Stage 4: Index Add

```bash
TASK_NAME=index_add python3 main.py \
  --video_dir outputs/video_000001 \
  --output_filename post_process_index.json \
  --enable_emotion True \
  --emotion_model_name enet_b0_8_best_vgaf \
  --emotion_device cuda \
  --enable_emotion_vlm False \
  --enable_body_pose True \
  --body_pose_detector yolo \
  --body_pose_device cuda:0 \
  --enable_mask_hole_quality_check True \
  --mask_hole_threshold 0 \
  --overwrite True
```

Stage 4 通过 `identity_matching/persons.jsonl` 找到人物，再优先读取每个 `person_xxxx/frame_manifest.jsonl`，输出：

```text
identity_matching/person_clusters/person_xxxx/post_process_index.json
```

### Stage 5: Generate Training Pairs

```bash
TASK_NAME=generate_training_pairs python3 main.py \
  --video_dir outputs/video_000001 \
  --index_filename post_process_index.json \
  --angle_ref_count 5 \
  --emo_ref_count 5 \
  --body_pose_ref_count 5 \
  --bucket_candidate_topk 5 \
  --min_same_prefix_shot_gap 3
```

Stage 5 读取当前 workspace 的 `person_clusters`，输出：

```text
outputs/video_000001/training_pairs/pairs.jsonl
outputs/video_000001/training_pairs/stats.json
outputs/video_000001/training_pairs/first_frames/
```

> `run_pipeline.sh` 的 Stage 5 默认值是 `bucket_candidate_topk=5`、`min_same_prefix_shot_gap=3`；直接调用 `main.py` 时 dataclass 默认值分别是 `8` 和 `0`。若希望结果完全一致，请显式传参。

---

## 各阶段说明

### Stage 1: `utils/shot_detection.py`

| 组件 | 作用 |
|---|---|
| `TransNetV2Wrapper` | 加载 TransNetV2，对原始视频预测镜头边界 |
| `VideoCutter` | 按场景区间裁剪 shot，并按配置裁掉头尾边界残留 |
| `ShotDetectionPipeline` | 处理当前 worker 的连续视频区间，写 shot JSONL 和 manifest |

每视频输入：

```text
pipeline task row: video_id + video_path + video_dir
```

主要输出：

```text
shot_detection/output.jsonl
shot_detection/transnet_scores.npy
shot_detection/shots/*.mp4
```

`output.jsonl` 每行对应一个 shot，核心字段包括：

```text
video_id
source_video_path
shot_id / shot_video_path
original_start_frame / original_end_frame
start_frame / end_frame / frame_count
trim_mode / head_offset_frames / tail_offset_frames
transnet_metadata.threshold / transnet_metadata.scores_path
```

Stage 1 会读取已存在的 `output.jsonl`，按 `source_video_path` 判断是否已经处理，用于断点续跑。

### Stage 1.5 (选用，独立 VLM 过滤脚本) `shot_filter_vlm.py`

```bash
python shot_filter_vlm.py \
    --input_jsonl outputs/shot_detection/output.jsonl \
    --output_jsonl outputs/shot_filter/accept.jsonl \
    --rejected_jsonl outputs/shot_filter/rejected.jsonl \
    --model_path /path/to/Qwen3-VL-8B-Instruct \
    --device cuda:0
```
若启用该部分，stage2 输入的 input_jsonl 文件则应改为 outputs/shot_filter/rejected.jsonl

### Stage 2: `utils/one_shot.py`

Stage 2 对每个 shot 执行人物检测、SAM3 跟踪与帧级数据保存。主要能力：

- YOLO11-pose 人体检测。
- SAM3 视频目标跟踪。
- InsightFace 或 facexlib 人脸检测与人脸 embedding。
- DINOv3 全身特征。
- DECA 头部欧拉角。
- 保存全身原图/白底图、裁剪人脸、SAM 人脸、mask 和可选可视化视频。

主要过滤逻辑：

- `min_duration_sec`：跳过过短 shot。
- `min_area_ratio`、`small_track_frame_ratio`：过滤长期过小的人物 track。
- `bg_width_ratio_threshold`：过滤长期覆盖画面大部分宽度的背景 track。
- `max_bbox_aspect_ratio`：过滤异常细长 bbox。
- 人脸预筛：无法检测到有效人脸的 track 不进入后续保存。
- `dino_consistency_check`：任一检查帧与基准帧低于阈值时丢弃 track。
- `min_face_crop_size`：小于最小尺寸的人脸 crop 不保存。

主要输出结构：

```text
one_shot_process/<shot_key>/id_<obj_id>/
├── full/
│   ├── full_pic_white/
│   ├── full_pic_orig/
│   ├── cropped_full_mask/
│   └── orig_full_mask/
├── face/
│   ├── cropped_face/
│   ├── sam/
│   └── face_euler_angles.jsonl
└── features/
    ├── dino_feature/*.npy
    └── face_feature/*.npy
```

`one_shot_process/output.jsonl` 每行对应一个 shot 内的人物 ID，除目录路径外还显式记录：

```text
face_feature_paths
dino_feature_paths
frame_index_mapping
source_shot_path
```

Stage 3 会优先复用这些显式特征路径，不需要递归搜索特征目录。Stage 2 会按 `source_shot_path` 读取已有输出并跳过已完成 shot。

### Stage 3: `utils/identity_matching.py`

`IdentityMatcher` 合并了旧流程中的质量筛选、正面帧选择和跨 shot 匹配：

1. 读取 Stage 2 ID 记录。
2. 按人脸清晰度、人脸面积和最小有效帧数过滤。
3. 从每个 ID 的正面帧聚合人脸特征。
4. 计算全局人脸相似度矩阵。
5. 按 `face_match_threshold` 建立跨 shot 匹配关系。
6. 使用 Union-Find 聚类得到 `person_xxxx`。
7. 构建人物帧池、角度库和 face/DINO diversity TOP-K。
8. 写出稳定人物清单 `persons.jsonl` 和逐人物 `frame_manifest.jsonl`。

主要输出：

```text
identity_matching/
├── output.jsonl
├── output_simple.jsonl
├── global_data.json
├── persons.jsonl
├── global_matrices/
│   ├── face_similarity_matrix.npy
│   ├── face_similarity_matrix.json        # 可关闭
│   ├── index_map.json
│   └── pairwise_similarities.json         # 可关闭
└── person_clusters/person_xxxx/
    ├── cluster_meta.json
    ├── frame_manifest.jsonl
    ├── face_white/
    ├── face_orig/
    ├── full_white/
    ├── full_orig/
    ├── face_angle_library/
    ├── face_diversity_topk/
    └── dino_diversity_topk/
```

`persons.jsonl` 每行核心字段：

```text
person_id
cluster_dir
cluster_meta_path
frame_manifest_path
post_process_index_path
```

其中 `post_process_index_path` 是 Stage 3 按规范路径预登记的目标位置；真正的索引文件要到 Stage 4 才生成。

`frame_manifest.jsonl` 每行对应一个人物帧，记录 `shot_key`、`obj_id`、`frame_idx`、姿态、质量信息和精确图片路径。它取代 Stage 4 历史上的目录遍历和文件名正则反推。

### Stage 4: `utils/index_add.py`

Stage 4 为每个 `person_xxxx` 构建结构化索引。默认输出文件名：

```text
post_process_index.json
```

可选或默认开启的属性：

- 头部姿态 `pose`。
- EmotiEffLib 表情 `expression`。
- Qwen3-VL 表情复核 `expression.emo_flag/final_expression`。
- 4D-Humans 身体朝向与 YOLO body extent `body_pose`。
- 白底图 SAM mask 孔洞质量 `quality/quality_label`。
- 可选的人脸 bbox 边界与 face-mask 覆盖质量。

主脚本默认开启：

```text
INDEX_ENABLE_EMOTION=True
INDEX_ENABLE_BODY_POSE=True
INDEX_BODY_POSE_DETECTOR=yolo
INDEX_ENABLE_MASK_HOLE_QUALITY_CHECK=True
INDEX_MASK_HOLE_THRESHOLD=0
INDEX_OVERWRITE=True
```

索引顶层结构大致为：

```json
{
  "person_id": "person_0000",
  "cluster_dir": ".../person_clusters/person_0000",
  "schema": "images.image_type.image_path.attributes",
  "enabled_features": ["pose", "emotion", "body_pose"],
  "images": {
    "face_orig": {
      "path/to/image.jpg": {
        "shot_key": "video_000001_shot_0003",
        "obj_id": 0,
        "frame_idx": 12,
        "source_shot_path": "...",
        "related_images": {},
        "pose": {},
        "expression": {},
        "body_pose": {},
        "quality_label": true,
        "quality": {}
      }
    },
    "face_white": {},
    "full_orig": {},
    "full_white": {}
  },
  "face_diversity_topk": [],
  "dino_diversity_topk": [],
  "stats": {}
}
```

增量更新已有索引时可使用：

```text
--update_features emotion emotion_vlm body_pose mask_hole_quality face_boundary_quality
```

只复核已有表情也可使用 `--update_emotion_vlm_only True`。增量模式会原地更新已有索引，不重新构建全部基础映射。

### Stage 5: `utils/generate_training_pairs.py`

Stage 5 从 `post_process_index.json` 中构造候选并选择：

- angle reference images
- emotion reference images
- body pose reference images
- target shot video
- target video first frame

Stage 4 写入的 `quality_label` 会影响参考图候选：face 参考要求相应人脸图质量通过，body 参考优先要求 full 图质量通过。Stage 5 本身不会重新计算 mask 质量。

每个人物会生成或复用：

```text
training_pairs/person_xxxx_candidate_similarity.npz
```

候选未变化时可直接加载；新候选是旧候选子集时可从旧矩阵取子矩阵；缓存不兼容或设置 `overwrite_similarity_matrix=True` 时重新计算。

`pairs.jsonl` 每行核心字段：

```text
person_id
source_uid
source_shot_key
source_frame_idx
first_frame
target_video
angle_ref / angle_ref_white / angle_ref_meta
emo_ref / emo_ref_white / emo_ref_meta
body_pose_ref / body_pose_ref_white / body_pose_ref_meta
selection_meta
selection_stats
```

Stage 5 会排除与当前 target 属于同一 `shot_key` 的参考候选，并用 `min_same_prefix_shot_gap` 约束同视频前缀下的 shot 距离。每个 target 最多尝试 20 次，以减少重复 reference 组合。

`stats.json` 记录：

```text
config
total_persons
rows_written
first_frame_error
persons.<person_id>.status
persons.<person_id>.candidate_count
persons.<person_id>.similarity_matrix
persons.<person_id>.target_video_count
persons.<person_id>.skipped
persons.<person_id>.per_target
```

---

## 并行与任务切分

### 脚本层：启动固定数量的同阶段进程

`run_pipeline.sh` 对每个 Stage 调用一次 `run_stage`。假设 Stage 2 配置 8 个 worker，脚本只启动 8 个 Python 进程：

```text
TASK_NAME=one_shot_process phase=0 total=8
TASK_NAME=one_shot_process phase=1 total=8
...
TASK_NAME=one_shot_process phase=7 total=8
```

不会为 100 个视频启动 100 个 Python 进程。该 Stage 的全部 worker 完成后，脚本才进入下一个 Stage。

### 代码层：按连续区间切视频

每个进程都会读取同一份任务 JSONL，然后由 `utils/pipeline_workspace.py` 的 `shard_range()` 计算连续区间。

因此 worker 数较多时，任务 JSONL 会被重复读取和解析多次，但这里只读取一个文本任务文件，不会递归扫描所有视频 workspace。

例如 100 个视频、8 个 worker：

```text
worker 0: [0, 13)
worker 1: [13, 26)
worker 2: [26, 39)
worker 3: [39, 52)
worker 4: [52, 64)
worker 5: [64, 76)
worker 6: [76, 88)
worker 7: [88, 100)
```

前 4 个 worker 各处理 13 个视频，后 4 个各处理 12 个视频；区间连续、互不重叠、无遗漏。

### 各阶段的内部任务粒度

| 阶段 | 脚本分配单位 | 进程内部继续处理的单位 | 模型/缓存复用 |
|---|---|---|---|
| Stage 1 | 视频连续区间 | 当前 worker 的多个原始视频 | TransNetV2 在进程内复用 |
| Stage 2 | 视频连续区间 | 每视频的多个 shot | One-shot 模型在进程内复用 |
| Stage 3 | 视频连续区间 | 每视频的 ID、person cluster | 同一 Python 进程连续处理多个视频 |
| Stage 4 | 视频连续区间 | 每视频 `person_clusters` 下的 `person_xxxx` | extractor 在 worker 内构建一次并跨视频复用 |
| Stage 5 | 视频连续区间 | 每视频 `person_clusters` 下的 `person_xxxx` 与 target video | 进程持续存在；特征/首帧缓存按视频重置 |

Stage 4 会把当前 worker 已分到的视频写入：

```text
OUTPUT_ROOT/_pipeline_runtime/stage4_phase_<phase>.txt
```

随后同一进程读取这些 `person_clusters`，在进程内部通过 `persons.jsonl` 得到 `person_xxxx`。它不会先全局展开整个 `OUTPUT_ROOT` 下的所有人物再分配。

---

## Manifest 与目录搜索控制

### `pipeline_manifest.json`

每个 workspace 的 manifest 保存：

```json
{
  "schema_version": 1,
  "video_id": "video_000001",
  "video_dir": "/abs/path/outputs/video_000001",
  "source_video_path": "/data/raw/video_000001.mp4",
  "stages": {
    "shot_detection": {
      "status": "complete",
      "output_jsonl": "shot_detection/output.jsonl",
      "record_count": 12
    }
  }
}
```

写 manifest 使用临时文件加 `os.replace()` 原子更新，避免写一半留下损坏 JSON。

当前各阶段会写入 `running` 和 `complete`。如果进程异常退出，代码没有统一捕获并写 `failed`，因此 manifest 可能停留在 `running`；此时应结合 worker 日志判断。

### 下游复用清单

| 清单 | 生产阶段 | 消费阶段 | 作用 |
|---|---|---|---|
| Stage-1 task JSONL | 用户 | Stage 1–5 | 唯一顶层视频任务来源 |
| `pipeline_manifest.json` | workspace helper/各阶段 | Stage 1–5、最终汇总 | 源视频、阶段状态、固定输出 |
| `one_shot_process/output.jsonl` | Stage 2 | Stage 3/4 | ID 记录和显式 feature 文件路径 |
| `identity_matching/persons.jsonl` | Stage 3 | Stage 4/5 | 精确人物目录注册表 |
| `person_xxxx/frame_manifest.jsonl` | Stage 3 | Stage 4 | 精确帧、图片和派生信息清单 |
| `post_process_index.json` | Stage 4 | Stage 5 | 训练对候选及质量/属性标签 |

新标准流水线不会递归搜索整个 `OUTPUT_ROOT`。为兼容缺失新 manifest 的旧数据，Stage 4/5 仍保留局部目录扫描回退：

- 缺少 `persons.jsonl` 时，只扫描当前视频的 `person_clusters`。
- 缺少 `frame_manifest.jsonl` 时，Stage 4 才回退到旧成员/目录恢复逻辑。

这些回退不会改变标准 v4 的推荐输入方式。

---

## 核心配置

所有 CLI 配置都在 `utils/config.py`。

### 公共 workspace 参数

| 字段 | 默认值 | 说明 |
|---|---|---|
| `pipeline_input_jsonl` | `None` | 批量模式的唯一任务 JSONL |
| `output_root` | `outputs` | 所有 per-video workspace 的根目录 |
| `video_dir` | `None` | 单 workspace 模式入口 |
| `video_path` | `None` | 单独初始化 Stage 1 时的源视频 |
| `video_id` | `None` | 单独初始化 Stage 1 时的视频 ID |
| `phase` | 阶段默认值 | 当前连续分片编号，脚本传入 `0..workers-1` |
| `total` | `1` | 当前阶段 worker 总数 |

### `ShotDetectionConfig`

| 字段 | 默认值 | 说明 |
|---|---|---|
| `transnetv2_checkpoint` | `./pretrained_models/TransNetV2/transnetv2-pytorch-weights.pt` | TransNetV2 权重 |
| `scene_threshold` | `0.5` | 镜头边界阈值 |
| `device` | CUDA 可用时 `cuda`，否则 `cpu` | 推理设备 |
| `enable_trim` | `True` | 是否裁掉 shot 边界残留 |
| `trim_mode` | `frames` | `frames` 或 `seconds` |
| `trim_head_frames / trim_tail_frames` | `6 / 12` | frame 模式头尾裁剪帧数 |
| `min_shot_duration_sec` | `2.0` | 裁剪后最短 shot 时长 |

### `OneShotProcessConfig`

| 字段 | 默认值 | 说明 |
|---|---|---|
| `sam3_checkpoint` | `./pretrained_models/sam3/sam3.pt` | SAM3 权重 |
| `dinov3_model` | `./pretrained_models/dinov3-vitl16-pretrain-lvd1689m` | DINOv3 模型 |
| `yolon11_pose_checkpoint` | `./pretrained_models/yolo/yolo11n-pose.pt` | YOLO11 pose 权重 |
| `deca_model_path` | `./pretrained_models/deca_model.tar` | DECA 头部姿态权重 |
| `face_detector_backend` | `insightface` | `insightface` 或 `facexlib` |
| `insightface_root` | `./pretrained_models/insightface` | InsightFace 模型根目录 |
| `device / device_sam` | `cuda:0 / 自动` | 通用模型与 SAM 设备 |
| `sample_fps` | `10` | SAM3 跟踪目标采样 FPS |
| `sam_conf` | `0.75` | SAM3 置信度阈值 |
| `min_area_ratio` | `0.1` | 最小 mask 面积比例 |
| `min_face_crop_size` | `128` | 最小人脸 crop 边长 |
| `dino_consistency_check` | `True` | 是否检查 track 的 DINO 一致性 |
| `dino_consistency_threshold` | `0.6` | DINO cosine 最低阈值 |
| `save_videos` | `False` | 是否额外保存可视化/裁剪视频 |
| `enable_timing` | `False` | 是否输出详细耗时统计 |

### `IdentityMatchingConfig`

| 字段 | 默认值 | 说明 |
|---|---|---|
| `min_face_sharpness` | `10.0` | 最小人脸清晰度 |
| `min_face_ratio` | `0.005` | 人脸占原图最小面积比例 |
| `min_valid_frames_per_id` | `2` | 单 ID 最少有效帧数 |
| `face_top_k_frontal_frames` | `5` | 聚合 embedding 使用的正面帧数 |
| `face_match_threshold` | `0.65` | 跨 shot 匹配阈值 |
| `top_k_matches` | `10` | 单 ID 最多保留的匹配数 |
| `keep_unmatched` | `False` | 是否保留没有跨 shot 匹配的 ID |
| `top_k_face_diversity` | `5` | 人脸角度 diversity TOP-K |
| `top_k_dino_diversity` | `5` | DINO diversity TOP-K |
| `save_global_matrix_as_npy` | `True` | 保存全局 NPY 相似度矩阵 |
| `save_global_matrix_as_json` | `True`（脚本传 `False`） | 保存完整矩阵 JSON |
| `save_pairwise_similarities` | `True`（脚本传 `False`） | 保存完整 pairwise JSON |

### `IndexAddConfig`

| 字段 | 默认值 | 说明 |
|---|---|---|
| `output_filename` | `post_process_index.json` | 每个人物的索引文件名 |
| `disable_pose` | `False` | 是否关闭已有头部姿态映射 |
| `enable_emotion` | `True` | 使用 EmotiEffLib 提取表情 |
| `emotion_model_name` | `enet_b0_8_best_vgaf` | EmotiEffLib 模型名 |
| `emotion_model_path` | `None` | 可选本地 `.pt` 权重 |
| `enable_emotion_vlm` | `False` | 使用 Qwen3-VL 复核表情 |
| `enable_body_pose` | `True` | 提取 4D-Humans 身体朝向和 body extent |
| `body_pose_detector` | `vitdet`（脚本传 `yolo`） | `vitdet/regnety/yolo` |
| `enable_mask_hole_quality_check` | `True` | 根据 SAM mask 孔洞写质量标签 |
| `mask_hole_threshold` | `0` | 大于该孔洞数时质量不通过 |
| `enable_face_boundary_quality_check` | `False` | 兼容开关：同时开启两种 face boundary 质量检查 |
| `overwrite` | `False`（脚本传 `True`） | 是否覆盖已有索引 |
| `update_features` | `None` | 只增量更新指定属性 |

### `TrainingPairsConfig`

| 字段 | 默认值 | 说明 |
|---|---|---|
| `index_filename` | `post_process_index.json` | Stage 4 索引文件名 |
| `ref_image_type` | `face_orig` | 首选人脸参考图类型 |
| `ref_fallback_image_type` | `face_white` | 缺失时的回退类型 |
| `angle_ref_count` | `5` | angle reference 数量 |
| `emo_ref_count` | `5` | emotion reference 数量 |
| `body_pose_ref_count` | `5` | body pose reference 数量 |
| `bucket_candidate_topk` | `8`（脚本传 `5`） | 每个 bucket 的候选池大小 |
| `seed` | `42` | 可复现采样种子 |
| `min_same_prefix_shot_gap` | `0`（脚本传 `3`） | 同视频前缀 shot 间隔约束 |
| `overwrite_first_frames` | `False` | 是否覆盖已缓存首帧 |
| `overwrite_similarity_matrix` | `False` | 是否重建人物候选相似度矩阵 |

配置类中仍保留若干旧版直接路径字段和 `unit_list_file` 字段，以兼容内部实现或旧逻辑；标准 v4 批量入口只需要 `pipeline_input_jsonl + output_root`，单视频入口只需要 `video_dir`（Stage 1 另加源视频）。

---

## 代码结构

```text
main.py                              # 唯一主入口，根据 TASK_NAME 分发
utils/
├── config.py                         # 固定 workspace 路径和五阶段 CLI 配置
├── pipeline_workspace.py             # 任务解析、连续切片、workspace、manifest
├── shot_detection.py                 # Stage 1
├── one_shot.py                       # Stage 2
├── identity_matching.py              # Stage 3
├── index_add.py                      # Stage 4
├── generate_training_pairs.py        # Stage 5
├── path_utils.py                     # 项目相对/绝对路径转换
└── human_orientation_4dhumans.py     # Stage 4 身体朝向封装
scripts/
├── run_pipeline.sh                   # 标准多设备五阶段脚本
├── run_8gpu_pipeline.sh              # GPU 名称封装，转发 run_pipeline.sh
├── run_8npu_pipeline.sh              # NPU 名称封装，转发 run_pipeline.sh
└── run_flask_viewers.sh              # 两个可视化入口
flask/
└── app.py                            # Identity Matching Viewer
flask2/
└── app.py                            # Training Pairs Viewer
```

---

## 依赖与模型

环境文件：

```bash
conda env create -f environment.yml
# 或按当前 CUDA/NPU 环境安装
pip install -r requirements.txt
```

主要模型默认路径：

```text
pretrained_models/TransNetV2/transnetv2-pytorch-weights.pt
pretrained_models/sam3/sam3.pt
pretrained_models/dinov3-vitl16-pretrain-lvd1689m
pretrained_models/yolo/yolo11n-pose.pt
pretrained_models/deca_model.tar
pretrained_models/insightface
pretrained_models/facexlib/weights
pretrained_models/4D-Humans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt
pretrained_models/Qwen3-VL-8B-Instruct
```

部分依赖与 CUDA/NPU、PyTorch 版本强相关，例如 `torch-npu`、`detectron2`、HMR2/4D-Humans。需要按照实验室机器环境安装匹配版本。

## Flask 可视化

### Identity Matching Viewer

查看某个视频的 Stage 3 聚类和 Stage 4 属性：

```bash
VIEWER=flask \
VIDEO_DIR=outputs/video_000001 \
bash scripts/run_flask_viewers.sh
```

等价显式路径：

```bash
VIEWER=flask \
DATA_DIR=outputs/video_000001/identity_matching \
PORT=9000 \
bash scripts/run_flask_viewers.sh
```

常用变量：

```text
DATA_DIR
PORT（默认 9000）
AFTER_PIPELINE_INDEX（默认 post_process_index.json）
SHOW_FACE_POSE
SHOW_BODY_POSE
SHOW_EMOTION
EMOTION_THRESHOLD
```

### Training Pairs Viewer

查看某个视频的 Stage 5 输出：

```bash
VIEWER=flask2 \
VIDEO_DIR=outputs/video_000001 \
bash scripts/run_flask_viewers.sh
```

等价显式路径：

```bash
VIEWER=flask2 \
PAIRS_JSONL=outputs/video_000001/training_pairs/pairs.jsonl \
PERSON_CLUSTERS_DIR=outputs/video_000001/identity_matching/person_clusters \
PORT=7893 \
bash scripts/run_flask_viewers.sh
```

`flask2` 当前一次读取一个 `pairs.jsonl`；v4 每个视频本来就有独立的训练对文件。

---

## 断点续跑与常见问题

### 断点续跑

- Stage 1：读取已有 shot JSONL，按源视频路径跳过已经写入的结果。
- Stage 2：读取已有 one-shot JSONL，按 `source_shot_path` 跳过已经处理的 shot。
- Stage 3：重新计算当前视频身份聚类，并重写相应输出。
- Stage 4：由 `overwrite` 控制是否重写已有索引；标准脚本默认 `True`。
- Stage 5：重写当前视频 `pairs.jsonl/stats.json`，但默认复用首帧和人物相似度矩阵缓存。

重跑 Stage 4、5：

```bash
RUN_STAGE1=False \
RUN_STAGE2=False \
RUN_STAGE3=False \
RUN_STAGE4=True \
RUN_STAGE5=True \
bash scripts/run_pipeline.sh
```

只重跑 Stage 5：

```bash
RUN_STAGE1=False \
RUN_STAGE2=False \
RUN_STAGE3=False \
RUN_STAGE4=False \
RUN_STAGE5=True \
bash scripts/run_pipeline.sh
```

### 为什么出现空目录

workspace 初始化会创建五阶段标准目录，因此后续阶段未运行时也可能提前看到空目录。不要用“目录是否存在”判断阶段完成，应检查：

```text
outputs/<video_id>/pipeline_manifest.json
outputs/logs/<stage>_worker_<phase>.log
```

### Stage 4 是否会检查 mask

标准脚本默认：

```text
INDEX_ENABLE_MASK_HOLE_QUALITY_CHECK=True
```

所以 Stage 4 会读取 SAM mask 计算孔洞质量。若明确不需要：

```bash
INDEX_ENABLE_MASK_HOLE_QUALITY_CHECK=False bash scripts/run_pipeline.sh
```

Stage 5 只读取 Stage 4 已写入的质量标签，不会重新检查 mask。

### 查看最终汇总

`run_pipeline.sh` 在所选阶段完成后读取任务 JSONL 中每个视频的 `pipeline_manifest.json`，打印类似：

```json
{
  "videos": 100,
  "stages": {
    "shot_detection:complete": 100,
    "one_shot_process:complete": 100,
    "identity_matching:complete": 100,
    "index_add:complete": 100,
    "generate_training_pairs:complete": 100
  }
}
```

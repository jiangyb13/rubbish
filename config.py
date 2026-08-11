from dataclasses import dataclass, field
from typing import List, Optional
import os
import json
import cv2
import torch

PIPELINE_PATH_DEFAULTS = {
    "manifest": "pipeline_manifest.json",
    "identity_dir": "identity_matching",
    "identity_jsonl": "identity_matching/output.jsonl",
    "person_clusters_dir": "identity_matching/person_clusters",
    "person_registry_jsonl": "identity_matching/persons.jsonl",
    "training_dir": "training_pairs",
    "pairs_jsonl": "training_pairs/pairs.jsonl",
    "rejected_pairs_jsonl": "training_pairs/rejected_pairs.jsonl",
    "training_stats_json": "training_pairs/stats.json",
    "first_frame_dir": "training_pairs/first_frames",
}

@dataclass
class ShotDetectionConfig:
    task_name: str = field(
        default="shot_detection",
        metadata={"help": "Name of the processing task, used for logging and output organization."
                    "E.g. 'shot_detection', 'one_shot_process', 'face_selection', 'cross_shot_matching'."},
    )
    transnetv2_checkpoint: str = field(
        default="./pretrained_models/TransNetV2/transnetv2-pytorch-weights.pt",
        metadata={"help": "TransNetV2 checkpoint path"},
    )
    input_json: str = field(
        default="./data/input_videos.json",
        metadata={"help": "Input JSON file containing a list of video objects. E.g. [{'video_path': '...'}]"},
    )
    output_dir: str = field(
        default="./outputs/shots_results",
        metadata={"help": "Directory to save the cropped shot videos"},
    )
    output_jsonl: str = field(
        default="./outputs/shots_results/processed_shots.jsonl",
        metadata={"help": "Output JSONL file to save metadata and support resume capability"},
    )
    scene_threshold: float = field(
        default=0.5,
        metadata={"help": "Threshold for TransNetV2 scene boundary prediction"},
    )
    device: str = field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        metadata={"help": "Device to run TransNetV2 on (cuda or cpu)"},
    )
    debug: bool = field(
        default=False,
        metadata={"help": "Enable debug mode. When enabled, prints elapsed time for each processing stage "
                          "in process_single_item (download, SAM3 tracking, postprocess, pair selection, save, cleanup)."},
    )
    phase: int = field(
        default=1,
        metadata={"help": "Current process id for parallel processing."},
    )
    total: int = field(
        default=1,
        metadata={"help": "Total number of processes for parallel processing."},
    )
    enable_trim: bool = field(
        default=True,
        metadata={"help": "Whether to trim head/tail seconds from each detected shot before saving."},
    )
    trim_mode: str = field(
        default="frames",
        metadata={"help": "Trim mode: 'seconds' uses trim_head_sec/trim_tail_sec (old behavior); "
                          "'frames' uses trim_head_frames offset like multi-shot-pipe (only trims head, "
                          "short shots are protected). Default: 'frames'."},
    )
    trim_head_sec: float = field(
        default=1,
        metadata={"help": "Seconds to trim from the beginning of each shot. Typical values: 0.5 or 1.0. "
                          "Only used when enable_trim is True and trim_mode is 'seconds'."},
    )
    trim_tail_sec: float = field(
        default=1,
        metadata={"help": "Seconds to trim from the end of each shot. Typical values: 0.5 or 1.0. "
                          "Only used when enable_trim is True and trim_mode is 'seconds'."},
    )
    trim_head_frames: int = field(
        default=6,
        metadata={"help": "Number of frames to skip at the beginning of each shot (removes TransNetV2 "
                          "boundary residual). Short shots (total frames <= 2*(head+tail)) are not trimmed. "
                          "Only used when enable_trim is True and trim_mode is 'frames'."},
    )
    trim_tail_frames: int = field(
        default=12,
        metadata={"help": "Number of frames to skip at the end of each shot (removes TransNetV2 "
                          "boundary residual from the next shot's beginning leaking into current shot's tail). "
                          "Short shots (total frames <= 2*(head+tail)) are not trimmed. "
                          "Only used when enable_trim is True and trim_mode is 'frames'."},
    )
    min_shot_duration_sec: float = field(
        default=2.0,
        metadata={"help": "Minimum duration (in seconds) of a shot after trimming. "
                          "Shots shorter than this value are discarded."},
    )

    def __post_init__(self):
        self.output_dir = os.path.abspath(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(self.output_jsonl)), exist_ok=True)

@dataclass
class OneShotProcessConfig:
    task_name: str = field(
        default="one_shot_process",
        metadata={"help": "Name of the processing task, used for logging and output organization."
                    "E.g. 'shot_detection', 'one_shot_process', 'face_selection', 'cross_shot_matching'."},
    )
    sam3_checkpoint: str = field(
        default="./pretrained_models/sam3/sam3.pt",
        metadata={"help": "SAM3 checkpoint path"},
    )
    sam2_checkpoint: str = field(
        default="./pretrained_models/sam2-hiera-large",
        metadata={"help": "[Deprecated] SAM2 checkpoint path. Currently unused; face segmentation is handled by SAM3."},
    )
    dinov3_model: str = field(
        default="./pretrained_models/dinov3-vitl16-pretrain-lvd1689m",
        metadata={"help": "DINOv3 model path"},
    )
    yolon11_pose_checkpoint: str = field(
        default="./pretrained_models/yolo/yolo11n-pose.pt",
        metadata={"help": "YOLON11 pose estimation checkpoint path"},
    )
    face_detector_backend: str = field(
        default="insightface",
        metadata={"help": "Face detection backend: 'facexlib' (RetinaFace+ArcFace) or 'insightface' (buffalo_l). "
                          "facexlib uses facexlib.detection + facexlib.recognition; "
                          "insightface uses insightface.app.FaceAnalysis."},
    )
    retinaface_checkpoint : str = field(
        default="./pretrained_models/facexlib/weights",
        metadata={"help": "RetinaFace checkpoint path (only used when face_detector_backend='facexlib')"},
    )
    insightface_model_name: str = field(
        default="buffalo_l",
        metadata={"help": "InsightFace model pack name (only used when face_detector_backend='insightface'). "
                          "Common options: 'buffalo_l', 'buffalo_m', 'buffalo_s', 'buffalo_sc'."},
    )
    insightface_root: str = field(
        default="./pretrained_models/insightface",
        metadata={"help": "Root directory for InsightFace model files (only used when face_detector_backend='insightface')."},
    )
    deca_model_path: str = field(
        default="./pretrained_models/deca_model.tar",
        metadata={"help": "Path to DECA E_flame encoder checkpoint (.tar file). "
                          "Used for head pose (Euler angle) estimation."},
    )
    device : str = field(
        default="cuda:0",
        metadata={"help": "Device to run models on (cuda or cpu)"},
    )
    sam_face : bool = field(
        default=True,
        metadata={"help": "Whether to use SAM3 for face segmentation and cropping. " }
    )
    cropped_face : bool = field(
        default=True,
        metadata={"help": "Whether to save cropped face videos based on SAM3 masks. "
                    "If False, only SAM3 tracking results are saved without face-focused cropping."},
    )
    sam_imgsz: int = field(
        default=1008,
        metadata={"help": "SAM3 inference image size. Must match the ViT backbone's img_size in model_builder "
                          "(default 1008 = 14 * 72). Changing this value requires rebuilding the ViT backbone "
                          "with a matching img_size, otherwise RoPE positional encoding will fail. "
                          "Do NOT change unless you also modify the SAM3 model config."},
    )
    sam_conf: float = field(
        default=0.75,
        metadata={"help": "SAM3 detection confidence threshold."},
    )
    output_dir: str = field(
        default="./outputs_new/processed_results_pairs_plan3_clear_with_mask_filter_005",
        metadata={"help": "Root directory for output results"},
    )
    output_jsonl: str = field(
        default="./outputs_new/output_sam3_plan3.jsonl",
        metadata={"help": "Output JSONL filename"},
    )
    input_jsonl: str = field(
        default="/primus_xpfs_workspace_T04/shuotao.wt/project/VideoID_Process/data/0122_shuf_geminioutput_ultra.jsonl",
        metadata={"help": "Input JSONL file path"},
    )
    imgsz: tuple = field(
        default=None,
        metadata={"help": "[Deprecated] Previously used to resize full-body images to a fixed size. "
                          "Now images are saved at their original cropped size. "
                          "Set to None to disable resize (default). "
                          "If set to a tuple like (1024, 1024), images will be resized to that size (legacy behavior)."},
    )
    save_videos: bool = field(
        default=False, 
        metadata={"help": "Whether to save visualization and cropped videos."}
    )
    min_duration_sec: float = field(
        default=3.0,
        metadata={"help": "Minimum video duration in seconds"},
    )
    device_sam: str = field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        metadata={"help": "Device to run TransNetV2 on (cuda or cpu)"},
    )
    
    min_area_ratio: float = field(
        default=0.1,
        metadata={"help": "Minimum mask area ratio; tracks always below this are filtered"},
    )
    total: int = field(
        default=1,
        metadata={"help": "Total number of GPUs/shards for parallel processing"},
    )
    phase: int = field(
        default=0,
        metadata={"help": "Current shard index (0-based), determines which subset of data to process"},
    )
    raw_tracking_output_root: str = field(
        default="./outputs_new/raw_tracking_results",
        metadata={"help": "Root directory for raw SAM3 tracking results (npy + visualization video)"},
    )
    mask_alpha: float = field(
        default=0.4,
        metadata={"help": "Transparency alpha for mask overlay in visualization"},
    )
    bbox_thickness: int = field(
        default=2,
        metadata={"help": "Thickness of bounding box lines in visualization"},
    )
    font_scale: float = field(
        default=0.6,
        metadata={"help": "Font scale for instance labels in visualization"},
    )
    max_size_change_ratio: float = field(
        default=0.5,
        metadata={"help": "Maximum allowed mask area change ratio relative to source frame. "
                          "E.g. 0.5 means target mask area must be within [0.5x, 1.5x] of source mask area."},
    )
    bg_width_ratio_threshold: float = field(
        default=0.75,
        metadata={"help": "Background filtering threshold. If a track's bbox width consistently exceeds "
                          "this fraction of the frame width across ALL frames, it is considered background and removed. "
                          "Set to 1.0 to disable this filter."},
    )
    sample_fps: int = field(
        default=10,
        metadata={"help": "Target inference FPS for uniform frame sampling via SAM3 vid_stride. "
                          "vid_stride = original_fps / sample_fps. "
                          "Set to 0 to disable frame sampling (process all frames, vid_stride=1)."},
    )
    enable_sharpness_pair_selection: bool = field(
        default=True,
        metadata={"help": "Enable sharpness-aware source/target pair selection. "
                          "When enabled, source is chosen from early frames by highest Laplacian sharpness, "
                          "and target is scored by combining distance and sharpness. "
                          "When disabled, falls back to original logic (first frame as source, max displacement as target)."},
    )
    sharpness_weight: float = field(
        default=0.3,
        metadata={"help": "Weight of sharpness score in the combined target scoring formula. "
                          "combined_score = (1 - sharpness_weight) * normalized_distance + sharpness_weight * normalized_sharpness. "
                          "Only used when enable_sharpness_pair_selection is True."},
    )
    source_candidate_ratio: float = field(
        default=0.3,
        metadata={"help": "Fraction of early frames to consider as source candidates. "
                          "The sharpest crop among the first source_candidate_ratio * total_frames frames is chosen as source. "
                          "Only used when enable_sharpness_pair_selection is True."},
    )
    min_sharpness_ratio: float = field(
        default=0.5,
        metadata={"help": "Minimum sharpness ratio relative to the source frame's sharpness. "
                          "Target candidates with sharpness below source_sharpness * min_sharpness_ratio are filtered out. "
                          "Only used when enable_sharpness_pair_selection is True."},
    )
    small_track_frame_ratio: float = field(
        default=0.5,
        metadata={"help": "Fraction of frames that must have mask area below min_area_ratio to filter out a track. "
                          "E.g. 0.5 means if >50% of frames have small masks, the track is removed. "
                          "This handles segmentation instability where mask size may jump between frames."},
    )
    max_bbox_aspect_ratio: float = field(
        default=5.0,
        metadata={"help": "Maximum allowed bbox aspect ratio (max(w/h, h/w)). "
                          "Tracks where >50% of frames have aspect ratio exceeding this value are filtered out. "
                          "This removes abnormally elongated objects like thin bars or edges."},
    )
    shared_target_max_size_change_ratio: float = field(
        default=0.3,
        metadata={"help": "Maximum allowed mask area change ratio for shared target selection. "
                          "Stricter than max_size_change_ratio used for individual target selection. "
                          "E.g. 0.3 means each subject's mask area must be within [0.7x, 1.3x] of its source mask area."},
    )
    shared_target_min_frame_sharpness: float = field(
        default=100.0,
        metadata={"help": "Absolute minimum Laplacian sharpness threshold for shared target frame. "
                          "Frames with whole-frame sharpness below this value are excluded."},
    )
    shared_target_min_frame_sharpness_ratio: float = field(
        default=0.7,
        metadata={"help": "Relative minimum sharpness ratio for shared target frame. "
                          "Frames with sharpness below max_candidate_sharpness * this_ratio are excluded. "
                          "The effective threshold is max(absolute_threshold, max_sharpness * this_ratio)."},
    )
    
    debug: bool = field(
        default=False,
        metadata={"help": "Enable debug mode. When enabled, prints elapsed time for each processing stage "
                          "in process_single_item (download, SAM3 tracking, postprocess, pair selection, save, cleanup)."},
    )
    min_face_crop_size: int = field(
        default=128,
        metadata={"help": "Minimum face crop size in pixels. Face crops where either width or height is smaller than "
                          "this value will be skipped (not saved). Default 128 means faces smaller than 128x128 are discarded."},
    )
    dino_consistency_check: bool = field(
        default=True,
        metadata={"help": "Whether to enable DINO feature consistency check for tracking validation. "
                          "When enabled, each frame's DINO feature is compared to the first frame's DINO feature. "
                          "Tracks where any frame's cosine similarity falls below dino_consistency_threshold are discarded."},
    )
    dino_consistency_threshold: float = field(
        default=0.6,
        metadata={"help": "Minimum cosine similarity between each frame's DINO feature and the first frame's DINO feature. "
                          "Frames below this threshold indicate tracking drift. "
                          "If any frame falls below this threshold, the entire track is discarded. "
                          "Only used when dino_consistency_check is True."},
    )
    enable_timing: bool = field(
        default=False,
        metadata={"help": "Enable detailed timing profiling for each processing stage in one-shot pipeline. "
                          "When enabled, prints per-shot and per-frame elapsed time breakdown for: "
                          "video loading, YOLO detection, SAM3 tracking, face pre-screening, "
                          "DINO consistency check, frame saving (image cropping, DINO feature extraction, "
                          "full-body image writing, face detection, euler angle estimation, "
                          "cropped face saving, SAM face segmentation), and JSONL writing. "
                          "A cumulative summary is printed at the end of all processing."},
    )

    def __post_init__(self):
        self.output_dir = os.path.abspath(self.output_dir)
        self.raw_tracking_output_root = os.path.abspath(self.raw_tracking_output_root)
        os.makedirs(self.output_dir, exist_ok=True)
        # os.makedirs(self.raw_tracking_output_root, exist_ok=True)
        if self.total > 1:
            base, ext = os.path.splitext(self.output_jsonl)
            self.output_jsonl = f"{base}_phase{self.phase}{ext}"

@dataclass
class IdentityMatchingConfig:
    """Merged Stage 3 (Face Selection) + Stage 4 (Cross-Shot Matching) = Identity Matching."""

    task_name: str = field(
        default="identity_matching",
        metadata={"help": "Name of the processing task."},
    )

    # ---- Input / Output ----
    input_jsonl: str = field(
        default="./outputs/one_shot_process/output.jsonl",
        metadata={"help": "Input JSONL file generated by one-shot pipeline (Stage 2 output)."},
    )
    output_dir: str = field(
        default="./outputs/identity_matching",
        metadata={"help": "Root directory for identity matching outputs."},
    )
    output_jsonl: str = field(
        default="./outputs/identity_matching/output.jsonl",
        metadata={"help": "Detailed output JSONL containing full matching results per id."},
    )
    output_jsonl_simple: str = field(
        default="./outputs/identity_matching/output_simple.jsonl",
        metadata={"help": "Simplified output JSONL containing concise matching results per id."},
    )
    output_global_json: str = field(
        default="./outputs/identity_matching/global_data.json",
        metadata={"help": "Global summary JSON for online tuning, including matrices and config snapshot."},
    )

    # ---- Quality Filtering (from original Stage 3) ----
    min_face_sharpness: float = field(
        default=10.0,
        metadata={"help": "Minimum Laplacian sharpness of face crop. Frames below this are filtered out."},
    )
    min_face_ratio: float = field(
        default=0.005,
        metadata={"help": "Minimum face area ratio in original frame. Frames below this are filtered out."},
    )
    min_valid_frames_per_id: int = field(
        default=2,
        metadata={"help": "Minimum number of frames that survive quality filtering for one id. Otherwise skip this id."},
    )

    # ---- Frontal Frame Selection ----
    face_top_k_frontal_frames: int = field(
        default=5,
        metadata={"help": "For each id, choose top-k most frontal quality-passed frames and average their face features."},
    )
    frontal_yaw_weight: float = field(
        default=1.0,
        metadata={"help": "Weight of |yaw| in frontal score."},
    )
    frontal_pitch_weight: float = field(
        default=1.0,
        metadata={"help": "Weight of |pitch| in frontal score."},
    )
    frontal_roll_weight: float = field(
        default=0.5,
        metadata={"help": "Weight of |roll| in frontal score."},
    )

    # ---- Cross-Shot Matching (from original Stage 4) ----
    similarity_metric: str = field(
        default="cosine",
        metadata={"help": "Similarity metric: cosine (recommended)."},
    )
    face_match_threshold: float = field(
        default=0.65,
        metadata={"help": "Pure face matching threshold. Only keep cross-shot pairs with face similarity >= this value."},
    )
    top_k_matches: int = field(
        default=10,
        metadata={"help": "Max number of matched ids kept for each source id. <=0 means keep all."},
    )
    min_feature_count: int = field(
        default=1,
        metadata={"help": "Minimum number of valid face feature files required for an id."},
    )

    # ---- Person Clustering & Face Pool ----
    person_cluster_output_dir: str = field(
        default="./outputs/identity_matching/person_clusters",
        metadata={"help": "Output directory for person cluster data (per-person organized with four image folders)."},
    )
    save_person_clusters: bool = field(
        default=True,
        metadata={"help": "Whether to save per-person cluster directories with face/full images and diversity TOP-K."},
    )

    # ---- Diversity TOP-K ----
    top_k_face_diversity: int = field(
        default=5,
        metadata={"help": "Global TOP-K: select K frames with largest face angle diversity across all shots of the same person."},
    )
    top_k_dino_diversity: int = field(
        default=5,
        metadata={"help": "Global TOP-K: select K frames with largest DINO feature diversity across all shots of the same person."},
    )
    face_angle_front_abs_yaw_max: float = field(
        default=20.0,
        metadata={"help": "Front-face yaw threshold (degrees). Frames with |yaw| <= this value are saved to front library."},
    )
    face_angle_side_abs_yaw_min: float = field(
        default=30.0,
        metadata={"help": "Side-face yaw threshold (degrees). Frames with yaw <= -threshold go to left, yaw >= threshold go to right."},
    )

    # ---- Output Control ----
    keep_unmatched: bool = field(
        default=False,
        metadata={"help": "Whether to keep ids with no cross-shot matches in output JSONL."},
    )
    save_global_matrix_as_npy: bool = field(
        default=True,
        metadata={"help": "Whether to save global face similarity matrix as .npy file."},
    )
    save_global_matrix_as_json: bool = field(
        default=True,
        metadata={"help": "Whether to save global face similarity matrix as .json file."},
    )
    save_pairwise_similarities: bool = field(
        default=True,
        metadata={"help": "Save detailed pairwise similarity JSON with histogram and statistics for threshold tuning."},
    )

    def __post_init__(self):
        self.output_dir = os.path.abspath(self.output_dir)
        self.output_jsonl = os.path.abspath(self.output_jsonl)
        self.output_jsonl_simple = os.path.abspath(self.output_jsonl_simple)
        self.output_global_json = os.path.abspath(self.output_global_json)
        self.person_cluster_output_dir = os.path.abspath(self.person_cluster_output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.output_jsonl), exist_ok=True)
        os.makedirs(os.path.dirname(self.output_jsonl_simple), exist_ok=True)
        os.makedirs(os.path.dirname(self.output_global_json), exist_ok=True)
        os.makedirs(os.path.dirname(self.output_global_json), exist_ok=True)

@dataclass
class IndexAddConfig:
    """Configuration for after-pipeline index building."""
    pipeline_input_jsonl: Optional[str] = field(default=None, metadata={"help": "Canonical Stage-1 task JSONL."})
    output_root: str = field(default="outputs", metadata={"help": "Root containing per-video workspaces."})
    video_dir: Optional[str] = field(default=None, metadata={"help": "Standalone outputs/<video_id> workspace."})
    video_path: Optional[str] = field(default=None, metadata={"help": "Optional standalone source video override."})
    video_id: Optional[str] = field(default=None, metadata={"help": "Optional standalone video id."})
    phase: int = field(default=0, metadata={"help": "Continuous shard rank."})
    total: int = field(default=1, metadata={"help": "Continuous shard worker count."})
    global_mode: bool = field(default=False, metadata={"help": "Process the explicitly configured identity_matching directory instead of workspace mode."})
    # identity_matching 输出根目录
    path: str = field(
        default="outputs/identity_matching",
        metadata={"help": "identity_matching output directory. Reads output.jsonl and person_clusters from here."}
    )
    # 可选直接指定 identity jsonl
    identity_jsonl: Optional[str] = field(
        default=None,
        metadata={"help": "Optional explicit identity_matching output.jsonl path."}
    )
    # 可选用于恢复 cluster_meta 成员的 jsonl
    member_recovery_jsonl: Optional[str] = field(
        default=None,
        metadata={"help": "Optional JSONL for recovering cluster_meta members not present in identity_matching/output.jsonl."}
    )
    # 可选单独指定 person_clusters 目录
    person_clusters_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Optional explicit identity_matching person_clusters directory."}
    )
    # 输出文件名
    output_filename: str = field(
        default="post_process_index.json",
        metadata={"help": "Filename written for each person."}
    )
    # 输出根目录（可选）
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Optional separate output root. Defaults to each input person folder."}
    )
    # v3 批量处理：每行 SRC_PERSON_CLUSTERS|VIDEO_NAME|PART_NAME|UUID_NAME
    unit_list_file: Optional[str] = field(
        default=None,
        metadata={"help": "Optional shard txt for index_add_v3. Each line is SRC_PERSON_CLUSTERS|VIDEO_NAME|PART_NAME|UUID_NAME."}
    )
    unit_list_input_base_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Input base dir used to mirror unit_list_file source paths under output_dir."}
    )
    # 只处理指定 person_id
    person_ids: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Optional person ids to process, e.g. person_0001 person_0002."}
    )
    # 是否禁用姿态属性
    disable_pose: bool = field(
        default=False,
        metadata={"help": "Disable the default pose attributes while keeping image mappings."}
    )
    # 是否启用情感提取
    enable_emotion: bool = field(
        default=True,
        metadata={"help": "Add DeepFace emotion attributes to each frame's image entries."}
    )
    # 是否用 Qwen3-VL 对表情结果做二次复筛
    enable_emotion_vlm: bool = field(
        default=False,
        metadata={"help": "Use Qwen3-VL to strictly verify the predicted emotion label and write expression.emo_flag."}
    )
    update_features: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Incrementally update existing after_pipeline_index.json files for selected features only. Supported: emotion, emotion_vlm, body_pose, mask_hole_quality, face_boundary_quality, face_occlusion_quality, image_clarity_quality."}
    )
    update_emotion_vlm_only: bool = field(
        default=False,
        metadata={"help": "Shortcut alias for --update_features emotion_vlm."}
    )
    emotion_vlm_model_path: str = field(
        default="pretrained_models/Qwen3-VL-8B-Instruct",
        metadata={"help": "Qwen3-VL model path used for emotion verification."}
    )
    emotion_vlm_device: str = field(
        default="cuda:0",
        metadata={"help": "Device map passed to Qwen3-VL for emotion verification, e.g. cuda:0 or auto."}
    )
    emotion_vlm_max_new_tokens: int = field(
        default=512,
        metadata={"help": "Max new tokens for Qwen3-VL emotion verification."}
    )
    # 是否启用 4D-Humans + YOLO body pose/body extent 提取
    enable_body_pose: bool = field(
        default=True,
        metadata={"help": "Add 4D-Humans body orientation and YOLO body extent attributes to each frame's image entries."}
    )
    body_pose_checkpoint: str = field(
        default="./pretrained_models/4D-Humans/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt",
        metadata={"help": "HMR2 checkpoint path used for 4D-Humans body orientation."}
    )
    body_pose_yolo_checkpoint: str = field(
        default="./pretrained_models/yolo/yolo11n-pose.pt",
        metadata={"help": "YOLO pose checkpoint path used for body extent classification."}
    )
    body_pose_detector: str = field(
        default="vitdet",
        metadata={"help": "4D-Humans detector backend: vitdet/regnety use Detectron2; yolo uses YOLO bbox and avoids Detectron2."}
    )
    body_pose_device: str = field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        metadata={"help": "Device for body pose models, for example cuda:0 or cpu."}
    )
    enable_mask_hole_quality_check: bool = field(
        default=True,
        metadata={"help": "Add quality_label for white-background images by counting holes in the corresponding SAM mask."}
    )
    mask_hole_threshold: int = field(
        default=0,
        metadata={"help": "White-background images whose SAM mask hole_count is greater than this value get quality_label=False."}
    )
    quality_update_overwrite: bool = field(
        default=True,
        metadata={"help": "In incremental quality updates, overwrite existing quality items. If False, only missing quality items are computed."}
    )
    enable_face_boundary_quality_check: bool = field(
        default=False,
        metadata={"help": "Compatibility switch: enable both Stage4 face bbox boundary and face mask coverage quality checks."}
    )
    enable_face_bbox_boundary_quality_check: bool = field(
        default=False,
        metadata={"help": "Use InsightFace in Stage4 to reject face crops whose expanded face bbox touches image boundary."}
    )
    enable_face_mask_coverage_quality_check: bool = field(
        default=False,
        metadata={"help": "Use InsightFace face bbox to reject crops whose original face bbox contains SAM background/white pixels."}
    )
    face_boundary_expand_ratio: float = field(
        default=1.1,
        metadata={"help": "Expansion ratio applied to detected face bbox before boundary checking. SAM mask coverage uses the original bbox."}
    )
    face_mask_min_foreground_ratio: float = field(
        default=0.98,
        metadata={"help": "Minimum foreground ratio inside the original face bbox on the SAM face mask. Default 0.98 tolerates small boundary/background noise."}
    )
    face_mask_coverage_max_abs_yaw: float = field(
        default=30.0,
        metadata={"help": "Run face mask coverage quality check only when abs(face yaw) is within this threshold in degrees."}
    )
    face_quality_device: str = field(
        default="cuda:0",
        metadata={"help": "Device for Stage4 face-boundary quality InsightFace detector, e.g. cuda:0 or cpu."}
    )
    face_quality_model_name: str = field(
        default="buffalo_l",
        metadata={"help": "InsightFace model pack used by Stage4 face-boundary quality check."}
    )
    face_quality_model_root: str = field(
        default="./pretrained_models/insightface",
        metadata={"help": "InsightFace model root used by Stage4 face-boundary quality check."}
    )
    face_quality_det_size: int = field(
        default=640,
        metadata={"help": "Square detector input size for Stage4 face-boundary quality check."}
    )
    face_quality_recompute_bbox: bool = field(
        default=False,
        metadata={"help": "In incremental face_boundary_quality updates, recompute face bboxes even when existing quality bbox is available."}
    )
    enable_face_occlusion_quality_check: bool = field(
        default=False,
        metadata={"help": "Use Qwen3-VL in Stage4 to reject images where the face is occluded."}
    )
    enable_image_clarity_quality_check: bool = field(
        default=False,
        metadata={"help": "Use image clarity checks in Stage4 to reject blurry face images."}
    )
    enable_image_clarity_vlm_check: bool = field(
        default=True,
        metadata={"help": "When image clarity quality is enabled, also use Qwen3-VL for clarity judgment. If False, only Laplacian sharpness is used."}
    )
    quality_vlm_model_path: str = field(
        default="pretrained_models/Qwen3-VL-8B-Instruct",
        metadata={"help": "Qwen3-VL model path used for Stage4 face occlusion and clarity quality checks."}
    )
    quality_vlm_device: str = field(
        default="cuda:0",
        metadata={"help": "Device map passed to Qwen3-VL for Stage4 quality checks, e.g. cuda:0 or auto."}
    )
    quality_vlm_max_new_tokens: int = field(
        default=512,
        metadata={"help": "Max new tokens for Qwen3-VL Stage4 quality checks."}
    )
    clarity_laplacian_threshold: float = field(
        default=10.0,
        metadata={"help": "Minimum Laplacian variance required for Stage4 image clarity quality."}
    )
    # 是否覆盖已有索引文件
    overwrite: bool = field(
        default=False,
        metadata={"help": "Overwrite existing after-pipeline index files."}
    )

    rank: int = field(
        default=0,
        metadata={"help": "Current shard index (0-based) for parallel processing. 0 means process all when total_rank is 0."}
    )
    total_rank: int = field(
        default=0,
        metadata={"help": "Total number of shards for parallel processing. 0 means process all persons at once."}
    )


@dataclass
class TrainingPairsConfig:
    """Configuration for building training-pair JSONL from after-pipeline indexes."""

    pipeline_input_jsonl: Optional[str] = field(default=None, metadata={"help": "Canonical Stage-1 task JSONL."})
    output_root: str = field(default="outputs", metadata={"help": "Root containing per-video workspaces."})
    video_dir: Optional[str] = field(default=None, metadata={"help": "Standalone outputs/<video_id> workspace."})
    video_path: Optional[str] = field(default=None, metadata={"help": "Optional standalone source video override."})
    video_id: Optional[str] = field(default=None, metadata={"help": "Optional standalone video id."})
    phase: int = field(default=0, metadata={"help": "Continuous shard rank."})
    total: int = field(default=1, metadata={"help": "Continuous shard worker count."})
    global_mode: bool = field(default=False, metadata={"help": "Generate pairs directly from the explicitly configured person_clusters directory instead of workspace mode."})

    person_clusters_dir: str = field(
        default="outputs_demo/identity_matching/person_clusters",
        metadata={"help": "Directory containing person_*/after_pipeline_index.json files."},
    )
    output_jsonl: str = field(
        default="outputs_demo/training_pairs/pairs.jsonl",
        metadata={"help": "Output JSONL path for generated training pairs."},
    )
    rejected_jsonl: Optional[str] = field(
        default=None,
        metadata={"help": "Optional JSONL path for rejected training-pair candidates. Defaults to rejected_pairs.jsonl beside output_jsonl."},
    )
    stats_json: str = field(
        default="outputs_demo/training_pairs/stats.json",
        metadata={"help": "Output JSON path for generation statistics."},
    )
    first_frame_dir: str = field(
        default="outputs_demo/training_pairs/first_frames",
        metadata={"help": "Directory used to save extracted first frames for target videos."},
    )
    unit_list_file: Optional[str] = field(
        default=None,
        metadata={"help": "Optional list file. Each line starts with a person_clusters dir, optionally followed by video|part|uuid."},
    )
    unit_list_input_base_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Input base dir used to mirror unit_list_file source paths under the output_jsonl/stats/first-frame bases."},
    )
    one_shot_process_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Optional root of one_shot_process. If set, face features are read from one_shot_process/<video>/<part>/<shot_key>/id_<obj>/features/face_feature/<frame>.npy."},
    )
    index_filename: str = field(
        default="post_process_index.json",
        metadata={"help": "Per-person index filename."},
    )
    person_ids: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Optional person ids to process, e.g. person_0001 person_0002."},
    )
    ref_image_type: str = field(
        default="face_orig",
        metadata={"help": "Reference face image type from after_pipeline_index images, usually face_orig or face_white."},
    )
    ref_fallback_image_type: str = field(
        default="face_white",
        metadata={"help": "Fallback image type when ref_image_type is missing."},
    )
    ignore_ref_quality: bool = field(
        default=False,
        metadata={"help": "In Stage5 ref selection, ignore all Stage4 quality labels for face/full refs."},
    )
    ignore_mask_hole_ref_quality: bool = field(
        default=False,
        metadata={"help": "In Stage5 ref selection, ignore only mask_hole quality failures while keeping other quality failures."},
    )
    angle_ref_count: int = field(
        default=5,
        metadata={"help": "Number of angle reference images. Current selector expects 5 semantic buckets."},
    )
    emo_ref_count: int = field(
        default=5,
        metadata={"help": "Number of emotion reference images."},
    )
    body_pose_ref_count: int = field(
        default=5,
        metadata={"help": "Number of body-pose reference images."},
    )
    bucket_candidate_topk: int = field(
        default=8,
        metadata={"help": "Per-bucket diverse candidate pool size before random sampling."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Base random seed for reproducible bucket/reference sampling."},
    )
    min_same_prefix_shot_gap: int = field(
        default=0,
        metadata={"help": "Same video prefix requires abs(shot_no_a - shot_no_b) > this value."},
    )
    angle_front_up_min_pitch: float = field(
        default=-10.0,
        metadata={"help": "Minimum pitch for front_up angle bucket. Prevents over-upward head poses."},
    )
    angle_front_up_max_pitch: float = field(
        default=20.0,
        metadata={"help": "Maximum pitch for front_up angle bucket."},
    )
    angle_front_down_min_pitch: float = field(
        default=40.0,
        metadata={"help": "Minimum pitch for front_down angle bucket."},
    )
    angle_front_down_max_pitch: float = field(
        default=70.0,
        metadata={"help": "Maximum pitch for front_down angle bucket. Prevents over-downward head poses."},
    )
    enable_dino_ref_diversity: bool = field(
        default=False,
        metadata={"help": "Use DINO feature similarity, instead of face similarity, to make refs inside each group more visually diverse."},
    )
    dino_max_pairwise_cosine: float = field(
        default=0.95,
        metadata={"help": "When DINO ref diversity is enabled, reject a generated pair if any ref group has max pairwise DINO cosine above this value."},
    )
    bucket_top_t: int = field(
        default=50,
        metadata={"help": "Max candidates kept per angle/emotion bucket before beam search."},
    )
    beam_size: int = field(
        default=200,
        metadata={"help": "Beam size used for diverse reference selection."},
    )
    cosine_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for mean pairwise cosine in reference selection."},
    )
    max_cosine_weight: float = field(
        default=0.25,
        metadata={"help": "Weight for max pairwise cosine in reference selection."},
    )
    emotion_confidence_weight: float = field(
        default=0.01,
        metadata={"help": "Reward weight for dominant emotion confidence."},
    )
    overwrite_first_frames: bool = field(
        default=False,
        metadata={"help": "Re-extract first frames even if cached files already exist."},
    )
    overwrite_similarity_matrix: bool = field(
        default=False,
        metadata={"help": "Recompute per-person candidate similarity matrix even if the cache file exists."},
    )

    def __post_init__(self):
        self.person_clusters_dir = os.path.abspath(self.person_clusters_dir)
        self.output_jsonl = os.path.abspath(self.output_jsonl)
        if self.rejected_jsonl:
            self.rejected_jsonl = os.path.abspath(self.rejected_jsonl)
        else:
            self.rejected_jsonl = os.path.join(os.path.dirname(self.output_jsonl), "rejected_pairs.jsonl")
        self.stats_json = os.path.abspath(self.stats_json)
        self.first_frame_dir = os.path.abspath(self.first_frame_dir)
        if self.one_shot_process_dir:
            self.one_shot_process_dir = os.path.abspath(self.one_shot_process_dir)
        os.makedirs(os.path.dirname(self.output_jsonl), exist_ok=True)
        os.makedirs(os.path.dirname(self.rejected_jsonl), exist_ok=True)
        os.makedirs(os.path.dirname(self.stats_json), exist_ok=True)
        os.makedirs(self.first_frame_dir, exist_ok=True)

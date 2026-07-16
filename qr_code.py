CUDA_VISIBLE_DEVICES=0 TASK_NAME=index_add python main.py \
  --path /mnt/nfs/data/peiyao/HUAWEI_CrossPairDataset/outputs/identity_matching \
  --member_recovery_jsonl /mnt/nfs/data/peiyao/HUAWEI_CrossPairDataset/outputs/one_shot_process/output.jsonl \
  --output_filename post_process_index_v3.json \
  --enable_emotion False \
  --enable_body_pose True \
  --body_pose_detector yolo \
  --enable_face_bbox_boundary_quality_check True \
  --enable_face_mask_coverage_quality_check True \
  --face_mask_min_foreground_ratio 0.98 \
  --face_mask_coverage_max_abs_yaw 30.0 \
  --face_quality_model_name buffalo_l \
  --face_quality_model_root /mnt/nfs/data/pretrained_models/insightface \
  --face_quality_device cuda:0 \
  --overwrite True \
  --rank 0 \
  --total_rank 1

# Incremental mode: update existing post_process_index_v3.json only for quality labels.
# CUDA_VISIBLE_DEVICES=0 TASK_NAME=index_add python main.py \
#   --path /mnt/nfs/data/peiyao/HUAWEI_CrossPairDataset/outputs/identity_matching \
#   --member_recovery_jsonl /mnt/nfs/data/peiyao/HUAWEI_CrossPairDataset/outputs/one_shot_process/output.jsonl \
#   --output_filename post_process_index_v3.json \
#   --update_features face_boundary_quality \
#   --enable_face_bbox_boundary_quality_check True \
#   --enable_face_mask_coverage_quality_check True \
#   --face_mask_min_foreground_ratio 0.98 \
#   --face_mask_coverage_max_abs_yaw 30.0 \
#   --face_quality_model_name buffalo_l \
#   --face_quality_model_root /mnt/nfs/data/pretrained_models/insightface \
#   --face_quality_device cuda:0 \
#   --rank 0 \
#   --total_rank 1

# Incremental mode: update both mask-hole quality and face-boundary quality.
# CUDA_VISIBLE_DEVICES=0 TASK_NAME=index_add python main.py \
#   --path /mnt/nfs/data/peiyao/HUAWEI_CrossPairDataset/outputs/identity_matching \
#   --member_recovery_jsonl /mnt/nfs/data/peiyao/HUAWEI_CrossPairDataset/outputs/one_shot_process/output.jsonl \
#   --output_filename post_process_index_v3.json \
#   --update_features mask_hole_quality face_boundary_quality \
#   --mask_hole_threshold 10 \
#   --enable_face_bbox_boundary_quality_check True \
#   --enable_face_mask_coverage_quality_check True \
#   --face_mask_min_foreground_ratio 0.98 \
#   --face_mask_coverage_max_abs_yaw 30.0 \
#   --face_quality_model_name buffalo_l \
#   --face_quality_model_root /mnt/nfs/data/pretrained_models/insightface \
#   --face_quality_device cuda:0 \
#   --rank 0 \
#   --total_rank 1

cd /mnt/nfs/data/peiyao/HUAWEI_CrossPairDataset_v4

VIDEO_DIR=/mnt/nfs/data/peiyao/HUAWEI_CrossPairDataset_v4/outputs/your_video_id

TASK_NAME=index_add python3 main.py \
  --video_dir "$VIDEO_DIR" \
  --update_features face_occlusion_quality image_clarity_quality \
  --enable_face_occlusion_quality_check true \
  --enable_image_clarity_quality_check true \
  --enable_image_clarity_vlm_check true \
  --quality_vlm_model_path pretrained_models/Qwen3-VL-8B-Instruct \
  --quality_vlm_device cuda:0 \
  --quality_vlm_max_new_tokens 512 \
  --clarity_laplacian_threshold 10.0 \
  --quality_update_overwrite true

TASK_NAME=generate_training_pairs python3 main.py \
  --video_dir "$VIDEO_DIR" \
  --enable_dino_ref_diversity true \
  --dino_max_pairwise_cosine 0.95 \
  --angle_front_up_min_pitch -10 \
  --angle_front_up_max_pitch 20 \
  --angle_front_down_min_pitch 40 \
  --angle_front_down_max_pitch 70

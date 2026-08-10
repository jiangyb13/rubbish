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

TASK_NAME=generate_training_pairs python main.py \
  --video_dir /data/huanan/code/jwx1520881/HUAWEI_CrossPairDataset_v4/outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b \
  --index_filename post_process_index.json \
  --angle_ref_count 5 \
  --emo_ref_count 5 \
  --body_pose_ref_count 5 \
  --bucket_candidate_topk 5 \
  --min_same_prefix_shot_gap 3 \
  --enable_dino_ref_diversity true \
  --dino_max_pairwise_cosine 0.95 \
  --angle_front_up_min_pitch -10 \
  --angle_front_up_max_pitch 20 \
  --angle_front_down_min_pitch 40 \
  --angle_front_down_max_pitch 70

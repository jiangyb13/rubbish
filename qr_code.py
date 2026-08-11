cd /mnt/nfs/data/peiyao/HUAWEI_CrossPairDataset_v4

TASK_NAME=generate_training_pairs python main.py \
  --video_dir /你的/VIDEO_DIR \
  --index_filename post_process_index.json \
  --angle_ref_count 5 \
  --emo_ref_count 5 \
  --body_pose_ref_count 5 \
  --allow_variable_ref_count true \
  --min_ref_count 3 \
  --bucket_candidate_topk 8 \
  --min_same_prefix_shot_gap 3 \
  --enable_dino_ref_diversity true \
  --dino_max_pairwise_cosine 0.95 \
  --dino_max_mean_pairwise_cosine 0.85 \
  --angle_front_up_min_pitch -10 \
  --angle_front_up_max_pitch 20 \
  --angle_front_down_min_pitch 40 \
  --angle_front_down_max_pitch 70

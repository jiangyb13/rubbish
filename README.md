torchrun --master_port 26599 --nproc_per_node 8 \
  scripts/extract_ref_attn_map.py \
  --config  configs/mimo/id_inference/inference.py \
  --ckpt-path  /cache/I2V_ID_model.pth \
  --test_jsonl  test.jsonl \
  --img_dir  /path/to/images \
  --face_aug_dir  /path/to/face_aug \
  --save-dir  /tmp/videos \
  --attn-save-dir  /tmp/attn_maps \
  --attn-layers  0 6 12 18 23 \
  --attn-step  25

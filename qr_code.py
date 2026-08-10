{
  "quality_label": false,
  "failed_quality_keys": [],
  "quality": {
    "face_occlusion": {
      "checked": true,
      "confidence": 0.95,
      "face_occluded": false,
      "image_path": "/cache/identity_matching/video_720p_15min_0/part_0079/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/person_clusters/person_0000/face_orig/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b_shot_832_id0_frame0003.jpg",
      "max_new_tokens": 512,
      "model": "/data/huanan/misc/models/vlm/Qwen3-VL-8B-Instruct/",
      "parse_status": "success",
      "passed": true,
      "raw_response": "{\"face_occluded\": false, \"confidence\": 0.95, \"reason\": \"Face fully visible, no obstructions or extreme cropping detected.\"}",
      "reason": "Face fully visible, no obstructions or extreme cropping detected.",
      "status": "ok"
    },
    "image_clarity_laplacian": {
      "checked": true,
      "image_path": "/cache/identity_matching/video_720p_15min_0/part_0079/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/person_clusters/person_0000/face_orig/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b_shot_832_id0_frame0003.jpg",
      "passed": true,
      "sharpness": 14.597181081924457,
      "status": "ok",
      "threshold": 10
    },
    "image_clarity_vlm": {
      "checked": true,
      "confidence": 0.85,
      "image_path": "/cache/identity_matching/video_720p_15min_0/part_0079/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/person_clusters/person_0000/face_orig/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b_shot_832_id0_frame0003.jpg",
      "is_clear": true,
      "max_new_tokens": 512,
      "model": "/data/huanan/misc/models/vlm/Qwen3-VL-8B-Instruct/",
      "parse_status": "success",
      "passed": true,
      "raw_response": "{\"is_clear\": true, \"confidence\": 0.85, \"reason\": \"Face is in focus with visible features, no motion blur or compression artifacts.\"}",
      "reason": "Face is in focus with visible features, no motion blur or compression artifacts.",
      "status": "ok"
    },
    "mask_hole": {
      "checked": true,
      "entry_image_type": "face_orig",
      "hole_count": 0,
      "is_mask_source_white_image": true,
      "is_white_image": false,
      "mask_image_type": "face_white",
      "mask_path": "/cache/one_shot_process/video_720p_15min_0/part_0079/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b_shot_832/id_0/face/cropped_face/face_mask_for_face/3.npy",
      "passed": true,
      "status": "ok",
      "threshold": 0
    }
  },
  "pose": {
    "pitch": 37.06200269756981,
    "roll": 7.129563264685443,
    "yaw": -28.53006563823389
  },
  "expression": {
    "backend": "emotiefflib",
    "dominant": "neutral",
    "model_name": "enet_b0_8_best_vgaf",
    "scores": {
      "angry": 17.03309565782547,
      "contempt": 15.641628205776215,
      "disgust": 0.9287634864449501,
      "fear": 0.12926830677315593,
      "happy": 16.53546541929245,
      "neutral": 30.79792559146881,
      "sad": 16.961726546287537,
      "surprise": 1.9721340388059616
    },
    "status": "success"
  },
  "body_pose": {
    "bbox": [
      11.699872016906738,
      11.207839965820312,
      743.7472534179688,
      887.67724609375
    ],
    "body_part": "half_body",
    "detector": "yolo",
    "heading": [
      -0.08940402418375015,
      0.06174793839454651,
      -0.9940795302391052
    ],
    "label": "front",
    "status": "success",
    "yaw_deg": -5.139154434204102
  },
  "image_path": "/cache/identity_matching/video_720p_15min_0/part_0079/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/person_clusters/person_0000/face_orig/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b_shot_832_id0_frame0003.jpg",
  "index_path": "/data/huanan/code/jwx1520881/HUAWEI_CrossPairDataset_v4/outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/person_clusters/person_0000/post_process_index.json"
}

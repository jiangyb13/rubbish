{
  "config": {
    "pipeline_input_jsonl": null,
    "output_root": "outputs",
    "video_dir": "/data/huanan/code/jwx1520881/HUAWEI_CrossPairDataset_v4/outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b",
    "video_path": null,
    "video_id": null,
    "phase": 0,
    "total": 1,
    "global_mode": false,
    "person_clusters_dir": "/data/huanan/code/jwx1520881/HUAWEI_CrossPairDataset_v4/outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/person_clusters",
    "output_jsonl": "/data/huanan/code/jwx1520881/HUAWEI_CrossPairDataset_v4/outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/pairs.jsonl",
    "rejected_jsonl": "/data/huanan/code/jwx1520881/HUAWEI_CrossPairDataset_v4/outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl",
    "stats_json": "/data/huanan/code/jwx1520881/HUAWEI_CrossPairDataset_v4/outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/stats.json",
    "first_frame_dir": "/data/huanan/code/jwx1520881/HUAWEI_CrossPairDataset_v4/outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/first_frames",
    "unit_list_file": null,
    "unit_list_input_base_dir": null,
    "one_shot_process_dir": null,
    "index_filename": "post_process_index.json",
    "person_ids": null,
    "ref_image_type": "face_orig",
    "ref_fallback_image_type": "face_white",
    "angle_ref_count": 5,
    "emo_ref_count": 5,
    "body_pose_ref_count": 5,
    "bucket_candidate_topk": 5,
    "seed": 42,
    "min_same_prefix_shot_gap": 3,
    "angle_front_up_min_pitch": -10.0,
    "angle_front_up_max_pitch": 20.0,
    "angle_front_down_min_pitch": 40.0,
    "angle_front_down_max_pitch": 70.0,
    "enable_dino_ref_diversity": true,
    "dino_max_pairwise_cosine": 0.95,
    "bucket_top_t": 50,
    "beam_size": 200,
    "cosine_weight": 1.0,
    "max_cosine_weight": 0.25,
    "emotion_confidence_weight": 0.01,
    "overwrite_first_frames": false,
    "overwrite_similarity_matrix": false
  },
  "total_persons": 48,
  "persons": {
    "person_0000": {
      "status": "ok",
      "candidate_count": 295,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0000_candidate_similarity.npz",
        "status": "built",
        "shape": [
          295,
          295
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 62,
      "skipped": {
        "missing_feature": 295,
        "missing_dino_feature": 295,
        "low_quality_face_ref": 295,
        "low_quality_full_ref": 295
      },
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 62,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0001": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0001_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0002": {
      "status": "ok",
      "candidate_count": 165,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0002_candidate_similarity.npz",
        "status": "built",
        "shape": [
          165,
          165
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 34,
      "skipped": {
        "missing_feature": 165,
        "missing_dino_feature": 165,
        "low_quality_face_ref": 165,
        "low_quality_full_ref": 165
      },
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 34,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0003": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0003_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0004": {
      "status": "ok",
      "candidate_count": 29,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0004_candidate_similarity.npz",
        "status": "built",
        "shape": [
          29,
          29
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 6,
      "skipped": {
        "missing_feature": 29,
        "missing_dino_feature": 29,
        "low_quality_face_ref": 29,
        "low_quality_full_ref": 29
      },
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 6,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0005": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0005_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0006": {
      "status": "ok",
      "candidate_count": 66,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0006_candidate_similarity.npz",
        "status": "built",
        "shape": [
          66,
          66
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 15,
      "skipped": {
        "missing_feature": 66,
        "missing_dino_feature": 66,
        "low_quality_face_ref": 66,
        "low_quality_full_ref": 66
      },
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 15,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0007": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0007_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0008": {
      "status": "ok",
      "candidate_count": 222,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0008_candidate_similarity.npz",
        "status": "built",
        "shape": [
          222,
          222
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 51,
      "skipped": {
        "missing_feature": 222,
        "missing_dino_feature": 222,
        "low_quality_face_ref": 222,
        "low_quality_full_ref": 222
      },
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 51,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0009": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0009_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0010": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0010_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0011": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0011_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0012": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0012_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0013": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0013_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0014": {
      "status": "ok",
      "candidate_count": 12,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0014_candidate_similarity.npz",
        "status": "built",
        "shape": [
          12,
          12
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 3,
      "skipped": {
        "missing_feature": 12,
        "missing_dino_feature": 12,
        "low_quality_face_ref": 12,
        "low_quality_full_ref": 12
      },
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 3,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0015": {
      "status": "ok",
      "candidate_count": 14,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0015_candidate_similarity.npz",
        "status": "built",
        "shape": [
          14,
          14
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 3,
      "skipped": {
        "missing_feature": 14,
        "missing_dino_feature": 14,
        "low_quality_full_ref": 2
      },
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 3,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0016": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0016_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0017": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0017_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0018": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0018_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0019": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0019_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0020": {
      "status": "ok",
      "candidate_count": 5,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0020_candidate_similarity.npz",
        "status": "built",
        "shape": [
          5,
          5
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 2,
      "skipped": {
        "missing_feature": 5,
        "missing_dino_feature": 5
      },
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 2,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0021": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0021_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0022": {
      "status": "ok",
      "candidate_count": 8,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0022_candidate_similarity.npz",
        "status": "built",
        "shape": [
          8,
          8
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 2,
      "skipped": {
        "missing_feature": 8,
        "missing_dino_feature": 8
      },
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 2,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0023": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0023_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0024": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0024_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0025": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0025_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0026": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0026_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0027": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0027_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0028": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0028_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0029": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0029_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0030": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0030_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0031": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0031_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0032": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0032_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0033": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0033_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0034": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0034_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0035": {
      "status": "ok",
      "candidate_count": 6,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0035_candidate_similarity.npz",
        "status": "built",
        "shape": [
          6,
          6
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 2,
      "skipped": {
        "missing_feature": 6,
        "missing_dino_feature": 6,
        "low_quality_full_ref": 4
      },
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 2,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0036": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0036_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0037": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0037_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0038": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0038_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0039": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0039_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0040": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0040_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0041": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0041_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0042": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0042_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0043": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0043_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0044": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0044_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0045": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0045_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0046": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0046_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    },
    "person_0047": {
      "status": "ok",
      "candidate_count": 0,
      "similarity_matrix": {
        "path": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/person_0047_candidate_similarity.npz",
        "status": "built",
        "shape": [
          0,
          0
        ],
        "valid_feature_count": 0
      },
      "target_video_count": 0,
      "skipped": {},
      "per_target": [],
      "dino_diversity_rejected": 0,
      "insufficient_refs_rejected": 0,
      "rejected_jsonl": "outputs/outputs_from_multi_person/0ae41a56-67d4-4635-b5e7-96e5a8d3de4b/training_pairs/rejected_pairs.jsonl"
    }
  },
  "rows_written": 0,
  "first_frame_error": 0
}

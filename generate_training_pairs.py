import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from .config import TrainingPairsConfig
from .path_utils import resolve_repo_path, to_repo_relative_path


WORKSPACE_REL_PATHS = {
    "manifest": "pipeline_manifest.json",
    "person_clusters_dir": "identity_matching/person_clusters",
    "training_dir": "training_pairs",
    "pairs_jsonl": "training_pairs/pairs.jsonl",
    "rejected_pairs_jsonl": "training_pairs/rejected_pairs.jsonl",
    "training_stats_json": "training_pairs/stats.json",
    "first_frame_dir": "training_pairs/first_frames",
}


def _safe_video_id(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value or "").strip())
    return text.strip("._") or "video"


def _workspace_path(video_dir: str, key: str) -> str:
    return os.path.join(video_dir, WORKSPACE_REL_PATHS[key])


def _workspace_person_clusters_dir(video_dir: str) -> str:
    flat_dir = os.path.join(video_dir, "person_clusters")
    if os.path.isdir(flat_dir):
        return flat_dir
    return _workspace_path(video_dir, "person_clusters_dir")


def _shard_range(total: int, rank: int, world_size: int) -> Tuple[int, int]:
    if world_size <= 0:
        raise ValueError("total must be positive")
    if not 0 <= rank < world_size:
        raise ValueError(f"phase must be in [0, {world_size}), got {rank}")
    base, remainder = divmod(total, world_size)
    start = rank * base + min(rank, remainder)
    end = start + base + (1 if rank < remainder else 0)
    return start, end


def _load_task_units(input_jsonl: str, output_root: str) -> List[Dict[str, str]]:
    units = []
    seen = set()
    with open(resolve_repo_path(input_jsonl), "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid task JSONL at line {line_number}: {exc}") from exc
            video_path = os.path.abspath(os.path.expanduser(str(record.get("video_path") or "")))
            if not video_path:
                raise ValueError(f"Missing video_path at line {line_number}")
            video_id = _safe_video_id(record.get("video_id") or os.path.splitext(os.path.basename(video_path))[0])
            if video_id in seen:
                raise ValueError(f"Duplicate video_id in task JSONL: {video_id}")
            seen.add(video_id)
            video_dir = record.get("video_dir")
            if video_dir:
                video_dir = os.path.abspath(os.path.expanduser(str(video_dir)))
            else:
                video_dir = os.path.abspath(os.path.join(resolve_repo_path(output_root), video_id))
            units.append({"video_id": video_id, "video_path": video_path, "video_dir": video_dir})
    return units


def _workspace_units_from_config(config: Any) -> List[Dict[str, str]]:
    phase = int(getattr(config, "phase", 0))
    total = int(getattr(config, "total", 1))
    if getattr(config, "video_dir", None):
        video_dir = os.path.abspath(os.path.expanduser(str(config.video_dir)))
        manifest_path = _workspace_path(video_dir, "manifest")
        manifest = {}
        if os.path.isfile(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        video_path = getattr(config, "video_path", None) or manifest.get("source_video_path") or ""
        video_id = _safe_video_id(getattr(config, "video_id", None) or manifest.get("video_id") or os.path.basename(video_dir))
        units = [{"video_id": video_id, "video_path": os.path.abspath(os.path.expanduser(str(video_path))) if video_path else "", "video_dir": video_dir}]
    else:
        if not getattr(config, "pipeline_input_jsonl", None):
            raise ValueError("Set --pipeline_input_jsonl for batch mode or --video_dir for standalone workspace mode.")
        units = _load_task_units(config.pipeline_input_jsonl, getattr(config, "output_root", "outputs"))
    st, en = _shard_range(len(units), phase, total)
    return units[st:en]


def _ensure_workspace_manifest(unit: Dict[str, str]) -> Dict[str, Any]:
    video_dir = unit["video_dir"]
    os.makedirs(_workspace_path(video_dir, "training_dir"), exist_ok=True)
    os.makedirs(_workspace_path(video_dir, "first_frame_dir"), exist_ok=True)
    manifest_path = _workspace_path(video_dir, "manifest")
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    else:
        os.makedirs(video_dir, exist_ok=True)
        manifest = {
            "schema_version": 1,
            "video_id": unit.get("video_id"),
            "video_dir": video_dir,
            "source_video_path": unit.get("video_path"),
            "stages": {},
        }
    manifest.setdefault("video_id", unit.get("video_id"))
    manifest.setdefault("video_dir", video_dir)
    if unit.get("video_path"):
        manifest.setdefault("source_video_path", unit.get("video_path"))
    manifest.setdefault("stages", {})
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest


def _update_workspace_stage(unit: Dict[str, str], stage: str, status: str, outputs: Optional[Dict[str, Any]] = None) -> None:
    manifest = _ensure_workspace_manifest(unit)
    stage_data = {"status": status}
    if outputs:
        stage_data.update(outputs)
    manifest.setdefault("stages", {})[stage] = stage_data
    with open(_workspace_path(unit["video_dir"], "manifest"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


SHOT_RE = re.compile(r"^(?P<prefix>.+)_shot_(?P<number>\d+)$")
EMOTIONS_8 = ("angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise")
BODY_LABEL_BUCKETS = ("front", "back", "left", "right")
BODY_POSE_LABEL_GRID = ("left", "right", "front")
BODY_PART_BUCKETS = ("full_body", "half_body", "Head_Close_up")
BODY_POSE_BUCKETS = BODY_LABEL_BUCKETS + BODY_PART_BUCKETS
BODY_POSE_GRID_BUCKETS = tuple(
    f"{label}__{body_part}"
    for label in BODY_POSE_LABEL_GRID
    for body_part in BODY_PART_BUCKETS
)
BODY_POSE_PRIORITY_BUCKETS = tuple(f"{label}__full_body" for label in BODY_POSE_LABEL_GRID)
PERSON_CLUSTERS_SUBDIR = "person_clusters"


class TrainingPairGenerator:
    def __init__(self, config: TrainingPairsConfig):
        self.config = config
        self.feature_cache: Dict[str, Optional[np.ndarray]] = {}
        self.dino_feature_cache: Dict[str, Optional[np.ndarray]] = {}
        self.first_frame_cache: Dict[str, Optional[str]] = {}

    # @staticmethod
    # def _load_json(path: str) -> dict:
    #     with open(path, "r", encoding="utf-8") as f:
    #         data = json.load(f)
    #     return data if isinstance(data, dict) else {}
    
    @staticmethod
    def _load_json(path: str) -> dict:
        # 过滤 JSON 文件里真实存在的非法控制字符，避免 json.load 报：
        # JSONDecodeError: Invalid control character
        ctrl_re = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        text = ctrl_re.sub("", text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to load JSON: {path}")
            print(f"[ERROR] {e}")
            raise

        return data if isinstance(data, dict) else {}

    @staticmethod
    def _safe_text(value: str) -> str:
        text = str(value or "").strip()
        return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text) or "unknown"

    @staticmethod
    def _parse_shot(shot_key: str) -> Tuple[str, Optional[int]]:
        match = SHOT_RE.match(str(shot_key or ""))
        if not match:
            return str(shot_key or ""), None
        return match.group("prefix"), int(match.group("number"))

    @staticmethod
    def _unique_key(item: dict) -> Tuple[str, str, int]:
        return (str(item.get("shot_key")), str(item.get("obj_id")), int(item.get("frame_idx", -1)))

    def _passes_shot_gap(self, candidate: dict, selected: List[dict]) -> bool:
        gap = int(self.config.min_same_prefix_shot_gap)
        if gap < 0:
            return True
        candidate_prefix = candidate.get("video_prefix")
        candidate_shot = candidate.get("shot_no")
        if candidate_prefix is None or candidate_shot is None:
            return True
        for item in selected:
            if item.get("video_prefix") != candidate_prefix:
                continue
            item_shot = item.get("shot_no")
            if item_shot is None:
                continue
            if abs(int(candidate_shot) - int(item_shot)) <= gap:
                return False
        return True

    def _rng(self, *parts) -> random.Random:
        text = "::".join(str(part) for part in (self.config.seed,) + parts)
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        return random.Random(int(digest[:12], 16))

    def _video_part_from_cluster_dir(self) -> Tuple[Optional[str], Optional[str]]:
        parts = list(os.path.abspath(self.config.person_clusters_dir).split(os.sep))
        if "identity_matching" not in parts:
            return None, None
        idx = len(parts) - 1 - parts[::-1].index("identity_matching")
        tail = parts[idx + 1 :]
        if len(tail) >= 3 and tail[-1] == PERSON_CLUSTERS_SUBDIR:
            return tail[0], tail[1]
        return None, None

    def _feature_path(self, item: dict) -> str:
        if self.config.one_shot_process_dir:
            video, part = self._video_part_from_cluster_dir()
            if video and part:
                return os.path.join(
                    self.config.one_shot_process_dir,
                    video,
                    part,
                    str(item["shot_key"]),
                    f"id_{item['obj_id']}",
                    "features",
                    "face_feature",
                    f"{int(item['frame_idx'])}.npy",
                )

        cluster_dir = os.path.abspath(self.config.person_clusters_dir)
        outputs_root = os.path.dirname(os.path.dirname(cluster_dir))
        return os.path.join(
            outputs_root,
            "one_shot_process",
            str(item["shot_key"]),
            f"id_{item['obj_id']}",
            "features",
            "face_feature",
            f"{int(item['frame_idx'])}.npy",
        )

    def _dino_feature_path(self, item: dict) -> str:
        if self.config.one_shot_process_dir:
            video, part = self._video_part_from_cluster_dir()
            if video and part:
                return os.path.join(
                    self.config.one_shot_process_dir,
                    video,
                    part,
                    str(item["shot_key"]),
                    f"id_{item['obj_id']}",
                    "features",
                    "dino_feature",
                    f"{int(item['frame_idx'])}.npy",
                )

        cluster_dir = os.path.abspath(self.config.person_clusters_dir)
        outputs_root = os.path.dirname(os.path.dirname(cluster_dir))
        return os.path.join(
            outputs_root,
            "one_shot_process",
            str(item["shot_key"]),
            f"id_{item['obj_id']}",
            "features",
            "dino_feature",
            f"{int(item['frame_idx'])}.npy",
        )

    def _load_feature(self, path: str) -> Optional[np.ndarray]:
        path = resolve_repo_path(path)
        if path in self.feature_cache:
            return self.feature_cache[path]
        try:
            feat = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(feat))
            if feat.size == 0 or norm <= 1e-12:
                self.feature_cache[path] = None
            else:
                self.feature_cache[path] = feat / norm
        except Exception:
            self.feature_cache[path] = None
        return self.feature_cache[path]

    def _load_dino_feature(self, path: str) -> Optional[np.ndarray]:
        path = resolve_repo_path(path)
        if path in self.dino_feature_cache:
            return self.dino_feature_cache[path]
        try:
            feat = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(feat))
            if feat.size == 0 or norm <= 1e-12:
                self.dino_feature_cache[path] = None
            else:
                self.dino_feature_cache[path] = feat / norm
        except Exception:
            self.dino_feature_cache[path] = None
        return self.dino_feature_cache[path]

    @staticmethod
    def _cosine(a: dict, b: dict) -> float:
        return float(np.dot(a["feature"], b["feature"]))

    def _similarity(self, a: dict, b: dict) -> float:
        matrix = getattr(self, "current_similarity_matrix", None)
        if matrix is not None and a.get("matrix_idx") is not None and b.get("matrix_idx") is not None:
            return float(matrix[int(a["matrix_idx"]), int(b["matrix_idx"])])
        if a.get("feature") is None or b.get("feature") is None:
            return 1.0
        return self._cosine(a, b)

    @staticmethod
    def _dino_cosine(a: dict, b: dict) -> float:
        return float(np.dot(a["dino_feature"], b["dino_feature"]))

    def _dino_similarity(self, a: dict, b: dict) -> float:
        if a.get("dino_feature") is None or b.get("dino_feature") is None:
            return 1.0
        return self._dino_cosine(a, b)

    def _selection_similarity(self, a: dict, b: dict) -> float:
        if bool(getattr(self.config, "enable_dino_ref_diversity", False)):
            return self._dino_similarity(a, b)
        return self._similarity(a, b)

    def _combo_pairwise_stats(self, selected: List[dict]) -> Tuple[Optional[float], Optional[float], Optional[dict]]:
        selected = [item for item in selected if item.get("feature") is not None]
        if len(selected) < 2:
            return None, None, None
        values = []
        max_pair = None
        max_value = None
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                value = self._similarity(selected[i], selected[j])
                values.append(value)
                if max_value is None or value > max_value:
                    max_value = value
                    max_pair = {
                        "similarity": float(value),
                        "left": self._pair_item_summary(selected[i]),
                        "right": self._pair_item_summary(selected[j]),
                    }
        return float(np.mean(values)), float(max_value), max_pair

    def _combo_dino_pairwise_stats(self, selected: List[dict]) -> Tuple[Optional[float], Optional[float], Optional[dict]]:
        selected = [item for item in selected if item.get("dino_feature") is not None]
        if len(selected) < 2:
            return None, None, None
        values = []
        max_pair = None
        max_value = None
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                value = self._dino_similarity(selected[i], selected[j])
                values.append(value)
                if max_value is None or value > max_value:
                    max_value = value
                    max_pair = {
                        "similarity": float(value),
                        "left": self._pair_item_summary(selected[i]),
                        "right": self._pair_item_summary(selected[j]),
                    }
        return float(np.mean(values)), float(max_value), max_pair

    def _combo_cos_stats(self, selected: List[dict]) -> Tuple[Optional[float], Optional[float]]:
        mean_value, max_value, _ = self._combo_pairwise_stats(selected)
        return mean_value, max_value

    @staticmethod
    def _pair_item_summary(item: dict) -> dict:
        return {
            "path": item.get("path"),
            "shot_key": item.get("shot_key"),
            "obj_id": item.get("obj_id"),
            "frame_idx": item.get("frame_idx"),
            "source_shot_path": item.get("source_shot_path"),
            "bucket": item.get("bucket"),
            "bucket_source": item.get("bucket_source"),
        }

    def _image_path(self, attrs: dict) -> Optional[str]:
        related = attrs.get("related_images") or {}
        path = related.get(self.config.ref_image_type) or related.get(self.config.ref_fallback_image_type)
        if not path:
            path = attrs.get("image_path")
        return path if path and os.path.isfile(resolve_repo_path(path)) else None

    @staticmethod
    def _full_image_path(attrs: dict) -> Optional[str]:
        related = attrs.get("related_images") or {}
        path = related.get("full_orig") or related.get("full_white")
        return path if path and os.path.isfile(resolve_repo_path(path)) else None

    @staticmethod
    def _exists(path: Optional[str]) -> Optional[str]:
        return path if path and os.path.isfile(resolve_repo_path(path)) else None

    def _white_image_path(self, attrs: dict) -> Optional[str]:
        related = attrs.get("related_images") or {}
        return self._exists(related.get(self.config.ref_fallback_image_type) or related.get("face_white"))

    def _full_white_image_path(self, attrs: dict) -> Optional[str]:
        related = attrs.get("related_images") or {}
        return self._exists(related.get("full_white"))

    @staticmethod
    def _final_expression(expression: dict) -> str:
        if not isinstance(expression, dict):
            return ""
        candidates = [
            expression.get("final_expression"),
            (expression.get("vlm_check") or {}).get("final_expression"),
            (expression.get("vlm_check") or {}).get("correct_emotion"),
            expression.get("dominant"),
        ]
        for value in candidates:
            value = str(value or "").strip().lower()
            if value in EMOTIONS_8:
                return value
        return ""

    def _build_candidates(self, person_index: dict) -> Tuple[List[dict], dict]:
        images = person_index.get("images") or {}
        primary = images.get(self.config.ref_image_type) or images.get("face_orig") or {}
        candidates = []
        skipped = Counter()
        entries_by_frame = {}
        for image_type, entries in images.items():
            if not isinstance(entries, dict):
                continue
            for entry in entries.values():
                if not isinstance(entry, dict):
                    continue
                key = (
                    image_type,
                    str(entry.get("shot_key") or ""),
                    str(entry.get("obj_id")),
                    str(entry.get("frame_idx")),
                )
                entries_by_frame[key] = entry

        def related_entry(attrs: dict, image_type: str) -> Optional[dict]:
            related = attrs.get("related_images") or {}
            path = related.get(image_type)
            entry = (images.get(image_type) or {}).get(path) if path else None
            if not isinstance(entry, dict):
                key = (
                    image_type,
                    str(attrs.get("shot_key") or ""),
                    str(attrs.get("obj_id")),
                    str(attrs.get("frame_idx")),
                )
                entry = entries_by_frame.get(key)
            return entry if isinstance(entry, dict) else None

        def related_quality_label(attrs: dict, image_type: str) -> bool:
            if bool(getattr(self.config, "ignore_ref_quality", False)):
                return True
            entry = related_entry(attrs, image_type)
            if not isinstance(entry, dict):
                return True
            if bool(getattr(self.config, "ignore_mask_hole_ref_quality", False)):
                quality = entry.get("quality") if isinstance(entry.get("quality"), dict) else {}
                checked_any = False
                for key, item in quality.items():
                    if key == "mask_hole":
                        continue
                    if not isinstance(item, dict):
                        continue
                    checked_any = True
                    if item.get("passed") is False:
                        return False
                return True if checked_any or "mask_hole" in quality else bool(entry.get("quality_label", True))
            return bool(entry.get("quality_label", True))

        def related_quality(attrs: dict, image_type: str) -> dict:
            entry = related_entry(attrs, image_type)
            if not isinstance(entry, dict):
                return {}
            return {
                "quality_label": entry.get("quality_label", True),
                "quality": entry.get("quality") if isinstance(entry.get("quality"), dict) else {},
            }

        for attrs in primary.values():
            if not isinstance(attrs, dict):
                continue
            image_path = self._image_path(attrs)
            if not image_path:
                skipped["missing_image"] += 1
                continue

            shot_key = str(attrs.get("shot_key") or "")
            obj_id = str(attrs.get("obj_id"))
            frame_idx = attrs.get("frame_idx")
            if not shot_key or frame_idx is None or obj_id in ("", "None"):
                skipped["missing_key"] += 1
                continue

            feature_path = self._feature_path({"shot_key": shot_key, "obj_id": obj_id, "frame_idx": frame_idx})
            feature = self._load_feature(feature_path)
            if feature is None:
                skipped["missing_feature"] += 1
            dino_feature_path = self._dino_feature_path({"shot_key": shot_key, "obj_id": obj_id, "frame_idx": frame_idx})
            dino_feature = self._load_dino_feature(dino_feature_path)
            if bool(getattr(self.config, "enable_dino_ref_diversity", False)) and dino_feature is None:
                skipped["missing_dino_feature"] += 1

            prefix, shot_no = self._parse_shot(shot_key)
            pose = attrs.get("pose") or {}
            expression = attrs.get("expression") or {}
            body_pose = attrs.get("body_pose") or {}
            quality_by_image_type = {
                "face_orig": related_quality_label(attrs, "face_orig"),
                "face_white": related_quality_label(attrs, "face_white"),
                "full_orig": related_quality_label(attrs, "full_orig"),
                "full_white": related_quality_label(attrs, "full_white"),
            }
            quality_detail_by_image_type = {
                "face_orig": related_quality(attrs, "face_orig"),
                "face_white": related_quality(attrs, "face_white"),
                "full_orig": related_quality(attrs, "full_orig"),
                "full_white": related_quality(attrs, "full_white"),
            }
            face_quality_label = quality_by_image_type["face_orig"] and quality_by_image_type["face_white"]
            full_quality_label = quality_by_image_type["full_orig"] and quality_by_image_type["full_white"]
            if not face_quality_label:
                skipped["low_quality_face_ref"] += 1
            if not full_quality_label:
                skipped["low_quality_full_ref"] += 1
            emotion = self._final_expression(expression)
            scores = expression.get("scores") or {}
            try:
                emotion_score = float(scores.get(emotion, 0.0)) if emotion else 0.0
            except Exception:
                emotion_score = 0.0

            candidates.append({
                "person_id": attrs.get("person_id") or person_index.get("person_id"),
                "uid": attrs.get("uid"),
                "shot_key": shot_key,
                "video_prefix": prefix,
                "shot_no": shot_no,
                "obj_id": obj_id,
                "frame_idx": int(frame_idx),
                "source_shot_frame_idx": attrs.get("source_shot_frame_idx"),
                "source_shot_path": attrs.get("source_shot_path"),
                "path": image_path,
                "white_path": self._white_image_path(attrs),
                "full_path": self._full_image_path(attrs),
                "full_white_path": self._full_white_image_path(attrs),
                "feature_path": to_repo_relative_path(feature_path),
                "feature": feature,
                "dino_feature_path": to_repo_relative_path(dino_feature_path),
                "dino_feature": dino_feature,
                "pitch": float((pose or {}).get("pitch", 0.0)),
                "yaw": float((pose or {}).get("yaw", 0.0)),
                "roll": float((pose or {}).get("roll", 0.0)),
                "emotion": emotion,
                "emotion_score": emotion_score,
                "expression_status": expression.get("status"),
                "body_pose": body_pose if isinstance(body_pose, dict) else {},
                "face_quality_label": face_quality_label,
                "full_quality_label": full_quality_label,
                "quality_label": attrs.get("quality_label", True),
                "quality": attrs.get("quality") if isinstance(attrs.get("quality"), dict) else {},
                "quality_by_image_type": quality_by_image_type,
                "quality_detail_by_image_type": quality_detail_by_image_type,
            })

        return candidates, dict(skipped)

    def _similarity_matrix_path(self, person_id: str) -> str:
        out_dir = os.path.dirname(os.path.abspath(self.config.output_jsonl))
        return os.path.join(out_dir, f"{self._safe_text(person_id)}_candidate_similarity.npz")

    def _build_or_load_similarity_matrix(self, candidates: List[dict], person_id: str) -> dict:
        matrix_path = self._similarity_matrix_path(person_id)
        keys = [self._unique_key(item) for item in candidates]
        key_strings = np.asarray([f"{key[0]}::id_{key[1]}::frame_{key[2]}" for key in keys], dtype=object)
        feature_paths = np.asarray([str(item.get("feature_path") or "") for item in candidates], dtype=object)

        if (
            os.path.isfile(matrix_path)
            and not bool(getattr(self.config, "overwrite_similarity_matrix", False))
        ):
            try:
                cached = np.load(matrix_path, allow_pickle=True)
                cached_keys = cached["keys"].astype(object).tolist()
                requested_keys = key_strings.tolist()
                matrix = np.asarray(cached["similarity"], dtype=np.float32)
                if cached_keys == requested_keys:
                    if matrix.shape == (len(candidates), len(candidates)):
                        for idx, item in enumerate(candidates):
                            item["matrix_idx"] = idx
                        self.current_similarity_matrix = matrix
                        return {
                            "path": to_repo_relative_path(matrix_path),
                            "status": "loaded",
                            "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
                        }
                elif len(cached_keys) == matrix.shape[0] == matrix.shape[1]:
                    cached_key_to_idx = {key: idx for idx, key in enumerate(cached_keys)}
                    if all(key in cached_key_to_idx for key in requested_keys):
                        subset_indices = np.asarray([cached_key_to_idx[key] for key in requested_keys], dtype=np.int64)
                        subset_matrix = matrix[np.ix_(subset_indices, subset_indices)].astype(np.float32, copy=False)
                        for idx, item in enumerate(candidates):
                            item["matrix_idx"] = idx
                        self.current_similarity_matrix = subset_matrix
                        return {
                            "path": to_repo_relative_path(matrix_path),
                            "status": "loaded_subset",
                            "shape": [int(subset_matrix.shape[0]), int(subset_matrix.shape[1])],
                            "cached_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
                        }
            except Exception as exc:
                tqdm.write(f"[TrainingPairs] Rebuild similarity matrix for {person_id}: failed to load cache ({exc})")

        feature_dim = 0
        for item in candidates:
            if item.get("feature") is not None:
                feature_dim = int(item["feature"].shape[0])
                break
        features = np.zeros((len(candidates), feature_dim), dtype=np.float32)
        valid = np.zeros((len(candidates),), dtype=bool)
        if feature_dim > 0:
            for idx, item in enumerate(candidates):
                feature = item.get("feature")
                if feature is not None and int(feature.shape[0]) == feature_dim:
                    features[idx] = feature
                    valid[idx] = True

        if feature_dim > 0 and len(candidates) > 0:
            matrix = np.matmul(features, features.T).astype(np.float32)
            invalid = ~valid
            if np.any(invalid):
                matrix[invalid, :] = 1.0
                matrix[:, invalid] = 1.0
            np.fill_diagonal(matrix, 1.0)
        else:
            matrix = np.ones((len(candidates), len(candidates)), dtype=np.float32)

        os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
        np.savez_compressed(
            matrix_path,
            similarity=matrix,
            keys=key_strings,
            feature_paths=feature_paths,
            valid_feature=valid,
        )
        for idx, item in enumerate(candidates):
            item["matrix_idx"] = idx
        self.current_similarity_matrix = matrix
        return {
            "path": to_repo_relative_path(matrix_path),
            "status": "built",
            "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "valid_feature_count": int(valid.sum()),
        }

    def _diverse_topk(self, items: List[dict], topk: int) -> List[dict]:
        if len(items) <= topk:
            return list(items)
        with_features = [item for item in items if item.get("feature") is not None and item.get("matrix_idx") is not None]
        if len(with_features) < 2:
            return sorted(items, key=lambda x: (x.get("source_shot_frame_idx") is None, x.get("source_shot_frame_idx") or x.get("frame_idx") or 0))[:topk]

        best_pair = None
        best_cos = float("inf")
        for i in range(len(with_features)):
            for j in range(i + 1, len(with_features)):
                cos = self._selection_similarity(with_features[i], with_features[j])
                if cos < best_cos:
                    best_cos = cos
                    best_pair = (i, j)

        selected = [with_features[best_pair[0]], with_features[best_pair[1]]]
        selected_keys = {self._unique_key(item) for item in selected}
        remaining = [item for item in with_features if self._unique_key(item) not in selected_keys]

        while len(selected) < topk and remaining:
            best_idx = 0
            best_score = float("inf")
            for idx, candidate in enumerate(remaining):
                max_cos = max(self._selection_similarity(candidate, item) for item in selected)
                if max_cos < best_score:
                    best_score = max_cos
                    best_idx = idx
            chosen = remaining.pop(best_idx)
            selected.append(chosen)
            selected_keys.add(self._unique_key(chosen))

        if len(selected) < topk:
            for item in items:
                if self._unique_key(item) not in selected_keys:
                    selected.append(item)
                    selected_keys.add(self._unique_key(item))
                if len(selected) >= topk:
                    break
        return selected[:topk]

    def _balanced_allocation(self, buckets: List[str], count: int, rng: random.Random) -> Dict[str, int]:
        if not buckets or count <= 0:
            return {}
        buckets = list(buckets)
        if len(buckets) > count:
            buckets = rng.sample(buckets, count)
        base = count // len(buckets)
        remainder = count % len(buckets)
        allocation = {bucket: base for bucket in buckets}
        if remainder:
            for bucket in rng.sample(buckets, remainder):
                allocation[bucket] += 1
        return allocation

    def _sample_from_bucket_candidates(
        self,
        buckets: Dict[str, List[dict]],
        count: int,
        person_id: str,
        ref_type: str,
        sample_key: str,
        priority_buckets: Optional[List[str]] = None,
        fallback_pool: Optional[List[dict]] = None,
    ) -> Tuple[List[dict], dict]:
        rng = self._rng(person_id, ref_type, sample_key)
        available = sorted([bucket for bucket, items in buckets.items() if items])
        priority_buckets = [bucket for bucket in (priority_buckets or []) if bucket in available]
        secondary_buckets = [bucket for bucket in available if bucket not in set(priority_buckets)]
        allocation = {}
        if priority_buckets:
            priority_count = min(count, len(priority_buckets))
            allocation.update(self._balanced_allocation(priority_buckets, priority_count, rng))
            remaining_count = count - sum(allocation.values())
            if remaining_count > 0:
                allocation.update(self._balanced_allocation(secondary_buckets, remaining_count, rng))
        else:
            allocation = self._balanced_allocation(available, count, rng)
        topk = max(1, int(self.config.bucket_candidate_topk))

        bucket_candidates = {
            bucket: self._diverse_topk(items, topk)
            for bucket, items in buckets.items()
            if items
        }

        selected = []
        selected_keys = set()

        def best_from_pool(pool: List[dict]) -> Optional[dict]:
            candidates = list(pool)
            rng.shuffle(candidates)
            best_candidate = None
            best_score = float("inf")
            for candidate in candidates:
                key = self._unique_key(candidate)
                if key in selected_keys:
                    continue
                if not self._passes_shot_gap(candidate, selected):
                    continue
                score = max((self._selection_similarity(candidate, item) for item in selected), default=-1.0)
                if score < best_score:
                    best_score = score
                    best_candidate = candidate
            if best_candidate is None:
                return None
            selected_keys.add(self._unique_key(best_candidate))
            return dict(best_candidate)

        def bucket_pool(bucket_names: List[str]) -> List[dict]:
            pool = []
            for bucket in bucket_names:
                for item in bucket_candidates.get(bucket, []):
                    pool.append(dict(item, bucket=item.get("bucket") or bucket))
            return pool

        def take_from_bucket(bucket: str) -> Optional[dict]:
            return best_from_pool(bucket_pool([bucket]))

        deficits = 0
        for bucket, quota in allocation.items():
            for _ in range(quota):
                item = take_from_bucket(bucket)
                if item is None:
                    deficits += 1
                else:
                    selected.append(item)

        allocated_buckets = list(allocation.keys())
        secondary_added = 0
        while deficits > 0:
            remaining_buckets = [bucket for bucket in available if bucket not in set(allocated_buckets)]
            item = best_from_pool(bucket_pool(remaining_buckets))
            if item is None:
                break
            selected.append(item)
            deficits -= 1
            secondary_added += 1

        global_added = 0
        while len(selected) < count:
            item = best_from_pool(bucket_pool(available))
            if item is None:
                break
            selected.append(item)
            global_added += 1

        meta = {
            "available_buckets": {bucket: len(items) for bucket, items in buckets.items()},
            "topk_bucket_sizes": {bucket: len(items) for bucket, items in bucket_candidates.items()},
            "bucket_allocation": allocation,
            "priority_buckets": priority_buckets,
            "bucket_candidate_topk": topk,
            "requested_count": count,
            "selected_count": len(selected),
            "fallback_added_without_shot_gap": 0,
            "strict_secondary_bucket_added": secondary_added,
            "strict_global_bucket_added": global_added,
            "strict_shot_gap_failed": len(selected) < count,
            "fallback_pool_size": len(fallback_pool or []),
            "fallback_pool_used": False,
            "dedup_key": "shot_key::obj_id::frame_idx",
            "min_same_prefix_shot_gap": int(self.config.min_same_prefix_shot_gap),
            "sample_key": sample_key,
        }
        return selected, meta

    def _angle_buckets(self, candidates: List[dict]) -> Dict[str, List[dict]]:
        buckets = defaultdict(list)
        for item in candidates:
            if item.get("face_quality_label") is False:
                continue
            yaw = float(item["yaw"])
            pitch = float(item["pitch"])
            if abs(yaw) <= 20.0 and abs(pitch - 30.0) <= 10.0:
                buckets["front"].append(dict(item, bucket="front"))
            up_min = float(getattr(self.config, "angle_front_up_min_pitch", -10.0))
            up_max = float(getattr(self.config, "angle_front_up_max_pitch", 20.0))
            down_min = float(getattr(self.config, "angle_front_down_min_pitch", 40.0))
            down_max = float(getattr(self.config, "angle_front_down_max_pitch", 70.0))
            if up_min <= pitch < up_max:
                buckets["front_up"].append(dict(item, bucket="front_up"))
            if down_min < pitch <= down_max:
                buckets["front_down"].append(dict(item, bucket="front_down"))
            if yaw <= -30.0:
                buckets["left"].append(dict(item, bucket="left"))
            if yaw >= 30.0:
                buckets["right"].append(dict(item, bucket="right"))
        return dict(buckets)

    def _emotion_buckets(self, candidates: List[dict]) -> Dict[str, List[dict]]:
        buckets = defaultdict(list)
        for item in candidates:
            if item.get("face_quality_label") is False:
                continue
            emotion = str(item.get("emotion") or "").lower()
            if emotion not in EMOTIONS_8:
                continue
            buckets[emotion].append(dict(item, bucket=emotion))
        return dict(buckets)

    def _body_pose_buckets(self, candidates: List[dict]) -> Dict[str, List[dict]]:
        buckets = defaultdict(list)
        for item in candidates:
            full_path = item.get("full_path")
            full_white_path = item.get("full_white_path")
            if full_path:
                if item.get("full_quality_label") is False:
                    continue
            else:
                if item.get("face_quality_label") is False:
                    continue
                full_path = item.get("path")
                full_white_path = item.get("white_path")
            body_pose = item.get("body_pose") or {}
            if body_pose.get("status") and body_pose.get("status") != "success":
                continue
            label = str(body_pose.get("label") or "").strip()
            body_part = str(body_pose.get("body_part") or "").strip()
            if label in BODY_POSE_LABEL_GRID and body_part in BODY_PART_BUCKETS:
                bucket = f"{label}__{body_part}"
                buckets[bucket].append(dict(
                    item,
                    bucket=bucket,
                    bucket_source="body_label_body_part",
                    body_label=label,
                    body_part_bucket=body_part,
                    path=full_path,
                    white_path=full_white_path,
                ))
        return dict(buckets)

    @staticmethod
    def _body_pose_fallback_pool(candidates: List[dict]) -> List[dict]:
        pool = []
        for item in candidates:
            full_path = item.get("full_path")
            full_white_path = item.get("full_white_path")
            if full_path:
                if item.get("full_quality_label") is False:
                    continue
            else:
                if item.get("face_quality_label") is False:
                    continue
                full_path = item.get("path")
                full_white_path = item.get("white_path")
            if not full_path:
                continue
            pool.append(dict(
                item,
                bucket="matrix_fallback",
                bucket_source="matrix_fallback",
                path=full_path,
                white_path=full_white_path,
            ))
        return pool

    @staticmethod
    def _strip_feature(item: dict) -> dict:
        body_pose = item.get("body_pose") or {}
        return {
            "path": item.get("path"),
            "white_path": item.get("white_path"),
            "uid": item.get("uid"),
            "shot_key": item.get("shot_key"),
            "video_prefix": item.get("video_prefix"),
            "shot_no": item.get("shot_no"),
            "obj_id": item.get("obj_id"),
            "frame_idx": item.get("frame_idx"),
            "feature_path": item.get("feature_path"),
            "dino_feature_path": item.get("dino_feature_path"),
            "bucket": item.get("bucket"),
            "bucket_source": item.get("bucket_source"),
            "body_label": item.get("body_label"),
            "body_part_bucket": item.get("body_part_bucket"),
            "yaw": round(float(item.get("yaw", 0.0)), 4),
            "pitch": round(float(item.get("pitch", 0.0)), 4),
            "roll": round(float(item.get("roll", 0.0)), 4),
            "emotion": item.get("emotion"),
            "emotion_score": round(float(item.get("emotion_score", 0.0)), 6),
            "body_pose": {
                "label": body_pose.get("label"),
                "body_part": body_pose.get("body_part"),
                "yaw_deg": body_pose.get("yaw_deg"),
                "status": body_pose.get("status"),
            },
            "quality_label": item.get("quality_label"),
            "quality": item.get("quality") or {},
            "face_quality_label": item.get("face_quality_label"),
            "full_quality_label": item.get("full_quality_label"),
            "quality_by_image_type": item.get("quality_by_image_type") or {},
            "quality_detail_by_image_type": item.get("quality_detail_by_image_type") or {},
        }

    def _first_frame_path(self, video_path: str) -> Optional[str]:
        if not video_path:
            return None
        resolved_video = resolve_repo_path(video_path)
        if resolved_video in self.first_frame_cache:
            return self.first_frame_cache[resolved_video]
        stem = self._safe_text(os.path.splitext(os.path.basename(video_path))[0])
        out_path = os.path.join(self.config.first_frame_dir, f"{stem}.jpg")
        if os.path.isfile(out_path) and not bool(self.config.overwrite_first_frames):
            self.first_frame_cache[resolved_video] = to_repo_relative_path(out_path)
            return self.first_frame_cache[resolved_video]
        cap = cv2.VideoCapture(resolved_video)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            self.first_frame_cache[resolved_video] = None
            return None
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if not cv2.imwrite(out_path, frame):
            self.first_frame_cache[resolved_video] = None
            return None
        self.first_frame_cache[resolved_video] = to_repo_relative_path(out_path)
        return self.first_frame_cache[resolved_video]

    def _target_videos(self, candidates: List[dict]) -> List[dict]:
        by_video = {}
        for item in candidates:
            video = item.get("source_shot_path")
            if not video:
                continue
            current = by_video.get(video)
            idx = item.get("source_shot_frame_idx")
            if current is None or (idx is not None and idx < current.get("source_shot_frame_idx", 10**12)):
                by_video[video] = item
        return [by_video[key] for key in sorted(by_video.keys())]

    def _person_dirs(self) -> List[str]:
        person_dirs = []
        wanted = set(self.config.person_ids or [])
        for name in sorted(os.listdir(self.config.person_clusters_dir)):
            path = os.path.join(self.config.person_clusters_dir, name)
            if not os.path.isdir(path) or not name.startswith("person_"):
                continue
            if wanted and name not in wanted:
                continue
            person_dirs.append(path)
        return person_dirs

    def _select_refs(self, candidates: List[dict], person_id: str, sample_key: str) -> dict:
        angle_refs, angle_meta = self._sample_from_bucket_candidates(
            self._angle_buckets(candidates),
            int(self.config.angle_ref_count),
            person_id,
            "angle",
            sample_key,
            fallback_pool=candidates,
        )
        emo_refs, emo_meta = self._sample_from_bucket_candidates(
            self._emotion_buckets(candidates),
            int(self.config.emo_ref_count),
            person_id,
            "emotion",
            sample_key,
            fallback_pool=candidates,
        )
        body_refs, body_meta = self._sample_from_bucket_candidates(
            self._body_pose_buckets(candidates),
            int(self.config.body_pose_ref_count),
            person_id,
            "body_pose",
            sample_key,
            priority_buckets=list(BODY_POSE_PRIORITY_BUCKETS),
            fallback_pool=self._body_pose_fallback_pool(candidates),
        )
        return {
            "angle": (angle_refs, angle_meta),
            "emotion": (emo_refs, emo_meta),
            "body_pose": (body_refs, body_meta),
        }

    @staticmethod
    def _exclude_target_shot_candidates(candidates: List[dict], target_shot_key: str) -> Tuple[List[dict], int]:
        if not target_shot_key:
            return list(candidates), 0
        filtered = [item for item in candidates if item.get("shot_key") != target_shot_key]
        return filtered, len(candidates) - len(filtered)

    @staticmethod
    def _ref_signature(angle_refs: List[dict], emo_refs: List[dict], body_refs: List[dict]) -> Tuple[Tuple[str, ...], ...]:
        def paths(items: List[dict]) -> Tuple[str, ...]:
            return tuple(sorted(str(item.get("path") or "") for item in items))

        return paths(angle_refs), paths(emo_refs), paths(body_refs)

    def _read_unit_list(self, path: str) -> List[Tuple[str, str, str, str]]:
        units = []
        with open(path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if not parts[0]:
                    tqdm.write(f"[TrainingPairs] Skip invalid unit_list line {line_number}: {line}")
                    continue
                person_clusters_dir = parts[0]
                video = parts[1] if len(parts) > 1 else ""
                part = parts[2] if len(parts) > 2 else ""
                uuid = parts[3] if len(parts) > 3 else ""
                units.append((person_clusters_dir, video, part, uuid))
        return units

    def _rejected_output_path(self) -> str:
        return self.config.rejected_jsonl or os.path.join(
            os.path.dirname(os.path.abspath(self.config.output_jsonl)),
            "rejected_pairs.jsonl",
        )

    @staticmethod
    def _reject_ref_meta(items: List[dict]) -> List[dict]:
        return [TrainingPairGenerator._strip_feature(item) for item in items]

    def _reject_record(
        self,
        person_id: str,
        target: dict,
        reason: str,
        detail: Optional[dict] = None,
        angle_refs: Optional[List[dict]] = None,
        emo_refs: Optional[List[dict]] = None,
        body_refs: Optional[List[dict]] = None,
    ) -> dict:
        return {
            "person_id": person_id,
            "source_uid": target.get("uid"),
            "source_shot_key": target.get("shot_key"),
            "source_frame_idx": target.get("frame_idx"),
            "target_video": to_repo_relative_path(target.get("source_shot_path")),
            "reject_reason": reason,
            "detail": detail or {},
            "angle_ref_meta": self._reject_ref_meta(angle_refs or []),
            "emo_ref_meta": self._reject_ref_meta(emo_refs or []),
            "body_pose_ref_meta": self._reject_ref_meta(body_refs or []),
        }

    def _unit_output_root(self, src_person_clusters: str, output_base_dir: str, input_base_dir: Optional[str]) -> str:
        src_person_clusters = resolve_repo_path(src_person_clusters)
        parent_dir = os.path.dirname(src_person_clusters)
        if os.path.basename(src_person_clusters) == PERSON_CLUSTERS_SUBDIR:
            src_root = parent_dir
        else:
            src_root = src_person_clusters
        if input_base_dir:
            try:
                rel_path = os.path.relpath(src_root, input_base_dir)
            except ValueError:
                rel_path = os.path.basename(src_root)
            if rel_path != "." and not rel_path.startswith(".."):
                return os.path.join(output_base_dir, rel_path)
        return os.path.join(output_base_dir, os.path.basename(src_root))

    def _run_unit_list(self) -> Tuple[int, int]:
        unit_list_file = resolve_repo_path(self.config.unit_list_file)
        input_base_dir = (
            resolve_repo_path(self.config.unit_list_input_base_dir)
            if self.config.unit_list_input_base_dir
            else None
        )
        output_base_dir = os.path.dirname(os.path.abspath(self.config.output_jsonl))
        stats_base_dir = os.path.dirname(os.path.abspath(self.config.stats_json))
        first_frame_base_dir = os.path.abspath(self.config.first_frame_dir)
        output_name = os.path.basename(self.config.output_jsonl)
        stats_name = os.path.basename(self.config.stats_json)

        units = self._read_unit_list(unit_list_file)
        if not units:
            tqdm.write(f"[TrainingPairs] No units found in unit_list_file: {unit_list_file}")
            return 0, 0

        total_persons = 0
        total_rows = 0
        progress = tqdm(units, desc="training_pair_roots", unit="root")
        for src_person_clusters, video, part, uuid in progress:
            src_person_clusters = resolve_repo_path(src_person_clusters)
            progress.set_postfix_str(f"{video}/{part}/{uuid}", refresh=False)
            if not os.path.isdir(src_person_clusters):
                tqdm.write(f"[TrainingPairs] Skip missing person_clusters: {src_person_clusters}")
                continue

            output_root = self._unit_output_root(src_person_clusters, output_base_dir, input_base_dir)
            try:
                stats_rel = os.path.relpath(output_root, output_base_dir)
            except ValueError:
                stats_rel = os.path.basename(output_root)
            stats_root = os.path.join(stats_base_dir, stats_rel) if stats_rel != "." else stats_base_dir
            first_frame_root = os.path.join(output_root, os.path.basename(first_frame_base_dir))

            os.makedirs(output_root, exist_ok=True)
            os.makedirs(stats_root, exist_ok=True)
            os.makedirs(first_frame_root, exist_ok=True)

            child_config = replace(
                self.config,
                person_clusters_dir=src_person_clusters,
                output_jsonl=os.path.join(output_root, output_name),
                rejected_jsonl=os.path.join(output_root, os.path.basename(self._rejected_output_path())),
                stats_json=os.path.join(stats_root, stats_name),
                first_frame_dir=first_frame_root,
                unit_list_file=None,
            )
            persons, rows = TrainingPairGenerator(child_config)._run_person_clusters()
            total_persons += persons
            total_rows += rows

        return total_persons, total_rows

    def run(self) -> Tuple[int, int]:
        if self.config.unit_list_file:
            return self._run_unit_list()
        workspace_mode = bool(getattr(self.config, "video_dir", None) or getattr(self.config, "pipeline_input_jsonl", None))
        if workspace_mode and not bool(getattr(self.config, "global_mode", False)):
            return self._run_workspace()
        return self._run_person_clusters()

    def _run_workspace(self) -> Tuple[int, int]:
        base_config = self.config
        units = _workspace_units_from_config(base_config)
        total_persons = 0
        total_rows = 0
        print(f"[generate_training_pairs] phase={base_config.phase}/{base_config.total} assigned={len(units)}")
        for unit in units:
            _update_workspace_stage(unit, "generate_training_pairs", "running")
            video_dir = unit["video_dir"]
            child_config = replace(
                base_config,
                person_clusters_dir=_workspace_person_clusters_dir(video_dir),
                output_jsonl=_workspace_path(video_dir, "pairs_jsonl"),
                rejected_jsonl=_workspace_path(video_dir, "rejected_pairs_jsonl"),
                stats_json=_workspace_path(video_dir, "training_stats_json"),
                first_frame_dir=_workspace_path(video_dir, "first_frame_dir"),
                unit_list_file=None,
                unit_list_input_base_dir=None,
            )
            self.config = child_config
            self.feature_cache = {}
            self.dino_feature_cache = {}
            self.first_frame_cache = {}
            persons, rows = self._run_person_clusters()
            total_persons += persons
            total_rows += rows
            _update_workspace_stage(unit, "generate_training_pairs", "complete", {
                "output_jsonl": os.path.relpath(child_config.output_jsonl, video_dir),
                "rejected_jsonl": os.path.relpath(child_config.rejected_jsonl, video_dir),
                "stats_json": os.path.relpath(child_config.stats_json, video_dir),
                "total_persons": persons,
                "rows_written": rows,
            })
        self.config = base_config
        return total_persons, total_rows

    def _run_person_clusters(self) -> Tuple[int, int]:
        person_dirs = self._person_dirs()
        rows_written = 0
        stats = {
            "config": asdict(self.config),
            "total_persons": len(person_dirs),
            "persons": {},
            "rows_written": 0,
            "first_frame_error": 0,
        }

        with open(self.config.output_jsonl, "w", encoding="utf-8") as fout, open(self._rejected_output_path(), "w", encoding="utf-8") as freject:
            for person_dir in tqdm(person_dirs, desc="Generating training pairs", unit="person"):
                person_id = os.path.basename(person_dir)
                index_path = os.path.join(person_dir, self.config.index_filename)
                if not os.path.isfile(index_path):
                    stats["persons"][person_id] = {"status": "missing_index"}
                    continue

                person_index = self._load_json(index_path)
                candidates, skipped = self._build_candidates(person_index)
                similarity_meta = self._build_or_load_similarity_matrix(candidates, person_id)
                target_items = self._target_videos(candidates)
                used_ref_signatures = set()
                per_target_stats = []

                stats["persons"][person_id] = {
                    "status": "ok",
                    "candidate_count": len(candidates),
                    "similarity_matrix": similarity_meta,
                    "target_video_count": len(target_items),
                    "skipped": skipped,
                    "per_target": per_target_stats,
                    "dino_diversity_rejected": 0,
                    "insufficient_refs_rejected": 0,
                    "rejected_jsonl": to_repo_relative_path(self._rejected_output_path()),
                }

                for target in target_items:
                    target_video = target.get("source_shot_path")
                    target_shot_key = target.get("shot_key")
                    ref_candidates, same_target_shot_filtered = self._exclude_target_shot_candidates(candidates, target_shot_key)
                    base_sample_key = f"{target.get('shot_key')}::{target.get('obj_id')}::{target.get('frame_idx')}::{target_video}"
                    refs = None
                    signature = None
                    duplicate_signature = False
                    sample_attempt = 0
                    max_attempts = 20
                    for attempt in range(max_attempts):
                        sample_key = base_sample_key if attempt == 0 else f"{base_sample_key}::retry_{attempt}"
                        refs = self._select_refs(ref_candidates, person_id, sample_key)
                        angle_refs, _ = refs["angle"]
                        emo_refs, _ = refs["emotion"]
                        body_refs, _ = refs["body_pose"]
                        signature = self._ref_signature(angle_refs, emo_refs, body_refs)
                        sample_attempt = attempt
                        if signature not in used_ref_signatures:
                            break
                    if signature in used_ref_signatures:
                        duplicate_signature = True
                    else:
                        used_ref_signatures.add(signature)

                    angle_refs, angle_selection_meta = refs["angle"]
                    emo_refs, emo_selection_meta = refs["emotion"]
                    body_refs, body_selection_meta = refs["body_pose"]

                    if (
                        len(angle_refs) < int(self.config.angle_ref_count) or
                        len(emo_refs) < int(self.config.emo_ref_count) or
                        len(body_refs) < int(self.config.body_pose_ref_count)
                    ):
                        stats["persons"][person_id]["insufficient_refs_rejected"] += 1
                        freject.write(json.dumps(self._reject_record(
                            person_id,
                            target,
                            "insufficient_refs",
                            {
                                "angle_selected": len(angle_refs),
                                "angle_required": int(self.config.angle_ref_count),
                                "emotion_selected": len(emo_refs),
                                "emotion_required": int(self.config.emo_ref_count),
                                "body_pose_selected": len(body_refs),
                                "body_pose_required": int(self.config.body_pose_ref_count),
                                "selection_meta": {
                                    "angle": angle_selection_meta,
                                    "emotion": emo_selection_meta,
                                    "body_pose": body_selection_meta,
                                },
                            },
                            angle_refs,
                            emo_refs,
                            body_refs,
                        ), ensure_ascii=False) + "\n")
                        continue

                    angle_mean, angle_max, angle_max_pair = self._combo_pairwise_stats(angle_refs)
                    emo_mean, emo_max, emo_max_pair = self._combo_pairwise_stats(emo_refs)
                    body_mean, body_max, body_max_pair = self._combo_pairwise_stats(body_refs)
                    angle_dino_mean, angle_dino_max, angle_dino_max_pair = self._combo_dino_pairwise_stats(angle_refs)
                    emo_dino_mean, emo_dino_max, emo_dino_max_pair = self._combo_dino_pairwise_stats(emo_refs)
                    body_dino_mean, body_dino_max, body_dino_max_pair = self._combo_dino_pairwise_stats(body_refs)
                    if bool(getattr(self.config, "enable_dino_ref_diversity", False)):
                        dino_threshold = float(getattr(self.config, "dino_max_pairwise_cosine", 0.95))
                        dino_groups = [angle_refs, emo_refs, body_refs]
                        missing_group_dino = any(
                            any(item.get("dino_feature") is None for item in group)
                            for group in dino_groups
                        )
                        dino_max_values = [angle_dino_max, emo_dino_max, body_dino_max]
                        if missing_group_dino or any(value is None or float(value) > dino_threshold for value in dino_max_values):
                            stats["persons"][person_id]["dino_diversity_rejected"] += 1
                            freject.write(json.dumps(self._reject_record(
                                person_id,
                                target,
                                "dino_diversity",
                                {
                                    "missing_group_dino": bool(missing_group_dino),
                                    "threshold": dino_threshold,
                                    "angle_dino_mean_pairwise_cosine": angle_dino_mean,
                                    "angle_dino_max_pairwise_cosine": angle_dino_max,
                                    "angle_dino_max_pairwise_cosine_pair": angle_dino_max_pair,
                                    "emo_dino_mean_pairwise_cosine": emo_dino_mean,
                                    "emo_dino_max_pairwise_cosine": emo_dino_max,
                                    "emo_dino_max_pairwise_cosine_pair": emo_dino_max_pair,
                                    "body_pose_dino_mean_pairwise_cosine": body_dino_mean,
                                    "body_pose_dino_max_pairwise_cosine": body_dino_max,
                                    "body_pose_dino_max_pairwise_cosine_pair": body_dino_max_pair,
                                },
                                angle_refs,
                                emo_refs,
                                body_refs,
                            ), ensure_ascii=False) + "\n")
                            continue

                    def _rel(p):
                        return to_repo_relative_path(p) if p else None

                    angle_paths = [_rel(item.get("path")) for item in angle_refs]
                    emo_paths = [_rel(item.get("path")) for item in emo_refs]
                    body_paths = [_rel(item.get("path")) for item in body_refs]
                    angle_paths_white = [_rel(item.get("white_path")) for item in angle_refs]
                    emo_paths_white = [_rel(item.get("white_path")) for item in emo_refs]
                    body_paths_white = [_rel(item.get("white_path")) for item in body_refs]
                    angle_meta = [self._strip_feature(item) for item in angle_refs]
                    emo_meta = [self._strip_feature(item) for item in emo_refs]
                    body_meta = [self._strip_feature(item) for item in body_refs]

                    first_frame = self._first_frame_path(target.get("source_shot_path"))
                    if not first_frame:
                        stats["first_frame_error"] += 1
                    row = {
                        "person_id": person_id,
                        "source_uid": target.get("uid"),
                        "source_shot_key": target.get("shot_key"),
                        "source_frame_idx": target.get("frame_idx"),
                        "first_frame": first_frame,
                        "target_video": to_repo_relative_path(target.get("source_shot_path")),
                        "angle_ref": angle_paths,
                        "emo_ref": emo_paths,
                        "body_pose_ref": body_paths,
                        "angle_ref_white": angle_paths_white,
                        "emo_ref_white": emo_paths_white,
                        "body_pose_ref_white": body_paths_white,
                        "angle_ref_meta": angle_meta,
                        "emo_ref_meta": emo_meta,
                        "body_pose_ref_meta": body_meta,
                        "selection_meta": {
                            "angle": angle_selection_meta,
                            "emotion": emo_selection_meta,
                            "body_pose": body_selection_meta,
                            "sample_attempt": sample_attempt,
                            "duplicate_ref_signature": duplicate_signature,
                            "target_shot_ref_filter": {
                                "target_shot_key": target_shot_key,
                                "target_video": to_repo_relative_path(target_video),
                                "filtered_count": same_target_shot_filtered,
                                "remaining_candidate_count": len(ref_candidates),
                            },
                        },
                        "selection_stats": {
                            "angle_mean_pairwise_cosine": angle_mean,
                            "angle_max_pairwise_cosine": angle_max,
                            "angle_max_pairwise_cosine_pair": angle_max_pair,
                            "angle_fallback_added_without_shot_gap": angle_selection_meta.get("fallback_added_without_shot_gap"),
                            "emo_mean_pairwise_cosine": emo_mean,
                            "emo_max_pairwise_cosine": emo_max,
                            "emo_max_pairwise_cosine_pair": emo_max_pair,
                            "emo_fallback_added_without_shot_gap": emo_selection_meta.get("fallback_added_without_shot_gap"),
                            "body_pose_mean_pairwise_cosine": body_mean,
                            "body_pose_max_pairwise_cosine": body_max,
                            "body_pose_max_pairwise_cosine_pair": body_max_pair,
                            "body_pose_fallback_added_without_shot_gap": body_selection_meta.get("fallback_added_without_shot_gap"),
                            "angle_dino_mean_pairwise_cosine": angle_dino_mean,
                            "angle_dino_max_pairwise_cosine": angle_dino_max,
                            "angle_dino_max_pairwise_cosine_pair": angle_dino_max_pair,
                            "emo_dino_mean_pairwise_cosine": emo_dino_mean,
                            "emo_dino_max_pairwise_cosine": emo_dino_max,
                            "emo_dino_max_pairwise_cosine_pair": emo_dino_max_pair,
                            "body_pose_dino_mean_pairwise_cosine": body_dino_mean,
                            "body_pose_dino_max_pairwise_cosine": body_dino_max,
                            "body_pose_dino_max_pairwise_cosine_pair": body_dino_max_pair,
                            "dino_max_pairwise_cosine_threshold": float(getattr(self.config, "dino_max_pairwise_cosine", 0.95)),
                        },
                    }
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows_written += 1
                    per_target_stats.append({
                        "target_video": to_repo_relative_path(target_video),
                        "source_shot_key": target.get("shot_key"),
                        "sample_attempt": sample_attempt,
                        "duplicate_ref_signature": duplicate_signature,
                        "target_shot_ref_filter": {
                            "target_shot_key": target_shot_key,
                            "target_video": to_repo_relative_path(target_video),
                            "filtered_count": same_target_shot_filtered,
                            "remaining_candidate_count": len(ref_candidates),
                        },
                        "angle": angle_selection_meta,
                        "emotion": emo_selection_meta,
                        "body_pose": body_selection_meta,
                    })

        stats["rows_written"] = rows_written
        with open(self.config.stats_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        return len(person_dirs), rows_written

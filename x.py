import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from .config import TrainingPairsConfig
from .path_utils import resolve_repo_path, to_repo_relative_path


SHOT_RE = re.compile(r"^(?P<prefix>.+)_shot_(?P<number>\d+)$")
EMOTIONS_8 = ("angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise")
BODY_LABEL_BUCKETS = ("front", "back", "left", "right")
BODY_PART_BUCKETS = ("full_body", "half_body", "Head_Close_up")
BODY_POSE_BUCKETS = BODY_LABEL_BUCKETS + BODY_PART_BUCKETS


class TrainingPairGenerator:
    def __init__(self, config: TrainingPairsConfig):
        self.config = config
        self.feature_cache: Dict[str, Optional[np.ndarray]] = {}
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

    def _feature_path(self, item: dict) -> str:
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

    @staticmethod
    def _cosine(a: dict, b: dict) -> float:
        return float(np.dot(a["feature"], b["feature"]))

    def _combo_cos_stats(self, selected: List[dict]) -> Tuple[Optional[float], Optional[float]]:
        selected = [item for item in selected if item.get("feature") is not None]
        if len(selected) < 2:
            return None, None
        values = [
            self._cosine(selected[i], selected[j])
            for i in range(len(selected))
            for j in range(i + 1, len(selected))
        ]
        return float(np.mean(values)), float(np.max(values))

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

            prefix, shot_no = self._parse_shot(shot_key)
            pose = attrs.get("pose") or {}
            expression = attrs.get("expression") or {}
            body_pose = attrs.get("body_pose") or {}
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
                "pitch": float((pose or {}).get("pitch", 0.0)),
                "yaw": float((pose or {}).get("yaw", 0.0)),
                "roll": float((pose or {}).get("roll", 0.0)),
                "emotion": emotion,
                "emotion_score": emotion_score,
                "expression_status": expression.get("status"),
                "body_pose": body_pose if isinstance(body_pose, dict) else {},
            })

        return candidates, dict(skipped)

    def _diverse_topk(self, items: List[dict], topk: int) -> List[dict]:
        if len(items) <= topk:
            return list(items)
        with_features = [item for item in items if item.get("feature") is not None]
        if len(with_features) < 2:
            return sorted(items, key=lambda x: (x.get("source_shot_frame_idx") is None, x.get("source_shot_frame_idx") or x.get("frame_idx") or 0))[:topk]

        best_pair = None
        best_cos = float("inf")
        for i in range(len(with_features)):
            for j in range(i + 1, len(with_features)):
                cos = self._cosine(with_features[i], with_features[j])
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
                max_cos = max(self._cosine(candidate, item) for item in selected)
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
    ) -> Tuple[List[dict], dict]:
        rng = self._rng(person_id, ref_type, sample_key)
        available = sorted([bucket for bucket, items in buckets.items() if items])
        allocation = self._balanced_allocation(available, count, rng)
        topk = max(1, int(self.config.bucket_candidate_topk))

        bucket_candidates = {
            bucket: self._diverse_topk(items, topk)
            for bucket, items in buckets.items()
            if items
        }

        selected = []
        selected_keys = set()

        def take_from_bucket(bucket: str) -> Optional[dict]:
            candidates = list(bucket_candidates.get(bucket, []))
            rng.shuffle(candidates)
            for candidate in candidates:
                key = self._unique_key(candidate)
                if key in selected_keys:
                    continue
                if not self._passes_shot_gap(candidate, selected):
                    continue
                selected_keys.add(key)
                return dict(candidate, bucket=bucket)
            return None

        deficits = 0
        for bucket, quota in allocation.items():
            for _ in range(quota):
                item = take_from_bucket(bucket)
                if item is None:
                    deficits += 1
                else:
                    selected.append(item)

        while deficits > 0:
            made_progress = False
            refill_buckets = sorted(bucket_candidates.keys())
            rng.shuffle(refill_buckets)
            for bucket in refill_buckets:
                item = take_from_bucket(bucket)
                if item is None:
                    continue
                selected.append(item)
                deficits -= 1
                made_progress = True
                if deficits <= 0:
                    break
            if not made_progress:
                break

        meta = {
            "available_buckets": {bucket: len(items) for bucket, items in buckets.items()},
            "bucket_allocation": allocation,
            "bucket_candidate_topk": topk,
            "requested_count": count,
            "selected_count": len(selected),
            "dedup_key": "shot_key::obj_id::frame_idx",
            "min_same_prefix_shot_gap": int(self.config.min_same_prefix_shot_gap),
            "sample_key": sample_key,
        }
        return selected, meta

    def _angle_buckets(self, candidates: List[dict]) -> Dict[str, List[dict]]:
        buckets = defaultdict(list)
        for item in candidates:
            yaw = float(item["yaw"])
            pitch = float(item["pitch"])
            if abs(yaw) <= 20.0 and abs(pitch - 30.0) <= 10.0:
                buckets["front"].append(dict(item, bucket="front"))
            if pitch < 20.0:
                buckets["front_up"].append(dict(item, bucket="front_up"))
            if pitch > 40.0:
                buckets["front_down"].append(dict(item, bucket="front_down"))
            if yaw <= -30.0:
                buckets["left"].append(dict(item, bucket="left"))
            if yaw >= 30.0:
                buckets["right"].append(dict(item, bucket="right"))
        return dict(buckets)

    def _emotion_buckets(self, candidates: List[dict]) -> Dict[str, List[dict]]:
        buckets = defaultdict(list)
        for item in candidates:
            emotion = str(item.get("emotion") or "").lower()
            if emotion not in EMOTIONS_8:
                continue
            buckets[emotion].append(dict(item, bucket=emotion))
        return dict(buckets)

    def _body_pose_buckets(self, candidates: List[dict]) -> Dict[str, List[dict]]:
        buckets = defaultdict(list)
        for item in candidates:
            body_pose = item.get("body_pose") or {}
            if body_pose.get("status") and body_pose.get("status") != "success":
                continue
            label = str(body_pose.get("label") or "").strip()
            body_part = str(body_pose.get("body_part") or "").strip()
            full_path = item.get("full_path") or item.get("path")
            full_white_path = item.get("full_white_path") or item.get("white_path")
            if label in BODY_LABEL_BUCKETS:
                buckets[label].append(dict(item, bucket=label, bucket_source="label", path=full_path, white_path=full_white_path))
            if body_part in BODY_PART_BUCKETS:
                buckets[body_part].append(dict(item, bucket=body_part, bucket_source="body_part", path=full_path, white_path=full_white_path))
        return dict(buckets)

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
            "bucket": item.get("bucket"),
            "bucket_source": item.get("bucket_source"),
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
        )
        emo_refs, emo_meta = self._sample_from_bucket_candidates(
            self._emotion_buckets(candidates),
            int(self.config.emo_ref_count),
            person_id,
            "emotion",
            sample_key,
        )
        body_refs, body_meta = self._sample_from_bucket_candidates(
            self._body_pose_buckets(candidates),
            int(self.config.body_pose_ref_count),
            person_id,
            "body_pose",
            sample_key,
        )
        return {
            "angle": (angle_refs, angle_meta),
            "emotion": (emo_refs, emo_meta),
            "body_pose": (body_refs, body_meta),
        }

    @staticmethod
    def _ref_signature(angle_refs: List[dict], emo_refs: List[dict], body_refs: List[dict]) -> Tuple[Tuple[str, ...], ...]:
        def paths(items: List[dict]) -> Tuple[str, ...]:
            return tuple(sorted(str(item.get("path") or "") for item in items))

        return paths(angle_refs), paths(emo_refs), paths(body_refs)

    def run(self) -> Tuple[int, int]:
        person_dirs = self._person_dirs()
        rows_written = 0
        stats = {
            "config": asdict(self.config),
            "total_persons": len(person_dirs),
            "persons": {},
            "rows_written": 0,
            "first_frame_error": 0,
        }

        with open(self.config.output_jsonl, "w", encoding="utf-8") as fout:
            for person_dir in tqdm(person_dirs, desc="Generating training pairs", unit="person"):
                person_id = os.path.basename(person_dir)
                index_path = os.path.join(person_dir, self.config.index_filename)
                if not os.path.isfile(index_path):
                    stats["persons"][person_id] = {"status": "missing_index"}
                    continue

                person_index = self._load_json(index_path)
                candidates, skipped = self._build_candidates(person_index)
                target_items = self._target_videos(candidates)
                used_ref_signatures = set()
                per_target_stats = []

                stats["persons"][person_id] = {
                    "status": "ok",
                    "candidate_count": len(candidates),
                    "target_video_count": len(target_items),
                    "skipped": skipped,
                    "per_target": per_target_stats,
                }

                for target in target_items:
                    target_video = target.get("source_shot_path")
                    base_sample_key = f"{target.get('shot_key')}::{target.get('obj_id')}::{target.get('frame_idx')}::{target_video}"
                    refs = None
                    signature = None
                    duplicate_signature = False
                    sample_attempt = 0
                    max_attempts = 20
                    for attempt in range(max_attempts):
                        sample_key = base_sample_key if attempt == 0 else f"{base_sample_key}::retry_{attempt}"
                        refs = self._select_refs(candidates, person_id, sample_key)
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

                    angle_mean, angle_max = self._combo_cos_stats(angle_refs)
                    emo_mean, emo_max = self._combo_cos_stats(emo_refs)
                    body_mean, body_max = self._combo_cos_stats(body_refs)

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
                        },
                        "selection_stats": {
                            "angle_mean_pairwise_cosine": angle_mean,
                            "angle_max_pairwise_cosine": angle_max,
                            "emo_mean_pairwise_cosine": emo_mean,
                            "emo_max_pairwise_cosine": emo_max,
                            "body_pose_mean_pairwise_cosine": body_mean,
                            "body_pose_max_pairwise_cosine": body_max,
                        },
                    }
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows_written += 1
                    per_target_stats.append({
                        "target_video": to_repo_relative_path(target_video),
                        "source_shot_key": target.get("shot_key"),
                        "sample_attempt": sample_attempt,
                        "duplicate_ref_signature": duplicate_signature,
                        "angle": angle_selection_meta,
                        "emotion": emo_selection_meta,
                        "body_pose": body_selection_meta,
                    })

        stats["rows_written"] = rows_written
        with open(self.config.stats_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        return len(person_dirs), rows_written

#!/usr/bin/env python3
"""
After-pipeline index builder (v3) —— 自包含版本，不依赖 index_add.py。

分发逻辑：
  1. 直接列出 person_clusters_dir 下的 person 目录列表；
  2. 先剔除「已处理且未开启 overwrite」的 person；
  3. 对剩余列表按 rank / total_rank 切出连续区间 [st, en)，得到本 rank 应处理的 person；
  4. 逐个构建索引。

v3 与 v2 的主要差异：
  - 输入目录与输出目录分离；
  - 当指定 output_dir 时，输出目录镜像输入目录在 identity_root 下的相对结构；
  - 只创建输出目录和 post-process json，不复制图片、cluster_meta 或其它原始文件。
"""
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

try:
    from .path_utils import (
        resolve_path_fields,
        resolve_repo_path as _resolve_repo_path,
        to_repo_relative_path,
    )
except ImportError:
    from path_utils import (
        resolve_path_fields,
        resolve_repo_path as _resolve_repo_path,
        to_repo_relative_path,
    )

# ---- constants ----
CORE_IMAGE_TYPES = ("face_orig", "face_white", "full_orig", "full_white")
DERIVED_IMAGE_TYPES = (
    "face_angle_left",
    "face_angle_front",
    "face_angle_right",
    "face_diversity_topk",
    "dino_diversity_topk",
)
IMAGE_TYPES = CORE_IMAGE_TYPES + DERIVED_IMAGE_TYPES
ONE_SHOT_DIR_FIELDS = {
    "face_orig": "cropped_id_face_orig_dir_path",
    "face_white": "cropped_id_face_white_dir_path",
    "full_orig": "id_full_orig_dir_path",
    "full_white": "id_full_white_dir_path",
}
WHITE_IMAGE_MASK_DIR_FIELDS = {
    "face_white": "cropped_id_face_mask_for_face_dir_path",
    "full_white": "id_cropped_full_mask_dir_path",
}
QUALITY_MASK_IMAGE_TYPES = {
    "face_orig": "face_white",
    "face_white": "face_white",
    "full_orig": "full_white",
    "full_white": "full_white",
}
IMAGE_EXTENSIONS = {
    "face_orig": ("jpg", "jpeg", "png"),
    "face_white": ("png", "jpg", "jpeg"),
    "full_orig": ("jpg", "jpeg", "png"),
    "full_white": ("png", "jpg", "jpeg"),
}
CLUSTER_IMAGE_RE = re.compile(
    r"^(?P<shot>.+)_id(?P<obj>[^_]+)_frame(?P<frame>\d+)\.[^.]+$"
)
FACE_ANGLE_RE = re.compile(
    r"^(?P<shot>.+)_id(?P<obj>[^_]+)_frame(?P<frame>\d+)_yaw(?P<yaw>[+-]?\d+(?:\.\d+)?)\.[^.]+$"
)
DIVERSITY_RE = re.compile(r"^rank_(?P<rank>\d+)_(?P<shot>.+)_frame(?P<frame>\d+)\.[^.]+$")


# ---- helper functions ----
def resolve_repo_path(value: str) -> str:
    return _resolve_repo_path(value)


def read_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[AfterPipelineV3] Skip invalid JSON at line {line_number}: {exc}")
    return records


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):  # numpy / torch scalar
        return value.item()
    return value


def relativize_index_output(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            to_repo_relative_path(key) if isinstance(key, str) else key: relativize_index_output(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [relativize_index_output(item) for item in value]
    if isinstance(value, str):
        return to_repo_relative_path(value)
    return value


def load_json_object(path: Optional[str]) -> Dict[str, dict]:
    path = resolve_repo_path(path) if path else path
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}


def member_key(record: dict) -> Tuple[str, str]:
    shot_key = record.get("shot_key")
    if not shot_key:
        shot_key = Path(record.get("source_shot_path", "unknown")).stem
    return str(shot_key), str(record.get("obj_id"))


def load_all_cluster_members(
    person_clusters_dir: str,
    identity_records: List[dict],
    recovery_records: List[dict],
    person_ids: Optional[set] = None,
) -> Dict[str, List[dict]]:
    grouped = defaultdict(list)
    seen = defaultdict(set)

    for record in identity_records:
        person_id = record.get("identity_matching_person_id")
        if not person_id:
            continue
        person_id = str(person_id)
        if person_ids is not None and person_id not in person_ids:
            continue
        key = member_key(record)
        grouped[person_id].append(record)
        seen[person_id].add(key)

    recovery_lookup = {member_key(record): record for record in recovery_records}
    if not recovery_lookup or not os.path.isdir(person_clusters_dir):
        return grouped

    for name in sorted(os.listdir(person_clusters_dir)):
        cluster_dir = os.path.join(person_clusters_dir, name)
        meta_path = os.path.join(cluster_dir, "cluster_meta.json")
        if not os.path.isdir(cluster_dir) or not os.path.isfile(meta_path):
            continue
        with open(meta_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
        person_id = str(metadata.get("person_id") or name)
        if person_ids is not None and person_id not in person_ids:
            continue
        grouped[person_id]
        for member in metadata.get("members", []) or []:
            key = member_key(member)
            if key in seen[person_id]:
                continue
            source = recovery_lookup.get(key)
            if source is None:
                print(
                    f"[AfterPipelineV3] Missing recovery record for "
                    f"{person_id}: {key[0]}::id_{key[1]}"
                )
                continue
            recovered = dict(source)
            recovered["identity_matching_person_id"] = person_id
            recovered["identity_matching_person_cluster_dir"] = cluster_dir
            grouped[person_id].append(recovered)
            seen[person_id].add(key)

    return grouped


def candidate_image_path(
    directory: Optional[str],
    frame_idx: int,
    extensions: Iterable[str],
) -> Optional[str]:
    if not directory:
        return None
    directory = resolve_repo_path(directory)
    for extension in extensions:
        path = os.path.join(directory, f"{frame_idx}.{extension}")
        if os.path.isfile(path):
            return os.path.abspath(path)
    first_extension = next(iter(extensions), None)
    if first_extension is None:
        return None
    return os.path.abspath(os.path.join(directory, f"{frame_idx}.{first_extension}"))


def load_one_shot_pose_map(record: dict) -> Dict[str, dict]:
    return load_json_object(record.get("face_euler_angles_jsonl_path"))


def source_shot_frame_idx(record: dict, frame_idx: int) -> Optional[int]:
    mapping = record.get("frame_index_mapping") or {}
    value = mapping.get(str(frame_idx), mapping.get(frame_idx))
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def build_one_shot_paths(record: dict, frame_idx: int) -> Dict[str, Optional[str]]:
    paths = {}
    for image_type in CORE_IMAGE_TYPES:
        directory = record.get(ONE_SHOT_DIR_FIELDS[image_type])
        paths[image_type] = candidate_image_path(
            directory, frame_idx, IMAGE_EXTENSIONS[image_type]
        )
    return paths


def white_image_mask_path(record: dict, image_type: str, frame_idx: int) -> Optional[str]:
    mask_image_type = QUALITY_MASK_IMAGE_TYPES.get(image_type)
    directory_field = WHITE_IMAGE_MASK_DIR_FIELDS.get(mask_image_type)
    if not directory_field:
        return None
    return candidate_image_path(record.get(directory_field), frame_idx, ("npy", "png"))


def _count_mask_holes(mask) -> int:
    import cv2
    import numpy as np

    fg = np.asarray(mask) > 0
    if not np.any(fg):
        return 0

    bg = (~fg).astype(np.uint8)
    h, w = bg.shape
    flood = bg.copy()
    mask_pad = np.zeros((h + 2, w + 2), dtype=np.uint8)
    for x in range(w):
        if flood[0, x]:
            cv2.floodFill(flood, mask_pad, (x, 0), 0)
        if flood[h - 1, x]:
            cv2.floodFill(flood, mask_pad, (x, h - 1), 0)
    for y in range(h):
        if flood[y, 0]:
            cv2.floodFill(flood, mask_pad, (0, y), 0)
        if flood[y, w - 1]:
            cv2.floodFill(flood, mask_pad, (w - 1, y), 0)

    return max(0, int(cv2.connectedComponents(flood, connectivity=8)[0]) - 1)


def mask_hole_quality(
    image_path: Optional[str],
    mask_path: Optional[str],
    image_type: str,
    threshold: int,
) -> dict:
    mask_image_type = QUALITY_MASK_IMAGE_TYPES.get(image_type)
    is_mask_checked_image = mask_image_type in WHITE_IMAGE_MASK_DIR_FIELDS
    result = {
        "checked": False,
        "entry_image_type": image_type,
        "mask_image_type": mask_image_type,
        "is_white_image": image_type in WHITE_IMAGE_MASK_DIR_FIELDS,
        "is_mask_source_white_image": mask_image_type in WHITE_IMAGE_MASK_DIR_FIELDS,
        "mask_path": mask_path,
        "hole_count": 0,
        "threshold": int(threshold),
        "passed": True,
        "status": "not_mask_checked_image",
    }
    if not is_mask_checked_image:
        return result

    try:
        import cv2
        import numpy as np

        mask = None
        resolved_mask_path = resolve_repo_path(mask_path) if mask_path else None
        if resolved_mask_path and os.path.isfile(resolved_mask_path):
            if resolved_mask_path.lower().endswith(".npy"):
                mask = np.load(resolved_mask_path)
            else:
                mask = cv2.imread(resolved_mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None and image_path:
            img = cv2.imread(resolve_repo_path(image_path), cv2.IMREAD_UNCHANGED)
            if img is not None and img.ndim == 3 and img.shape[2] >= 4:
                mask = img[:, :, 3]
                result["status"] = "fallback_alpha"

        if mask is None:
            result["status"] = "missing_mask"
            return result

        hole_count = _count_mask_holes(mask)
        passed = hole_count <= int(threshold)
        result.update({
            "checked": True,
            "hole_count": hole_count,
            "passed": passed,
            "status": result["status"] if result["status"] == "fallback_alpha" else "ok",
        })
        return result
    except Exception as exc:
        result.update({"status": "error", "error_msg": str(exc)})
        return result


def _mask_hole_quality_for_entry(quality: dict, entry_image_type: str) -> dict:
    value = dict(quality)
    value["entry_image_type"] = entry_image_type
    value["is_white_image"] = entry_image_type in WHITE_IMAGE_MASK_DIR_FIELDS
    return value


def _quality_passed(entry: dict) -> bool:
    quality = entry.get("quality")
    if isinstance(quality, dict):
        for item in quality.values():
            if isinstance(item, dict) and item.get("passed") is False:
                return False
    return bool(entry.get("quality_label", True))


def _set_quality_item(entry: dict, name: str, value: dict) -> None:
    quality = entry.get("quality")
    if not isinstance(quality, dict):
        quality = {}
    quality[name] = dict(value)
    entry["quality"] = quality
    entry["quality_label"] = _quality_passed(entry)


def _has_quality_item(entry: Optional[dict], name: str) -> bool:
    if not isinstance(entry, dict):
        return False
    quality = entry.get("quality")
    if not isinstance(quality, dict):
        return False
    item = quality.get(name)
    if not isinstance(item, dict):
        return False
    if name == "mask_hole":
        return item.get("hole_count") is not None
    status = item.get("status")
    if name == "face_bbox_boundary":
        if status == "no_face_detected":
            return True
        return item.get("touches_boundary") is not None and item.get("expanded_face_bbox") is not None
    if name == "face_mask_coverage":
        if status in {"no_face_detected", "skipped_non_frontal_face"}:
            return True
        return item.get("mask_foreground_ratio") is not None and item.get("mask_face_bbox") is not None
    return True


def _load_binary_mask(mask_path: Optional[str]):
    if not mask_path:
        return None
    resolved = resolve_repo_path(mask_path)
    if not os.path.isfile(resolved):
        return None
    import cv2
    import numpy as np

    if resolved.lower().endswith(".npy"):
        return np.load(resolved) > 0
    mask = cv2.imread(resolved, cv2.IMREAD_GRAYSCALE)
    return None if mask is None else mask > 0


def _expanded_bbox(bbox, ratio: float):
    x1, y1, x2, y2 = map(float, bbox)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    nw, nh = w * float(ratio), h * float(ratio)
    return (cx - nw / 2.0, cy - nh / 2.0, cx + nw / 2.0, cy + nh / 2.0)


def _bbox_touches_image_boundary(bbox, image_shape) -> bool:
    if bbox is None or image_shape is None:
        return False
    h, w = image_shape[:2]
    x1, y1, x2, y2 = map(float, bbox)
    return x1 <= 0 or y1 <= 0 or x2 >= w or y2 >= h


class FaceBoundaryQualityChecker:
    """Stage4 face quality: expanded bbox boundary + original bbox SAM coverage."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        model_root: str = "./pretrained_models/insightface",
        device: str = "cuda:0",
        det_size: int = 640,
        expand_ratio: float = 1.1,
        min_foreground_ratio: float = 1.0,
        max_abs_yaw_for_mask_coverage: float = 30.0,
        check_boundary: bool = True,
        check_mask_coverage: bool = True,
    ):
        self._model_name = model_name
        self._model_root = model_root
        self._device = device
        self._det_size = int(det_size)
        self._expand_ratio = float(expand_ratio)
        self._min_foreground_ratio = float(min_foreground_ratio)
        self._max_abs_yaw_for_mask_coverage = float(max_abs_yaw_for_mask_coverage)
        self._check_boundary = bool(check_boundary)
        self._check_mask_coverage = bool(check_mask_coverage)
        self._app = None
        self._cv2 = None
        self._np = None

    def _load_cv_backend(self) -> None:
        if self._cv2 is not None and self._np is not None:
            return
        import cv2
        import numpy as np

        self._cv2 = cv2
        self._np = np

    def _load_backend(self) -> None:
        if self._app is not None:
            return
        self._load_cv_backend()
        from insightface.app import FaceAnalysis

        providers = ["CPUExecutionProvider"] if str(self._device).lower() == "cpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = -1
        if str(self._device).startswith("cuda"):
            try:
                ctx_id = int(str(self._device).split(":", 1)[1])
            except (IndexError, ValueError):
                ctx_id = 0
        self._app = FaceAnalysis(
            name=self._model_name,
            root=resolve_repo_path(self._model_root),
            providers=providers,
        )
        self._app.prepare(ctx_id=ctx_id, det_size=(self._det_size, self._det_size))

    def check(
        self,
        image_path: Optional[str],
        mask_path: Optional[str],
        pose: Optional[dict] = None,
        existing_face_bbox: Optional[List[float]] = None,
        existing_det_score: Optional[float] = None,
        requested_quality_names: Optional[Iterable[str]] = None,
    ) -> dict:
        requested_quality_names = {str(name) for name in (requested_quality_names or []) if str(name)}
        check_boundary = self._check_boundary and (not requested_quality_names or "face_bbox_boundary" in requested_quality_names)
        check_mask_coverage = self._check_mask_coverage and (not requested_quality_names or "face_mask_coverage" in requested_quality_names)
        common = {
            "image_path": image_path,
            "mask_path": mask_path,
            "face_bbox": None,
            "det_score": None,
            "image_size": None,
            "bbox_source": None,
        }
        boundary = {
            **common,
            "checked": False,
            "expanded_face_bbox": None,
            "expand_ratio": self._expand_ratio,
            "touches_boundary": False,
            "passed": True,
            "status": "disabled" if not check_boundary else "missing_face_orig",
        }
        mask_coverage = {
            **common,
            "checked": False,
            "mask_face_bbox": None,
            "mask_foreground_ratio": None,
            "mask_background_pixel_count": None,
            "min_foreground_ratio": self._min_foreground_ratio,
            "max_abs_yaw": self._max_abs_yaw_for_mask_coverage,
            "yaw": None,
            "is_frontal": None,
            "passed": True,
            "status": "disabled" if not check_mask_coverage else "missing_face_orig",
        }
        result = {}
        if check_boundary:
            result["face_bbox_boundary"] = boundary
        if check_mask_coverage:
            result["face_mask_coverage"] = mask_coverage
        if not image_path:
            return result

        try:
            self._load_cv_backend()
            image = self._cv2.imread(resolve_repo_path(image_path))
            if image is None:
                for item in result.values():
                    item["status"] = "cannot_read_image"
                return result

            bbox = None
            det_score = existing_det_score
            bbox_source = None
            if existing_face_bbox is not None:
                try:
                    candidate = self._np.asarray(existing_face_bbox, dtype=float).reshape(-1)
                    if candidate.size == 4 and self._np.all(self._np.isfinite(candidate)):
                        bbox = candidate
                        bbox_source = "existing"
                except Exception:
                    bbox = None

            if bbox is None:
                self._load_backend()
                faces = self._app.get(image)
                if faces is None or len(faces) == 0:
                    for item in result.values():
                        item.update({"checked": True, "status": "no_face_detected", "passed": False})
                    return result

                face = max(faces, key=lambda item: item.det_score)
                bbox = face.bbox.astype(float)
                det_score = float(face.det_score)
                bbox_source = "detected"

            common_update = {
                "checked": True,
                "face_bbox": [float(v) for v in bbox],
                "det_score": float(det_score) if det_score is not None else None,
                "image_size": [int(image.shape[1]), int(image.shape[0])],
                "bbox_source": bbox_source,
                "status": "ok",
            }
            for item in result.values():
                item.update(common_update)

            if check_boundary:
                expanded = _expanded_bbox(bbox, self._expand_ratio)
                touches_boundary = _bbox_touches_image_boundary(expanded, image.shape)
                boundary.update({
                    "expanded_face_bbox": [float(v) for v in expanded],
                    "touches_boundary": bool(touches_boundary),
                    "passed": not touches_boundary,
                    "status": "face_bbox_touches_boundary" if touches_boundary else "ok",
                })

            if not check_mask_coverage:
                return result

            yaw = None
            if isinstance(pose, dict):
                try:
                    yaw = float(pose.get("yaw"))
                except (TypeError, ValueError):
                    yaw = None
            is_frontal = yaw is not None and abs(yaw) <= self._max_abs_yaw_for_mask_coverage
            mask_coverage.update({
                "yaw": yaw,
                "is_frontal": bool(is_frontal),
            })
            if not is_frontal:
                mask_coverage.update({
                    "checked": False,
                    "passed": True,
                    "status": "skipped_non_frontal_face",
                })
                return result

            mask = _load_binary_mask(mask_path)
            if mask is None:
                mask_coverage["status"] = "missing_mask"
                return result
            if mask.shape[:2] != image.shape[:2]:
                mask = self._cv2.resize(
                    mask.astype("uint8"),
                    (image.shape[1], image.shape[0]),
                    interpolation=self._cv2.INTER_NEAREST,
                ) > 0

            h, w = mask.shape[:2]
            x1, y1, x2, y2 = map(int, [
                self._np.floor(bbox[0]),
                self._np.floor(bbox[1]),
                self._np.ceil(bbox[2]),
                self._np.ceil(bbox[3]),
            ])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                mask_coverage.update({"status": "empty_face_bbox", "passed": False})
                return result
            roi = mask[y1:y2, x1:x2]
            fg_count = int(self._np.count_nonzero(roi))
            total = int(roi.size)
            bg_count = total - fg_count
            fg_ratio = float(fg_count / total) if total else 0.0
            mask_passed = fg_ratio >= self._min_foreground_ratio
            mask_coverage.update({
                "mask_face_bbox": [int(x1), int(y1), int(x2), int(y2)],
                "mask_foreground_ratio": fg_ratio,
                "mask_background_pixel_count": bg_count,
                "passed": bool(mask_passed),
                "status": "ok" if mask_passed else "face_bbox_contains_mask_background",
            })
            return result
        except Exception as exc:
            for item in result.values():
                item.update({"status": "error", "error_msg": str(exc)})
            return result


def load_cluster_meta(cluster_dir: str) -> dict:
    meta_path = os.path.join(cluster_dir, "cluster_meta.json")
    if not os.path.isfile(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as file:
        return json.load(file)


def scan_cluster_images(
    cluster_dir: str,
) -> Tuple[
    Dict[str, Dict[Tuple[str, str, int], str]],
    Dict[str, Dict[Tuple[str, str, int], dict]],
]:
    indexes = {image_type: {} for image_type in IMAGE_TYPES}
    metadata = {image_type: {} for image_type in IMAGE_TYPES}

    for image_type in CORE_IMAGE_TYPES:
        directory = os.path.join(cluster_dir, image_type)
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            if not os.path.isfile(path):
                continue
            match = CLUSTER_IMAGE_RE.match(name)
            if not match:
                continue
            key = (
                match.group("shot"),
                str(match.group("obj")),
                int(match.group("frame")),
            )
            indexes[image_type][key] = os.path.abspath(path)

    for bucket in ("left", "front", "right"):
        image_type = f"face_angle_{bucket}"
        bucket_dir = os.path.join(cluster_dir, "face_angle_library", bucket)
        if not os.path.isdir(bucket_dir):
            continue
        for name in sorted(os.listdir(bucket_dir)):
            path = os.path.join(bucket_dir, name)
            if not os.path.isfile(path):
                continue
            match = FACE_ANGLE_RE.match(name)
            if not match:
                continue
            key = (
                match.group("shot"),
                str(match.group("obj")),
                int(match.group("frame")),
            )
            indexes[image_type][key] = os.path.abspath(path)
            metadata[image_type][key] = {
                "derived_image_type": "face_angle_library",
                "angle_bucket": bucket,
                "filename_smoothed_yaw": float(match.group("yaw")),
            }

    cluster_meta = load_cluster_meta(cluster_dir)
    diversity_specs = (
        ("face_diversity_topk", "face_diversity_topk"),
        ("dino_diversity_topk", "dino_diversity_topk"),
    )
    for image_type, meta_key in diversity_specs:
        directory = os.path.join(cluster_dir, image_type)
        if not os.path.isdir(directory):
            continue
        by_rank = {
            int(item.get("rank", 0)): item
            for item in (cluster_meta.get(meta_key, []) or [])
        }
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            if not os.path.isfile(path):
                continue
            match = DIVERSITY_RE.match(name)
            if not match:
                continue
            rank = int(match.group("rank"))
            item = by_rank.get(rank, {})
            shot_key = item.get("shot_key") or match.group("shot")
            obj_id = item.get("obj_id")
            frame_idx = item.get("frame_idx")
            if obj_id is None or frame_idx is None:
                continue
            key = (str(shot_key), str(obj_id), int(frame_idx))
            indexes[image_type][key] = os.path.abspath(path)
            metadata[image_type][key] = {
                "derived_image_type": image_type,
                "rank": rank,
                "stage3_diversity_metadata": item or None,
            }

    return indexes, metadata


def cluster_frame_keys(
    cluster_images: Dict[str, Dict[Tuple[str, str, int], str]],
    shot_key: str,
    obj_id: str,
) -> List[Tuple[str, str, int]]:
    keys = set()
    for image_index in cluster_images.values():
        for key in image_index:
            if key[0] == shot_key and key[1] == str(obj_id):
                keys.add(key)
    return sorted(keys, key=lambda item: item[2])


def empty_output(person_id: str, cluster_dir: str, enabled_features: List[str]) -> dict:
    return {
        "person_id": person_id,
        "cluster_dir": cluster_dir,
        "schema": "images.image_type.image_path.attributes",
        "enabled_features": enabled_features,
        "images": {image_type: {} for image_type in IMAGE_TYPES},
        "face_diversity_topk": [],
        "dino_diversity_topk": [],
        "stats": {
            "member_count": 0,
            "frame_count": 0,
            "image_count": 0,
            "members_without_pose": 0,
        },
    }



# ---- VLM quality checkers ----
class FaceQualityVLMChecker:
    def __init__(
        self,
        model_path: str = "pretrained_models/Qwen3-VL-8B-Instruct",
        device: str = "cuda:0",
        max_new_tokens: int = 512,
        laplacian_threshold: float = 10.0,
        check_occlusion: bool = True,
        check_clarity: bool = True,
        check_clarity_vlm: bool = True,
    ):
        self._model_path = model_path
        self._device = device
        self._max_new_tokens = int(max_new_tokens)
        self._laplacian_threshold = float(laplacian_threshold)
        self._check_occlusion = bool(check_occlusion)
        self._check_clarity = bool(check_clarity)
        self._check_clarity_vlm = bool(check_clarity_vlm)
        self._vlm_model = None
        self._vlm_processor = None
        self._torch = None

    @property
    def quality_names(self) -> Tuple[str, ...]:
        names = []
        if self._check_occlusion:
            names.append("face_occlusion")
        if self._check_clarity:
            names.append("image_clarity_laplacian")
            if self._check_clarity_vlm:
                names.append("image_clarity_vlm")
        return tuple(names)

    def _load_vlm_backend(self) -> None:
        if self._vlm_model is not None and self._vlm_processor is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        model_path = resolve_repo_path(self._model_path)
        self._vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=self._device,
        )
        self._vlm_processor = AutoProcessor.from_pretrained(model_path)
        self._torch = torch

    @staticmethod
    def _parse_json_response(text: str) -> Tuple[dict, str]:
        raw = (text or "").strip()
        json_text = raw
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
        if fence_match:
            json_text = fence_match.group(1).strip()
        elif "{" in raw and "}" in raw:
            json_text = raw[raw.find("{"):raw.rfind("}") + 1]
        try:
            data = json.loads(json_text)
            return (data if isinstance(data, dict) else {}), "success"
        except json.JSONDecodeError:
            return {}, "parse_error"

    @staticmethod
    def _bool_value(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "1", "y", "pass", "passed"}
        return default

    def _generate_vlm_response(self, image_path: str, prompt: str) -> Tuple[str, int]:
        self._load_vlm_backend()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self._vlm_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._vlm_model.device)
        max_new_tokens = max(128, int(self._max_new_tokens))
        with self._torch.no_grad():
            generated_ids = self._vlm_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._vlm_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output_text, max_new_tokens

    def face_occlusion_quality(self, image_path: Optional[str]) -> dict:
        result = {
            "checked": False,
            "image_path": image_path,
            "passed": False,
            "status": "missing_image",
            "face_occluded": None,
            "model": self._model_path,
        }
        if not image_path or not os.path.isfile(resolve_repo_path(image_path)):
            return result
        try:
            prompt = (
                "You are checking face quality for a portrait dataset. Determine whether the visible face is occluded "
                "by objects, hands, hair, masks, text, extreme cropping, or any obstruction that hides important facial features. "
                "Return pure JSON only: {\"face_occluded\": true or false, \"confidence\": 0.0 to 1.0, \"reason\": \"within 20 words\"}."
            )
            output_text, max_new_tokens = self._generate_vlm_response(resolve_repo_path(image_path), prompt)
            data, parse_status = self._parse_json_response(output_text)
            face_occluded = self._bool_value(data.get("face_occluded"), default=True)
            result.update({
                "checked": parse_status == "success",
                "passed": parse_status == "success" and not face_occluded,
                "status": "ok" if parse_status == "success" else parse_status,
                "face_occluded": face_occluded,
                "confidence": data.get("confidence"),
                "reason": str(data.get("reason", "")).strip(),
                "parse_status": parse_status,
                "max_new_tokens": max_new_tokens,
                "raw_response": output_text,
            })
            return result
        except Exception as exc:
            result.update({"status": "error", "error_msg": str(exc)})
            return result

    def laplacian_clarity_quality(self, image_path: Optional[str]) -> dict:
        result = {
            "checked": False,
            "image_path": image_path,
            "passed": False,
            "status": "missing_image",
            "sharpness": None,
            "threshold": self._laplacian_threshold,
        }
        if not image_path or not os.path.isfile(resolve_repo_path(image_path)):
            return result
        try:
            import cv2

            img = cv2.imread(resolve_repo_path(image_path), cv2.IMREAD_COLOR)
            if img is None:
                result["status"] = "read_failed"
                return result
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            result.update({
                "checked": True,
                "passed": sharpness >= self._laplacian_threshold,
                "status": "ok",
                "sharpness": sharpness,
            })
            return result
        except Exception as exc:
            result.update({"status": "error", "error_msg": str(exc)})
            return result

    def vlm_clarity_quality(self, image_path: Optional[str]) -> dict:
        result = {
            "checked": False,
            "image_path": image_path,
            "passed": False,
            "status": "missing_image",
            "is_clear": None,
            "model": self._model_path,
        }
        if not image_path or not os.path.isfile(resolve_repo_path(image_path)):
            return result
        try:
            prompt = (
                "You are checking image clarity for a face dataset. Decide whether the face image is clear enough for identity, "
                "expression, and pose supervision. Mark unclear if it is blurry, motion-blurred, defocused, very low resolution, "
                "or has compression artifacts that obscure facial details. Return pure JSON only: "
                "{\"is_clear\": true or false, \"confidence\": 0.0 to 1.0, \"reason\": \"within 20 words\"}."
            )
            output_text, max_new_tokens = self._generate_vlm_response(resolve_repo_path(image_path), prompt)
            data, parse_status = self._parse_json_response(output_text)
            is_clear = self._bool_value(data.get("is_clear"), default=False)
            result.update({
                "checked": parse_status == "success",
                "passed": parse_status == "success" and is_clear,
                "status": "ok" if parse_status == "success" else parse_status,
                "is_clear": is_clear,
                "confidence": data.get("confidence"),
                "reason": str(data.get("reason", "")).strip(),
                "parse_status": parse_status,
                "max_new_tokens": max_new_tokens,
                "raw_response": output_text,
            })
            return result
        except Exception as exc:
            result.update({"status": "error", "error_msg": str(exc)})
            return result

    def check(self, image_path: Optional[str]) -> Dict[str, dict]:
        quality = {}
        if self._check_occlusion:
            quality["face_occlusion"] = self.face_occlusion_quality(image_path)
        if self._check_clarity:
            quality["image_clarity_laplacian"] = self.laplacian_clarity_quality(image_path)
            if self._check_clarity_vlm:
                quality["image_clarity_vlm"] = self.vlm_clarity_quality(image_path)
        return quality

# ---- feature extractors ----
class EmotionExtractor:
    def __init__(
        self,
        enable_vlm: bool = False,
        vlm_model_path: str = "pretrained_models/Qwen3-VL-8B-Instruct",
        vlm_device: str = "cuda:0",
        vlm_max_new_tokens: int = 2048,
    ):
        self._cv2 = None
        self._np = None
        self._torch = None
        self._recognizer = None
        self._model_name = None
        self._enable_vlm = bool(enable_vlm)
        self._vlm_model_path = vlm_model_path
        self._vlm_device = vlm_device
        self._vlm_max_new_tokens = int(vlm_max_new_tokens)
        self._vlm_model = None
        self._vlm_processor = None
        self._emotion_labels = [
            "angry",
            "contempt",
            "disgust",
            "fear",
            "happy",
            "neutral",
            "sad",
            "surprise",
        ]

    @staticmethod
    def _normalize_emotion_label(label: Any) -> str:
        text = str(label).strip().lower()
        aliases = {
            "anger": "angry",
            "happiness": "happy",
            "sadness": "sad",
        }
        return aliases.get(text, text)

    def _load_backend(self) -> None:
        if self._cv2 is not None and self._recognizer is not None:
            return
        import cv2
        import numpy as np
        import torch
        from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list

        self._cv2 = cv2
        self._np = np
        self._torch = torch
        self._model_name = get_model_list()[0]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._recognizer = EmotiEffLibRecognizer(
            engine="torch",
            model_name=self._model_name,
            device=device,
        )
        idx_to_class = getattr(self._recognizer, "idx_to_emotion_class", None)
        if isinstance(idx_to_class, dict) and idx_to_class:
            self._emotion_labels = [
                self._normalize_emotion_label(idx_to_class[key])
                for key in sorted(idx_to_class)
            ]
            return
        for attr_name in ("emotion_labels", "class_names", "classes", "idx_to_class"):
            labels = getattr(self._recognizer, attr_name, None)
            if isinstance(labels, dict):
                labels = [labels[key] for key in sorted(labels)]
            if isinstance(labels, (list, tuple)) and labels:
                self._emotion_labels = [self._normalize_emotion_label(label) for label in labels]
                break

    def _load_vlm_backend(self) -> None:
        if self._vlm_model is not None and self._vlm_processor is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        model_path = resolve_repo_path(self._vlm_model_path)
        self._vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=self._vlm_device,
        )
        self._vlm_processor = AutoProcessor.from_pretrained(model_path)
        self._torch = torch

    def _parse_vlm_response(self, text: str, expected_emotion: str = "") -> dict:
        raw = (text or "").strip()
        json_text = raw
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
        if fence_match:
            json_text = fence_match.group(1).strip()
        elif "{" in raw and "}" in raw:
            json_text = raw[raw.find("{"):raw.rfind("}") + 1]

        valid_emotions = set(self._emotion_labels)
        expected_emotion = self._normalize_emotion_label(expected_emotion)
        try:
            data = json.loads(json_text)
            if isinstance(data, dict):
                final_expression = self._normalize_emotion_label(data.get("final_expression", ""))
                if final_expression not in valid_emotions:
                    final_expression = self._normalize_emotion_label(data.get("correct_emotion", ""))
                if final_expression not in valid_emotions:
                    final_expression = ""

                is_correct = data.get("is_emotiefflib_correct", data.get("match", False))
                if isinstance(is_correct, str):
                    is_correct = is_correct.strip().lower() in {"true", "yes", "1", "correct", "match"}
                else:
                    is_correct = bool(is_correct)
                if final_expression and expected_emotion:
                    is_correct = final_expression == expected_emotion

                correct_emotion = self._normalize_emotion_label(data.get("correct_emotion", ""))
                if correct_emotion not in valid_emotions and final_expression:
                    correct_emotion = "" if is_correct else final_expression
                elif correct_emotion not in valid_emotions:
                    correct_emotion = ""

                return {
                    "match": is_correct,
                    "correct_emotion": correct_emotion,
                    "final_expression": final_expression,
                    "facial_analysis": str(data.get("facial_analysis", "")).strip(),
                    "reasoning": str(data.get("reasoning", "")).strip(),
                    "parse_status": "success",
                }
        except json.JSONDecodeError:
            pass

        lowered = raw.lower()
        if '"match"' in lowered and "true" in lowered and "false" not in lowered:
            return {"match": True, "correct_emotion": "", "final_expression": "", "facial_analysis": "", "reasoning": "", "parse_status": "fallback"}
        if lowered in {"true", "yes", "match"}:
            return {"match": True, "correct_emotion": "", "final_expression": "", "facial_analysis": "", "reasoning": "", "parse_status": "fallback"}
        detected_emotion = ""
        for emotion in valid_emotions:
            if emotion in lowered:
                detected_emotion = emotion
                break
        return {
            "match": bool(detected_emotion and expected_emotion and detected_emotion == expected_emotion),
            "correct_emotion": "" if detected_emotion == expected_emotion else detected_emotion,
            "final_expression": detected_emotion,
            "facial_analysis": "",
            "reasoning": "",
            "parse_status": "fallback",
        }

    def _generate_vlm_response(self, inputs: Any, max_new_tokens: int) -> str:
        with self._torch.no_grad():
            generated_ids = self._vlm_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self._vlm_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def _verify_emotion_with_vlm(self, img_path: str, emotion: str) -> dict:
        if not emotion:
            return {
                "emo_flag": False,
                "final_expression": "",
                "vlm_check": {
                    "status": "missing_emotion",
                    "expected_emotion": emotion,
                    "final_expression": "",
                    "model": self._vlm_model_path,
                },
            }
        try:
            self._load_vlm_backend()
            prompt = (
                "You are a professional micro-expression analysis and computer vision expert. "
                "Your task is to perform a second-stage verification / double check of a facial expression label.\n\n"

                'The standard facial expression categories are: '
                '["angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"].\n\n'

                "Please analyze the provided image carefully and output the final classification result. "
                "You must return pure JSON only. Do not include any Markdown formatting or explanatory text outside the JSON.\n\n"

                f"The preliminary prediction made by the previous model (EmotiEffLib) is: {emotion}.\n\n"

                "Please perform the following verification steps:\n\n"

                "1. Facial action analysis: Carefully observe the person's eyebrows, eyes, mouth, "
                "and facial muscle patterns, such as frown lines, nasolabial folds, and other visible cues, "
                "to determine the emotional features.\n\n"

                f"2. Logical comparison: Evaluate whether the preliminary prediction {emotion} is reasonable. "
                "If it is unreasonable, correct it.\n\n"

                '3. Final decision: Choose the most accurate category from '
                '["angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"].\n\n'

                "Please strictly return the result in the following JSON format:\n"
                "{\n"
                '  "facial_analysis": "Briefly describe the key facial features you observed, within 25 words.",\n'
                '  "reasoning": "Explain why the final classification is supported, within 25 words.",\n'
                '  "is_emotiefflib_correct": true or false,\n'
                '  "final_expression": "Must be one of the 8 standard categories."\n'
                "}"
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            inputs = self._vlm_processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self._vlm_model.device)
            max_new_tokens = max(256, int(self._vlm_max_new_tokens))
            output_text = self._generate_vlm_response(inputs, max_new_tokens)
            parsed = self._parse_vlm_response(output_text, expected_emotion=emotion)
            retry_count = 0
            if parsed["parse_status"] != "success":
                retry_count = 1
                retry_max_new_tokens = max(max_new_tokens * 2, 1024)
                output_text = self._generate_vlm_response(inputs, retry_max_new_tokens)
                parsed = self._parse_vlm_response(output_text, expected_emotion=emotion)
                max_new_tokens = retry_max_new_tokens
            final_expression = parsed["final_expression"] or emotion
            return {
                "emo_flag": parsed["match"],
                "final_expression": final_expression,
                "vlm_check": {
                    "status": "success",
                    "expected_emotion": emotion,
                    "is_emotiefflib_correct": parsed["match"],
                    "final_expression": final_expression,
                    "correct_emotion": parsed["correct_emotion"],
                    "facial_analysis": parsed["facial_analysis"],
                    "reasoning": parsed["reasoning"],
                    "parse_status": parsed["parse_status"],
                    "retry_count": retry_count,
                    "max_new_tokens": max_new_tokens,
                    "model": self._vlm_model_path,
                    "raw_response": output_text,
                },
            }
        except Exception as exc:
            return {
                "emo_flag": False,
                "final_expression": "",
                "vlm_check": {
                    "status": "error",
                    "expected_emotion": emotion,
                    "final_expression": "",
                    "model": self._vlm_model_path,
                    "error_msg": str(exc),
                },
            }

    def verify_existing_emotion(self, img_path: str, emotion: str) -> dict:
        return self._verify_emotion_with_vlm(img_path, self._normalize_emotion_label(emotion))

    def _format_scores(self, raw_scores: Any) -> Dict[str, float]:
        if raw_scores is None:
            return {}
        if self._np is None:
            import numpy as np

            self._np = np
        if isinstance(raw_scores, dict):
            return {
                self._normalize_emotion_label(key): float(value)
                * (100.0 if float(value) <= 1.0 else 1.0)
                for key, value in raw_scores.items()
            }

        arr = self._np.asarray(raw_scores, dtype=float).reshape(-1)
        if arr.size == 0:
            return {}
        if arr.size == len(self._emotion_labels):
            labels = self._emotion_labels
        else:
            labels = [f"emotion_{idx}" for idx in range(arr.size)]

        if arr.min(initial=0.0) < 0.0 or arr.max(initial=0.0) > 1.0:
            arr = arr - arr.max()
            exp = self._np.exp(arr)
            denom = exp.sum()
            arr = exp / denom if denom > 0 else exp

        return {label: float(score) * 100.0 for label, score in zip(labels, arr)}

    def _analyze(self, img_path: str) -> dict:
        self._load_backend()
        img = self._cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        face_img = self._cv2.cvtColor(img, self._cv2.COLOR_BGR2RGB)
        emotions, raw_scores = self._recognizer.predict_emotions(face_img, logits=False)
        dominant = ""
        if emotions is not None and len(emotions) > 0:
            dominant = self._normalize_emotion_label(emotions[0])
        scores = self._format_scores(raw_scores)
        if not dominant and scores:
            dominant = max(scores.items(), key=lambda item: item[1])[0]
        expression = {
            "scores": scores,
            "dominant": dominant,
            "status": "success",
            "backend": "emotiefflib",
            "model_name": self._model_name,
        }
        if self._enable_vlm:
            expression.update(self._verify_emotion_with_vlm(img_path, dominant))
        return {
            "expression": expression
        }

    def extract(self, context: dict) -> dict:
        img_path = (
            context["identity_matching_paths"].get("face_orig")
            or context["one_shot_paths"].get("face_orig")
        )
        if not img_path:
            expression = {
                "scores": {},
                "dominant": "",
                "status": "missing_face_orig",
            }
            if self._enable_vlm:
                expression.update({
                    "emo_flag": False,
                    "final_expression": "",
                    "vlm_check": {
                        "status": "skipped_missing_face_orig",
                        "expected_emotion": "",
                        "final_expression": "",
                        "model": self._vlm_model_path,
                    },
                })
            return {
                "expression": expression
            }
        try:
            return self._analyze(img_path)
        except Exception as exc:
            expression = {
                "scores": {},
                "dominant": "",
                "status": "error",
                "error_msg": str(exc),
            }
            if self._enable_vlm:
                expression.update({
                    "emo_flag": False,
                    "final_expression": "",
                    "vlm_check": {
                        "status": "skipped_emotion_error",
                        "expected_emotion": "",
                        "final_expression": "",
                        "model": self._vlm_model_path,
                    },
                })
            return {
                "expression": expression
            }


class BodyPoseExtractor:
    """Estimate body extent and coarse body orientation from a full-body crop."""

    POSE_DET_CONF_THR = 0.45
    POSE_KEYPOINT_CONF_THR = 0.5

    def __init__(
        self,
        checkpoint_path: str,
        pose_checkpoint: str,
        detector_name: str = "vitdet",
        device: Optional[str] = None,
    ):
        self._checkpoint_path = checkpoint_path
        self._pose_checkpoint = pose_checkpoint
        self._detector_name = detector_name
        self._device_name = device
        self._device = None
        self._orientation_estimator = None
        self._pose_model = None
        self._cv2 = None
        self._np = None
        self._torch = None

    def _load_backend(self) -> None:
        if self._orientation_estimator is not None and self._pose_model is not None:
            return
        import cv2
        import numpy as np
        import torch
        from ultralytics import YOLO

        try:
            from .human_orientation_4dhumans import HumanBodyOrientationEstimator
        except ImportError:
            from human_orientation_4dhumans import HumanBodyOrientationEstimator

        self._cv2 = cv2
        self._np = np
        self._torch = torch
        self._device = self._device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        self._orientation_estimator = HumanBodyOrientationEstimator(
            checkpoint_path=resolve_repo_path(self._checkpoint_path),
            detector_name=self._detector_name,
            yolo_checkpoint=resolve_repo_path(self._pose_checkpoint),
            device=self._device,
        )
        self._pose_model = YOLO(resolve_repo_path(self._pose_checkpoint))
        self._pose_model.to(self._device)
        self._pose_model.eval()

    def _person_keypoints_to_body_part(
        self,
        keypoints: Any,
        confidence: Any,
        bbox: Any,
    ) -> str:
        del keypoints, bbox
        confidence = self._np.asarray(confidence)
        if confidence.size == 0:
            return "unknown"

        visible = confidence >= self.POSE_KEYPOINT_CONF_THR
        if not self._np.any(visible):
            return "unknown"

        visible_idxs = set(self._np.where(visible)[0].tolist())
        upper_idxs = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
        torso_idxs = {5, 6, 11, 12}
        lower_idxs = {13, 14, 15, 16}
        ankle_idxs = {15, 16}
        upper_visible = len(upper_idxs.intersection(visible_idxs))
        torso_visible = len(torso_idxs.intersection(visible_idxs))
        lower_visible = len(lower_idxs.intersection(visible_idxs))
        ankle_visible = len(ankle_idxs.intersection(visible_idxs))

        if lower_visible >= 2 or ankle_visible >= 1:
            return "full_body"
        if torso_visible >= 2:
            return "half_body"
        if upper_visible >= 1:
            return "head_closeup"
        return "unknown"

    def _predict(self, img_path: str) -> Optional[Dict[str, Any]]:
        self._load_backend()
        image_bgr = self._cv2.imread(img_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        pose_result = self._pose_model.predict(
            image_bgr,
            save=False,
            verbose=False,
            device=self._device,
            conf=self.POSE_DET_CONF_THR,
        )[0]
        if pose_result.boxes is None or len(pose_result.boxes) == 0:
            return None

        valid_idx = pose_result.boxes.conf >= self.POSE_DET_CONF_THR
        valid_idx_np = valid_idx.detach().cpu().numpy() if self._torch.is_tensor(valid_idx) else valid_idx
        if not self._np.any(valid_idx_np):
            return None
        if pose_result.keypoints is None:
            return None

        person_idx = int(self._np.argmax(pose_result.boxes.conf.detach().cpu().numpy()))
        bbox = pose_result.boxes.xyxy[person_idx].detach().cpu().numpy()
        keypoints = pose_result.keypoints.xy[person_idx].detach().cpu().numpy()
        confidence = pose_result.keypoints.conf[person_idx].detach().cpu().numpy()
        body_part = self._person_keypoints_to_body_part(keypoints, confidence, bbox)

        orientation = self._orientation_estimator.predict(image_bgr)
        if orientation is None:
            return {
                "body_part": body_part,
                "orientation": None,
                "status": "success",
            }

        orientation["body_part"] = body_part
        orientation["status"] = "success"
        return orientation

    def extract(self, context: dict) -> dict:
        img_path = context["identity_matching_paths"].get("full_orig")
        if not img_path:
            return {
                "body_pose": {
                    "body_part": "unknown",
                    "orientation": None,
                    "status": "missing_full_orig",
                    "available_identity_image_types": sorted(context["identity_matching_paths"].keys()),
                    "available_one_shot_image_types": sorted(context["one_shot_paths"].keys()),
                }
            }
        try:
            result = self._predict(img_path)
            if result is None:
                result = {
                    "body_part": "unknown",
                    "orientation": None,
                    "status": "no_person_detected",
                }
            return {"body_pose": result}
        except Exception as exc:
            return {
                "body_pose": {
                    "body_part": "unknown",
                    "orientation": None,
                    "status": "error",
                    "error_msg": str(exc),
                }
            }


def build_face_boundary_quality_checker(config, force: bool = False) -> Optional[FaceBoundaryQualityChecker]:
    del force
    enable_all = bool(getattr(config, "enable_face_boundary_quality_check", False))
    check_boundary = enable_all or bool(getattr(config, "enable_face_bbox_boundary_quality_check", False))
    check_mask_coverage = enable_all or bool(getattr(config, "enable_face_mask_coverage_quality_check", False))
    if not check_boundary and not check_mask_coverage:
        return None
    return FaceBoundaryQualityChecker(
        model_name=getattr(config, "face_quality_model_name", "buffalo_l"),
        model_root=getattr(config, "face_quality_model_root", "./pretrained_models/insightface"),
        device=getattr(config, "face_quality_device", "cuda:0"),
        det_size=getattr(config, "face_quality_det_size", 640),
        expand_ratio=getattr(config, "face_boundary_expand_ratio", 1.1),
        min_foreground_ratio=getattr(config, "face_mask_min_foreground_ratio", 1.0),
        max_abs_yaw_for_mask_coverage=getattr(config, "face_mask_coverage_max_abs_yaw", 30.0),
        check_boundary=check_boundary,
        check_mask_coverage=check_mask_coverage,
    )


def build_face_quality_vlm_checker(
    config,
    check_occlusion: Optional[bool] = None,
    check_clarity: Optional[bool] = None,
    check_clarity_vlm: Optional[bool] = None,
) -> Optional[FaceQualityVLMChecker]:
    if check_occlusion is None:
        check_occlusion = bool(getattr(config, "enable_face_occlusion_quality_check", False))
    if check_clarity is None:
        check_clarity = bool(getattr(config, "enable_image_clarity_quality_check", False))
    if check_clarity_vlm is None:
        check_clarity_vlm = bool(getattr(config, "enable_image_clarity_vlm_check", True))
    if not check_occlusion and not check_clarity:
        return None
    return FaceQualityVLMChecker(
        model_path=getattr(config, "quality_vlm_model_path", getattr(config, "emotion_vlm_model_path", "pretrained_models/Qwen3-VL-8B-Instruct")),
        device=getattr(config, "quality_vlm_device", getattr(config, "emotion_vlm_device", "cuda:0")),
        max_new_tokens=getattr(config, "quality_vlm_max_new_tokens", getattr(config, "emotion_vlm_max_new_tokens", 512)),
        laplacian_threshold=getattr(config, "clarity_laplacian_threshold", 10.0),
        check_occlusion=bool(check_occlusion),
        check_clarity=bool(check_clarity),
        check_clarity_vlm=bool(check_clarity_vlm),
    )


def build_feature_extractors(config) -> Dict[str, Callable[[dict], dict]]:
    extractors = {}
    if config.enable_emotion:
        extractors["emotion"] = EmotionExtractor(
            enable_vlm=config.enable_emotion_vlm,
            vlm_model_path=config.emotion_vlm_model_path,
            vlm_device=config.emotion_vlm_device,
            vlm_max_new_tokens=config.emotion_vlm_max_new_tokens,
        ).extract
    if config.enable_body_pose:
        extractors["body_pose"] = BodyPoseExtractor(
            checkpoint_path=config.body_pose_checkpoint,
            pose_checkpoint=config.body_pose_yolo_checkpoint,
            detector_name=config.body_pose_detector,
            device=config.body_pose_device,
        ).extract
    return extractors


def build_person_index(
    person_id: str,
    cluster_dir: str,
    member_records: List[dict],
    include_pose: bool,
    feature_extractors: Dict[str, Callable[[dict], dict]],
    progress_position: int = 1,
    enable_mask_hole_quality_check: bool = True,
    mask_hole_threshold: int = 0,
    face_boundary_quality_checker: Optional[FaceBoundaryQualityChecker] = None,
    face_quality_vlm_checker: Optional[FaceQualityVLMChecker] = None,
) -> dict:
    enabled_features = []
    if include_pose:
        enabled_features.append("pose")
    if enable_mask_hole_quality_check:
        enabled_features.append("mask_hole_quality")
    if face_boundary_quality_checker is not None:
        enabled_features.append("face_boundary_quality")
    if face_quality_vlm_checker is not None:
        if face_quality_vlm_checker._check_occlusion:
            enabled_features.append("face_occlusion_quality")
        if face_quality_vlm_checker._check_clarity:
            enabled_features.append("image_clarity_quality")
    enabled_features.extend(feature_extractors.keys())

    output = empty_output(person_id, cluster_dir, enabled_features)
    output["stats"]["member_count"] = len(member_records)
    cluster_images, cluster_image_metadata = scan_cluster_images(cluster_dir)
    frame_contexts = []

    for record in member_records:
        shot_key, obj_id = member_key(record)
        uid = f"{shot_key}::id_{obj_id}"
        pose_map = load_one_shot_pose_map(record)
        if not pose_map:
            output["stats"]["members_without_pose"] += 1
            continue

        for key in cluster_frame_keys(cluster_images, shot_key, str(obj_id)):
            frame_key = str(key[2])
            pose = pose_map.get(frame_key)
            if not isinstance(pose, dict):
                continue
            frame_idx = int(frame_key)
            one_shot_paths = build_one_shot_paths(record, frame_idx)
            identity_paths = {
                image_type: cluster_images[image_type].get(key)
                for image_type in IMAGE_TYPES
            }
            related_images = {
                image_type: identity_paths.get(image_type) or one_shot_paths.get(image_type)
                for image_type in CORE_IMAGE_TYPES
            }
            face_boundary_quality_by_type = {}
            if face_boundary_quality_checker is not None:
                for orig_type, white_type in (("face_orig", "face_white"), ("full_orig", "full_white")):
                    group_quality = face_boundary_quality_checker.check(
                        image_path=related_images.get(orig_type),
                        mask_path=white_image_mask_path(record, white_type, frame_idx),
                        pose=pose,
                    )
                    face_boundary_quality_by_type[orig_type] = group_quality
                    face_boundary_quality_by_type[white_type] = group_quality
            face_quality_vlm = {}
            if face_quality_vlm_checker is not None:
                face_quality_vlm = face_quality_vlm_checker.check(
                    related_images.get("face_orig") or related_images.get("face_white")
                )
            context = {
                "person_id": person_id,
                "uid": uid,
                "shot_key": shot_key,
                "obj_id": obj_id,
                "frame_idx": frame_idx,
                "record": record,
                "pose": pose,
                "identity_matching_paths": identity_paths,
                "one_shot_paths": one_shot_paths,
            }

            output["stats"]["frame_count"] += 1
            frame_entries = []
            for image_type in IMAGE_TYPES:
                image_path = related_images.get(image_type)
                if image_type in DERIVED_IMAGE_TYPES:
                    image_path = identity_paths.get(image_type)
                if not image_path:
                    continue
                attrs = {
                    "person_id": person_id,
                    "uid": uid,
                    "shot_key": shot_key,
                    "obj_id": obj_id,
                    "frame_idx": frame_idx,
                    "source_shot_frame_idx": source_shot_frame_idx(record, frame_idx),
                    "source_shot_path": record.get("source_shot_path"),
                    "image_type": image_type,
                    "image_path": image_path,
                    "identity_matching_image_path": identity_paths.get(image_type),
                    "one_shot_image_path": one_shot_paths.get(image_type),
                    "related_images": related_images,
                    "related_identity_matching_images": identity_paths,
                    "related_one_shot_images": one_shot_paths,
                }
                derived_metadata = cluster_image_metadata.get(image_type, {}).get(key)
                if derived_metadata:
                    attrs.update(derived_metadata)
                attrs["quality_label"] = True
                if enable_mask_hole_quality_check:
                    quality = mask_hole_quality(
                        image_path=image_path,
                        mask_path=white_image_mask_path(record, image_type, frame_idx),
                        image_type=image_type,
                        threshold=mask_hole_threshold,
                    )
                    _set_quality_item(attrs, "mask_hole", quality)
                face_boundary_quality = face_boundary_quality_by_type.get(image_type)
                if face_boundary_quality is not None:
                    for quality_name, quality_value in face_boundary_quality.items():
                        _set_quality_item(attrs, quality_name, quality_value)
                if face_quality_vlm:
                    for quality_name, quality_value in face_quality_vlm.items():
                        _set_quality_item(attrs, quality_name, quality_value)
                if include_pose:
                    pose_attrs = {
                        "pitch": float(pose.get("pitch", 0.0)),
                        "yaw": float(pose.get("yaw", 0.0)),
                        "roll": float(pose.get("roll", 0.0)),
                    }
                    attrs["pose"] = pose_attrs
                output["images"][image_type][image_path] = attrs
                if image_type in ("face_diversity_topk", "dino_diversity_topk"):
                    output[image_type].append(dict(attrs))
                frame_entries.append(attrs)
                output["stats"]["image_count"] += 1
            if frame_entries:
                frame_contexts.append((context, frame_entries))

    for image_type in ("face_diversity_topk", "dino_diversity_topk"):
        output[image_type].sort(key=lambda item: int(item.get("rank") or 0))

    tqdm.write(
        f"[AfterPipelineV3] {person_id}: pose/image mapping done, "
        f"frames={output['stats']['frame_count']}, "
        f"images={output['stats']['image_count']}"
    )

    for feature_name, extractor in feature_extractors.items():
        total = len(frame_contexts)
        tqdm.write(f"[AfterPipelineV3] {person_id}: start {feature_name}, total_frames={total}")
        progress = tqdm(
            frame_contexts,
            desc=f"{person_id} {feature_name}",
            unit="frame",
            position=progress_position,
            leave=False,
        )
        for context, frame_entries in progress:
            progress.set_postfix_str(f"{context['uid']} frame={context['frame_idx']}", refresh=False)
            feature_attrs = extractor(context)
            for entry in frame_entries:
                entry.update(feature_attrs)
        tqdm.write(f"[AfterPipelineV3] {person_id}: {feature_name} done")

    return output


def output_cluster_dir(
    person_id: str,
    input_cluster_dir: str,
    output_dir: Optional[str],
) -> str:
    if output_dir:
        return os.path.join(output_dir, person_id)
    return input_cluster_dir


# ---- 增量更新（对已建好的索引补加 emotion / emotion_vlm / body_pose） ----
def _entry_face_orig_path(entry: dict) -> Optional[str]:
    for field in ("related_identity_matching_images", "related_images", "related_one_shot_images"):
        paths = entry.get(field) or {}
        if isinstance(paths, dict) and paths.get("face_orig"):
            return paths["face_orig"]
    if entry.get("image_type") == "face_orig":
        return (
            entry.get("identity_matching_image_path")
            or entry.get("image_path")
            or entry.get("one_shot_image_path")
        )
    return None


def _entry_full_orig_path(entry: dict) -> Optional[str]:
    for field in ("related_identity_matching_images", "related_images", "related_one_shot_images"):
        paths = entry.get(field) or {}
        if isinstance(paths, dict) and paths.get("full_orig"):
            return paths["full_orig"]
    if entry.get("image_type") == "full_orig":
        return (
            entry.get("identity_matching_image_path")
            or entry.get("image_path")
            or entry.get("one_shot_image_path")
        )
    return None


def _existing_entry_context(entry: dict) -> dict:
    identity_paths = dict(entry.get("related_identity_matching_images") or {})
    one_shot_paths = dict(entry.get("related_one_shot_images") or {})
    if entry.get("image_type") and entry.get("identity_matching_image_path"):
        identity_paths.setdefault(entry["image_type"], entry.get("identity_matching_image_path"))
    if entry.get("image_type") and entry.get("one_shot_image_path"):
        one_shot_paths.setdefault(entry["image_type"], entry.get("one_shot_image_path"))
    full_orig_path = _entry_full_orig_path(entry)
    if full_orig_path:
        identity_paths.setdefault("full_orig", full_orig_path)
    face_orig_path = _entry_face_orig_path(entry)
    if face_orig_path:
        identity_paths.setdefault("face_orig", face_orig_path)
    return {
        "person_id": entry.get("person_id"),
        "uid": entry.get("uid"),
        "shot_key": entry.get("shot_key"),
        "obj_id": entry.get("obj_id"),
        "frame_idx": entry.get("frame_idx"),
        "record": {},
        "pose": entry.get("pose") or {},
        "identity_matching_paths": identity_paths,
        "one_shot_paths": one_shot_paths,
    }


def _expression_dominant(expression: dict) -> str:
    dominant = EmotionExtractor._normalize_emotion_label(expression.get("dominant", ""))
    if dominant:
        return dominant
    scores = expression.get("scores") or {}
    if not isinstance(scores, dict) or not scores:
        return ""
    try:
        return EmotionExtractor._normalize_emotion_label(
            max(scores.items(), key=lambda item: float(item[1]))[0]
        )
    except (TypeError, ValueError):
        return ""


def _iter_after_pipeline_entries_for_update(index_data: dict):
    images = index_data.get("images", {}) if isinstance(index_data, dict) else {}
    for image_type_entries in images.values():
        if not isinstance(image_type_entries, dict):
            continue
        for image_path, entry in image_type_entries.items():
            if isinstance(entry, dict):
                yield image_path, entry


def _entry_image_path(entry: dict) -> Optional[str]:
    return (
        entry.get("image_path")
        or entry.get("identity_matching_image_path")
        or entry.get("one_shot_image_path")
    )


def _mask_path_from_white_image_path(image_path: Optional[str], image_type: str) -> Optional[str]:
    if not image_path:
        return None
    resolved = resolve_repo_path(image_path)
    dirname, filename = os.path.split(resolved)
    stem, _ = os.path.splitext(filename)
    parent = os.path.dirname(dirname)
    if image_type == "face_white" and os.path.basename(dirname) == "face_pic_white":
        mask_dir = os.path.join(parent, "face_mask_for_face")
    elif image_type == "full_white" and os.path.basename(dirname) == "full_pic_white":
        mask_dir = os.path.join(parent, "cropped_full_mask")
    else:
        return None
    for extension in ("npy", "png"):
        candidate = os.path.join(mask_dir, f"{stem}.{extension}")
        if os.path.isfile(candidate):
            return candidate
    return None


def update_group_mask_hole_quality(
    entries: List[dict],
    threshold: int,
    recovery_record: Optional[dict] = None,
    overwrite_existing_quality: bool = True,
) -> int:
    entries_by_type = {entry.get("image_type"): entry for entry in entries if entry.get("image_type")}
    updated = 0
    for mask_image_type, affected_types in (
        ("face_white", ("face_orig", "face_white")),
        ("full_white", ("full_orig", "full_white")),
    ):
        source_entry = entries_by_type.get(mask_image_type)
        if not source_entry:
            continue
        target_entries = [entries_by_type.get(image_type) for image_type in affected_types]
        if not overwrite_existing_quality and all(_has_quality_item(entry, "mask_hole") for entry in target_entries if entry):
            continue
        image_path = _entry_image_path(source_entry)
        frame_idx = source_entry.get("frame_idx")
        try:
            frame_idx = int(frame_idx)
        except (TypeError, ValueError):
            frame_idx = None
        recovery_mask_path = (
            white_image_mask_path(recovery_record, mask_image_type, frame_idx)
            if recovery_record and frame_idx is not None
            else None
        )
        quality = mask_hole_quality(
            image_path=image_path,
            mask_path=recovery_mask_path or _mask_path_from_white_image_path(image_path, mask_image_type),
            image_type=mask_image_type,
            threshold=threshold,
        )
        for image_type in affected_types:
            entry = entries_by_type.get(image_type)
            if not entry:
                continue
            if overwrite_existing_quality or not _has_quality_item(entry, "mask_hole"):
                _set_quality_item(entry, "mask_hole", _mask_hole_quality_for_entry(quality, image_type))
                updated += 1
    return updated


def _existing_quality_face_bbox(*entries: Optional[dict]) -> Tuple[Optional[List[float]], Optional[float]]:
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        quality = entry.get("quality") or {}
        if not isinstance(quality, dict):
            continue
        for quality_name in ("face_bbox_boundary", "face_mask_coverage"):
            item = quality.get(quality_name) or {}
            if not isinstance(item, dict):
                continue
            bbox = item.get("face_bbox")
            if (
                (not isinstance(bbox, (list, tuple)) or len(bbox) != 4)
                and quality_name == "face_mask_coverage"
            ):
                bbox = item.get("mask_face_bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                bbox_values = [float(value) for value in bbox]
            except (TypeError, ValueError):
                continue
            det_score = item.get("det_score")
            try:
                det_score = float(det_score) if det_score is not None else None
            except (TypeError, ValueError):
                det_score = None
            return bbox_values, det_score
    return None, None


def update_group_face_boundary_quality(
    entries: List[dict],
    checker: FaceBoundaryQualityChecker,
    recovery_record: Optional[dict] = None,
    reuse_existing_bbox: bool = True,
    overwrite_existing_quality: bool = True,
) -> int:
    entries_by_type = {entry.get("image_type"): entry for entry in entries if entry.get("image_type")}
    updated = 0
    for orig_type, white_type in (("face_orig", "face_white"), ("full_orig", "full_white")):
        orig_entry = entries_by_type.get(orig_type)
        if not orig_entry:
            continue
        target_entries = [entry for entry in (orig_entry, entries_by_type.get(white_type)) if entry]
        quality_names = []
        if checker._check_boundary:
            quality_names.append("face_bbox_boundary")
        if checker._check_mask_coverage:
            quality_names.append("face_mask_coverage")
        requested_quality_names = quality_names
        if not overwrite_existing_quality:
            requested_quality_names = [
                name
                for name in quality_names
                if any(not _has_quality_item(entry, name) for entry in target_entries)
            ]
            if not requested_quality_names:
                continue
        image_path = _entry_image_path(orig_entry)
        frame_idx = orig_entry.get("frame_idx")
        try:
            frame_idx = int(frame_idx)
        except (TypeError, ValueError):
            frame_idx = None
        recovery_mask_path = (
            white_image_mask_path(recovery_record, white_type, frame_idx)
            if recovery_record and frame_idx is not None
            else None
        )
        mask_path = recovery_mask_path
        if not mask_path:
            source_entry = entries_by_type.get(white_type)
            mask_path = _mask_path_from_white_image_path(_entry_image_path(source_entry), white_type) if source_entry else None
        existing_bbox, existing_det_score = (None, None)
        if reuse_existing_bbox:
            existing_bbox, existing_det_score = _existing_quality_face_bbox(
                orig_entry,
                entries_by_type.get(white_type),
            )
        quality = checker.check(
            image_path=image_path,
            mask_path=mask_path,
            pose=orig_entry.get("pose") or {},
            existing_face_bbox=existing_bbox,
            existing_det_score=existing_det_score,
            requested_quality_names=requested_quality_names,
        )
        for image_type in (orig_type, white_type):
            entry = entries_by_type.get(image_type)
            if not entry:
                continue
            wrote_entry = False
            for quality_name, quality_value in quality.items():
                if overwrite_existing_quality or not _has_quality_item(entry, quality_name):
                    _set_quality_item(entry, quality_name, quality_value)
                    wrote_entry = True
            if wrote_entry:
                updated += 1
    return updated


def update_group_face_quality_vlm(
    entries: List[dict],
    checker: FaceQualityVLMChecker,
    overwrite_existing_quality: bool = True,
) -> int:
    target_entries = [entry for entry in entries if isinstance(entry, dict)]
    if not target_entries:
        return 0
    quality_names = checker.quality_names
    if not overwrite_existing_quality and all(
        all(_has_quality_item(entry, name) for name in quality_names)
        for entry in target_entries
    ):
        return 0
    entries_by_type = {entry.get("image_type"): entry for entry in target_entries if entry.get("image_type")}
    source_entry = entries_by_type.get("face_orig") or entries_by_type.get("face_white") or target_entries[0]
    quality = checker.check(_entry_image_path(source_entry))
    updated = 0
    for entry in target_entries:
        wrote_entry = False
        for quality_name, quality_value in quality.items():
            if overwrite_existing_quality or not _has_quality_item(entry, quality_name):
                _set_quality_item(entry, quality_name, quality_value)
                wrote_entry = True
        if wrote_entry:
            updated += 1
    return updated


def build_update_feature_extractors(feature_names: List[str], config: Any) -> Dict[str, Any]:
    feature_names = {str(name).strip().lower() for name in feature_names if str(name).strip()}
    extractors: Dict[str, Any] = {}
    if "emotion" in feature_names or "emotion_vlm" in feature_names:
        extractors["emotion"] = EmotionExtractor(
            enable_vlm="emotion_vlm" in feature_names,
            vlm_model_path=config.emotion_vlm_model_path,
            vlm_device=config.emotion_vlm_device,
            vlm_max_new_tokens=config.emotion_vlm_max_new_tokens,
        )
    if "body_pose" in feature_names:
        extractors["body_pose"] = BodyPoseExtractor(
            checkpoint_path=config.body_pose_checkpoint,
            pose_checkpoint=config.body_pose_yolo_checkpoint,
            detector_name=config.body_pose_detector,
            device=config.body_pose_device,
        )
    if "face_boundary_quality" in feature_names:
        checker = build_face_boundary_quality_checker(config, force=True)
        if checker is not None:
            extractors["face_boundary_quality"] = checker
    if "face_occlusion_quality" in feature_names or "image_clarity_quality" in feature_names:
        checker = build_face_quality_vlm_checker(
            config,
            check_occlusion="face_occlusion_quality" in feature_names,
            check_clarity="image_clarity_quality" in feature_names,
            check_clarity_vlm=bool(getattr(config, "enable_image_clarity_vlm_check", True)),
        )
        if checker is not None:
            extractors["face_quality_vlm"] = checker
    return extractors


def update_existing_index_features(
    index_path: str,
    feature_names: List[str],
    config: Any,
    update_extractors: Optional[Dict[str, Any]] = None,
    progress_position: int = 1,
    recovery_lookup: Optional[Dict[Tuple[str, str], dict]] = None,
) -> Tuple[int, int, int]:
    feature_names = [str(name).strip().lower() for name in feature_names if str(name).strip()]
    valid_features = {"emotion", "emotion_vlm", "body_pose", "mask_hole_quality", "face_boundary_quality", "face_occlusion_quality", "image_clarity_quality"}
    unknown_features = sorted(set(feature_names) - valid_features)
    if unknown_features:
        raise ValueError(f"Unknown update_features: {', '.join(unknown_features)}")

    with open(index_path, "r", encoding="utf-8") as file:
        index_data = json.load(file)

    frame_groups: Dict[Tuple[str, str, str], dict] = {}
    for _, entry in _iter_after_pipeline_entries_for_update(index_data):
        key = (
            str(entry.get("uid") or ""),
            str(entry.get("shot_key") or ""),
            str(entry.get("frame_idx") or ""),
        )
        group = frame_groups.setdefault(key, {"entries": [], "context_entry": entry})
        group["entries"].append(entry)
        if _entry_face_orig_path(entry) or _entry_full_orig_path(entry):
            group["context_entry"] = entry

    update_extractors = update_extractors or build_update_feature_extractors(feature_names, config)
    emotion_extractor = update_extractors.get("emotion")
    body_pose_extractor = update_extractors.get("body_pose")
    face_boundary_checker = update_extractors.get("face_boundary_quality")
    face_quality_vlm_checker = update_extractors.get("face_quality_vlm")

    updated_frames = skipped_frames = updated_entries = 0
    progress = tqdm(
        list(frame_groups.values()),
        desc="update_features",
        unit="frame",
        position=progress_position,
        leave=False,
    )
    for group in progress:
        context = _existing_entry_context(group["context_entry"])
        progress.set_postfix_str(
            f"{context.get('uid') or context.get('shot_key')} frame={context.get('frame_idx')}",
            refresh=False,
        )
        frame_attrs: Dict[str, Any] = {}

        if "emotion" in feature_names and emotion_extractor is not None:
            frame_attrs.update(emotion_extractor.extract(context))
        elif "emotion_vlm" in feature_names and emotion_extractor is not None:
            expression = None
            for entry in group["entries"]:
                if isinstance(entry.get("expression"), dict):
                    expression = entry["expression"]
                    break
            dominant = _expression_dominant(expression or {})
            face_orig_path = context["identity_matching_paths"].get("face_orig") or context["one_shot_paths"].get("face_orig")
            if not dominant or not face_orig_path:
                frame_attrs["expression"] = {
                    **(expression or {}),
                    "emo_flag": False,
                    "final_expression": "",
                    "vlm_check": {
                        "status": "skipped_missing_face_or_emotion",
                        "expected_emotion": dominant,
                        "final_expression": "",
                        "model": emotion_extractor._vlm_model_path,
                    },
                }
                skipped_frames += 1
            else:
                vlm_attrs = emotion_extractor.verify_existing_emotion(resolve_repo_path(face_orig_path), dominant)
                frame_attrs["expression"] = {**(expression or {}), **vlm_attrs}

        if "body_pose" in feature_names and body_pose_extractor is not None:
            frame_attrs.update(body_pose_extractor.extract(context))

        quality_updated_entries = 0
        recovery_record = None
        if recovery_lookup:
            recovery_record = recovery_lookup.get((
                str(context.get("shot_key")),
                str(context.get("obj_id")),
            ))
        if "mask_hole_quality" in feature_names:
            quality_updated_entries += update_group_mask_hole_quality(
                group["entries"],
                threshold=config.mask_hole_threshold,
                recovery_record=recovery_record,
                overwrite_existing_quality=bool(getattr(config, "quality_update_overwrite", True)),
            )
        if "face_boundary_quality" in feature_names:
            if face_boundary_checker is None:
                face_boundary_checker = build_face_boundary_quality_checker(config, force=True)
            if face_boundary_checker is not None:
                quality_updated_entries += update_group_face_boundary_quality(
                    group["entries"],
                    checker=face_boundary_checker,
                    recovery_record=recovery_record,
                    reuse_existing_bbox=not bool(getattr(config, "face_quality_recompute_bbox", False)),
                    overwrite_existing_quality=bool(getattr(config, "quality_update_overwrite", True)),
                )
        if "face_occlusion_quality" in feature_names or "image_clarity_quality" in feature_names:
            if face_quality_vlm_checker is None:
                face_quality_vlm_checker = build_face_quality_vlm_checker(
                    config,
                    check_occlusion="face_occlusion_quality" in feature_names,
                    check_clarity="image_clarity_quality" in feature_names,
                    check_clarity_vlm=bool(getattr(config, "enable_image_clarity_vlm_check", True)),
                )
            if face_quality_vlm_checker is not None:
                quality_updated_entries += update_group_face_quality_vlm(
                    group["entries"],
                    checker=face_quality_vlm_checker,
                    overwrite_existing_quality=bool(getattr(config, "quality_update_overwrite", True)),
                )

        if not frame_attrs and not quality_updated_entries:
            skipped_frames += 1
            continue

        frame_attr_updated_entries = 0
        if frame_attrs:
            for entry in group["entries"]:
                for feature_name, feature_value in frame_attrs.items():
                    if feature_name == "expression" and "emotion" in feature_names:
                        entry["expression"] = feature_value
                    elif feature_name == "expression" and isinstance(entry.get("expression"), dict) and isinstance(feature_value, dict):
                        entry["expression"].update(feature_value)
                    else:
                        entry[feature_name] = feature_value
                frame_attr_updated_entries += 1
        updated_entries += max(frame_attr_updated_entries, quality_updated_entries)
        updated_frames += 1

    enabled_features = index_data.get("enabled_features")
    if isinstance(enabled_features, list):
        for feature_name in feature_names:
            if feature_name not in enabled_features:
                enabled_features.append(feature_name)

    with open(index_path, "w", encoding="utf-8") as file:
        json.dump(
            to_jsonable(relativize_index_output(index_data)),
            file,
            ensure_ascii=False,
            indent=2,
        )

    return updated_frames, skipped_frames, updated_entries


# ---- v3 分发逻辑 ----
# 目录结构：identity_root / <video> / identity_matching / {output.jsonl, person_clusters/<person_id>}
#   即 video 与 person_clusters 之间还有一层 identity_matching（含该 video 的 output.jsonl）。
# video_base_dir() 会自动探测这一中间层，并兼容没有它的旧结构。
PERSON_CLUSTERS_SUBDIR = "person_clusters"
IDENTITY_SUBDIR = "identity_matching"   # video 与 person_clusters 之间的中间层目录名


def list_person_dirs(person_clusters_dir: str) -> List[str]:
    """列出 person_clusters_dir 下所有 person 子目录名（即 person_id），按名称排序保证多卡一致。"""
    names = []
    if not os.path.isdir(person_clusters_dir):
        return names
    for name in sorted(os.listdir(person_clusters_dir)):
        if os.path.isdir(os.path.join(person_clusters_dir, name)):
            names.append(name)
    return names


def video_base_dir(identity_root: str, video: str) -> str:
    """
    返回某 video 下「含 output.jsonl 与 person_clusters」的基目录。
    优先新结构 <root>/<video>/identity_matching；若该层下没有 person_clusters，则回退 <root>/<video>。
    （video 为 "" 时 os.path.join 自动忽略该层，可表示单 video / 旧结构。）
    """
    cand = os.path.join(identity_root, video, IDENTITY_SUBDIR)
    if os.path.isdir(os.path.join(cand, PERSON_CLUSTERS_SUBDIR)):
        return cand
    return os.path.join(identity_root, video)


def enumerate_units(identity_root: str) -> List[Tuple[str, str]]:
    """
    枚举所有处理单元 (video, person_id)。

    主结构：identity_root/<video>/identity_matching/person_clusters/<person_id>。
    兼容旧结构（视为单 video，video 名为 ""）：
      - identity_root/person_clusters/<person_id>
      - identity_root/identity_matching/person_clusters/<person_id>
    """
    # 旧结构兼容：identity_root[/identity_matching]/person_clusters/<person_id>
    for direct in (
        os.path.join(identity_root, PERSON_CLUSTERS_SUBDIR),
        os.path.join(identity_root, IDENTITY_SUBDIR, PERSON_CLUSTERS_SUBDIR),
    ):
        if os.path.isdir(direct):
            return [("", person_id) for person_id in list_person_dirs(direct)]

    # 新结构：逐 video -> identity_matching -> person
    units: List[Tuple[str, str]] = []
    for video in sorted(os.listdir(identity_root)):
        if not os.path.isdir(os.path.join(identity_root, video)):
            continue
        person_clusters_dir = os.path.join(video_base_dir(identity_root, video), PERSON_CLUSTERS_SUBDIR)
        for person_id in list_person_dirs(person_clusters_dir):
            units.append((video, person_id))
    return units


def unit_dirs(
    identity_root: str,
    video: str,
    person_id: str,
    output_dir: Optional[str],
) -> Tuple[str, str]:
    """返回某单元的 (输入聚类目录, 输出目录)。

    输入按 video_base_dir 定位（含 identity_matching 中间层）。
    v3 输出策略：若指定 output_dir，则在 output_dir 下镜像 input_cluster_dir
    相对 identity_root 的目录结构，只写 post-process json，不复制原始文件。
    """
    input_cluster_dir = os.path.join(video_base_dir(identity_root, video), PERSON_CLUSTERS_SUBDIR, person_id)
    if output_dir:
        relative_cluster_dir = os.path.relpath(input_cluster_dir, identity_root)
        target_dir = os.path.join(output_dir, relative_cluster_dir)
    else:
        target_dir = input_cluster_dir
    return input_cluster_dir, target_dir


def compute_shard_range(total: int, rank: int, total_rank: int) -> Tuple[int, int]:
    """
    把长度为 total 的列表按 rank 连续切片，返回本 rank 负责的 [st, en)。

    - total_rank <= 0 或 rank 非法时不分片，处理全部 [0, total)；
    - 余数均摊给前面的 rank，保证各分片长度最多相差 1，且无重叠、无遗漏。
    """
    if total_rank <= 0 or not (0 <= rank < total_rank):
        return 0, total
    base = total // total_rank          # 每个 rank 至少处理的数量
    remainder = total % total_rank      # 多出来的若干个，分给前 remainder 个 rank
    st = rank * base + min(rank, remainder)
    en = st + base + (1 if rank < remainder else 0)
    return st, en


class IndexAddPipeline:
    def __init__(self, config):
        self.config = config

    def run(self) -> Tuple[int, int, int, int, int]:
        """
        Returns:
            total_units, written, skipped, total_frames, total_images
        （total_units 为所有 (video, person) 处理单元数）
        """
        config = self.config

        if config.unit_list_file:
            return self._run_unit_list(config)

        # ---- 1. 解析路径 ----
        identity_root = resolve_repo_path(config.path)
        output_dir = resolve_repo_path(config.output_dir) if config.output_dir else None
        # identity jsonl 是「每个 video 一份」：取 basename，在各 video 目录下分别查找
        identity_name = os.path.basename(config.identity_jsonl) if config.identity_jsonl else "output.jsonl"
        # recovery jsonl 是「单一全局文件」（上游 one_shot 输出，含全部成员），直接整路径解析，所有 video 共用
        recovery_jsonl = resolve_repo_path(config.member_recovery_jsonl) if config.member_recovery_jsonl else None

        if not os.path.isdir(identity_root):
            raise FileNotFoundError(f"identity_matching root does not exist: {identity_root}")

        # ---- 2. 解析运行模式：有 update_features -> 增量更新；否则 -> 构建 ----
        update_features = list(config.update_features or [])
        if config.update_emotion_vlm_only and "emotion_vlm" not in update_features:
            update_features.append("emotion_vlm")
        update_features = [str(name).strip().lower() for name in update_features if str(name).strip()]
        is_update = bool(update_features)

        # ---- 3. 枚举 (video, person) 单元，按模式筛出「待处理列表」 ----
        #   构建模式：剔除「已存在结果且未 overwrite」的单元（已处理则跳过）；
        #   增量模式：只保留「已存在结果」的单元（缺索引无法更新则跳过）。
        all_units = enumerate_units(identity_root)
        pending_units: List[Tuple[str, str]] = []
        skipped = 0
        for video, person_id in all_units:
            _, target_dir = unit_dirs(identity_root, video, person_id, output_dir)
            output_path = os.path.join(target_dir, config.output_filename)
            exists = os.path.isfile(output_path)
            if is_update:
                if not exists:
                    skipped += 1
                    continue
            else:
                if exists and not config.overwrite:
                    skipped += 1
                    continue
            pending_units.append((video, person_id))

        # ---- 4. 对「待处理列表」按 rank 切出连续区间 [st, en) ----
        st, en = compute_shard_range(len(pending_units), config.rank, config.total_rank)
        my_units = pending_units[st:en]
        tqdm.write(
            f"[AfterPipelineV3] mode={'update' if is_update else 'build'}, "
            f"total={len(all_units)}, skipped={skipped}, pending={len(pending_units)}, "
            f"rank={config.rank}/{config.total_rank}, shard=[{st}:{en}] -> my_units={len(my_units)}"
        )
        if not my_units:
            tqdm.write("[AfterPipelineV3] No unit to process for this rank.")
            return len(all_units), 0, skipped, 0, 0

        if is_update:
            return self._run_update(config, identity_root, output_dir, my_units, update_features, all_units, skipped)
        return self._run_build(
            config, identity_root, output_dir, my_units, identity_name, recovery_jsonl, all_units, skipped,
        )

    def _read_unit_list(self, path: str) -> List[Tuple[str, str, str, str]]:
        units = []
        with open(path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) < 1 or not parts[0]:
                    tqdm.write(f"[AfterPipelineV3] Skip invalid unit_list line {line_number}: {line}")
                    continue
                src_person_clusters = parts[0]
                video = parts[1] if len(parts) > 1 else ""
                part = parts[2] if len(parts) > 2 else ""
                uuid = parts[3] if len(parts) > 3 else ""
                units.append((src_person_clusters, video, part, uuid))
        return units

    def _unit_output_root(self, src_uuid_dir: str, output_base_dir: Optional[str], input_base_dir: Optional[str]) -> Optional[str]:
        if not output_base_dir:
            return None
        if input_base_dir:
            try:
                rel_uuid_path = os.path.relpath(src_uuid_dir, input_base_dir)
            except ValueError:
                rel_uuid_path = os.path.basename(src_uuid_dir)
            if rel_uuid_path != "." and not rel_uuid_path.startswith(".."):
                return os.path.join(output_base_dir, rel_uuid_path)
        return os.path.join(output_base_dir, os.path.basename(src_uuid_dir))

    def _run_unit_list(self, config) -> Tuple[int, int, int, int, int]:
        """Batch mode: one Python process handles many UUID roots from a shard txt.

        This keeps model extractors alive across UUIDs in the same process and avoids
        reloading Qwen/body/emotion models for every directory.
        """
        unit_list_file = resolve_repo_path(config.unit_list_file)
        output_base_dir = resolve_repo_path(config.output_dir) if config.output_dir else None
        input_base_dir = (
            resolve_repo_path(config.unit_list_input_base_dir)
            if config.unit_list_input_base_dir
            else None
        )
        identity_name = os.path.basename(config.identity_jsonl) if config.identity_jsonl else "output.jsonl"
        recovery_jsonl = resolve_repo_path(config.member_recovery_jsonl) if config.member_recovery_jsonl else None
        recovery_records = (
            [resolve_path_fields(record) for record in read_jsonl(recovery_jsonl)]
            if recovery_jsonl and os.path.isfile(recovery_jsonl)
            else []
        )
        if config.member_recovery_jsonl and not recovery_records:
            tqdm.write(f"[AfterPipelineV3] WARNING: member_recovery_jsonl not found or empty: {recovery_jsonl}")
        recovery_lookup = {member_key(record): record for record in recovery_records}

        update_features = list(config.update_features or [])
        if config.update_emotion_vlm_only and "emotion_vlm" not in update_features:
            update_features.append("emotion_vlm")
        update_features = [str(name).strip().lower() for name in update_features if str(name).strip()]
        is_update = bool(update_features)
        include_pose = not config.disable_pose
        feature_extractors = {} if is_update else build_feature_extractors(config)
        face_boundary_quality_checker = None if is_update else build_face_boundary_quality_checker(config)
        face_quality_vlm_checker = None if is_update else build_face_quality_vlm_checker(config)
        update_extractors = build_update_feature_extractors(update_features, config) if is_update else {}

        src_units = self._read_unit_list(unit_list_file)
        if not src_units:
            tqdm.write(f"[AfterPipelineV3] No units found in unit_list_file: {unit_list_file}")
            return 0, 0, 0, 0, 0

        written = skipped = 0
        total_frames = total_images = 0
        total_person_units = 0
        progress = tqdm(src_units, desc="uuid_roots", unit="uuid", position=0)
        for src_person_clusters, video_name, part_name, uuid_name in progress:
            src_person_clusters = resolve_repo_path(src_person_clusters)
            src_uuid_dir = os.path.dirname(src_person_clusters)
            dst_uuid_dir = self._unit_output_root(src_uuid_dir, output_base_dir, input_base_dir)
            progress.set_postfix_str(f"{video_name}/{part_name}/{uuid_name}", refresh=False)

            if not os.path.isdir(src_person_clusters):
                tqdm.write(f"[AfterPipelineV3] Skip missing person_clusters: {src_person_clusters}")
                skipped += 1
                continue
            identity_jsonl = os.path.join(src_uuid_dir, identity_name)
            if not is_update and not os.path.isfile(identity_jsonl):
                tqdm.write(f"[AfterPipelineV3] Skip UUID (missing identity jsonl): {identity_jsonl}")
                skipped += 1
                continue

            persons = list_person_dirs(src_person_clusters)
            if config.person_ids:
                wanted = set(config.person_ids)
                persons = [person_id for person_id in persons if person_id in wanted]
            total_person_units += len(persons)
            if not persons:
                continue

            grouped: Dict[str, List[dict]] = {}
            if not is_update:
                identity_records = [resolve_path_fields(record) for record in read_jsonl(identity_jsonl)]
                grouped = load_all_cluster_members(
                    src_person_clusters,
                    identity_records,
                    recovery_records,
                    person_ids=set(persons),
                )

            for person_id in persons:
                input_cluster_dir = os.path.join(src_person_clusters, person_id)
                target_uuid_dir = dst_uuid_dir or src_uuid_dir
                target_dir = os.path.join(target_uuid_dir, PERSON_CLUSTERS_SUBDIR, person_id)
                output_path = os.path.join(target_dir, config.output_filename)

                if is_update:
                    if not os.path.isfile(output_path):
                        skipped += 1
                        continue
                    updated_frames, skipped_frames, updated_entries = update_existing_index_features(
                        output_path,
                        update_features,
                        config,
                        update_extractors=update_extractors,
                        progress_position=1,
                        recovery_lookup=recovery_lookup,
                    )
                    total_frames += updated_frames
                    total_images += updated_entries
                    written += 1
                    tqdm.write(
                        f"[AfterPipelineV3] {video_name}/{part_name}/{uuid_name}/{person_id}: "
                        f"update_features={','.join(update_features)} updated_frames={updated_frames}, "
                        f"skipped_frames={skipped_frames}, updated_entries={updated_entries} -> {output_path}"
                    )
                    continue

                if os.path.isfile(output_path) and not config.overwrite:
                    skipped += 1
                    continue

                os.makedirs(target_dir, exist_ok=True)
                member_records = grouped.get(person_id, [])
                person_index = build_person_index(
                    person_id,
                    input_cluster_dir,
                    member_records,
                    include_pose=include_pose,
                    feature_extractors=feature_extractors,
                    progress_position=1,
                    enable_mask_hole_quality_check=config.enable_mask_hole_quality_check,
                    mask_hole_threshold=config.mask_hole_threshold,
                    face_boundary_quality_checker=face_boundary_quality_checker,
                    face_quality_vlm_checker=face_quality_vlm_checker,
                )
                with open(output_path, "w", encoding="utf-8") as file:
                    json.dump(
                        to_jsonable(relativize_index_output(person_index)),
                        file,
                        ensure_ascii=False,
                        indent=2,
                    )

                stats = person_index["stats"]
                total_frames += stats["frame_count"]
                total_images += stats["image_count"]
                written += 1
                tqdm.write(
                    f"[AfterPipelineV3] {video_name}/{part_name}/{uuid_name}/{person_id}: "
                    f"members={stats['member_count']}, frames={stats['frame_count']}, "
                    f"images={stats['image_count']}, features={','.join(person_index['enabled_features'])} "
                    f"-> {output_path}"
                )

        tqdm.write(
            f"[AfterPipelineV3] Unit-list {'update' if is_update else 'build'} done. "
            f"total={total_person_units}, written={written}, skipped={skipped}, "
            f"frames={total_frames}, images={total_images}"
        )
        return total_person_units, written, skipped, total_frames, total_images

    def _run_build(
        self, config, identity_root, output_dir, my_units, identity_name, recovery_jsonl,
        all_units, skipped,
    ) -> Tuple[int, int, int, int, int]:
        """构建模式：按 video 各读自己的 jsonl，扫聚类图片、抽特征，写回新索引。"""
        include_pose = not config.disable_pose
        feature_extractors = build_feature_extractors(config)
        face_boundary_quality_checker = build_face_boundary_quality_checker(config)

        # recovery 为单一全局文件，只读一次，供所有 video 共用（用于补回 cluster_meta 里
        # 但不在 identity jsonl 中的成员）
        recovery_records = (
            [resolve_path_fields(record) for record in read_jsonl(recovery_jsonl)]
            if recovery_jsonl and os.path.isfile(recovery_jsonl)
            else []
        )
        if config.member_recovery_jsonl and not recovery_records:
            tqdm.write(f"[AfterPipelineV3] WARNING: member_recovery_jsonl not found or empty: {recovery_jsonl}")

        # 把本 rank 的单元按 video 分组，使每个 video 的 identity jsonl 只读一次
        units_by_video: Dict[str, List[str]] = defaultdict(list)
        for video, person_id in my_units:
            units_by_video[video].append(person_id)

        written = 0
        total_frames = total_images = 0
        for video in sorted(units_by_video):
            persons = units_by_video[video]
            base = video_base_dir(identity_root, video)
            person_clusters_dir = os.path.join(base, PERSON_CLUSTERS_SUBDIR)
            identity_jsonl = os.path.join(base, identity_name)

            if not os.path.isfile(identity_jsonl):
                tqdm.write(f"[AfterPipelineV3] Skip video (missing identity jsonl): {identity_jsonl}")
                continue

            # 仅为本 video 的待处理 person 加载聚类成员（person_ids 过滤）
            identity_records = [resolve_path_fields(record) for record in read_jsonl(identity_jsonl)]
            grouped = load_all_cluster_members(
                person_clusters_dir,
                identity_records,
                recovery_records,
                person_ids=set(persons),
            )

            progress = tqdm(persons, desc=f"{video or 'root'} persons", unit="person", position=0, leave=False)
            for person_id in progress:
                progress.set_postfix_str(person_id, refresh=False)
                input_cluster_dir, target_dir = unit_dirs(identity_root, video, person_id, output_dir)
                os.makedirs(target_dir, exist_ok=True)
                output_path = os.path.join(target_dir, config.output_filename)

                member_records = grouped.get(person_id, [])
                person_index = build_person_index(
                    person_id,
                    input_cluster_dir,
                    member_records,
                    include_pose=include_pose,
                    feature_extractors=feature_extractors,
                    progress_position=1,
                    enable_mask_hole_quality_check=config.enable_mask_hole_quality_check,
                    mask_hole_threshold=config.mask_hole_threshold,
                    face_boundary_quality_checker=face_boundary_quality_checker,
                    face_quality_vlm_checker=face_quality_vlm_checker,
                )
                with open(output_path, "w", encoding="utf-8") as file:
                    json.dump(
                        to_jsonable(relativize_index_output(person_index)),
                        file,
                        ensure_ascii=False,
                        indent=2,
                    )

                stats = person_index["stats"]
                total_frames += stats["frame_count"]
                total_images += stats["image_count"]
                written += 1
                tqdm.write(
                    f"[AfterPipelineV3] {video}/{person_id}: members={stats['member_count']}, "
                    f"frames={stats['frame_count']}, images={stats['image_count']}, "
                    f"features={','.join(person_index['enabled_features'])} -> {output_path}"
                )

        tqdm.write(
            f"[AfterPipelineV3] Build done. total={len(all_units)}, written={written}, "
            f"skipped={skipped}, frames={total_frames}, images={total_images}"
        )
        return len(all_units), written, skipped, total_frames, total_images

    def _run_update(
        self, config, identity_root, output_dir, my_units, update_features, all_units, skipped,
    ) -> Tuple[int, int, int, int, int]:
        """增量模式：对已建好的索引补加 emotion / emotion_vlm / body_pose，原地写回。"""
        update_extractors = build_update_feature_extractors(update_features, config)
        recovery_jsonl = resolve_repo_path(config.member_recovery_jsonl) if config.member_recovery_jsonl else None
        recovery_records = (
            [resolve_path_fields(record) for record in read_jsonl(recovery_jsonl)]
            if recovery_jsonl and os.path.isfile(recovery_jsonl)
            else []
        )
        recovery_lookup = {member_key(record): record for record in recovery_records}
        if config.member_recovery_jsonl and not recovery_lookup:
            tqdm.write(f"[AfterPipelineV3] WARNING: member_recovery_jsonl not found or empty: {recovery_jsonl}")

        written = 0
        total_frames = total_images = 0
        progress = tqdm(my_units, desc="persons", unit="person", position=0)
        for video, person_id in progress:
            progress.set_postfix_str(f"{video}/{person_id}", refresh=False)
            _, target_dir = unit_dirs(identity_root, video, person_id, output_dir)
            output_path = os.path.join(target_dir, config.output_filename)

            updated_frames, skipped_frames, updated_entries = update_existing_index_features(
                output_path,
                update_features,
                config,
                update_extractors=update_extractors,
                progress_position=1,
                recovery_lookup=recovery_lookup,
            )
            total_frames += updated_frames
            total_images += updated_entries
            written += 1
            tqdm.write(
                f"[AfterPipelineV3] {video}/{person_id}: update_features={','.join(update_features)} "
                f"updated_frames={updated_frames}, skipped_frames={skipped_frames}, "
                f"updated_entries={updated_entries} -> {output_path}"
            )

        tqdm.write(
            f"[AfterPipelineV3] Update done. total={len(all_units)}, written={written}, "
            f"skipped={skipped}, frames={total_frames}, images={total_images}"
        )
        return len(all_units), written, skipped, total_frames, total_images

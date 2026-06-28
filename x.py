#!/usr/bin/env python3
"""
After-pipeline index builder (v2) —— 自包含版本，不依赖 index_add.py。

分发逻辑（与旧版的差异）：
  1. 直接列出 person_clusters_dir 下的 person 目录列表；
  2. 先剔除「已处理且未开启 overwrite」的 person；
  3. 对剩余列表按 rank / total_rank 切出连续区间 [st, en)，得到本 rank 应处理的 person；
  4. 逐个构建索引，结果写回每个 person 目录下的 post-process json。
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
                print(f"[AfterPipelineV2] Skip invalid JSON at line {line_number}: {exc}")
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
                    f"[AfterPipelineV2] Missing recovery record for "
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
) -> dict:
    enabled_features = []
    if include_pose:
        enabled_features.append("pose")
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
        f"[AfterPipelineV2] {person_id}: pose/image mapping done, "
        f"frames={output['stats']['frame_count']}, "
        f"images={output['stats']['image_count']}"
    )

    for feature_name, extractor in feature_extractors.items():
        total = len(frame_contexts)
        tqdm.write(f"[AfterPipelineV2] {person_id}: start {feature_name}, total_frames={total}")
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
        tqdm.write(f"[AfterPipelineV2] {person_id}: {feature_name} done")

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
    return extractors


def update_existing_index_features(
    index_path: str,
    feature_names: List[str],
    config: Any,
    update_extractors: Optional[Dict[str, Any]] = None,
    progress_position: int = 1,
) -> Tuple[int, int, int]:
    feature_names = [str(name).strip().lower() for name in feature_names if str(name).strip()]
    valid_features = {"emotion", "emotion_vlm", "body_pose"}
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

        if not frame_attrs:
            skipped_frames += 1
            continue

        for entry in group["entries"]:
            for feature_name, feature_value in frame_attrs.items():
                if feature_name == "expression" and "emotion" in feature_names:
                    entry["expression"] = feature_value
                elif feature_name == "expression" and isinstance(entry.get("expression"), dict) and isinstance(feature_value, dict):
                    entry["expression"].update(feature_value)
                else:
                    entry[feature_name] = feature_value
            updated_entries += 1
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


# ---- v2 分发逻辑 ----
# 目录结构：identity_root / <video> / person_clusters / <person_id>
# 每个 video 目录下各有一份 identity 输出 jsonl（默认 output.jsonl）。
PERSON_CLUSTERS_SUBDIR = "person_clusters"


def list_person_dirs(person_clusters_dir: str) -> List[str]:
    """列出 person_clusters_dir 下所有 person 子目录名（即 person_id），按名称排序保证多卡一致。"""
    names = []
    if not os.path.isdir(person_clusters_dir):
        return names
    for name in sorted(os.listdir(person_clusters_dir)):
        if os.path.isdir(os.path.join(person_clusters_dir, name)):
            names.append(name)
    return names


def enumerate_units(identity_root: str) -> List[Tuple[str, str]]:
    """
    枚举所有处理单元 (video, person_id)。

    主结构：identity_root/<video>/person_clusters/<person_id>。
    兼容旧结构：若 identity_root 下直接有 person_clusters，则视为单 video（video 名为 ""）。
    """
    # 旧结构兼容：identity_root/person_clusters/<person_id>
    direct_pcd = os.path.join(identity_root, PERSON_CLUSTERS_SUBDIR)
    if os.path.isdir(direct_pcd):
        return [("", person_id) for person_id in list_person_dirs(direct_pcd)]

    # 新结构：逐 video -> person
    units: List[Tuple[str, str]] = []
    for video in sorted(os.listdir(identity_root)):
        video_dir = os.path.join(identity_root, video)
        if not os.path.isdir(video_dir):
            continue
        person_clusters_dir = os.path.join(video_dir, PERSON_CLUSTERS_SUBDIR)
        for person_id in list_person_dirs(person_clusters_dir):
            units.append((video, person_id))
    return units


def unit_dirs(
    identity_root: str,
    video: str,
    person_id: str,
    output_dir: Optional[str],
) -> Tuple[str, str]:
    """返回某单元的 (输入聚类目录, 输出目录)。video 为 "" 时 os.path.join 会自动忽略该层。"""
    input_cluster_dir = os.path.join(identity_root, video, PERSON_CLUSTERS_SUBDIR, person_id)
    target_dir = os.path.join(output_dir, video, person_id) if output_dir else input_cluster_dir
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
            f"[AfterPipelineV2] mode={'update' if is_update else 'build'}, "
            f"total={len(all_units)}, skipped={skipped}, pending={len(pending_units)}, "
            f"rank={config.rank}/{config.total_rank}, shard=[{st}:{en}] -> my_units={len(my_units)}"
        )
        if not my_units:
            tqdm.write("[AfterPipelineV2] No unit to process for this rank.")
            return len(all_units), 0, skipped, 0, 0

        if is_update:
            return self._run_update(config, identity_root, output_dir, my_units, update_features, all_units, skipped)
        return self._run_build(
            config, identity_root, output_dir, my_units, identity_name, recovery_jsonl, all_units, skipped,
        )

    def _run_build(
        self, config, identity_root, output_dir, my_units, identity_name, recovery_jsonl,
        all_units, skipped,
    ) -> Tuple[int, int, int, int, int]:
        """构建模式：按 video 各读自己的 jsonl，扫聚类图片、抽特征，写回新索引。"""
        include_pose = not config.disable_pose
        feature_extractors = build_feature_extractors(config)

        # recovery 为单一全局文件，只读一次，供所有 video 共用（用于补回 cluster_meta 里
        # 但不在 identity jsonl 中的成员）
        recovery_records = (
            [resolve_path_fields(record) for record in read_jsonl(recovery_jsonl)]
            if recovery_jsonl and os.path.isfile(recovery_jsonl)
            else []
        )
        if config.member_recovery_jsonl and not recovery_records:
            tqdm.write(f"[AfterPipelineV2] WARNING: member_recovery_jsonl not found or empty: {recovery_jsonl}")

        # 把本 rank 的单元按 video 分组，使每个 video 的 identity jsonl 只读一次
        units_by_video: Dict[str, List[str]] = defaultdict(list)
        for video, person_id in my_units:
            units_by_video[video].append(person_id)

        written = 0
        total_frames = total_images = 0
        for video in sorted(units_by_video):
            persons = units_by_video[video]
            video_dir = os.path.join(identity_root, video)
            person_clusters_dir = os.path.join(video_dir, PERSON_CLUSTERS_SUBDIR)
            identity_jsonl = os.path.join(video_dir, identity_name)

            if not os.path.isfile(identity_jsonl):
                tqdm.write(f"[AfterPipelineV2] Skip video (missing identity jsonl): {identity_jsonl}")
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
                    f"[AfterPipelineV2] {video}/{person_id}: members={stats['member_count']}, "
                    f"frames={stats['frame_count']}, images={stats['image_count']}, "
                    f"features={','.join(person_index['enabled_features'])} -> {output_path}"
                )

        tqdm.write(
            f"[AfterPipelineV2] Build done. total={len(all_units)}, written={written}, "
            f"skipped={skipped}, frames={total_frames}, images={total_images}"
        )
        return len(all_units), written, skipped, total_frames, total_images

    def _run_update(
        self, config, identity_root, output_dir, my_units, update_features, all_units, skipped,
    ) -> Tuple[int, int, int, int, int]:
        """增量模式：对已建好的索引补加 emotion / emotion_vlm / body_pose，原地写回。"""
        update_extractors = build_update_feature_extractors(update_features, config)

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
            )
            total_frames += updated_frames
            total_images += updated_entries
            written += 1
            tqdm.write(
                f"[AfterPipelineV2] {video}/{person_id}: update_features={','.join(update_features)} "
                f"updated_frames={updated_frames}, skipped_frames={skipped_frames}, "
                f"updated_entries={updated_entries} -> {output_path}"
            )

        tqdm.write(
            f"[AfterPipelineV2] Update done. total={len(all_units)}, written={written}, "
            f"skipped={skipped}, frames={total_frames}, images={total_images}"
        )
        return len(all_units), written, skipped, total_frames, total_images

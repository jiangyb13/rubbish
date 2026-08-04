#!/usr/bin/env python3
"""Locate the SAM2 face image corresponding to an identity angle-library image."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ANGLE_NAME_RE = re.compile(
    r"^(?P<shot_key>.+)_id(?P<obj_id>\d+)_frame(?P<frame_idx>\d+)(?:_yaw[+-]\d+(?:\.\d+)?)?\.[^.]+$"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def _resolve_workspace_path(path_value: str | None, workspace: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else workspace / path


def _infer_workspace_from_output_jsonl(output_jsonl: Path) -> Path:
    # .../<workspace>/outputs/<video>/identity_matching/output.jsonl
    for parent in output_jsonl.resolve().parents:
        if parent.name == "outputs":
            return parent.parent
    return output_jsonl.resolve().parents[2]


def _parse_angle_filename(angle_image: Path) -> dict[str, Any]:
    match = ANGLE_NAME_RE.match(angle_image.name)
    if not match:
        raise ValueError(
            "Cannot parse angle-library filename. Expected something like "
            "video15_shot_0004_id0_frame0002_yaw-000.0.png"
        )
    return {
        "shot_key": match.group("shot_key"),
        "obj_id": match.group("obj_id"),
        "frame_idx": int(match.group("frame_idx")),
    }


def _find_manifest_entry(angle_image: Path, workspace: Path) -> dict[str, Any] | None:
    parts = angle_image.resolve().parts
    if "face_angle_library" not in parts:
        return None
    angle_root_index = parts.index("face_angle_library")
    cluster_dir = Path(*parts[:angle_root_index])
    manifest_path = cluster_dir / "frame_manifest.jsonl"
    if not manifest_path.is_file():
        return None

    target = angle_image.resolve()
    for row in _read_jsonl(manifest_path):
        images = row.get("images") or {}
        for key in ("face_angle_left", "face_angle_front", "face_angle_right"):
            candidate = _resolve_workspace_path(images.get(key), workspace)
            if candidate and candidate.resolve() == target:
                return row
    return None


def _find_output_record(
    rows: list[dict[str, Any]], shot_key: str, obj_id: str
) -> dict[str, Any] | None:
    suffix = f"{shot_key}/id_{obj_id}"
    for row in rows:
        id_dir_path = str(row.get("id_dir_path") or "")
        if id_dir_path.endswith(suffix):
            return row
    return None


def locate_sam2(angle_image: Path, output_jsonl: Path, workspace: Path | None = None) -> dict[str, Any]:
    output_jsonl = output_jsonl.resolve()
    workspace = workspace.resolve() if workspace else _infer_workspace_from_output_jsonl(output_jsonl)
    angle_image = angle_image.resolve()

    output_rows = _read_jsonl(output_jsonl)
    manifest_entry = _find_manifest_entry(angle_image, workspace)
    identity = manifest_entry or _parse_angle_filename(angle_image)

    shot_key = str(identity["shot_key"])
    obj_id = str(identity["obj_id"])
    frame_idx = int(identity["frame_idx"])
    record = _find_output_record(output_rows, shot_key, obj_id)
    if record is None:
        raise LookupError(f"No output.jsonl record found for shot_key={shot_key}, obj_id={obj_id}")

    sam_face_white = _resolve_workspace_path(record.get("sam_id_face_white_dir_path"), workspace)
    sam_face_orig = _resolve_workspace_path(record.get("sam_id_face_orig_dir_path"), workspace)
    sam_full_mask = _resolve_workspace_path(record.get("sam_id_cropped_from_full_face_mask_dir_path"), workspace)
    sam_orig_mask = _resolve_workspace_path(record.get("sam_id_orig_face_mask_dir_path"), workspace)

    result = {
        "angle_image": str(angle_image),
        "person_id": identity.get("person_id") or record.get("identity_matching_person_id"),
        "shot_key": shot_key,
        "obj_id": obj_id,
        "frame_idx": frame_idx,
        "source_shot_frame_idx": identity.get("source_shot_frame_idx"),
        "id_dir_path": str(_resolve_workspace_path(record.get("id_dir_path"), workspace)),
        "identity_matching_images": (manifest_entry or {}).get("images"),
        "one_shot_images": (manifest_entry or {}).get("derived"),
        "sam2_face_white": str(sam_face_white / f"{frame_idx}.png") if sam_face_white else None,
        "sam2_face_orig": str(sam_face_orig / f"{frame_idx}.jpg") if sam_face_orig else None,
        "sam2_face_mask_for_full": str(sam_full_mask / f"{frame_idx}.npy") if sam_full_mask else None,
        "sam2_face_mask_for_orig": str(sam_orig_mask / f"{frame_idx}.npy") if sam_orig_mask else None,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--angle-image", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--workspace", type=Path, default=None)
    args = parser.parse_args()

    print(json.dumps(locate_sam2(args.angle_image, args.output_jsonl, args.workspace), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

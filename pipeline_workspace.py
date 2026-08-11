"""Canonical per-video workspace layout and batch-unit helpers."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from .config import PIPELINE_PATH_DEFAULTS


@dataclass(frozen=True)
class WorkspacePaths:
    video_dir: str

    @property
    def manifest(self) -> str:
        return self.path(PIPELINE_PATH_DEFAULTS["manifest"])

    @property
    def shot_dir(self) -> str:
        return self.path(PIPELINE_PATH_DEFAULTS["shot_dir"])

    @property
    def shot_video_dir(self) -> str:
        return self.path(PIPELINE_PATH_DEFAULTS["shot_video_dir"])

    @property
    def shot_jsonl(self) -> str:
        return self.path(PIPELINE_PATH_DEFAULTS["shot_jsonl"])

    @property
    def one_shot_dir(self) -> str:
        return self.path(PIPELINE_PATH_DEFAULTS["one_shot_dir"])

    @property
    def one_shot_jsonl(self) -> str:
        return self.path(PIPELINE_PATH_DEFAULTS["one_shot_jsonl"])

    @property
    def identity_dir(self) -> str:
        return self.path(PIPELINE_PATH_DEFAULTS["identity_dir"])

    @property
    def identity_jsonl(self) -> str:
        return self.path(PIPELINE_PATH_DEFAULTS["identity_jsonl"])

    @property
    def identity_simple_jsonl(self) -> str:
        return self.path(PIPELINE_PATH_DEFAULTS["identity_simple_jsonl"])

    @property
    def identity_global_json(self) -> str:
        return self.path(PIPELINE_PATH_DEFAULTS["identity_global_json"])

    @property
    def person_clusters_dir(self) -> str:
        return self.path(PIPELINE_PATH_DEFAULTS["person_clusters_dir"])

    @property
    def person_registry_jsonl(self) -> str:
        return self.path(PIPELINE_PATH_DEFAULTS["person_registry_jsonl"])

    @property
    def training_dir(self) -> str:
        return self.path(PIPELINE_PATH_DEFAULTS["training_dir"])

    @property
    def pairs_jsonl(self) -> str:
        return self.path(PIPELINE_PATH_DEFAULTS["pairs_jsonl"])

    @property
    def training_stats_json(self) -> str:
        return self.path(PIPELINE_PATH_DEFAULTS["training_stats_json"])

    @property
    def first_frame_dir(self) -> str:
        return self.path(PIPELINE_PATH_DEFAULTS["first_frame_dir"])

    def path(self, *parts: str) -> str:
        if len(parts) == 1:
            return os.path.join(self.video_dir, parts[0])
        return os.path.join(self.video_dir, *parts)

    def ensure(self) -> None:
        for directory in (
            self.video_dir,
            self.shot_video_dir,
            self.one_shot_dir,
            self.identity_dir,
            self.person_clusters_dir,
            self.training_dir,
            self.first_frame_dir,
        ):
            os.makedirs(directory, exist_ok=True)


def _safe_video_id(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value or "").strip())
    return text.strip("._") or "video"


def load_units(input_jsonl: str, output_root: str) -> List[Dict[str, str]]:
    """Load the single canonical Stage-1 task JSONL.

    Each row must contain ``video_path``. ``video_id`` is optional and defaults
    to the source filename stem. A caller may explicitly set ``video_dir``;
    otherwise it is ``output_root/video_id``.
    """
    units: List[Dict[str, str]] = []
    seen_ids = set()
    with open(os.path.abspath(input_jsonl), "r", encoding="utf-8") as handle:
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
            raw_id = record.get("video_id") or os.path.splitext(os.path.basename(video_path))[0]
            video_id = _safe_video_id(raw_id)
            if video_id in seen_ids:
                raise ValueError(f"Duplicate video_id in task JSONL: {video_id}")
            seen_ids.add(video_id)
            video_dir_value = record.get("video_dir")
            video_dir = (
                os.path.abspath(os.path.expanduser(str(video_dir_value)))
                if video_dir_value
                else os.path.abspath(os.path.join(output_root, video_id))
            )
            units.append({
                "video_id": video_id,
                "video_path": video_path,
                "video_dir": video_dir,
            })
    return units


def shard_range(total: int, rank: int, world_size: int) -> Tuple[int, int]:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    base, remainder = divmod(total, world_size)
    start = rank * base + min(rank, remainder)
    end = start + base + (1 if rank < remainder else 0)
    return start, end


def assigned_units(units: List[Dict[str, str]], rank: int, world_size: int) -> List[Dict[str, str]]:
    start, end = shard_range(len(units), rank, world_size)
    return units[start:end]


def units_from_config(config: Any) -> List[Dict[str, str]]:
    """Resolve batch or standalone units for one stage configuration."""
    video_dir_value = getattr(config, "video_dir", None)
    phase = int(getattr(config, "phase", 0))
    total = int(getattr(config, "total", 1))
    if video_dir_value:
        video_dir = os.path.abspath(os.path.expanduser(str(video_dir_value)))
        paths = WorkspacePaths(video_dir)
        manifest: Dict[str, Any] = {}
        if os.path.isfile(paths.manifest):
            with open(paths.manifest, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        video_path = getattr(config, "video_path", None) or manifest.get("source_video_path")
        if not video_path:
            raise ValueError(
                "Standalone workspace has no source_video_path; pass --video_path when initializing Stage 1."
            )
        unit = {
            "video_id": str(
                getattr(config, "video_id", None)
                or manifest.get("video_id")
                or os.path.basename(video_dir)
            ),
            "video_path": os.path.abspath(os.path.expanduser(str(video_path))),
            "video_dir": video_dir,
        }
        return assigned_units([unit], phase, total)

    task_jsonl = getattr(config, "pipeline_input_jsonl", None)
    if not task_jsonl:
        raise ValueError("Set --pipeline_input_jsonl for batch mode or --video_dir for standalone mode.")
    output_root = os.path.abspath(getattr(config, "output_root", "outputs"))
    return assigned_units(load_units(task_jsonl, output_root), phase, total)


def _atomic_json_dump(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".manifest.", suffix=".tmp", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def initialize_manifest(unit: Dict[str, str]) -> WorkspacePaths:
    paths = WorkspacePaths(unit["video_dir"])
    paths.ensure()
    if os.path.isfile(paths.manifest):
        return paths
    _atomic_json_dump(paths.manifest, {
        "schema_version": 1,
        "video_id": unit["video_id"],
        "video_dir": unit["video_dir"],
        "source_video_path": unit["video_path"],
        "stages": {},
    })
    return paths


def update_stage_manifest(
    unit: Dict[str, str],
    stage: str,
    status: str,
    outputs: Dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    paths = initialize_manifest(unit)
    try:
        with open(paths.manifest, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except Exception:
        manifest = {
            "schema_version": 1,
            "video_id": unit["video_id"],
            "video_dir": unit["video_dir"],
            "source_video_path": unit["video_path"],
            "stages": {},
        }
    stage_data: Dict[str, Any] = {"status": status}
    if outputs:
        stage_data.update(outputs)
    if error:
        stage_data["error"] = error
    manifest.setdefault("stages", {})[stage] = stage_data
    _atomic_json_dump(paths.manifest, manifest)


def count_jsonl(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for line in handle if line.strip())


def write_person_cluster_units(path: str, units: Iterable[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for unit in units:
            workspace = WorkspacePaths(unit["video_dir"])
            handle.write(
                f"{workspace.person_clusters_dir}|{unit['video_id']}||{unit['video_id']}\n"
            )

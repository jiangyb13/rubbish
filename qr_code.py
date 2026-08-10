#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path


IMAGE_TYPES = ("face_orig", "face_white", "full_orig", "full_white")


def parse_args():
    parser = argparse.ArgumentParser(description="Recompute quality_label in post_process_index.json files.")
    parser.add_argument("--video_dir", required=True, help="Video workspace root.")
    parser.add_argument("--index_name", default="post_process_index.json")
    parser.add_argument("--dry_run", action="store_true", help="Only print changes; do not write JSON files.")
    parser.add_argument("--backup", action="store_true", help="Write a .bak copy before modifying each changed JSON.")
    return parser.parse_args()


def person_clusters_dirs(video_dir: Path):
    candidates = [
        video_dir / "person_clusters",
        video_dir / "identity_matching" / "person_clusters",
    ]
    return [path for path in candidates if path.is_dir()]


def index_files(video_dir: Path, index_name: str):
    found = []
    for cluster_dir in person_clusters_dirs(video_dir):
        for person_dir in sorted(path for path in cluster_dir.iterdir() if path.is_dir()):
            index_path = person_dir / index_name
            if index_path.is_file():
                found.append(index_path)
    return found


def recompute_quality_label(entry: dict) -> bool:
    quality = entry.get("quality")
    if isinstance(quality, dict):
        for item in quality.values():
            if isinstance(item, dict) and item.get("passed") is False:
                return False
    return True


def update_index(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    images = data.get("images") if isinstance(data, dict) else {}
    if not isinstance(images, dict):
        return 0, 0, data

    checked = 0
    changed = 0
    for image_type in IMAGE_TYPES:
        entries = images.get(image_type) or {}
        if not isinstance(entries, dict):
            continue
        for entry in entries.values():
            if not isinstance(entry, dict):
                continue
            old_value = bool(entry.get("quality_label", True))
            new_value = recompute_quality_label(entry)
            checked += 1
            if old_value != new_value:
                entry["quality_label"] = new_value
                changed += 1
    return checked, changed, data


def main():
    args = parse_args()
    video_dir = Path(os.path.expanduser(args.video_dir)).resolve()
    files = index_files(video_dir, args.index_name)
    if not files:
        raise SystemExit(f"No {args.index_name} found under {video_dir}/person_clusters or identity_matching/person_clusters")

    total_checked = 0
    total_changed = 0
    changed_files = 0

    for path in files:
        checked, changed, data = update_index(path)
        total_checked += checked
        total_changed += changed
        if changed:
            changed_files += 1
            print(f"[fix_quality_labels] {path}: changed={changed}, checked={checked}")
            if not args.dry_run:
                if args.backup:
                    backup_path = path.with_suffix(path.suffix + ".bak")
                    if not backup_path.exists():
                        backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
                with path.open("w", encoding="utf-8") as handle:
                    json.dump(data, handle, ensure_ascii=False, indent=2)

    mode = "dry_run" if args.dry_run else "written"
    print(
        f"[fix_quality_labels] done mode={mode}, files={len(files)}, "
        f"changed_files={changed_files}, checked_entries={total_checked}, changed_entries={total_changed}"
    )


if __name__ == "__main__":
    main()

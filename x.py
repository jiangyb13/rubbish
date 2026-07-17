import argparse
import json
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSONL = PROJECT_ROOT / "outputs_demo" / "training_pairs" / "pairs.jsonl"
DEFAULT_PERSON_CLUSTERS = PROJECT_ROOT / "outputs_demo" / "identity_matching" / "person_clusters"
DEFAULT_FILTED_POOL = PROJECT_ROOT / "filted_pool"


def resolve_repo_path(path):
    if not path:
        return None
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return Path(path).resolve()
    return (PROJECT_ROOT / path).resolve()


def repo_relative(path):
    if not path:
        return None
    resolved = resolve_repo_path(path)
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def path_info(path):
    if not path:
        return None
    raw = str(path)
    resolved = resolve_repo_path(raw)
    rel = None
    try:
        rel = str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        rel = raw if not os.path.isabs(raw) else None
    return {
        "raw": raw,
        "abs": str(resolved),
        "rel": rel,
        "exists": bool(resolved.exists()),
        "is_absolute": os.path.isabs(os.path.expanduser(raw)),
    }


def add_path_info(row):
    paths = []

    def add(value):
        if isinstance(value, str) and value:
            paths.append(value)
        elif isinstance(value, list):
            for item in value:
                add(item)

    for key in (
        "first_frame",
        "target_video",
        "angle_ref",
        "emo_ref",
        "body_pose_ref",
        "angle_ref_white",
        "emo_ref_white",
        "body_pose_ref_white",
    ):
        add(row.get(key))
    for meta_key in ("angle_ref_meta", "emo_ref_meta", "body_pose_ref_meta"):
        for meta in row.get(meta_key) or []:
            if isinstance(meta, dict):
                add(meta.get("path"))
                add(meta.get("white_path"))
                add(meta.get("feature_path"))
    row["_path_info"] = {path: path_info(path) for path in sorted(set(paths))}
    return row


def add_pool_path_info(row):
    paths = []

    def add(value):
        if isinstance(value, str) and value:
            paths.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                add(item)

    for key in (
        "pool_image_path",
        "image_path",
        "identity_matching_image_path",
        "one_shot_image_path",
        "source_shot_path",
    ):
        add(row.get(key))
    add(row.get("related_images"))
    add(row.get("related_identity_matching_images"))
    add(row.get("related_one_shot_images"))
    row["_path_info"] = {path: path_info(path) for path in sorted(set(paths))}
    return row


def read_jsonl(path):
    rows = []
    if not path or not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def iter_jsonl(path):
    if not path or not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def person_dirs(root):
    if not root or not root.exists():
        return []
    return [
        path
        for path in sorted(root.iterdir())
        if path.is_dir() and path.name.startswith("person_")
    ]


PAIRWISE_MAX_KEYS = (
    "angle_max_pairwise_cosine",
    "emo_max_pairwise_cosine",
    "body_pose_max_pairwise_cosine",
)


def parse_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def row_max_pairwise_similarity(row):
    stats = row.get("selection_stats") or {}
    values = []
    for key in PAIRWISE_MAX_KEYS:
        value = parse_float(stats.get(key))
        if value is not None:
            values.append(value)
    return max(values) if values else None


def pass_filter(row, flag_filter, person_id, max_similarity_threshold=None):
    if person_id and row.get("person_id") != person_id:
        return False
    if max_similarity_threshold is not None:
        max_similarity = row_max_pairwise_similarity(row)
        if max_similarity is not None and max_similarity >= max_similarity_threshold:
            return False
    if flag_filter == "has_angle":
        return bool(row.get("angle_ref"))
    if flag_filter == "has_emo":
        return bool(row.get("emo_ref"))
    if flag_filter == "has_body_pose":
        return bool(row.get("body_pose_ref"))
    return True


def parse_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def make_app(pairs_jsonl):
    app = Flask(__name__)
    app.config["PAIRS_JSONL"] = resolve_repo_path(pairs_jsonl)
    app.config["PERSON_CLUSTERS_DIR"] = DEFAULT_PERSON_CLUSTERS
    app.config["FILTED_POOL_DIR"] = DEFAULT_FILTED_POOL

    @app.route("/")
    def index():
        return render_template(
            "index.html",
            pairs_jsonl=repo_relative(app.config["PAIRS_JSONL"]),
        )

    @app.route("/api/summary")
    def api_summary():
        pairs_path = app.config["PAIRS_JSONL"]
        payload = {
            "pairs_jsonl": repo_relative(pairs_path),
            "total": None,
            "persons": [],
            "person_count": None,
            "angle_refs": None,
            "emo_refs": None,
            "body_pose_refs": None,
            "summary_is_fast": True,
        }
        if request.args.get("full") == "1":
            rows = read_jsonl(pairs_path)
            persons = sorted({row.get("person_id") for row in rows if row.get("person_id")})
            payload.update(
                {
                    "total": len(rows),
                    "persons": persons,
                    "person_count": len(persons),
                    "angle_refs": sum(len(row.get("angle_ref") or []) for row in rows),
                    "emo_refs": sum(len(row.get("emo_ref") or []) for row in rows),
                    "body_pose_refs": sum(len(row.get("body_pose_ref") or []) for row in rows),
                    "summary_is_fast": False,
                }
            )
        return jsonify(payload)

    @app.route("/api/pairs")
    def api_pairs():
        pairs_path = app.config["PAIRS_JSONL"]
        flag_filter = request.args.get("flag", "all")
        person_id = request.args.get("person_id", "")
        max_similarity_threshold = parse_float(request.args.get("max_similarity_threshold", ""))
        offset = max(0, parse_int(request.args.get("offset", 0), 0))
        limit = max(1, min(200, parse_int(request.args.get("limit", 30), 30)))

        matched = 0
        page = []
        has_next = False
        total = None
        for row in iter_jsonl(pairs_path):
            if not pass_filter(row, flag_filter, person_id, max_similarity_threshold):
                continue
            if matched < offset:
                matched += 1
                continue
            if len(page) < limit:
                page.append(row)
                matched += 1
                continue
            has_next = True
            break
        else:
            total = matched

        return jsonify(
            {
                "pairs_jsonl": repo_relative(pairs_path),
                "total": total,
                "total_known": total is not None,
                "has_next": has_next,
                "offset": offset,
                "limit": limit,
                "max_similarity_threshold": max_similarity_threshold,
                "rows": [add_path_info(dict(row)) for row in page],
            }
        )

    @app.route("/api/pool/summary")
    def api_pool_summary():
        people = []
        total_frames = 0
        for person_dir in person_dirs(app.config["FILTED_POOL_DIR"]):
            meta_path = person_dir / "filtered_pool_meta.json"
            jsonl_path = person_dir / "filtered_pool.jsonl"
            if not jsonl_path.exists():
                continue
            meta = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    meta = {}
            count = int(meta.get("output_frame_count") or len(read_jsonl(jsonl_path)))
            total_frames += count
            people.append(
                {
                    "person_id": person_dir.name,
                    "count": count,
                    "jsonl_path": repo_relative(jsonl_path),
                    "meta_path": repo_relative(meta_path) if meta_path.exists() else None,
                    "distance": meta.get("distance"),
                    "max_per_shot": meta.get("max_per_shot"),
                }
            )
        return jsonify(
            {
                "person_count": len(people),
                "total_frames": total_frames,
                "persons": people,
            }
        )

    @app.route("/api/pool")
    def api_pool():
        person_id = request.args.get("person_id", "")
        offset = max(0, int(request.args.get("offset", 0)))
        limit = max(1, min(300, int(request.args.get("limit", 60))))
        if not person_id:
            return jsonify({"total": 0, "offset": offset, "limit": limit, "rows": []})
        jsonl_path = app.config["FILTED_POOL_DIR"] / person_id / "filtered_pool.jsonl"
        rows = read_jsonl(jsonl_path)
        page = rows[offset : offset + limit]
        return jsonify({"total": len(rows), "offset": offset, "limit": limit, "rows": [add_pool_path_info(dict(row)) for row in page]})

    @app.route("/media")
    def media():
        raw_path = request.args.get("path", "")
        resolved = resolve_repo_path(raw_path)
        if not resolved or not resolved.exists() or not resolved.is_file():
            return "not found", 404
        return send_file(str(resolved))

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize generated training pairs.")
    parser.add_argument("--pairs-jsonl", default=os.environ.get("PAIRS_JSONL", str(DEFAULT_JSONL)))
    parser.add_argument("--person-clusters-dir", default=os.environ.get("PERSON_CLUSTERS_DIR", str(DEFAULT_PERSON_CLUSTERS)))
    parser.add_argument("--filted-pool-dir", default=os.environ.get("FILTED_POOL_DIR", str(DEFAULT_FILTED_POOL)))
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7893")))
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = make_app(args.pairs_jsonl)
    app.config["PERSON_CLUSTERS_DIR"] = resolve_repo_path(args.person_clusters_dir)
    app.config["FILTED_POOL_DIR"] = resolve_repo_path(args.filted_pool_dir)
    app.run(host=args.host, port=args.port, debug=args.debug)

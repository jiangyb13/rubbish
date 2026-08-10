#!/usr/bin/env python3
import argparse
import json
import os
from collections import Counter
from pathlib import Path

from flask import Flask, abort, jsonify, render_template_string, request, send_file


IMAGE_TYPES = ("face_orig", "face_white", "full_orig", "full_white")
QUALITY_KEYS = (
    "mask_hole",
    "face_bbox_boundary",
    "face_mask_coverage",
    "face_occlusion",
    "image_clarity_laplacian",
    "image_clarity_vlm",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Stage4 post_process_index quality results.")
    parser.add_argument("--root", required=True, help="video_dir, person_clusters dir, or a single post_process_index.json.")
    parser.add_argument("--index_name", default="post_process_index.json")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7897)
    parser.add_argument("--max_rows", type=int, default=0, help="Load at most this many image rows; 0 means all.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def find_index_files(root, index_name):
    root = Path(root).expanduser().resolve()
    if root.is_file():
        return [root]

    def collect(cluster_dir):
        if not cluster_dir.is_dir():
            return []
        return sorted(p / index_name for p in cluster_dir.iterdir() if p.is_dir() and (p / index_name).is_file())

    matches = []
    if root.name == "person_clusters":
        matches.extend(collect(root))
    matches.extend(collect(root / "person_clusters"))
    matches.extend(collect(root / "identity_matching" / "person_clusters"))
    return sorted(set(matches))


def quality_state(item):
    if not isinstance(item, dict):
        return "missing"
    if item.get("passed") is False:
        return "fail"
    if item.get("passed") is True:
        return "pass"
    status = str(item.get("status") or "")
    if status.startswith("skipped"):
        return "skip"
    return "missing"


def summarize_quality(item):
    if not isinstance(item, dict):
        return {"state": "missing"}
    data = {
        "state": quality_state(item),
        "status": item.get("status"),
        "passed": item.get("passed"),
        "checked": item.get("checked"),
    }
    for key in (
        "hole_count", "threshold", "sharpness", "face_occluded", "is_clear",
        "confidence", "reason", "parse_status", "touches_boundary",
        "mask_foreground_ratio", "yaw", "is_frontal", "mask_path",
    ):
        if key in item:
            data[key] = item.get(key)
    return data


def load_rows(root, index_name, max_rows):
    root_abs = str(Path(root).expanduser().resolve())
    rows = []
    index_files = find_index_files(root, index_name)
    for index_path in index_files:
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[quality-viewer] skip {index_path}: {exc}")
            continue
        person_id = data.get("person_id") or index_path.parent.name
        images = data.get("images") if isinstance(data, dict) else {}
        if not isinstance(images, dict):
            continue
        for image_type in IMAGE_TYPES:
            entries = images.get(image_type) or {}
            if not isinstance(entries, dict):
                continue
            for image_path, attrs in entries.items():
                if not isinstance(attrs, dict):
                    continue
                quality = attrs.get("quality") if isinstance(attrs.get("quality"), dict) else {}
                states = {key: summarize_quality(quality.get(key)) for key in QUALITY_KEYS}
                failed = [key for key, value in states.items() if value.get("state") == "fail"]
                row = {
                    "id": len(rows),
                    "index_path": str(index_path),
                    "relative_index_path": os.path.relpath(index_path, root_abs) if os.path.isdir(root_abs) else str(index_path),
                    "person_id": attrs.get("person_id") or person_id,
                    "uid": attrs.get("uid"),
                    "shot_key": attrs.get("shot_key"),
                    "obj_id": attrs.get("obj_id"),
                    "frame_idx": attrs.get("frame_idx"),
                    "image_type": image_type,
                    "image_path": attrs.get("image_path") or image_path,
                    "one_shot_image_path": attrs.get("one_shot_image_path"),
                    "quality_label": attrs.get("quality_label", True),
                    "failed_quality_keys": failed,
                    "quality_states": states,
                    "quality": {key: quality.get(key) for key in QUALITY_KEYS if key in quality},
                    "related_images": attrs.get("related_images") or {},
                    "related_one_shot_images": attrs.get("related_one_shot_images") or {},
                    "pose": attrs.get("pose") or {},
                    "expression": attrs.get("expression") or {},
                    "body_pose": attrs.get("body_pose") or {},
                }
                rows.append(row)
                if max_rows and len(rows) >= max_rows:
                    return rows, index_files
    return rows, index_files


def create_app(args):
    root = Path(args.root).expanduser().resolve()
    rows, index_files = load_rows(str(root), args.index_name, args.max_rows)
    allowed_roots = [root]
    for row in rows:
        for value in (row.get("image_path"), row.get("one_shot_image_path")):
            if value:
                try:
                    allowed_roots.append(Path(value).expanduser().resolve().parents[3])
                except Exception:
                    pass

    def resolve_image(path_value):
        if not path_value:
            return None
        path = Path(str(path_value)).expanduser()
        if not path.is_absolute():
            path = root / path
        try:
            path = path.resolve()
        except Exception:
            return None
        if not path.is_file():
            return None
        if not any(str(path).startswith(str(base)) for base in allowed_roots):
            return None
        return str(path)

    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template_string(HTML)

    @app.get("/data")
    def data():
        by_type = Counter(row["image_type"] for row in rows)
        by_quality = Counter()
        for row in rows:
            for key in row["failed_quality_keys"]:
                by_quality[key] += 1
        return jsonify({
            "root": str(root),
            "index_name": args.index_name,
            "index_file_count": len(index_files),
            "rows": rows,
            "total": len(rows),
            "failed": sum(1 for row in rows if row.get("quality_label") is False),
            "by_type": dict(by_type),
            "by_failed_quality": dict(by_quality),
        })

    @app.get("/image")
    def image():
        path = resolve_image(request.args.get("path"))
        if not path:
            abort(404)
        return send_file(path)

    return app


HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Stage4 Quality Viewer</title>
  <style>
    body { margin:0; font-family:Arial,sans-serif; background:#f6f7f9; color:#1f2933; }
    header { position:sticky; top:0; z-index:10; background:#fff; border-bottom:1px solid #d7dde8; padding:14px 18px; }
    h1 { margin:0 0 8px; font-size:20px; }
    .summary,.controls { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
    .summary span { padding:4px 8px; background:#eef1f5; border-radius:999px; font-size:12px; }
    .controls { margin-top:10px; }
    input,select,button { height:32px; border:1px solid #c8d0dc; border-radius:6px; background:#fff; padding:0 8px; }
    main { padding:16px 18px 48px; }
    .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(390px,1fr)); gap:12px; }
    .card { background:#fff; border:1px solid #d7dde8; border-radius:8px; padding:12px; }
    .head { display:flex; justify-content:space-between; gap:12px; }
    .title { font-weight:700; font-size:13px; }
    .sub { color:#667085; font-size:12px; margin-top:3px; word-break:break-all; }
    .badge { height:22px; padding:3px 8px; border-radius:999px; font-size:12px; border:1px solid #c8d0dc; }
    .badge.pass { background:#e8f5e9; border-color:#9ccc9c; color:#166534; }
    .badge.fail { background:#fff0f0; border-color:#ef9a9a; color:#991b1b; }
    .image { margin-top:10px; }
    img { width:100%; max-height:260px; object-fit:contain; background:#111827; border-radius:6px; display:block; }
    .chips { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:7px; margin-top:10px; }
    .chip { border:1px solid #d7dde8; border-radius:6px; padding:7px; background:#fafafa; font-size:12px; min-height:64px; }
    .chip.pass { background:#edf7ee; border-color:#a7d8a7; }
    .chip.fail { background:#fff0f0; border-color:#ef9a9a; }
    .chip.skip { background:#eef6ff; border-color:#9ec5fe; }
    .chip.missing { color:#7b8494; }
    .chip b { display:block; margin-bottom:3px; }
    .metric { color:#4b5563; font-size:11px; line-height:1.35; overflow:hidden; text-overflow:ellipsis; }
    pre { white-space:pre-wrap; word-break:break-all; max-height:260px; overflow:auto; font-size:11px; background:#111827; color:#d1d5db; border-radius:6px; padding:9px; }
    details { margin-top:8px; }
    summary { cursor:pointer; font-size:12px; color:#344054; }
  </style>
</head>
<body>
  <header>
    <h1>Stage4 Quality Viewer</h1>
    <div class="summary" id="summary"><span>loading...</span></div>
    <div class="controls">
      <input id="search" placeholder="person / shot / reason / path">
      <select id="label"><option value="">all labels</option><option value="fail">quality_label=false</option><option value="pass">quality_label=true</option></select>
      <select id="imageType"><option value="">all image types</option></select>
      <select id="failedKey"><option value="">all failure keys</option></select>
      <input id="limit" type="number" min="1" step="100" value="300">
      <button id="apply">Apply</button>
    </div>
  </header>
  <main><div class="grid" id="grid"></div></main>
<script>
let rows = [];
const keys = ["mask_hole","face_bbox_boundary","face_mask_coverage","face_occlusion","image_clarity_laplacian","image_clarity_vlm"];
const $ = id => document.getElementById(id);
function enc(v){ return encodeURIComponent(v || ""); }
function esc(v){ return String(v ?? "").replace(/[&<>"']/g, s => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[s])); }
function metric(k,v){ return (v === undefined || v === null || v === "") ? "" : `<div class="metric">${esc(k)}: ${esc(v)}</div>`; }
function chip(key, q) {
  q = q || {state:"missing"};
  const lines = [
    metric("status", q.status),
    metric("passed", q.passed),
    metric("holes", q.hole_count),
    metric("sharpness", q.sharpness == null ? null : Number(q.sharpness).toFixed(3)),
    metric("threshold", q.threshold),
    metric("occluded", q.face_occluded),
    metric("clear", q.is_clear),
    metric("confidence", q.confidence),
    metric("reason", q.reason),
  ].join("");
  return `<div class="chip ${esc(q.state)}"><b>${esc(key)}: ${esc(q.state)}</b>${lines}</div>`;
}
function card(row, idx) {
  const pass = row.quality_label !== false;
  const image = row.image_path ? `<img loading="lazy" src="/image?path=${enc(row.image_path)}">` : "";
  const chips = keys.map(k => chip(k, (row.quality_states || {})[k])).join("");
  return `<article class="card">
    <div class="head">
      <div>
        <div class="title">#${idx + 1} ${esc(row.person_id)} · ${esc(row.image_type)}</div>
        <div class="sub">${esc(row.shot_key)} / id_${esc(row.obj_id)} / frame ${esc(row.frame_idx)}</div>
      </div>
      <div class="badge ${pass ? "pass" : "fail"}">${pass ? "pass" : "fail"}</div>
    </div>
    <div class="image">${image}</div>
    <div class="chips">${chips}</div>
    <details><summary>raw quality</summary><pre>${esc(JSON.stringify({quality_label: row.quality_label, failed_quality_keys: row.failed_quality_keys, quality: row.quality, pose: row.pose, expression: row.expression, body_pose: row.body_pose, image_path: row.image_path, index_path: row.index_path}, null, 2))}</pre></details>
  </article>`;
}
function addOptions(id, values) {
  for (const [key, count] of Object.entries(values).sort()) {
    const opt = document.createElement("option");
    opt.value = key; opt.textContent = `${key} (${count})`;
    $(id).appendChild(opt);
  }
}
function render() {
  const needle = $("search").value.toLowerCase();
  const label = $("label").value;
  const imageType = $("imageType").value;
  const failedKey = $("failedKey").value;
  const limit = Number($("limit").value || 300);
  const filtered = rows.filter(row => {
    if (needle && !JSON.stringify(row).toLowerCase().includes(needle)) return false;
    if (label === "fail" && row.quality_label !== false) return false;
    if (label === "pass" && row.quality_label === false) return false;
    if (imageType && row.image_type !== imageType) return false;
    if (failedKey && !(row.failed_quality_keys || []).includes(failedKey)) return false;
    return true;
  });
  $("grid").innerHTML = filtered.slice(0, limit).map(card).join("");
}
fetch("/data").then(r => r.json()).then(data => {
  rows = data.rows;
  $("summary").innerHTML = `<span>rows: <b>${data.total}</b></span><span>failed: <b>${data.failed}</b></span><span>index files: <b>${data.index_file_count}</b></span><span>root: ${esc(data.root)}</span>`;
  addOptions("imageType", data.by_type);
  addOptions("failedKey", data.by_failed_quality);
  render();
});
["search","label","imageType","failedKey","limit"].forEach(id => $(id).addEventListener("input", render));
$("apply").addEventListener("click", render);
</script>
</body>
</html>
"""


def main():
    args = parse_args()
    app = create_app(args)
    print(f"Viewer: http://{args.host}:{args.port}")
    print(f"Root: {Path(args.root).expanduser().resolve()}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

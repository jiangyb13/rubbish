#!/usr/bin/env python3
import argparse
import json
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_file

QUALITY_KEYS = ("mask_hole", "face_bbox_boundary", "face_mask_coverage")
IMAGE_TYPES = ("face_orig", "face_white", "full_orig", "full_white")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize post-process index quality labels.")
    parser.add_argument("--root", default="/mnt/nfs/data/peiyao/HUAWEI_CrossPairDataset/outputs")
    parser.add_argument("--index-name", default="post_process_index_v3.json")
    parser.add_argument("--pattern", default=None, help="Optional glob pattern under root, e.g. '**/post_process_index*.json'.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7895)
    parser.add_argument("--max-rows", type=int, default=0, help="Limit loaded rows; 0 means no limit.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def safe_path(value):
    if not value:
        abort(404)
    path = os.path.abspath(os.path.expanduser(value))
    if not os.path.isfile(path):
        abort(404)
    return path


def find_index_files(root, index_name, pattern):
    root_path = Path(root).expanduser().resolve()
    if root_path.is_file():
        return [root_path]
    if pattern:
        return sorted(p for p in root_path.glob(pattern) if p.is_file())

    prune_dirs = {
        "face_orig", "face_white", "full_orig", "full_white",
        "face_angle_library", "face_diversity_topk", "dino_diversity_topk",
        "vis", "visualization", "raw_tracking_results",
    }
    matches = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [name for name in dirnames if name not in prune_dirs]
        if index_name in filenames:
            matches.append(Path(dirpath) / index_name)
    return sorted(matches)


def quality_state(item):
    if not isinstance(item, dict):
        return "missing"
    if item.get("passed") is False:
        return "fail"
    status = str(item.get("status") or "")
    if item.get("checked") is False and status.startswith("skipped"):
        return "skip"
    if item.get("checked") is False and status in {"disabled", "not_mask_checked_image"}:
        return "missing"
    return "pass" if item.get("passed", True) else "fail"


def short_quality(item):
    if not isinstance(item, dict):
        return {"state": "missing", "status": "missing"}
    data = {
        "state": quality_state(item),
        "status": item.get("status"),
        "passed": item.get("passed"),
        "checked": item.get("checked"),
    }
    for key in (
        "hole_count", "threshold", "touches_boundary", "mask_foreground_ratio",
        "mask_background_pixel_count", "yaw", "is_frontal", "max_abs_yaw",
        "det_score", "mask_path",
    ):
        if key in item:
            data[key] = item.get(key)
    return data


def row_from_entry(index_path, image_path, attrs, image_type, root):
    quality = attrs.get("quality") if isinstance(attrs.get("quality"), dict) else {}
    states = {key: short_quality(quality.get(key)) for key in QUALITY_KEYS}
    failed = [key for key, value in states.items() if value.get("state") == "fail"]
    rel_index = os.path.relpath(index_path, root)
    return {
        "index_path": str(index_path),
        "relative_index_path": rel_index,
        "video_or_group": rel_index.split(os.sep)[0] if rel_index else "",
        "person_id": attrs.get("person_id"),
        "uid": attrs.get("uid"),
        "shot_key": attrs.get("shot_key"),
        "obj_id": attrs.get("obj_id"),
        "frame_idx": attrs.get("frame_idx"),
        "image_type": image_type,
        "image_path": attrs.get("image_path") or image_path,
        "quality_label": attrs.get("quality_label", True),
        "failed_quality_keys": failed,
        "quality_states": states,
        "quality": {key: quality.get(key) for key in QUALITY_KEYS if key in quality},
        "related_images": attrs.get("related_images") or {},
        "pose": attrs.get("pose") or {},
        "source_shot_path": attrs.get("source_shot_path"),
    }


def load_rows(root, index_name, pattern, max_rows):
    rows = []
    index_files = find_index_files(root, index_name, pattern)
    root_abs = str(Path(root).expanduser().resolve())
    for index_path in index_files:
        try:
            with open(index_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except Exception as exc:
            print(f"[viewer] skip invalid json: {index_path}: {exc}")
            continue
        images = data.get("images") if isinstance(data, dict) else None
        if not isinstance(images, dict):
            continue
        for image_type, entries in images.items():
            if image_type not in IMAGE_TYPES or not isinstance(entries, dict):
                continue
            for image_path, attrs in entries.items():
                if not isinstance(attrs, dict):
                    continue
                rows.append(row_from_entry(index_path, image_path, attrs, image_type, root_abs))
                if max_rows and len(rows) >= max_rows:
                    return rows, index_files
    return rows, index_files


HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Post Process Quality Viewer</title>
  <style>
    body { margin:0; font-family:Arial,sans-serif; background:#f5f5f2; color:#202020; }
    .top { position:sticky; top:0; z-index:4; background:#fff; border-bottom:1px solid #d7d7d0; padding:12px 16px; }
    h1 { margin:0 0 8px; font-size:19px; }
    .summary,.controls { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
    .summary span { background:#eeeeea; padding:5px 8px; border-radius:4px; font-size:12px; }
    .controls { margin-top:10px; }
    input,select,button { height:32px; border:1px solid #bbb; border-radius:4px; background:#fff; padding:0 8px; }
    button { cursor:pointer; }
    main { padding:16px; }
    .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(420px,1fr)); gap:12px; }
    .card { background:#fff; border:1px solid #d9d9d2; border-radius:6px; padding:10px; }
    .head { display:flex; justify-content:space-between; gap:8px; }
    .title { font-weight:700; font-size:13px; }
    .sub { margin-top:3px; color:#666; font-size:12px; }
    .overall { border-radius:4px; padding:4px 7px; font-size:12px; border:1px solid #ccc; white-space:nowrap; }
    .overall.pass { background:#e8f5e9; border-color:#a5d6a7; }
    .overall.fail { background:#ffebee; border-color:#ef9a9a; }
    .media { margin-top:9px; }
    .label { font-size:11px; color:#555; margin-bottom:3px; }
    img { width:100%; max-height:260px; object-fit:contain; border:1px solid #ddd; background:#fafafa; }
    .chips { display:grid; grid-template-columns:repeat(3,1fr); gap:6px; margin-top:8px; }
    .chip { border:1px solid #ddd; border-radius:5px; padding:6px; font-size:12px; background:#f8f8f6; min-height:54px; }
    .chip.pass { background:#edf7ee; border-color:#a5d6a7; }
    .chip.fail { background:#fff0f0; border-color:#ef9a9a; }
    .chip.skip { background:#eef4fb; border-color:#90caf9; }
    .chip.missing { color:#777; }
    .chip b { display:block; margin-bottom:3px; }
    .metric { color:#555; font-size:11px; line-height:1.35; }
    details { margin-top:8px; }
    summary { cursor:pointer; font-size:12px; }
    .related { display:grid; grid-template-columns:repeat(4,1fr); gap:7px; margin-top:7px; }
    pre { white-space:pre-wrap; word-break:break-all; font-size:11px; background:#f7f7f5; padding:8px; border-radius:4px; max-height:240px; overflow:auto; }
  </style>
</head>
<body>
  <div class="top">
    <h1>Post Process Quality Viewer</h1>
    <div class="summary" id="summary"><span>loading...</span></div>
    <div class="controls">
      <input id="q" placeholder="search person / shot / path" />
      <select id="overall"><option value="">all labels</option><option value="fail">failed only</option><option value="pass">passed only</option></select>
      <select id="type"><option value="">all image types</option></select>
      <select id="quality"><option value="">all quality dims</option><option value="mask_hole">mask_hole failed</option><option value="face_bbox_boundary">face_bbox_boundary failed</option><option value="face_mask_coverage">face_mask_coverage failed</option></select>
      <select id="video"><option value="">all videos/groups</option></select>
      <input id="limit" type="number" value="300" min="1" step="100" />
      <button id="apply">apply</button>
    </div>
  </div>
  <main><div class="grid" id="grid"></div></main>
<script>
let rows = [];
const $ = id => document.getElementById(id);
const grid = $('grid');
function enc(v) { return encodeURIComponent(v || ''); }
function text(v) { return (v === null || v === undefined) ? '' : String(v); }
function metricLine(k, v) { return (v === undefined || v === null || v === '') ? '' : `<div class="metric">${k}: ${text(v)}</div>`; }
function chip(name, q) {
  q = q || {state:'missing', status:'missing'};
  const extra = [
    metricLine('status', q.status), metricLine('holes', q.hole_count), metricLine('threshold', q.threshold),
    metricLine('touches', q.touches_boundary), metricLine('fg', q.mask_foreground_ratio == null ? null : Number(q.mask_foreground_ratio).toFixed(4)),
    metricLine('bg_px', q.mask_background_pixel_count), metricLine('yaw', q.yaw == null ? null : Number(q.yaw).toFixed(2)),
    metricLine('frontal', q.is_frontal),
  ].join('');
  return `<div class="chip ${q.state || 'missing'}"><b>${name}: ${q.state}</b>${extra}</div>`;
}
function card(row, idx) {
  const q = row.quality_states || {};
  const rel = row.related_images || {};
  const relatedHtml = ['face_orig','face_white','full_orig','full_white'].map(k => rel[k] ? `<div><div class="label">${k}</div><img loading="lazy" src="/image?path=${enc(rel[k])}"></div>` : '').join('');
  const overall = row.quality_label === false ? 'fail' : 'pass';
  return `<article class="card">
    <div class="head"><div><div class="title">#${idx+1} ${text(row.person_id)} · ${text(row.image_type)}</div><div class="sub">${text(row.video_or_group)} · ${text(row.shot_key)} / id_${text(row.obj_id)} / frame ${text(row.frame_idx)}</div></div><div class="overall ${overall}">${overall}</div></div>
    <div class="media"><div><div class="label">image</div><img loading="lazy" src="/image?path=${enc(row.image_path)}"></div></div>
    <div class="chips">${chip('mask_hole', q.mask_hole)}${chip('face_bbox_boundary', q.face_bbox_boundary)}${chip('face_mask_coverage', q.face_mask_coverage)}</div>
    <details><summary>related images</summary><div class="related">${relatedHtml}</div></details>
    <details><summary>raw quality / paths</summary><pre>${text(JSON.stringify({quality: row.quality, pose: row.pose, image_path: row.image_path, index_path: row.index_path}, null, 2))}</pre></details>
  </article>`;
}
function render() {
  const needle = $('q').value.toLowerCase();
  const overall = $('overall').value;
  const type = $('type').value;
  const quality = $('quality').value;
  const video = $('video').value;
  const limit = Number($('limit').value || 300);
  let filtered = rows.filter(row => {
    if (needle && !JSON.stringify(row).toLowerCase().includes(needle)) return false;
    if (overall === 'fail' && row.quality_label !== false) return false;
    if (overall === 'pass' && row.quality_label === false) return false;
    if (type && row.image_type !== type) return false;
    if (video && row.video_or_group !== video) return false;
    if (quality && !(row.failed_quality_keys || []).includes(quality)) return false;
    return true;
  });
  grid.innerHTML = filtered.slice(0, limit).map(card).join('');
}
function addOptions(id, counts) {
  for (const [k, v] of Object.entries(counts).sort()) {
    const opt = document.createElement('option'); opt.value = k; opt.textContent = `${k} (${v})`; $(id).appendChild(opt);
  }
}
fetch('/data').then(r => r.json()).then(payload => {
  rows = payload.rows;
  $('summary').innerHTML = `<span>rows: <b>${payload.rows.length}</b></span><span>index files: <b>${payload.index_file_count}</b></span><span>failed: <b>${payload.failed_count}</b></span><span>root: ${payload.root}</span>`;
  addOptions('type', payload.by_type);
  addOptions('video', payload.by_video);
  render();
});
['q','overall','type','quality','video','limit'].forEach(id => $(id).addEventListener('input', render));
$('apply').addEventListener('click', render);
</script>
</body>
</html>'''


def make_app(args):
    app = Flask(__name__)
    rows, index_files = load_rows(args.root, args.index_name, args.pattern, args.max_rows)
    by_type = Counter(row.get("image_type") for row in rows)
    by_video = Counter(row.get("video_or_group") for row in rows)
    failed_count = sum(1 for row in rows if row.get("quality_label") is False)

    @app.get("/")
    def index():
        return HTML

    @app.get("/data")
    def data():
        return jsonify({
            "root": str(Path(args.root).expanduser().resolve()),
            "index_name": args.index_name,
            "index_file_count": len(index_files),
            "rows": rows,
            "failed_count": failed_count,
            "by_type": dict(by_type),
            "by_video": dict(by_video),
        })

    @app.get("/image")
    def image():
        return send_file(safe_path(request.args.get("path", "")))

    @app.get("/mask")
    def mask():
        path = safe_path(request.args.get("path", ""))
        if path.lower().endswith(".npy"):
            arr = (np.load(path) > 0).astype(np.uint8) * 255
        else:
            arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if arr is None:
                abort(404)
        ok, buf = cv2.imencode(".png", arr)
        if not ok:
            abort(500)
        return Response(buf.tobytes(), mimetype="image/png")

    return app


if __name__ == "__main__":
    args = parse_args()
    app = make_app(args)
    print(f"Viewer: http://{args.host}:{args.port}")
    print(f"Root: {Path(args.root).expanduser().resolve()}")
    print(f"Index name: {args.index_name}")
    app.run(host=args.host, port=args.port, debug=args.debug)

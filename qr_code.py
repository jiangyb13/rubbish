#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from flask import Flask, abort, jsonify, render_template_string, request, send_file


APP_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize kept and rejected training pairs.")
    parser.add_argument("--video_dir", required=True, help="Workspace root that contains training_pairs/.")
    parser.add_argument("--pairs_jsonl", default=None, help="Kept pairs JSONL. Defaults to video_dir/training_pairs/pairs.jsonl.")
    parser.add_argument("--rejected_jsonl", default=None, help="Rejected pairs JSONL. Defaults to video_dir/training_pairs/rejected_pairs.jsonl.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7896)
    parser.add_argument("--max_rows", type=int, default=0, help="Maximum rows loaded from each JSONL; 0 means no limit.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def read_jsonl(path, max_rows=0):
    rows = []
    if not path or not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                row = {"_load_error": f"line {line_number}: {exc}", "_raw": line}
            row["_row_id"] = len(rows)
            rows.append(row)
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def compact_detail(value):
    if value is None:
        return None
    text = json.dumps(value, ensure_ascii=False, indent=2)
    return text if len(text) <= 4000 else text[:4000] + "\n..."


def first_existing_path(candidates):
    for value in candidates:
        if not value:
            continue
        path = Path(os.path.expanduser(str(value)))
        if path.is_file():
            return str(path.resolve())
    return None


def create_app(args):
    video_dir = Path(args.video_dir).expanduser().resolve()
    pairs_jsonl = Path(args.pairs_jsonl).expanduser().resolve() if args.pairs_jsonl else video_dir / "training_pairs" / "pairs.jsonl"
    rejected_jsonl = Path(args.rejected_jsonl).expanduser().resolve() if args.rejected_jsonl else video_dir / "training_pairs" / "rejected_pairs.jsonl"

    kept_rows = read_jsonl(str(pairs_jsonl), args.max_rows)
    rejected_rows = read_jsonl(str(rejected_jsonl), args.max_rows)

    allowed_roots = [video_dir, APP_ROOT, Path.cwd().resolve()]

    def resolve_image_path(value):
        if not value:
            return None
        raw = str(value)
        candidates = []
        path = Path(os.path.expanduser(raw))
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.extend([
                video_dir / raw,
                APP_ROOT / raw,
                Path.cwd().resolve() / raw,
            ])
        found = first_existing_path(candidates)
        if not found:
            return None
        resolved = Path(found).resolve()
        if not any(str(resolved).startswith(str(root)) for root in allowed_roots):
            return None
        return str(resolved)

    def image_url(value):
        resolved = resolve_image_path(value)
        if not resolved:
            return None
        return f"/image?path={resolved}"

    def ref_images(row, key, meta_key):
        values = []
        for value in row.get(key) or []:
            values.append({"path": value, "url": image_url(value), "meta": {}})
        metas = row.get(meta_key) or []
        if metas and not values:
            for meta in metas:
                path = meta.get("path") or meta.get("white_path")
                values.append({"path": path, "url": image_url(path), "meta": meta})
        else:
            for idx, meta in enumerate(metas):
                if idx < len(values):
                    values[idx]["meta"] = meta
        return values

    def normalize_row(row, status):
        if status == "kept":
            detail = row.get("selection_stats") or {}
            reason = "kept"
            target_path = row.get("first_frame")
        else:
            detail = row.get("detail") or {}
            reason = row.get("reject_reason") or "rejected"
            target_path = row.get("first_frame")
        return {
            "id": row.get("_row_id"),
            "status": status,
            "person_id": row.get("person_id"),
            "source_uid": row.get("source_uid"),
            "source_shot_key": row.get("source_shot_key"),
            "source_frame_idx": row.get("source_frame_idx"),
            "target_video": row.get("target_video"),
            "target_url": image_url(target_path),
            "target_path": target_path,
            "reason": reason,
            "detail": detail,
            "detail_text": compact_detail(detail),
            "selection_stats": row.get("selection_stats") or {},
            "angle": ref_images(row, "angle_ref", "angle_ref_meta"),
            "emotion": ref_images(row, "emo_ref", "emo_ref_meta"),
            "body_pose": ref_images(row, "body_pose_ref", "body_pose_ref_meta"),
            "raw_text": compact_detail({k: v for k, v in row.items() if not k.startswith("_")}),
        }

    kept = [normalize_row(row, "kept") for row in kept_rows]
    rejected = [normalize_row(row, "rejected") for row in rejected_rows]

    app = Flask(__name__)

    @app.route("/")
    def index():
        status = request.args.get("status", "kept")
        if status not in {"kept", "rejected"}:
            status = "kept"
        rows = kept if status == "kept" else rejected
        reason = request.args.get("reason", "")
        person = request.args.get("person", "")
        if reason:
            rows = [row for row in rows if row["reason"] == reason]
        if person:
            rows = [row for row in rows if str(row.get("person_id") or "") == person]
        reasons = sorted({row["reason"] for row in rejected})
        return render_template_string(TEMPLATE, rows=rows, status=status, reasons=reasons, reason=reason, person=person, kept_count=len(kept), rejected_count=len(rejected), video_dir=str(video_dir), pairs_jsonl=str(pairs_jsonl), rejected_jsonl=str(rejected_jsonl))

    @app.route("/image")
    def image():
        path = resolve_image_path(request.args.get("path"))
        if not path:
            abort(404)
        return send_file(path)

    @app.route("/api/summary")
    def summary():
        counts = {}
        for row in rejected:
            counts[row["reason"]] = counts.get(row["reason"], 0) + 1
        return jsonify({
            "video_dir": str(video_dir),
            "pairs_jsonl": str(pairs_jsonl),
            "rejected_jsonl": str(rejected_jsonl),
            "kept": len(kept),
            "rejected": len(rejected),
            "reject_reasons": counts,
        })

    return app


TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Training Pair Viewer</title>
  <style>
    body { margin: 0; font-family: Arial, sans-serif; background: #f6f7f9; color: #1f2933; }
    header { position: sticky; top: 0; z-index: 10; background: #ffffff; border-bottom: 1px solid #d9dee7; padding: 14px 20px; }
    .title { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
    .title h1 { font-size: 20px; margin: 0; }
    .pill { border: 1px solid #c8d0dc; border-radius: 999px; padding: 4px 10px; background: #fff; color: #4b5563; font-size: 13px; text-decoration: none; }
    .pill.active { background: #1f2937; color: #fff; border-color: #1f2937; }
    .meta { margin-top: 8px; font-size: 12px; color: #667085; line-height: 1.5; word-break: break-all; }
    .filters { display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }
    input, select, button { height: 32px; border: 1px solid #c8d0dc; border-radius: 6px; padding: 0 9px; background: #fff; }
    main { padding: 18px 20px 48px; }
    .card { background: #fff; border: 1px solid #d9dee7; border-radius: 8px; padding: 14px; margin-bottom: 16px; }
    .row-head { display: flex; justify-content: space-between; gap: 16px; flex-wrap: wrap; border-bottom: 1px solid #eef1f5; padding-bottom: 10px; margin-bottom: 12px; }
    .reason { font-weight: 700; color: #9a3412; }
    .kept { color: #166534; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(136px, 1fr)); gap: 10px; }
    .section { margin-top: 12px; }
    .section h3 { font-size: 14px; margin: 0 0 8px; color: #344054; }
    figure { margin: 0; border: 1px solid #e5e7eb; border-radius: 6px; overflow: hidden; background: #fafafa; }
    figure img { width: 100%; aspect-ratio: 1 / 1; object-fit: contain; background: #111827; display: block; }
    figcaption { font-size: 11px; color: #475467; padding: 6px; line-height: 1.35; word-break: break-all; }
    .target { max-width: 340px; }
    .target img { aspect-ratio: 16 / 9; }
    pre { max-height: 260px; overflow: auto; white-space: pre-wrap; background: #111827; color: #d1d5db; border-radius: 6px; padding: 10px; font-size: 12px; }
    .empty { padding: 32px; text-align: center; color: #667085; }
  </style>
</head>
<body>
  <header>
    <div class="title">
      <h1>Training Pair Viewer</h1>
      <a class="pill {{ 'active' if status == 'kept' else '' }}" href="/?status=kept">Kept {{ kept_count }}</a>
      <a class="pill {{ 'active' if status == 'rejected' else '' }}" href="/?status=rejected">Rejected {{ rejected_count }}</a>
    </div>
    <div class="meta">
      video_dir: {{ video_dir }}<br>
      pairs: {{ pairs_jsonl }}<br>
      rejected: {{ rejected_jsonl }}
    </div>
    <form class="filters" method="get">
      <input type="hidden" name="status" value="{{ status }}">
      <input name="person" placeholder="person_id" value="{{ person }}">
      {% if status == 'rejected' %}
      <select name="reason">
        <option value="">all reasons</option>
        {% for item in reasons %}
        <option value="{{ item }}" {{ 'selected' if item == reason else '' }}>{{ item }}</option>
        {% endfor %}
      </select>
      {% endif %}
      <button type="submit">Filter</button>
      <a class="pill" href="/?status={{ status }}">Reset</a>
    </form>
  </header>
  <main>
    {% if not rows %}
      <div class="empty">No rows to show.</div>
    {% endif %}
    {% for row in rows %}
    <article class="card">
      <div class="row-head">
        <div>
          <div><strong>#{{ row.id }}</strong> <span class="{{ 'kept' if row.status == 'kept' else 'reason' }}">{{ row.reason }}</span></div>
          <div class="meta">person={{ row.person_id }} | uid={{ row.source_uid }} | shot={{ row.source_shot_key }} | frame={{ row.source_frame_idx }}</div>
          <div class="meta">target_video={{ row.target_video }}</div>
        </div>
      </div>

      {% if row.target_url %}
      <div class="section">
        <h3>Target first frame</h3>
        <figure class="target">
          <img src="{{ row.target_url }}">
          <figcaption>{{ row.target_path }}</figcaption>
        </figure>
      </div>
      {% endif %}

      {% if row.status == 'rejected' %}
      <div class="section">
        <h3>Reject detail</h3>
        <pre>{{ row.detail_text }}</pre>
      </div>
      {% else %}
      <div class="section">
        <h3>Selection stats</h3>
        <pre>{{ row.detail_text }}</pre>
      </div>
      {% endif %}

      {% for group_name, refs in [('Angle refs', row.angle), ('Emotion refs', row.emotion), ('Body pose refs', row.body_pose)] %}
      <div class="section">
        <h3>{{ group_name }} ({{ refs|length }})</h3>
        <div class="grid">
          {% for ref in refs %}
          <figure>
            {% if ref.url %}
              <img src="{{ ref.url }}">
            {% else %}
              <div style="height:136px;display:flex;align-items:center;justify-content:center;color:#667085;">missing image</div>
            {% endif %}
            <figcaption>
              {{ ref.path }}<br>
              yaw={{ ref.meta.get('yaw') }} pitch={{ ref.meta.get('pitch') }} emotion={{ ref.meta.get('emotion') }} quality={{ ref.meta.get('quality_label') }}
            </figcaption>
          </figure>
          {% endfor %}
        </div>
      </div>
      {% endfor %}

      <details class="section">
        <summary>Raw row</summary>
        <pre>{{ row.raw_text }}</pre>
      </details>
    </article>
    {% endfor %}
  </main>
</body>
</html>
"""


def main():
    args = parse_args()
    app = create_app(args)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

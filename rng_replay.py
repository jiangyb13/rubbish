import argparse
import hashlib
import json
import os
import threading
from collections import defaultdict
from pathlib import PurePosixPath

import cv2
import moxing as mox

from flask import (
    Flask,
    jsonify,
    render_template_string,
    send_file,
    abort,
)


# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--jsonl",
        type=str,
        required=True,
        help="输入 JSONL 文件",
    )

    parser.add_argument(
        "--s3_root",
        type=str,
        required=True,
        help="S3 视频根目录，例如 s3://bucket/path/videos",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./video_cache",
        help="本地视频缓存目录",
    )

    parser.add_argument(
        "--video_key",
        type=str,
        default="video",
        help="JSONL 中视频路径字段名，默认 video",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--debug",
        action="store_true",
    )

    return parser.parse_args()


args = parse_args()

os.makedirs(args.cache_dir, exist_ok=True)


# ============================================================
# 读取 JSONL
# ============================================================

def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)

                # 保存原始行号，方便定位
                row["_jsonl_line_idx"] = line_idx

                rows.append(row)

            except Exception as e:
                print(
                    f"[JSON ERROR] "
                    f"line={line_idx}, error={e}"
                )

    return rows


rows = load_jsonl(args.jsonl)

print("=" * 80)
print(f"Loaded JSONL: {args.jsonl}")
print(f"Total samples: {len(rows)}")
print(f"S3 root:       {args.s3_root}")
print(f"Cache dir:     {args.cache_dir}")
print("=" * 80)


# ============================================================
# 路径工具
# ============================================================

def get_video_path(row):
    if args.video_key not in row:
        raise KeyError(
            f"Sample does not contain video key: {args.video_key}"
        )

    video_path = row[args.video_key]

    if not isinstance(video_path, str):
        raise TypeError(
            f"{args.video_key} must be str, "
            f"got {type(video_path)}"
        )

    return video_path


def get_s3_path(video_path):
    """
    video 本身已经是 s3:// 时直接使用。

    否则：
        s3_root + video
    """

    if video_path.startswith("s3://"):
        return video_path

    return (
        args.s3_root.rstrip("/")
        + "/"
        + video_path.lstrip("/")
    )


def get_cache_path(video_path):
    """
    防止：
        a/001.mp4
        b/001.mp4

    basename 相同导致冲突。
    """

    suffix = PurePosixPath(video_path).suffix

    if not suffix:
        suffix = ".mp4"

    hash_name = hashlib.md5(
        video_path.encode("utf-8")
    ).hexdigest()

    return os.path.join(
        args.cache_dir,
        hash_name + suffix
    )


# ============================================================
# 下载锁
#
# 防止浏览器对一个视频同时发多个 Range 请求，
# 导致同一个视频被重复下载。
# ============================================================

video_locks = defaultdict(threading.Lock)


def ensure_cached(index):
    """
    如果视频不存在于 cache：
        S3 -> cache

    如果已经存在：
        直接返回 cache

    Returns:
        cache_path
    """

    if index < 0 or index >= len(rows):
        raise IndexError(index)

    row = rows[index]

    video_path = get_video_path(row)
    s3_path = get_s3_path(video_path)
    cache_path = get_cache_path(video_path)

    # 已经下载
    if os.path.exists(cache_path):
        if os.path.getsize(cache_path) > 0:
            return cache_path

    lock = video_locks[cache_path]

    with lock:

        # double check
        if os.path.exists(cache_path):
            if os.path.getsize(cache_path) > 0:
                return cache_path

        print()
        print("=" * 80)
        print(f"[DOWNLOAD]")
        print(f"index: {index}")
        print(f"video: {video_path}")
        print(f"s3:    {s3_path}")
        print(f"cache: {cache_path}")
        print("=" * 80)

        # 临时文件
        tmp_path = (
            cache_path
            + f".tmp.{os.getpid()}.{threading.get_ident()}"
        )

        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

            # S3 -> local
            mox.file.copy(
                s3_path,
                tmp_path
            )

            if not os.path.exists(tmp_path):
                raise RuntimeError(
                    f"Download finished but file does not exist: "
                    f"{tmp_path}"
                )

            if os.path.getsize(tmp_path) <= 0:
                raise RuntimeError(
                    f"Downloaded empty file: {tmp_path}"
                )

            # 原子替换
            os.replace(
                tmp_path,
                cache_path
            )

            size_mb = (
                os.path.getsize(cache_path)
                / 1024
                / 1024
            )

            print(
                f"[DOWNLOAD DONE] "
                f"index={index}, "
                f"size={size_mb:.2f} MB"
            )

            return cache_path

        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

            # 避免留下损坏 cache
            if os.path.exists(cache_path):
                try:
                    if os.path.getsize(cache_path) == 0:
                        os.remove(cache_path)
                except Exception:
                    pass

            raise


# ============================================================
# 视频信息
# ============================================================

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        cap.release()

        return {
            "width": None,
            "height": None,
            "fps": None,
            "frame_count": None,
            "duration": None,
        }

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps > 0:
        duration = frame_count / fps
    else:
        duration = None

    cap.release()

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
    }


# ============================================================
# Flask
# ============================================================

app = Flask(__name__)


HTML = r"""
<!DOCTYPE html>
<html lang="zh-CN">

<head>
    <meta charset="UTF-8">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <title>JSONL Video Viewer</title>

    <style>
        body {
            margin: 0;
            background: #111;
            color: #eee;
            font-family:
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif;
        }

        .container {
            width: min(1500px, 96%);
            margin: 20px auto;
        }

        .top {
            display: flex;
            gap: 12px;
            align-items: center;
            flex-wrap: wrap;
            margin-bottom: 16px;
        }

        button {
            padding: 10px 18px;
            border: 0;
            border-radius: 6px;
            cursor: pointer;
            font-size: 15px;
        }

        button:hover {
            opacity: 0.85;
        }

        input {
            padding: 9px;
            font-size: 15px;
            width: 100px;
        }

        .index {
            font-size: 18px;
            font-weight: bold;
        }

        .main {
            display: grid;
            grid-template-columns: minmax(0, 2fr) minmax(350px, 1fr);
            gap: 20px;
        }

        .video-box {
            background: #000;
            min-height: 400px;
            display: flex;
            justify-content: center;
            align-items: center;
            border-radius: 8px;
            overflow: hidden;
        }

        video {
            max-width: 100%;
            max-height: 78vh;
            display: block;
        }

        .panel {
            background: #1b1b1b;
            padding: 16px;
            border-radius: 8px;
            overflow: hidden;
        }

        .field {
            margin-bottom: 12px;
        }

        .label {
            color: #888;
            font-size: 13px;
            margin-bottom: 3px;
        }

        .value {
            word-break: break-all;
        }

        pre {
            background: #090909;
            padding: 12px;
            border-radius: 6px;
            overflow: auto;
            max-height: 500px;
            white-space: pre-wrap;
            word-break: break-all;
        }

        #status {
            color: #ffcc66;
            font-weight: bold;
        }

        .cached {
            color: #75d975 !important;
        }

        .error {
            color: #ff6b6b !important;
        }

        @media (max-width: 900px) {
            .main {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>


<body>

<div class="container">

    <div class="top">

        <button id="prev">
            ← 上一个
        </button>

        <button id="next">
            下一个 →
        </button>

        <span class="index">
            <span id="current-index">0</span>
            /
            <span id="total">{{ total }}</span>
        </span>

        <input
            id="jump-input"
            type="number"
            min="0"
            max="{{ total - 1 }}"
            placeholder="index"
        >

        <button id="jump">
            跳转
        </button>

        <span id="status">
            Waiting
        </span>

    </div>


    <div class="main">

        <div class="video-box">

            <video
                id="video"
                controls
                autoplay
                preload="metadata"
            >
            </video>

        </div>


        <div class="panel">

            <div class="field">
                <div class="label">
                    Index
                </div>

                <div
                    class="value"
                    id="info-index"
                ></div>
            </div>


            <div class="field">
                <div class="label">
                    Video
                </div>

                <div
                    class="value"
                    id="info-video"
                ></div>
            </div>


            <div class="field">
                <div class="label">
                    S3
                </div>

                <div
                    class="value"
                    id="info-s3"
                ></div>
            </div>


            <div class="field">
                <div class="label">
                    Cache
                </div>

                <div
                    class="value"
                    id="info-cache"
                ></div>
            </div>


            <div class="field">
                <div class="label">
                    Resolution
                </div>

                <div
                    class="value"
                    id="info-resolution"
                >
                    -
                </div>
            </div>


            <div class="field">
                <div class="label">
                    JSON
                </div>

                <pre id="json-content"></pre>
            </div>

        </div>

    </div>

</div>


<script>

const total = {{ total }};
let currentIndex = 0;


const video =
    document.getElementById("video");

const statusElement =
    document.getElementById("status");


async function loadVideo(index) {

    if (index < 0 || index >= total) {
        return;
    }

    currentIndex = index;

    document.getElementById(
        "current-index"
    ).textContent = index;

    statusElement.textContent =
        "读取样例...";

    statusElement.className = "";


    try {

        // ======================================
        // 先获取 JSON / metadata
        // 这个接口不会下载视频
        // ======================================

        const response = await fetch(
            `/api/item/${index}`
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.error || "Unknown error"
            );
        }


        document.getElementById(
            "info-index"
        ).textContent = index;


        document.getElementById(
            "info-video"
        ).textContent =
            data.video;


        document.getElementById(
            "info-s3"
        ).textContent =
            data.s3;


        document.getElementById(
            "info-cache"
        ).textContent =
            data.cached
                ? "已缓存"
                : "未缓存，即将下载";


        document.getElementById(
            "json-content"
        ).textContent =
            JSON.stringify(
                data.row,
                null,
                2
            );


        document.getElementById(
            "info-resolution"
        ).textContent = "-";


        if (data.cached) {

            statusElement.textContent =
                "读取本地缓存...";

            statusElement.className =
                "cached";

        } else {

            statusElement.textContent =
                "正在从 S3 下载视频...";

            statusElement.className = "";

        }


        // ======================================
        // 设置 src 后浏览器请求 /video/index
        //
        // 后端此时才真正执行：
        // S3 -> local cache
        // ======================================

        video.pause();

        video.removeAttribute("src");

        video.load();


        video.src =
            `/video/${index}`;

        video.load();


        // 可选自动播放
        video.play().catch(() => {});


    } catch (error) {

        console.error(error);

        statusElement.textContent =
            "Error: " + error.message;

        statusElement.className =
            "error";
    }
}


// ==========================================
// 视频真正加载成功
// ==========================================

video.addEventListener(
    "loadedmetadata",
    function () {

        statusElement.textContent =
            "加载完成";

        statusElement.className =
            "cached";


        document.getElementById(
            "info-cache"
        ).textContent =
            "已缓存";


        document.getElementById(
            "info-resolution"
        ).textContent =
            `${video.videoWidth} × ${video.videoHeight}`;
    }
);


// ==========================================
// 视频加载失败
// ==========================================

video.addEventListener(
    "error",
    function () {

        statusElement.textContent =
            "视频加载失败";

        statusElement.className =
            "error";
    }
);


// ==========================================
// Previous
// ==========================================

document.getElementById(
    "prev"
).addEventListener(
    "click",
    function () {

        if (currentIndex > 0) {
            loadVideo(
                currentIndex - 1
            );
        }
    }
);


// ==========================================
// Next
// ==========================================

document.getElementById(
    "next"
).addEventListener(
    "click",
    function () {

        if (
            currentIndex
            <
            total - 1
        ) {

            loadVideo(
                currentIndex + 1
            );
        }
    }
);


// ==========================================
// Jump
// ==========================================

document.getElementById(
    "jump"
).addEventListener(
    "click",
    function () {

        const input =
            document.getElementById(
                "jump-input"
            );

        const index =
            parseInt(
                input.value
            );

        if (
            !Number.isNaN(index)
            &&
            index >= 0
            &&
            index < total
        ) {

            loadVideo(index);
        }
    }
);


// Enter 跳转
document.getElementById(
    "jump-input"
).addEventListener(
    "keydown",
    function (event) {

        if (event.key === "Enter") {

            document.getElementById(
                "jump"
            ).click();
        }
    }
);


// ==========================================
// ← / → 快捷键
// ==========================================

document.addEventListener(
    "keydown",
    function (event) {

        // 输入 index 时不要触发
        if (
            document.activeElement.tagName
            === "INPUT"
        ) {
            return;
        }

        if (event.key === "ArrowLeft") {

            if (currentIndex > 0) {
                loadVideo(
                    currentIndex - 1
                );
            }

        } else if (
            event.key === "ArrowRight"
        ) {

            if (
                currentIndex
                <
                total - 1
            ) {

                loadVideo(
                    currentIndex + 1
                );
            }
        }
    }
);


// ==========================================
// 首个视频
// ==========================================

if (total > 0) {
    loadVideo(0);
}

</script>

</body>
</html>
"""


# ============================================================
# 首页
# ============================================================

@app.route("/")
def index():
    return render_template_string(
        HTML,
        total=len(rows),
    )


# ============================================================
# 样本信息
#
# 注意：
# 这个 endpoint 不下载视频
# ============================================================

@app.route("/api/item/<int:index>")
def api_item(index):

    if index < 0 or index >= len(rows):
        return jsonify({
            "error": "Index out of range"
        }), 404

    try:
        row = rows[index]

        video_path = get_video_path(row)
        s3_path = get_s3_path(video_path)
        cache_path = get_cache_path(video_path)

        cached = (
            os.path.exists(cache_path)
            and
            os.path.getsize(cache_path) > 0
        )

        return jsonify({
            "index": index,
            "video": video_path,
            "s3": s3_path,
            "cached": cached,
            "row": row,
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# 真正的视频接口
#
# 请求这个接口时才下载视频。
# ============================================================

@app.route("/video/<int:index>")
def video(index):

    if index < 0 or index >= len(rows):
        abort(404)

    try:

        # ==========================================
        # 关键：
        #
        # 没 cache：
        #     S3 下载
        #
        # 有 cache：
        #     直接读取
        # ==========================================

        cache_path = ensure_cached(
            index
        )

        return send_file(
            cache_path,
            conditional=True,
            as_attachment=False,
        )

    except Exception as e:

        print(
            f"[VIDEO ERROR] "
            f"index={index}, "
            f"error={e}"
        )

        abort(
            500,
            description=str(e)
        )


# ============================================================
# 可选：查询当前视频详细信息
# ============================================================

@app.route("/api/video_info/<int:index>")
def api_video_info(index):

    if index < 0 or index >= len(rows):
        return jsonify({
            "error": "Index out of range"
        }), 404

    try:
        row = rows[index]

        video_path = get_video_path(row)
        cache_path = get_cache_path(video_path)

        if not os.path.exists(cache_path):
            return jsonify({
                "cached": False
            })

        info = get_video_info(
            cache_path
        )

        return jsonify({
            "cached": True,
            **info,
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True,
    )

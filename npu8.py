import argparse
import hashlib
import json
import os
import traceback
from pathlib import PurePosixPath

import cv2
import moxing as mox
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="输入 jsonl"
    )

    parser.add_argument(
        "--s3_root",
        type=str,
        required=True,
        help="远程 S3 视频根目录"
    )

    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="720p 数据输出 jsonl 路径"
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./video_cache",
        help="本地视频缓存目录"
    )

    parser.add_argument(
        "--video_key",
        type=str,
        default="video",
        help="jsonl 中视频路径的 key"
    )

    # =========================
    # 多卡 / 多进程参数
    # =========================
    parser.add_argument(
        "--total_rank",
        type=int,
        default=1,
        help="总进程数，例如 8"
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="当前进程 rank，范围 [0, total_rank)"
    )

    parser.add_argument(
        "--allow_portrait",
        action="store_true",
        help="是否把 720x1280 也认为是 720p"
    )

    return parser.parse_args()


def join_s3_path(root, video_path):
    """
    拼接 S3 路径。

    如果 video 本身就是完整 s3:// 路径，则直接返回。
    """
    if video_path.startswith("s3://"):
        return video_path

    return (
        root.rstrip("/")
        + "/"
        + video_path.lstrip("/")
    )


def get_cache_path(cache_dir, video_path, rank):
    """
    生成唯一 cache 文件名。

    不直接使用 basename，避免：
        a/001.mp4
        b/001.mp4

    发生冲突。

    同时 rank 之间使用独立目录。
    """
    suffix = PurePosixPath(video_path).suffix

    if not suffix:
        suffix = ".mp4"

    hash_name = hashlib.md5(
        video_path.encode("utf-8")
    ).hexdigest()

    rank_cache_dir = os.path.join(
        cache_dir,
        f"rank_{rank}"
    )

    os.makedirs(
        rank_cache_dir,
        exist_ok=True
    )

    return os.path.join(
        rank_cache_dir,
        hash_name + suffix
    )


def get_video_resolution(video_path):
    """
    返回：
        width, height

    OpenCV 的 shape 对应通常为：
        (height, width)
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        cap.release()

        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    cap.release()

    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"Invalid resolution: "
            f"{width}x{height}"
        )

    return width, height


def is_720p(
    width,
    height,
    allow_portrait=False
):
    """
    默认严格判断：
        1280 x 720

    如果 allow_portrait：
        1280 x 720
        720 x 1280
    """
    if width == 1280 and height == 720:
        return True

    if allow_portrait:
        if width == 720 and height == 1280:
            return True

    return False


def safe_remove(path):
    try:
        if os.path.exists(path):
            os.remove(path)

    except Exception as e:
        print(
            f"[WARNING] Failed to remove: "
            f"{path}, error={e}"
        )


def get_rank_output_path(
    output_jsonl,
    rank,
    total_rank
):
    """
    多 rank 时，每个 rank 写自己的 jsonl。

    比如：
        output_720p.jsonl

    8 rank 会变成：
        output_720p.rank0.jsonl
        output_720p.rank1.jsonl
        ...
        output_720p.rank7.jsonl

    避免多个进程同时写一个文件。
    """

    if total_rank == 1:
        return output_jsonl

    if output_jsonl.endswith(".jsonl"):
        prefix = output_jsonl[:-6]

        return (
            f"{prefix}.rank{rank}.jsonl"
        )

    return (
        f"{output_jsonl}.rank{rank}.jsonl"
    )


def main():
    args = parse_args()

    # =============================
    # 参数检查
    # =============================

    if args.total_rank <= 0:
        raise ValueError(
            "total_rank must be > 0"
        )

    if args.rank < 0:
        raise ValueError(
            "rank must be >= 0"
        )

    if args.rank >= args.total_rank:
        raise ValueError(
            f"rank={args.rank} must be smaller than "
            f"total_rank={args.total_rank}"
        )

    # =============================
    # cache
    # =============================

    os.makedirs(
        args.cache_dir,
        exist_ok=True
    )

    rank_output_jsonl = (
        get_rank_output_path(
            args.output_jsonl,
            args.rank,
            args.total_rank
        )
    )

    output_dir = os.path.dirname(
        os.path.abspath(
            rank_output_jsonl
        )
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    print(
        "========================================"
    )
    print(
        f"Rank:          "
        f"{args.rank}/{args.total_rank}"
    )
    print(
        f"Input:         "
        f"{args.input_jsonl}"
    )
    print(
        f"Output:        "
        f"{rank_output_jsonl}"
    )
    print(
        f"Cache:         "
        f"{args.cache_dir}/rank_{args.rank}"
    )
    print(
        "========================================"
    )

    # 当前 rank 的输出文件先清空
    with open(
        rank_output_jsonl,
        "w",
        encoding="utf-8"
    ):
        pass

    # =============================
    # 统计量
    # =============================

    assigned_cnt = 0
    valid_cnt = 0
    deleted_cnt = 0

    download_failed_cnt = 0
    video_failed_cnt = 0
    missing_video_key_cnt = 0
    json_failed_cnt = 0

    # =============================
    # 开始读取
    # =============================

    with open(
        args.input_jsonl,
        "r",
        encoding="utf-8"
    ) as fin:

        for sample_idx, line in enumerate(
            tqdm(
                fin,
                desc=f"rank {args.rank}"
            )
        ):

            # =====================================
            # 多进程切分
            #
            # rank 0:
            #   0, 8, 16, 24 ...
            #
            # rank 1:
            #   1, 9, 17, 25 ...
            # =====================================
            if (
                sample_idx
                % args.total_rank
                != args.rank
            ):
                continue

            assigned_cnt += 1

            line = line.strip()

            if not line:
                continue

            # =====================================
            # 解析 json
            # =====================================

            try:
                row = json.loads(line)

            except Exception as e:
                json_failed_cnt += 1

                print(
                    f"[JSON ERROR] "
                    f"sample_idx={sample_idx}, "
                    f"error={e}"
                )

                continue

            # =====================================
            # video key
            # =====================================

            if args.video_key not in row:
                missing_video_key_cnt += 1

                print(
                    f"[MISSING VIDEO KEY] "
                    f"sample_idx={sample_idx}"
                )

                continue

            video_path = row[
                args.video_key
            ]

            s3_path = join_s3_path(
                args.s3_root,
                video_path
            )

            cache_path = get_cache_path(
                args.cache_dir,
                video_path,
                args.rank
            )

            # 清理可能存在的旧缓存
            safe_remove(
                cache_path
            )

            try:
                # =====================================
                # 1. 从 S3 下载
                # =====================================

                try:
                    mox.file.copy(
                        s3_path,
                        cache_path
                    )

                except Exception as e:
                    download_failed_cnt += 1

                    print(
                        "\n"
                        f"[DOWNLOAD FAILED] "
                        f"rank={args.rank}\n"
                        f"video={video_path}\n"
                        f"s3={s3_path}\n"
                        f"error={e}"
                    )

                    safe_remove(
                        cache_path
                    )

                    continue

                # =====================================
                # 2. 获取 shape
                # =====================================

                try:
                    width, height = (
                        get_video_resolution(
                            cache_path
                        )
                    )

                except Exception as e:
                    video_failed_cnt += 1

                    print(
                        "\n"
                        f"[VIDEO READ FAILED] "
                        f"rank={args.rank}\n"
                        f"video={video_path}\n"
                        f"error={e}"
                    )

                    safe_remove(
                        cache_path
                    )

                    continue

                # OpenCV image/video convention：
                # shape = (height, width)
                video_shape = (
                    height,
                    width
                )

                # =====================================
                # 3. 判断 720p
                # =====================================

                if is_720p(
                    width,
                    height,
                    args.allow_portrait
                ):

                    valid_cnt += 1

                    # 立即写入，避免中途挂掉后结果全丢
                    with open(
                        rank_output_jsonl,
                        "a",
                        encoding="utf-8"
                    ) as fout:

                        fout.write(
                            json.dumps(
                                row,
                                ensure_ascii=False
                            )
                            + "\n"
                        )

                    # =================================
                    # 合格视频仅用于检测，
                    # 检查结束后删除本地 cache
                    # =================================
                    safe_remove(
                        cache_path
                    )

                else:
                    # =================================
                    # 非 720p：删除 + 打印 shape
                    # =================================

                    deleted_cnt += 1

                    print(
                        "\n"
                        f"[DELETE] "
                        f"rank={args.rank}, "
                        f"video={video_path}, "
                        f"shape={video_shape}, "
                        f"resolution="
                        f"{width}x{height}"
                    )

                    safe_remove(
                        cache_path
                    )

            except Exception:
                video_failed_cnt += 1

                print(
                    "\n"
                    f"[UNKNOWN ERROR] "
                    f"rank={args.rank}, "
                    f"video={video_path}"
                )

                traceback.print_exc()

                safe_remove(
                    cache_path
                )

            # =====================================
            # 每 100 个当前 rank 样本打印统计
            # =====================================

            if (
                assigned_cnt % 100
                == 0
            ):
                print(
                    "\n"
                    "========================================\n"
                    f"Rank:             "
                    f"{args.rank}/{args.total_rank}\n"
                    f"Assigned:         "
                    f"{assigned_cnt}\n"
                    f"Valid 720p:       "
                    f"{valid_cnt}\n"
                    f"Deleted non720p:  "
                    f"{deleted_cnt}\n"
                    f"Download failed:  "
                    f"{download_failed_cnt}\n"
                    f"Video failed:     "
                    f"{video_failed_cnt}\n"
                    f"Missing key:      "
                    f"{missing_video_key_cnt}\n"
                    f"JSON failed:      "
                    f"{json_failed_cnt}\n"
                    "========================================"
                )

    # =============================
    # 最终统计
    # =============================

    print(
        "\n"
        "========================================\n"
        f"Rank {args.rank} Finished\n"
        "========================================\n"
        f"Assigned samples: "
        f"{assigned_cnt}\n"
        f"Valid 720p:       "
        f"{valid_cnt}\n"
        f"Deleted non720p:  "
        f"{deleted_cnt}\n"
        f"Download failed:  "
        f"{download_failed_cnt}\n"
        f"Video failed:     "
        f"{video_failed_cnt}\n"
        f"Missing key:      "
        f"{missing_video_key_cnt}\n"
        f"JSON failed:      "
        f"{json_failed_cnt}\n"
        f"Output:           "
        f"{rank_output_jsonl}\n"
        "========================================"
    )


if __name__ == "__main__":
    main()

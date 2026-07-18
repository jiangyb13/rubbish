import argparse
import json
from pathlib import Path


def merge_jsonl_by_part(
    root_dir: str,
    output_name: str = "outputs.jsonl",
    validate_json: bool = True,
) -> None:
    """
    按 part 合并 JSONL 文件。

    输入目录结构：
        root/
            part_00/
                video_001/outputs.jsonl
                video_002/outputs.jsonl
            part_01/
                video_003/outputs.jsonl

    输出：
        root/part_00/outputs.jsonl
        root/part_01/outputs.jsonl

    Args:
        root_dir:
            根目录路径。
        output_name:
            每个 part 下保存的合并文件名。
        validate_json:
            是否检查每一行是不是合法 JSON。
            True：跳过非法行并打印警告。
            False：直接原样合并非空行。
    """
    root_path = Path(root_dir).expanduser().resolve()

    if not root_path.exists():
        raise FileNotFoundError(f"root 目录不存在：{root_path}")

    if not root_path.is_dir():
        raise NotADirectoryError(f"给定路径不是目录：{root_path}")

    # 只处理 root 直接下属的 part_* 目录
    part_dirs = sorted(
        path
        for path in root_path.glob("part_*")
        if path.is_dir()
    )

    if not part_dirs:
        print(f"没有找到 part_* 目录：{root_path}")
        return

    total_input_files = 0
    total_output_records = 0

    for part_dir in part_dirs:
        output_path = part_dir / output_name

        # 只找 part 下一级视频目录中的 outputs.jsonl。
        # 不会把 part 自身已经生成的 outputs.jsonl 再次合并进去。
        input_paths = sorted(
            path
            for path in part_dir.glob(f"*/{output_name}")
            if path.is_file()
        )

        if not input_paths:
            print(f"[跳过] {part_dir.name} 下没有找到 */{output_name}")
            continue

        part_record_count = 0
        invalid_count = 0

        # 先写临时文件，完成后再替换，避免程序中途退出破坏旧结果
        temp_output_path = part_dir / f".{output_name}.tmp"

        try:
            with temp_output_path.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as writer:
                for input_path in input_paths:
                    file_record_count = 0

                    with input_path.open(
                        "r",
                        encoding="utf-8",
                    ) as reader:
                        for line_number, line in enumerate(reader, start=1):
                            line = line.strip()

                            # 跳过空行
                            if not line:
                                continue

                            if validate_json:
                                try:
                                    record = json.loads(line)
                                except json.JSONDecodeError as error:
                                    invalid_count += 1
                                    print(
                                        f"[非法 JSON] {input_path}"
                                        f":{line_number}，已跳过：{error}"
                                    )
                                    continue

                                # 重新序列化，保证每条数据占一行
                                writer.write(
                                    json.dumps(
                                        record,
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )
                            else:
                                writer.write(line + "\n")

                            file_record_count += 1
                            part_record_count += 1

                    print(
                        f"  - {input_path.parent.name}: "
                        f"{file_record_count} 条"
                    )

            # 原子替换最终输出文件
            temp_output_path.replace(output_path)

        except Exception:
            if temp_output_path.exists():
                temp_output_path.unlink()
            raise

        total_input_files += len(input_paths)
        total_output_records += part_record_count

        print(
            f"[完成] {part_dir.name}: "
            f"合并 {len(input_paths)} 个文件，"
            f"写入 {part_record_count} 条数据，"
            f"非法数据 {invalid_count} 条"
        )
        print(f"       保存位置：{output_path}")

    print("\n全部处理完成")
    print(f"输入文件数量：{total_input_files}")
    print(f"输出记录数量：{total_output_records}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按照 part_* 目录合并视频文件夹中的 outputs.jsonl"
    )
    parser.add_argument(
        "root",
        type=str,
        help="数据根目录，例如 /data/dataset/root",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="outputs.jsonl",
        help="合并后的文件名，默认：outputs.jsonl",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="不校验 JSON，直接原样合并所有非空行",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    merge_jsonl_by_part(
        root_dir=args.root,
        output_name=args.output_name,
        validate_json=not args.no_validate,
    )

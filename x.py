build_person_clusters_unit_list() {
    local identity_root=$1
    local output_list=$2
    mkdir -p "$(dirname "$output_list")"
    "$PYTHON_BIN" -c '
import sys
from pathlib import Path
from tqdm import tqdm

identity_root = Path(sys.argv[1]).absolute()
output_list = Path(sys.argv[2])
rows = []

uuid_dirs = [d for d in identity_root.iterdir() if d.is_dir()]

for uuid_dir in tqdm(uuid_dirs, desc="Processing clusters", unit="dir"):
    path = uuid_dir / "identity_matching" / "person_clusters"

    if not path.is_dir():
        continue
    if not any(child.is_dir() and child.name.startswith("person_") for child in path.iterdir()):
        continue

    rel_parent = path.parent.absolute().relative_to(identity_root)
    parts = rel_parent.parts

    video = parts[0] if len(parts) > 0 else ""
    part = parts[1] if len(parts) > 1 else ""
    uuid = parts[2] if len(parts) > 2 else ""

    rows.append(f"{path.absolute()}|{video}|{part}|{uuid}")

output_list.parent.mkdir(parents=True, exist_ok=True)
output_list.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
print(f"[unit_list] {output_list}: {len(rows)} person_clusters roots")
' "$identity_root" "$output_list"
}

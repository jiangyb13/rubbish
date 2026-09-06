#!/usr/bin/env bash
set -euo pipefail

cd /data/jyb/Bagel-Reca

META="/data/jyb/geneval/prompts/evaluation_metadata.jsonl"
ROOT="/data/jyb/Bagel-Reca/selector_research/native_ce_v45_full_seed1234"
OUTDIR="$ROOT/images"
MODEL="/data/jyb/pretrained_models/BAGEL-7B-MoT"
EVAL_PY="/data/jyb/geneval/evaluation/evaluate_images.py"
EVAL_MODEL="/data/jyb/pretrained_models"
BAGEL_PY="/data/ccx/miniconda3/envs/bagel/bin/python"
GENEVAL_PY="/data/ccx/miniconda3/envs/geneval/bin/python"
OUTJSONL="/data/jyb/geneval/Bagel/outputs_native_ce_v45_constraint_first_scale25_k3_tok48_seed1234.jsonl"

mkdir -p "$ROOT/logs" "$OUTDIR" "/data/jyb/geneval/Bagel"

echo "Full v45 Native CE run"
echo "seed: 1234"
echo "per-sample seed reset: disabled by gen_reca.py; setup_seeds is called once before the 4-sample loop"
echo "images: $OUTDIR"
echo "eval jsonl: $OUTJSONL"
echo "metadata: $META"

for rank in 0 1 2 3 4 5 6 7; do
  (
    export CUDA_VISIBLE_DEVICES="$rank"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export BAGEL_MODEL_IMPL=native_ce_profile
    export BAGEL_RECA_NATIVE_CE=1
    export BAGEL_IDEAL_PROMPT_STYLE=reflection_constraint_first_then_evidence_v45
    export BAGEL_RECA_UPDATE_SCALE=25
    export BAGEL_RECA_RE_UPDATE_NUM=3
    export BAGEL_CE_MAX_TOKENS=48
    export BAGEL_CE_VIT_MAX_SIDE=168
    export BAGEL_REFRESH_IDEAL_EACH_TIMESTEP=1
    export BAGEL_REFRESH_IDEAL_EACH_UPDATE=0
    export BAGEL_REUSE_RECA_GRAD=0
    export BAGEL_USE_CE_SELECTION=0
    "$BAGEL_PY" -u GenEval/gen_reca.py \
      --metadata_file "$META" \
      --outdir "$OUTDIR" \
      --model "$MODEL" \
      --n_samples 4 \
      --seed 1234 \
      --skip_grid \
      --rank "$rank" \
      --total_rank 8 \
      > "$ROOT/logs/rank${rank}.log" 2>&1
  ) &
done
wait

"$GENEVAL_PY" "$EVAL_PY" "$OUTDIR" \
  --outfile "$OUTJSONL" \
  --model-path "$EVAL_MODEL" \
  > "$ROOT/eval.log" 2>&1

"$BAGEL_PY" - <<'PY' > "$ROOT/summary.txt"
import json
from collections import defaultdict
p="/data/jyb/geneval/Bagel/outputs_native_ce_v45_constraint_first_scale25_k3_tok48_seed1234.jsonl"
rows=[json.loads(l) for l in open(p)]
print("overall", sum(r["correct"] for r in rows), "/", len(rows), "=", sum(r["correct"] for r in rows)/len(rows))
by=defaultdict(lambda:[0,0])
for r in rows:
    by[r["tag"]][1]+=1
    by[r["tag"]][0]+=int(r["correct"])
for k in sorted(by):
    a,b=by[k]
    print(k, a, "/", b, "=", a/b)
PY

cat "$ROOT/summary.txt"


#!/usr/bin/env python3
"""Time Base / Think / native ReCA / UiG / TiR with one shared Bagel backend.

Example (use an otherwise idle GPU):
  python benchmark_methods_speed.py --device 0 --outdir speed_results
  python benchmark_methods_speed.py --methods ours think --dry-run

Ours calls InterleaveInferencer.resc_for_gen directly. No reflection, gradient,
image-transform, or target-reuse patches are installed. UiG/TiR use Bagel as
both generator and understanding model. UiG/TiR workflows, prompts and RNG
helpers are embedded; only the Bagel project, weights and input JSONL are needed.
This is a speed benchmark, not an
assertion that shared-backend outputs equal the official reproduction outputs.
"""
import argparse
import contextlib
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent
METHODS = ('base', 'think', 'ours', 'uig', 'tir')


# Embedded protocol snapshots (no runtime dependency on either paper repository):
# UiG QC-LY/UiG c96430bad5f734efcaf850b4e93775d928f4aeb2, evaluation/edit flow.
# TiR hafeezkhan909/Test-time-Image-Refinement b7bf34fd4061c84c2c36131291f7b3b416dbb197,
# released Qwen prompt; Bagel is the understanding backend here.
UIG_EVALUATION_PROMPT = 'Please carefully examine this generated image and compare it with the original prompt: "{original_prompt}"\n\nAnalyze the following aspects:\n1. Does the image accurately represent the main subject described in the prompt?\n2. Are the visual details (clothing, environment, style, etc.) consistent with the prompt?\n3. Is the overall mood and atmosphere matching the intended description?\n4. Are there any missing elements or incorrect interpretations?\n\nIf the image matches the prompt well, respond with: "MATCH: The image successfully represents the prompt."\n\nIf there are discrepancies, respond with: "EDIT_NEEDED: [specific editing instructions]"\nFor example: "EDIT_NEEDED: The character should be wearing a red dress instead of blue, and the background should be a forest not a city."\n\nPlease be specific about what needs to be changed:'


def capture():
    """Capture full RNG streams; kept local so rng_replay.py is not required."""
    import random
    import numpy as np
    import torch
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state().clone(),
            'cuda': [state.clone() for state in torch.cuda.get_rng_state_all()]}


def restore(state):
    import random
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def tir_prompt(original_prompt, history):
    last_prompt = history[-1] if history else original_prompt
    history_text = ''
    if history:
        history_text = '### Previous Prompt Refinements:\n' + ''.join(
            f'- Refinement {i+1}: "{prompt}"\n' for i, prompt in enumerate(history))
    return f"""\n                    ### Evaluation Task:\n                    You are an **Image Improvement Assistant**. Your job is to help make the image more aligned with the ORIGINAL prompt.\n\n                    ### **Given Inputs:**  \n                    1. **Original User Prompt:**  \n                    - {original_prompt}  \n\n                    2. **Last Used Prompt:**\n                    - {last_prompt}\n\n                    3. **Prompt History:**\n                    {history_text}\n\n                    4. **Current Image Analysis:**\n                    - Look at the image and identify what aspects DIFFER from what the ORIGINAL prompt requested\n                    - Analyze what essential elements from the ORIGINAL prompt are missing or incorrectly represented\n                    - Ignore image quality issues like noise, blurriness, or artifacts\n                    \n                    ### **Your Task:**  \n                    1. Create a NEW PROMPT that will help generate an image that better matches the ORIGINAL prompt\n                    2. Your new prompt should be a modification of the last used prompt\n                    3. Focus on fixing what's missing or incorrectly represented in the current image\n                    4. The goal is to get progressively closer to fulfilling the ORIGINAL prompt\n                    \n                    ### **Decision Process:**\n                    1. If the image ALREADY closely represents the ORIGINAL prompt:\n                    \n                    DECISION: "True"\n                    REFINED PROMPT: "<An enhanced version of the last prompt that maintains alignment>"\n\n                    2. If the image DOES NOT adequately represent the ORIGINAL prompt:\n                    \n                    DECISION: "False"\n                    REFINED PROMPT: "<Your NEW prompt that addresses the specific misalignments>"\n\n                    Follow this exact output format:\n                    DECISION: "True" or "False"\n                    REFINED PROMPT: "<Your new prompt here>"\n                    """


def parse_tir(raw):
    decision = prompt = None
    for line in raw.splitlines():
        if line.startswith('DECISION:'):
            decision = line.replace('DECISION:', '').strip().strip('"')
        elif line.startswith('REFINED PROMPT:'):
            prompt = line.replace('REFINED PROMPT:', '').strip().strip('"')
    if decision not in ('True', 'False') or not prompt or prompt == 'None':
        raise ValueError(f'Malformed TiR response: {raw!r}')
    return decision == 'True', prompt


def run_uig(call, prompt, common, max_edits, think_max_tokens):
    """Native Bagel calls with the released UiG evaluation/edit protocol."""
    image = call(text=prompt, **common)['image']
    history, raw_responses, iterations, edits = [], [], 0, 0
    editing = dict(common, cfg_img_scale=2., cfg_interval=[0., 1.],
                   cfg_renorm_type='text_channel')
    for iteration in range(max_edits):
        raw = call(image=image, text=UIG_EVALUATION_PROMPT.format(original_prompt=prompt),
                   understanding_output=True, think=True, do_sample=False,
                   max_think_token_n=think_max_tokens)['text']
        raw_responses.append(raw)
        # Preserve released parser exactly, including MATCH: precedence.
        if 'MATCH:' in raw:
            needs_editing, instructions = False, ''
        elif 'EDIT_NEEDED:' in raw:
            needs_editing, instructions = True, raw.split('EDIT_NEEDED:')[1].strip()
        else:
            needs_editing, instructions = True, raw
        history.append(dict(iteration=iteration + 1, needs_editing=needs_editing,
                            instructions=instructions))
        if not needs_editing:
            break
        if instructions:
            image = call(image=image, text=instructions, think=True, do_sample=False,
                         max_think_token_n=think_max_tokens, **editing)['image']
            edits += 1
        iterations = iteration + 1
    return image, dict(evaluation_history=history, raw_responses=raw_responses,
                       iterations=iterations, edits=edits)


def run_tir(call, prompt, common, rounds, max_tokens):
    """Compute all TiR rounds, selecting first matching draft or the last image."""
    image = call(text=prompt, think=False, **common)['image']
    history, decisions, raw_responses = [], [], []
    chosen = None
    for _ in range(rounds):
        raw = call(image=image, text=tir_prompt(prompt, history), understanding_output=True,
                   think=False, do_sample=False, max_think_token_n=max_tokens)['text']
        matched, rewritten = parse_tir(raw)
        if matched and chosen is None:
            chosen = image.copy()
        history.append(rewritten)
        decisions.append(matched)
        raw_responses.append(raw)
        image = call(text=rewritten, think=False, **common)['image']
    if chosen is not None:
        image = chosen
    return image, dict(prompt_history=history, decisions=decisions, raw_responses=raw_responses)


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--device', default='0', help='One physical CUDA GPU index or UUID')
    p.add_argument('--model-path', default='/data/jyb/pretrained_models/BAGEL-7B-MoT')
    p.add_argument('--model-impl', default='bagel_ce', choices=['bagel_ce', 'bagel'],
                   help='Shared backend for all methods; native module is used without algorithm patches')
    p.add_argument('--metadata', type=Path, default=Path('/data/jyb/geneval/prompts/evaluation_metadata.jsonl'))
    p.add_argument('--outdir', type=Path, default=Path('speed_results'))
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    p.add_argument('--num-prompts', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--resolution', type=int, default=1024)
    p.add_argument('--think-max-tokens', type=int, default=1024)
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--update-start', type=int, default=5)
    p.add_argument('--update-end', type=int, default=10)
    p.add_argument('--update-scale', type=float, default=200.)
    p.add_argument('--uig-max-edits', type=int, default=4)
    p.add_argument('--tir-rounds', type=int, default=2)
    p.add_argument('--tir-max-tokens', type=int, default=128)
    p.add_argument('--save-images', action='store_true')
    p.add_argument('--allow-shared-gpu', action='store_true', help='Allow other GPU processes; timing may be contaminated')
    p.add_argument('--dry-run', action='store_true', help='Validate inputs and show manifest without importing torch or loading a model')
    a = p.parse_args()
    if ',' in a.device:
        p.error('--device must specify exactly one GPU')
    if a.num_prompts < 1 or a.warmup < 0 or a.steps < 2 or a.resolution < 16 or a.resolution % 16:
        p.error('Invalid prompt count, warmup, steps, or resolution (must be a multiple of16)')
    if not 0 <= a.update_start <= a.update_end < a.steps - 1:
        p.error('Update window must fit the actual steps-1 denoising iterations')
    if min(a.k, a.uig_max_edits, a.tir_rounds, a.think_max_tokens, a.tir_max_tokens) < 1:
        p.error('K, round counts and token caps must be positive')
    if len(set(a.methods)) != len(a.methods):
        p.error('Duplicate methods')
    return a


def select_prompts(path, count):
    groups = {}
    for i, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get('prompt'), str) or not row['prompt'].strip():
            raise ValueError(f'Invalid prompt at line {i + 1}')
        row = dict(row, id=f'{i:05}')
        groups.setdefault(row.get('tag', 'untagged'), []).append(row)
    if count > sum(map(len, groups.values())):
        raise ValueError('Requested more prompts than available')
    # Round-robin categories, deterministic within each category; no label selection.
    for rows in groups.values():
        rows.sort(key=lambda r: hashlib.sha256(('speed-seed42|' + r['id']).encode()).hexdigest())
    chosen = []
    while len(chosen) < count:
        for tag in sorted(groups):
            if groups[tag] and len(chosen) < count:
                chosen.append(groups[tag].pop(0))
    return chosen


def summarize(rows):
    result = {}
    for method in METHODS:
        subset = [r for r in rows if r['method'] == method]
        if not subset:
            continue
        values = sorted(r['seconds'] for r in subset)
        slowest = max(subset, key=lambda r: r['seconds'])
        result[method] = dict(n=len(values), mean_seconds=statistics.mean(values),
                             median_seconds=statistics.median(values), min_seconds=values[0],
                             max_seconds=values[-1], p90_seconds=values[math.ceil(.9 * len(values)) - 1],
                             peak_allocated_gib=max(r['peak_allocated_gib'] for r in subset),
                             slowest_sample={k: slowest[k] for k in ('id', 'prompt', 'seconds')})
    if 'base' in result:
        for row in result.values():
            row['mean_times_base'] = row['mean_seconds'] / result['base']['mean_seconds']
    return result


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def benchmark(a, selected):
    # Set visibility before importing the model loader, so its VAE stays on this GPU.
    os.environ['CUDA_VISIBLE_DEVICES'] = a.device
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if not a.allow_shared_gpu:
        import subprocess
        query = subprocess.run(['nvidia-smi', '-i', a.device, '--query-compute-apps=pid',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        others = [int(x.strip()) for x in query.stdout.splitlines() if x.strip().isdigit() and int(x.strip()) != os.getpid()]
        if others:
            raise RuntimeError(f'GPU {a.device} has active processes {others}; use an idle GPU or --allow-shared-gpu')
    import inference
    from inferencer import InterleaveInferencer
    backend = importlib.import_module('modeling.bagel.' + a.model_impl)
    inference.Bagel, inference.BagelConfig = backend.Bagel, backend.BagelConfig
    model, vae, tok, ids, vaet, vitt = inference.initialize_full_model(a.model_path)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    inf = InterleaveInferencer(model, vae, tok, vaet, vitt, ids)
    # CE backend accepts legacy CLIP arguments but does not require CLIP weights.
    inf.clip_model = None
    inf.clip_processor = None
    common = dict(image_shapes=(a.resolution, a.resolution), num_timesteps=a.steps,
                  cfg_text_scale=4., cfg_img_scale=1., cfg_interval=[.4, 1.],
                  timestep_shift=3., cfg_renorm_min=0., cfg_renorm_type='global')
    active = {}
    native_prepare = model.prepare_vae_latent

    def prepare(*args, **kwargs):
        # Replay at noise construction too: VAE encoding during an edit can consume RNG.
        restore(active['state'])
        return native_prepare(*args, **kwargs)

    model.prepare_vae_latent = prepare

    def call(**kwargs):
        if not kwargs.get('understanding_output', False):
            restore(active['state'])
        with torch.no_grad():
            return inf(**kwargs)

    def run(method, prompt):
        inference.setup_seeds(a.seed)
        active['state'] = capture()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        detail = {}
        if method in ('base', 'think'):
            out = call(text=prompt, think=method == 'think', do_sample=False,
                       max_think_token_n=a.think_max_tokens, **common)
            image = out['image']
            detail['text'] = out.get('text')
        elif method == 'ours':
            # Normal public ReCA interface: do NOT wrap in no_grad/inference_mode.
            out = inf.resc_for_gen(prompt=prompt, think=False, use_longclip=True,
                                   re_update_num=a.k, update_lr=[(a.update_start, a.update_end)],
                                   update_scale=a.update_scale, use_save_pic=False, **common)
            image = out['image']
        elif method == 'uig':
            image, detail = run_uig(call, prompt, common, a.uig_max_edits, a.think_max_tokens)
        else:
            image, detail = run_tir(call, prompt, common, a.tir_rounds, a.tir_max_tokens)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        return image, dict(seconds=seconds, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                           peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30, **detail)

    rows = []
    for method in a.methods:
        for i in range(a.warmup):
            write_json(a.outdir / 'status.json', dict(stage='warmup', method=method, index=i))
            _, record = run(method, 'A red cube to the left of a blue sphere.')
            write_json(a.outdir / f'warmup_{method}_{i}.json', record)
    for i, prompt in enumerate(selected):
        # Rotate method order to reduce systematic warm-cache/order bias.
        offset = i % len(a.methods)
        for method in a.methods[offset:] + a.methods[:offset]:
            write_json(a.outdir / 'status.json', dict(stage='measuring', method=method, id=prompt['id'],
                       completed=len(rows), total=len(selected) * len(a.methods)))
            image, record = run(method, prompt['prompt'])
            record.update(method=method, id=prompt['id'], prompt=prompt['prompt'], tag=prompt.get('tag'))
            rows.append(record)
            with (a.outdir / 'measurements.jsonl').open('a') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            if a.save_images:
                folder = a.outdir / 'images' / method
                folder.mkdir(parents=True, exist_ok=True)
                image.save(folder / (prompt['id'] + '.png'))
            write_json(a.outdir / 'summary.json', summarize(rows))
            print(f'MEASURED {method} {prompt["id"]}: {record["seconds"]:.3f}s', flush=True)
    write_json(a.outdir / 'status.json', dict(stage='complete', completed=len(rows)))
    return summarize(rows)


def main():
    a = arguments()
    selected = select_prompts(a.metadata, a.num_prompts)
    if a.dry_run:
        print(json.dumps(dict(args=vars(a), manifest=selected), default=str, ensure_ascii=False, indent=2))
        return
    # Never append measurements from a different run/configuration by accident.
    if a.outdir.exists() and any(a.outdir.iterdir()):
        raise RuntimeError(f'Output directory is not empty: {a.outdir}; choose a new directory')
    a.outdir.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(embedded_protocols={'uig_commit':'c96430bad5f734efcaf850b4e93775d928f4aeb2', 'tir_commit':'b7bf34fd4061c84c2c36131291f7b3b416dbb197'}, ours_interface='InterleaveInferencer.resc_for_gen -> gen_image_reca; native backend algorithm',
                  timing='CUDA-synchronized end-to-end; includes initial generation and all rounds; excludes loading, warmup, disk image saving',
                  seed_protocol='One image per prompt/method; reset seed before each sample; replay same RNG at every diffusion noise construction. Not the four-continuous-images full evaluation protocol.',
                  tir_selection='Compute all requested refinement rounds; select first matched pre-refinement image, else final image',
                  reflection='Unmodified native backend prompt/token cap/refresh/gradient policy; no speed patches',
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [Path(__file__), ROOT/'inferencer.py', ROOT/'inference.py', ROOT/'modeling/bagel'/f'{a.model_impl}.py']})
    write_json(a.outdir / 'config.json', config)
    write_json(a.outdir / 'manifest.json', selected)
    print(f'Log: {a.outdir.resolve() / "run.log"}', flush=True)
    with (a.outdir / 'run.log').open('w', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            result = benchmark(a, selected)
        except Exception:
            import traceback
            error = traceback.format_exc()
            write_json(a.outdir / 'status.json', dict(stage='failed', error=error))
            print(error, flush=True)
            raise
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

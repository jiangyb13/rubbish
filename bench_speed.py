#!/usr/bin/env python3
"""Time Base / Think / native ReCA / UiG / TiR with one shared Bagel backend.

Example (use an otherwise idle GPU):
  python benchmark_methods_speed.py --device 0 --outdir speed_results
  python benchmark_methods_speed.py --methods ours think --dry-run

Ours calls InterleaveInferencer.resc_for_gen directly. No reflection, gradient,
image-transform, or target-reuse patches are installed. UiG/TiR use Bagel as
both generator and understanding model. This is a speed benchmark, not an
assertion that shared-backend outputs equal the official reproduction outputs.
"""
import argparse
import contextlib
import hashlib
import importlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent
METHODS = ('base', 'think', 'ours', 'uig', 'tir')


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
    p.add_argument('--uig-root', type=Path, default=Path('/data/jyb/paper_reproductions/UiG'))
    p.add_argument('--tir-root', type=Path, default=Path('/data/jyb/paper_reproductions/TIR'))
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
    sys.path.insert(0, str(ROOT / 'scripts/paper_repro'))
    from rng_replay import capture, restore
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

    pipeline = None
    if 'uig' in a.methods:
        spec = importlib.util.spec_from_file_location('speed_official_uig', a.uig_root / 'uni_reasoner.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        class Adapter:
            def __call__(self, **kwargs):
                return call(**kwargs)
        pipeline = module.UiGReasoner(Adapter(), logging.getLogger('speed.uig'))
        for name in ('generation_hyper', 'generation_hyper_think', 'editing_hyper'):
            getattr(pipeline, name).update(num_timesteps=a.steps, image_shapes=(a.resolution, a.resolution))
        for name in ('understanding_hyper', 'generation_hyper_think', 'editing_hyper'):
            getattr(pipeline, name)['max_think_token_n'] = a.think_max_tokens
    if 'tir' in a.methods:
        import run_bagel
        run_bagel.T = a.tir_root
        # Parse the released prompt once; don't count repeated file/AST parsing as model time.
        import ast
        tree = ast.parse((a.tir_root / 'src/qwen_integration.py').read_text())
        templates = [n for n in ast.walk(tree) if isinstance(n, ast.JoinedStr) and
                     any(isinstance(c, ast.Constant) and '### Evaluation Task:' in str(c.value) for c in n.values)]
        if len(templates) != 4:
            raise RuntimeError('Unexpected released TiR prompt structure')
        compiled = [compile(ast.Expression(n), 'tir_prompt', 'eval') for n in templates]
        def tir_prompt(prompt, history):
            env = dict(original_prompt=prompt, last_prompt=history[-1] if history else prompt,
                       history_text=('### Previous Prompt Refinements:\n' + ''.join(
                           f'- Refinement {i+1}: "{p}"\n' for i, p in enumerate(history))) if history else '')
            texts = [eval(c, {'__builtins__': {}}, env) for c in compiled]
            assert len(set(texts)) == 1
            return texts[0]

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
            out = pipeline.generate_image_with_pipeline(prompt, max_iterations=a.uig_max_edits,
                    save_intermediate=False, decompose_prompt=False, think=False,
                    prompt_dir=str(a.uig_root / 'prompts'))
            image = out['final_image']
            detail.update(evaluation_history=out['evaluation_history'], iterations=out['iterations'])
        else:
            image = call(text=prompt, think=False, **common)['image']
            history, decisions, raw_responses = [], [], []
            chosen = None
            for _ in range(a.tir_rounds):
                raw = call(image=image, text=tir_prompt(prompt, history), understanding_output=True,
                           think=False, do_sample=False, max_think_token_n=a.tir_max_tokens)['text']
                matched, rewritten = run_bagel.parse_tir(raw)
                if matched and chosen is None:
                    chosen = image.copy()
                history.append(rewritten)
                decisions.append(matched)
                raw_responses.append(raw)
                image = call(text=rewritten, think=False, **common)['image']
            if chosen is not None:
                image = chosen
            detail.update(prompt_history=history, decisions=decisions, raw_responses=raw_responses)
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
    config.update(ours_interface='InterleaveInferencer.resc_for_gen -> gen_image_reca; native backend algorithm',
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

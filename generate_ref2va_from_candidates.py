#!/usr/bin/env python3
"""Prepare seeded 4x3 candidates, then generate Ref2VA prompts using a cloud VLM."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import math
import mimetypes
import os
from pathlib import Path
import random
import re
import time
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
CATEGORIES = ('head_pose', 'facial_expression', 'body_motion', 'camera')
SECTION_NAMES = ('subject_definitions', 'summary', 'retention_analysis',
                 'detailed_description', 'overall_soundscape', 'non_diegetic_music')
DEFAULT_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
DEFAULT_MODEL = 'qwen3.5-27b'


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{line_no}: invalid JSON') from exc
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{line_no}: expected an object')
            rows.append(row)
    if not rows or any(not isinstance(r.get('id'), str) or not r['id'] for r in rows):
        raise ValueError('Input must contain nonempty string IDs')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate sample IDs')
    return rows


def parse_vocabulary(path):
    """Parse the supplied bilingual document, preserving every entry verbatim."""
    groups = {key: [] for key in CATEGORIES}
    category = subgroup = None
    current = None
    section_index = -1
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if re.match(r'^# [一二三四]、', line):
            section_index += 1
            if section_index >= 4:
                raise ValueError('Expected exactly four top-level categories')
            category = CATEGORIES[section_index]
            subgroup = None
        elif line.startswith('## '):
            subgroup = line[3:].strip()
        else:
            match = re.match(r'^(\d{3})\s*\|\s*([^|]+)\|\s*(.+)$', line)
            if match:
                if category is None or (current and 'en' not in current):
                    raise ValueError('Malformed vocabulary entry/category')
                current = dict(id=match[1], subgroup=subgroup, label=match[2].strip(), zh=match[3].strip())
                groups[category].append(current)
            elif line.strip().startswith('EN:'):
                if current is None or 'en' in current:
                    raise ValueError('Unexpected EN line')
                current['en'] = line.strip()[3:].strip()
    ids = [r['id'] for rows in groups.values() for r in rows]
    if section_index != 3 or ids != [f'{i:03}' for i in range(1, 198)]:
        raise ValueError('Expected all 197 entries, numbered 001 through 197 in order')
    if any(not r.get('en') for rows in groups.values() for r in rows):
        raise ValueError('Missing English vocabulary text')
    return groups


def validate_input(row, check_files=True):
    if not isinstance(row.get('refs'), list) or len(row['refs']) != 3 or len(set(row['refs'])) != 3:
        raise ValueError(f"{row['id']}: expected three distinct refs")
    if not isinstance(row.get('first_frame'), str):
        raise ValueError(f"{row['id']}: missing first_frame")
    if check_files:
        for value in [row['first_frame']] + row['refs']:
            if not Path(value).is_absolute() or not Path(value).is_file():
                raise ValueError(f"{row['id']}: missing/relative image: {value}")
    if 'candidates' in row:
        if set(row['candidates']) != set(CATEGORIES):
            raise ValueError('Expected exactly four candidate categories')
        ranges = {'head_pose': (1,26), 'facial_expression': (27,85), 'body_motion': (86,157), 'camera': (158,197)}
        for cat in CATEGORIES:
            choices = row['candidates'][cat]
            if not isinstance(choices, list) or len(choices) != 3:
                raise ValueError(f'{cat}: expected exactly three candidates')
            if len({x['id'] for x in choices}) != 3:
                raise ValueError(f'{cat}: duplicate candidate')
            lo, hi = ranges[cat]
            for choice in choices:
                if not re.fullmatch(r'\d{3}', choice['id']) or not lo <= int(choice['id']) <= hi:
                    raise ValueError(f'{cat}: wrong category/ID')
                if not choice.get('zh') or not choice.get('en'):
                    raise ValueError('Incomplete candidate')


def expand_rows(rows, vocabulary, seed, duration, vocab_sha256):
    expanded = []
    for row in rows:
        validate_input(row)
        if 'candidates' in row:
            raise ValueError('Input is already expanded; refusing to silently resample')
        # Stable per-ID seed means file reordering or subsetting does not alter a sample.
        local_seed = int.from_bytes(hashlib.sha256(f'{seed}:{row["id"]}'.encode()).digest()[:8], 'big')
        rng = random.Random(local_seed)
        candidates = {cat: rng.sample(vocabulary[cat], 3) for cat in CATEGORIES}
        result = {**row, 'duration_seconds': duration, 'candidates': candidates,
                  'sampling': {'seed': seed, 'sample_seed': local_seed,
                               'algorithm': 'sha256(seed:id)[:8] big-endian -> Python random.Random.sample',
                               'policy': 'uniform without replacement within each full category; no image-based filtering or resampling',
                               'vocabulary_sha256': vocab_sha256}}
        validate_input(result)
        expanded.append(result)
    return expanded


def prepare(args):
    vocab = parse_vocabulary(args.vocabulary)
    rows = expand_rows(read_jsonl(args.input), vocab, args.seed, args.duration, digest(args.vocabulary))
    out = args.output.resolve()
    if out.exists():
        raise ValueError(f'Refusing to overwrite {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    dump(out.with_suffix('.vocabulary.json'), vocab)
    dump(out.with_suffix('.sampling.json'), {'input': str(args.input.resolve()), 'vocabulary': str(args.vocabulary.resolve()),
         'seed': args.seed, 'duration_seconds': args.duration, 'samples': len(rows),
         'category_counts': {k: len(v) for k,v in vocab.items()}, 'output_sha256': digest(out)})
    print(f'Prepared {len(rows)} rows: {out}', flush=True)
    for row in rows[:2]:
        print(row['id'], {k: [v['id'] + ':' + v['zh'] for v in vs] for k,vs in row['candidates'].items()}, flush=True)


def image_content(path):
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0]
    if mime not in {'image/jpeg', 'image/png', 'image/webp'}:
        raise ValueError(f'Unsupported image format: {path}')
    # Remote APIs cannot open file:// paths. Send only the explicitly supplied four images.
    return {'type':'image_url','image_url':{'url':f'data:{mime};base64,' + base64.b64encode(path.read_bytes()).decode('ascii')}}


def build_messages(row, instructions):
    text = ('Generate one new continuous shot. No source video or audio is supplied.\n'
            f'Target duration: {row["duration_seconds"]:g} seconds.\n'
            'Select one ID from each of the four candidate lists.\nCANDIDATES:\n' +
            json.dumps(row['candidates'], ensure_ascii=False, indent=2))
    content = [{'type':'text','text':text},
               {'type':'text','text':'<Picture 1>: exact first frame; authoritative opening composition, scene and target appearance.'},
               image_content(row['first_frame'])]
    for i, p in enumerate(row['refs'], 2):
        content += [{'type':'text','text':f'<Picture {i}>: same-person HEAD identity reference only; not a keyframe, scene, or clothing reference.'},image_content(p)]
    return [{'role':'system','content':instructions},{'role':'user','content':content}]


def parse_response(text):
    text = text.strip()
    match = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```', text, re.S)
    return json.loads(match[1] if match else text)


def validate_response(result, row):
    errors = []
    if not isinstance(result, dict):
        return ['Expected a JSON object']
    expected = {'status','selected_ids','selection_rationale','compatibility_notes','prompt'}
    if set(result) != expected:
        errors.append('Expected exactly the five response fields')
    selected = result.get('selected_ids')
    if not isinstance(selected,dict) or set(selected) != set(CATEGORIES):
        errors.append('Select exactly one ID per category')
    else:
        for cat in CATEGORIES:
            if selected[cat] not in [x['id'] for x in row['candidates'][cat]]:
                errors.append(f'{cat}: selection not among supplied candidates')
    rationale = result.get('selection_rationale')
    if not isinstance(rationale,dict) or set(rationale) != set(CATEGORIES) or any(not isinstance(v,str) or not v.strip() for v in rationale.values()):
        errors.append('Missing per-category selection rationale')
    if not isinstance(result.get('compatibility_notes'),str) or not result['compatibility_notes'].strip():
        errors.append('Missing compatibility notes')
    if result.get('status') == 'incompatible':
        if result.get('prompt') is not None:
            errors.append('Incompatible result must not have a generated prompt')
        return errors
    if result.get('status') != 'ok':
        errors.append('Unknown status')
    text = result.get('prompt')
    if not isinstance(text,str) or not text:
        return errors + ['Missing generated prompt']
    matches = list(re.finditer(r'(?m)^(' + '|'.join(SECTION_NAMES) + r'):\s*$',text))
    if [m[1] for m in matches] != list(SECTION_NAMES):
        return errors + ['Expected six sections exactly once and in order']
    sections = {m[1]:text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)].strip() for i,m in enumerate(matches)}
    if not text.startswith('subject_definitions:'):
        errors.append('Prompt has a preamble')
    definitions = sections['subject_definitions']
    if not re.search(r'(?m)^<Picture 1> is',definitions) or not re.search(r'(?m)^<Subject 1> is',definitions):
        errors.append('Missing first-frame or subject definition')
    subject = definitions.split('<Subject 1> is',1)[-1]
    for i in range(2,5):
        if f'<Picture {i}>' not in subject:
            errors.append(f'Missing head reference {i} in subject definition')
    if set(re.findall(r'<Picture (\d+)>',text)) != {'1','2','3','4'} or set(re.findall(r'<Subject (\d+)>',text)) != {'1'}:
        errors.append('Missing/unavailable picture or subject labels')
    if re.search(r'(?m)^<Picture [234]>',definitions + '\n' + sections['retention_analysis']):
        errors.append('Head references must not be standalone defined/retained items')
    if not sections['summary'].startswith('[keyframe completion + reference generation]') or '<Subject 1>' not in sections['summary']:
        errors.append('Incorrect summary mode/subject')
    retention = sections['retention_analysis'].splitlines()
    if len(retention)!=2 or not retention[0].startswith('<Picture 1> ([Shot 1] first frame): fully_preserved - ') or not retention[1].startswith('<Subject 1> (appears in [Shot 1]): fully_preserved - '):
        errors.append('Incorrect two-entry retention analysis')
    detail=sections['detailed_description']
    words=len(re.findall(r"\b[A-Za-z]+(?:[-'][A-Za-z]+)*\b",detail))
    if not 350<=words<=500:
        errors.append(f'Detailed description has {words} words; expected 350-500')
    if re.findall(r'\[Shot (\d+)\]',detail)!=['1'] or not detail.split('[Shot 1]')[0].strip() or '<Subject 1>' not in detail:
        errors.append('Expected style opening, exactly one Shot 1, and Subject 1')
    if re.search(r'<Picture [234]>|\d{2}:\d{2}\.\d{3}',detail):
        errors.append('Head-reference label or timestamp in timeline')
    if re.search(r'<(?:Video|Audio) \d+>|<think>|<d>|\[Shot [2-9]\d*\]',text):
        errors.append('Unavailable media/shot or reasoning/speech wrapper')
    if any(sections[s]!='N/A' for s in SECTION_NAMES[-2:]):
        errors.append('Audio fields must each be exactly N/A')
    return errors


def infer(client, messages, args):
    chunks=[]; reasoning_chars=0; usage=None; finish=None; model=None; response_id=None
    extra={'enable_thinking':args.thinking}
    if args.thinking:
        extra['thinking_budget']=args.thinking_budget
    started=time.monotonic()
    with client.chat.completions.create(model=args.model,messages=messages,stream=True,
            stream_options={'include_usage':True},max_tokens=args.max_tokens,
            temperature=args.temperature,top_p=0.9,extra_body=extra) as stream:
        for chunk in stream:
            model=chunk.model or model;response_id=chunk.id or response_id
            if chunk.usage:usage=chunk.usage.model_dump()
            for choice in chunk.choices:
                if choice.delta.content:chunks.append(choice.delta.content)
                reasoning_chars+=len(getattr(choice.delta,'reasoning_content',None) or '')
                if choice.finish_reason:finish=choice.finish_reason
    return {'content':''.join(chunks),'usage':usage,'finish_reason':finish,'model':model,
            'response_id':response_id,'reasoning_chars':reasoning_chars,'elapsed_seconds':round(time.monotonic()-started,2)}


def generate(args):
    rows=read_jsonl(args.input)
    rows=rows[args.offset: args.offset+args.limit if args.limit is not None else None]
    if not rows:raise ValueError('No rows selected')
    for row in rows:
        validate_input(row)
        if 'candidates' not in row or not isinstance(row.get('duration_seconds'),(int,float)) or not math.isfinite(row['duration_seconds']) or row['duration_seconds']<=0:
            raise ValueError('Run prepare first; candidates and positive duration required')
    out=args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError(f'Nonempty output directory: {out}')
    instructions=args.instruction_file.read_text(encoding='utf-8')
    if not instructions.strip():raise ValueError('Empty instruction file')
    key=os.environ.get('DASHSCOPE_API_KEY','')
    if not key and not args.dry_run:raise ValueError('Set DASHSCOPE_API_KEY in the environment')
    parsed=urlparse(args.base_url)
    if parsed.scheme!='https' or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Use a clean HTTPS API base URL')
    out.mkdir(parents=True,exist_ok=True)
    (out/'instruction_snapshot.txt').write_text(instructions,encoding='utf-8')
    manifest={'input':str(args.input.resolve()),'input_sha256':digest(args.input),'instruction_sha256':digest(args.instruction_file),
              'model':args.model,'base_url':args.base_url,'thinking':args.thinking,'thinking_budget':args.thinking_budget,
              'max_tokens':args.max_tokens,'temperature':args.temperature,'samples':[],'dry_run':args.dry_run}
    client=None
    if not args.dry_run:
        from openai import OpenAI
        client=OpenAI(api_key=key,base_url=args.base_url,timeout=600,max_retries=2)
    failed=False
    with (out/'generated.jsonl').open('w',encoding='utf-8') as generated, (out/'failures.jsonl').open('w',encoding='utf-8') as failures:
        for number,row in enumerate(rows,1):
            prefix=out/f'sample_{number:03}'
            dump(prefix.with_suffix('.input.json'),row)
            entry={'id':row['id'],'status':'planned','attempts':[],'images':[{'label':f'<Picture {i}>','path':p,'sha256':digest(p)} for i,p in enumerate([row['first_frame']]+row['refs'],1)]}
            manifest['samples'].append(entry);dump(out/'manifest.json',manifest)
            print(f'[{number}/{len(rows)}] {row["id"]}: {args.model}',flush=True)
            if args.dry_run:continue
            try:
                messages=build_messages(row,instructions)
                for attempt in range(args.repair_attempts+1):
                    raw=infer(client,messages,args)
                    dump(out/f'{prefix.name}.attempt{attempt+1}.response.json',raw)
                    try:
                        result=parse_response(raw['content']);errors=validate_response(result,row)
                    except (ValueError,TypeError,KeyError) as exc:
                        result=None;errors=[f'Invalid response JSON/schema: {type(exc).__name__}']
                    if raw['finish_reason']!='stop':errors.append(f'Incomplete output: {raw["finish_reason"]}')
                    entry['attempts'].append({k:v for k,v in raw.items() if k!='content'} | {'errors':errors})
                    dump(out/'manifest.json',manifest)
                    if not errors:break
                    if attempt<args.repair_attempts:
                        print(f'  Repairing validation errors: {errors}',flush=True)
                        messages += [{'role':'assistant','content':raw['content']},{'role':'user','content':'Correct these output validation failures without changing the supplied options or image roles. Return the entire JSON envelope again:\n'+'\n'.join(errors)}]
                if errors:
                    entry.update(status='invalid',errors=errors);failed=True
                elif result['status']=='incompatible':
                    entry.update(status='incompatible',result=result);failed=True
                else:
                    chosen={cat:next(c for c in row['candidates'][cat] if c['id']==result['selected_ids'][cat]) for cat in CATEGORIES}
                    output={**row,**result,'selected_candidates':chosen,'generation':{k:v for k,v in raw.items() if k!='content'}}
                    generated.write(json.dumps(output,ensure_ascii=False)+'\n');generated.flush()
                    prefix.with_suffix('.prompt.txt').write_text(result['prompt']+'\n',encoding='utf-8')
                    dump(prefix.with_suffix('.selection.json'),{k:v for k,v in result.items() if k!='prompt'} | {'selected_candidates':chosen})
                    entry.update(status='ok',selected_ids=result['selected_ids'],prompt_file=str(prefix.with_suffix('.prompt.txt')))
            except Exception as exc:
                # Never log request objects, headers, environment or image payloads.
                message=re.sub(r'sk-[A-Za-z0-9_-]+','[REDACTED]',str(exc).replace(key,'[REDACTED]') if key else str(exc))
                entry.update(status='error',error=message[:1500]);failed=True
            if entry['status']!='ok':
                failures.write(json.dumps({'id':row['id'],**entry},ensure_ascii=False)+'\n');failures.flush()
            dump(out/'manifest.json',manifest)
            print(' ',entry['status'],entry.get('errors',entry.get('error','')),flush=True)
    if client:client.close()
    print(f'Manifest: {out / "manifest.json"}',flush=True)
    if failed:raise SystemExit(2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare',help='Uniform random sampling only; no VLM/GPU required')
    p.add_argument('--input',type=Path,required=True);p.add_argument('--vocabulary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--duration',type=float,default=6.0);p.set_defaults(func=prepare)
    g=sub.add_parser('generate',help='Cloud API; sends four still images per sample, never source video')
    g.add_argument('--input',type=Path,required=True);g.add_argument('--output-dir',type=Path,required=True)
    g.add_argument('--instruction-file',type=Path,default=HERE/'ref2va_candidate_prompt.txt')
    g.add_argument('--model',default=DEFAULT_MODEL);g.add_argument('--base-url',default=os.environ.get('DASHSCOPE_BASE_URL',DEFAULT_URL))
    g.add_argument('--limit',type=int);g.add_argument('--offset',type=int,default=0)
    g.add_argument('--max-tokens',type=int,default=10000);g.add_argument('--temperature',type=float,default=0.4)
    g.add_argument('--thinking',action=argparse.BooleanOptionalAction,default=True)
    g.add_argument('--thinking-budget',type=int,default=2048)
    g.add_argument('--repair-attempts',type=int,default=1);g.add_argument('--dry-run',action='store_true')
    g.set_defaults(func=generate)
    args=parser.parse_args()
    if args.command=='prepare' and (not math.isfinite(args.duration) or args.duration<=0):parser.error('Duration must be finite and positive')
    if args.command=='generate' and (args.offset<0 or (args.limit is not None and args.limit<1) or args.max_tokens<1 or args.thinking_budget<1 or args.repair_attempts<0):parser.error('Invalid numeric argument')
    args.func(args)

if __name__=='__main__':main()

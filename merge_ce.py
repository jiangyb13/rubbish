#!/usr/bin/env python3
"""Run seven GSS policies directly from official DPG-Bench result TXT files.

Requires original prompt .txt files and evaluated image grids. Official crop
order: top-left, top-right, bottom-left, bottom-right. All pairs are CE-scored.
"""
import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

def read_results(path, pic_num):
    files = sorted(path.glob('rank*.txt')) if path.is_dir() else [path]
    files = [f for f in files if not f.name.endswith('_detail.txt')]
    if not files: raise ValueError(f'No rank*.txt result files in {path}')
    rows = {}
    for file in files:
        for lineno,line in enumerate(file.read_text().splitlines(),1):
            # Official TXT appends category and overall summary prose.
            if not re.search(r'\.(png|jpg|jpeg|webp)\s*,',line,re.I): continue
            fields = line.rsplit(',',pic_num+1)
            if len(fields)!=pic_num+2: raise ValueError(f'{file}:{lineno}: wrong number of crop scores')
            name=fields[0].strip()
            try: values=[float(v.strip()) for v in fields[1:]]
            except ValueError as e: raise ValueError(f'{file}:{lineno}: not a main result TXT (do not pass _detail.txt)') from e
            if not all(math.isfinite(v) and 0<=v<=1 for v in values): raise ValueError('Expected DPG scores in [0,1]')
            if not math.isclose(sum(values[:-1])/pic_num,values[-1],abs_tol=1e-6): raise ValueError(f'Crop mean mismatch: {name}')
            key=Path(name).stem
            row={'image':name,'scores':values[:-1],'result_file':str(file.resolve())}
            if key in rows: raise ValueError(f'Duplicate prompt ID {key}; use results from one evaluation only')
            rows[key]=row
    if not rows: raise ValueError('No per-image result lines found; overall score alone is insufficient')
    return rows

def prepare(rows,prompts,image_root,out,pic_num,resolution):
    from PIL import Image
    records=[]
    for key,r in sorted(rows.items()):
        promptfile=prompts/f'{key}.txt'
        prompt=promptfile.read_text().strip()
        if not prompt: raise ValueError(f'Empty prompt: {promptfile}')
        original=Path(r['image'])
        imagepath=image_root/original.name if image_root else original
        if not imagepath.is_absolute():imagepath=Path(r['result_file']).parent/imagepath
        imagepath=imagepath.resolve()
        with Image.open(imagepath) as im:
            if pic_num==4:
                side=resolution or im.width//2
                if im.size!=(2*side,2*side):raise ValueError(f'{imagepath}: expected a square 2x2 grid without borders')
                boxes=[(0,0,side,side),(side,0,2*side,side),(0,side,side,2*side),(side,side,2*side,2*side)]
            else:
                if resolution:
                    if min(im.size)<resolution:raise ValueError('Image smaller than evaluation resolution')
                    boxes=[(0,0,resolution,resolution)]
                else:boxes=[(0,0,im.width,im.height)]
            for sample,box in enumerate(boxes):
                dest=out/'crops'/key/f'{sample:05d}.png'
                dest.parent.mkdir(parents=True,exist_ok=True)
                # Lossless crop, no resizing. Matches official evaluated pixels.
                im.crop(box).save(dest)
                records.append({'id':key,'sample':sample,'filename':str(dest.resolve()),'prompt':prompt,'dpg_score':r['scores'][sample],'source_grid':str(imagepath),'crop_box':box})
    manifest=out/'input.jsonl'
    manifest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('base_results',type=Path,help='Official results.txt or directory containing rank*.txt')
    p.add_argument('method_results',type=Path)
    p.add_argument('--prompts-dir',type=Path,required=True,help='Original DPG prompts: <image-stem>.txt')
    p.add_argument('--base-images',type=Path,help='Override old image paths with this directory of grids')
    p.add_argument('--method-images',type=Path)
    p.add_argument('--pic-num',type=int,choices=[1,4],default=4)
    p.add_argument('--resolution',type=int,help='Original evaluation crop resolution; default infer from 2x2 grid')
    p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--model',default='/data/jyb/pretrained_models/BAGEL-7B-MoT')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--ce-vit-max-side',type=int,default=336)
    p.add_argument('--ce-max-tokens',type=int,default=192)
    p.add_argument('--prepare-only',action='store_true',help='Validate TXT and extract crops, without loading BAGEL')
    a=p.parse_args()
    b,m=read_results(a.base_results,a.pic_num),read_results(a.method_results,a.pic_num)
    if b.keys()!=m.keys():raise ValueError('Base/method prompt IDs differ')
    a.outdir.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(a.outdir/'prepare.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    bp=prepare(b,a.prompts_dir,a.base_images,a.outdir/'base_input',a.pic_num,a.resolution)
    mp=prepare(m,a.prompts_dir,a.method_images,a.outdir/'method_input',a.pic_num,a.resolution)
    report={'prompts':len(b),'image_pairs':len(b)*a.pic_num,'base_dpg':sum(sum(r['scores'])/a.pic_num for r in b.values())/len(b)*100,'method_dpg':sum(sum(r['scores'])/a.pic_num for r in m.values())/len(m)*100,'note':'Input scores already dependency-adjusted by DPG; do not binarize. No DPG labels enter CE selection. Category subscores require separate QA details.'}
    (a.outdir/'input_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    if not a.prepare_only:
        subprocess.run([sys.executable,str(Path(__file__).with_name('merge_dpg_gss.py')),str(bp),str(mp),'--outdir',str(a.outdir/'gss'),'--model',a.model,'--device',a.device,'--ce-vit-max-side',str(a.ce_vit_max_side),'--ce-max-tokens',str(a.ce_max_tokens)],check=True)

if __name__=='__main__':main()

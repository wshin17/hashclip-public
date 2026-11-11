"""D-012 low-priority single-code deletion after both higher priorities finish."""
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import shutil
from statistics import mean
import time

from .evaluate_cpu import write_csv
from .final_cpu import quiet
from .opq_timing import read_json, save_json
from .paths import ROOT
from .systems_cpu import BUILD_KEYS, execute, resolve_path


def run(config):
    root = resolve_path(config['output'])
    if not (root/'fastscan_complete.json').exists() or not (root/'breakeven/complete.json').exists():
        raise ValueError('Complete FastScan and cost supplements before deletion timing')
    output = root/'deletion'
    output.mkdir(exist_ok=True)
    done = read_json(output/'results.json') if (output/'results.json').exists() else []
    keys = {(r['dataset'],r['direction'],r['seed']) for r in done}
    for dataset,options in config['datasets'].items():
        final = resolve_path(config['final_systems'])/dataset
        cfg = read_json(final/'config.json')['config']
        method = options['anchor_methods'][0]
        workers = [json.loads(line)['spec'] for line in (final/'workers.jsonl').read_text().splitlines()]
        templates = {(s['direction'],s['seed']):s for s in workers if s['split']=='test' and s['method']==method}
        for (direction,seed),template in sorted(templates.items()):
            if (dataset,direction,seed) in keys:
                continue
            if time.time()+60 >= datetime.fromisoformat(config['deadline']).timestamp():
                save_json(output/'incomplete.json',{'reason':'D-012 deadline; low-priority deletion incomplete','groups':len(done)})
                return
            quiet()
            print('Single-code deletion',dataset,direction,seed,flush=True)
            result = execute({**template,'config':cfg,'mode':'delete'},output/'workers.jsonl')
            done.append({'dataset':dataset,'direction':direction,'seed':seed,**result})
            save_json(output/'results.json',done)
    groups = defaultdict(list)
    for r in done:
        groups[r['dataset'],r['direction']].append(r)
    summary = [{'dataset':dataset,'direction':direction,'seeds':len(rows),
                'trials_per_seed':100,'gallery_size':rows[0]['gallery_size'],
                'delete_p50_ms_mean':mean(r['delete_p50_ms'] for r in rows),
                'delete_p95_ms_mean':mean(r['delete_p95_ms'] for r in rows),
                'scope':'binary code index only; fresh full index per trial; no embedding deletion'}
               for (dataset,direction),rows in groups.items()]
    write_csv(output/'summary.csv',summary)
    save_json(output/'complete.json',{'groups':len(done),'completed_utc':time.time()})
    archive = ROOT/'outputs/tables/icde_systems/breakeven/deletion'
    archive.mkdir(parents=True,exist_ok=True)
    write_csv(archive/'per_seed.csv',[{k:v for k,v in r.items() if k not in ('delete_samples_ms','delete_ids','load')} for r in done])
    for name in ('summary.csv','complete.json'):
        shutil.copyfile(output/name,archive/name)

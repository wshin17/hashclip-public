"""Retime validation frontiers, freeze test curves and selected RSS.

Baseline quality comes from the completed, identical validation grid. Binary
curves include the original fractional M grid and validation target budgets.
Test curves use this frozen set; test measurements never choose configurations.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import time

from .evaluate_cpu import configure, input_identity, write_csv
from .paths import ROOT, runs_root
from .refine_ivf import read_rows
from .systems_cpu import BUILD_KEYS, POINT_KEYS, execute, select_points, selected_groups, report_selected


def frontier(rows, metric='final_recall_at_10'):
    """Minimize latency, maximize quality, retaining a stable representative."""
    ordered = sorted(rows, key=lambda r: (r['amortized_p50_ms'], -r[metric]))
    best, keep = -1.0, []
    for row in ordered:
        if row[metric] > best:
            keep.append(row)
            best = row[metric]
    return keep


def point_key(row):
    return tuple(row.get(k) for k in POINT_KEYS)


def frontiers(rows):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in ('method','seed','direction','batch')), []).append(r)
    selected = {}
    for group in groups.values():
        # Include complete-recovery frontier as well, since both curves are
        # requested. This does not revisit the full parameter grid.
        for metric in ('final_recall_at_10', 'final_complete_at_10'):
            for r in frontier(group, metric):
                selected[point_key(r)] = r
    return list(selected.values())


def cost_points(selected, validation):
    """Include a labeled cost point when a method misses every quality target.

    Choose its highest observed validation quality (latency breaks ties).
    This point supplies build/insertion/RSS coverage, never a passed target.
    """
    keys = ('method','seed','direction','batch')
    points = [p for p in selected if p['target_met_on_validation']]
    covered = {tuple(p[k] for k in keys) for p in points}
    groups = {}
    for r in validation:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    for key, rows in groups.items():
        if key not in covered:
            best = max(rows, key=lambda r: (r['final_recall_at_10'], -r['amortized_p50_ms']))
            points.append({**{k:best.get(k) for k in POINT_KEYS},
                           'validation_recall_at_10':best['final_recall_at_10'],
                           'target_mean_recall_at_10':None, 'target_met_on_validation':False,
                           'cost_only':True})
    return points


def quiet():
    """Wait for authorized training/encoding and high P-core load to finish."""
    from .cpu_load import snapshot, difference
    while True:
        busy = []
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():
                continue
            try:
                args = (p/'cmdline').read_bytes().decode(errors='replace').split('\0')
                if any(m in args for m in ('hashclip.reselect','hashclip.coco_large_encode','hashclip.representation_quality','hashclip.ablation_quality')):
                    busy.append(int(p.name))
            except OSError:
                pass
        before = snapshot()
        time.sleep(2)
        sample = difference(before, snapshot())
        others = [p for p in sample['processes_above_5_percent']
                  if p['cpu_percent'] >= 20 and set(p['affinity']) & set(range(16))
                  and p['name'] not in ('final_cpu',)]
        if not busy and not others:
            return sample
        print('Waiting for quiet CPU/GPU execution window', {'training_or_quality':busy,'p_core_load':others}, flush=True)
        time.sleep(30)


def run_dataset(dataset, source, output, training=None, *, domain_quality=None, enforce_quiet=True):
    import yaml
    source = Path(source)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Use a new final measurement directory')
    meta = json.loads((source/'config.json').read_text())
    for item in meta['inputs']:
        if input_identity(Path(item['path'])) != item:
            raise ValueError(f"Input changed since validation: {item['path']}")
    cfg = {**meta['config'], 'record_load':True, 'quality_workers':8}
    configure(cfg)
    workers = [json.loads(line) for line in (source/'workers.jsonl').read_text().splitlines()]
    templates = {tuple(w['spec'].get(k) for k in BUILD_KEYS): w['spec']
                 for w in workers if w['spec']['split']=='val'}
    old = read_rows(source/'val.csv')
    natives = frontiers([r for r in old if r['method'] not in ('hashclip','lsh')])
    all_points = list(natives)
    domain_inputs = []
    for key, spec in list(templates.items()):
        if spec['method'] not in ('hashclip','lsh'):
            continue
        n = next(r['gallery_size'] for r in old if tuple(r.get(k) for k in BUILD_KEYS)==key)
        counts = {10, *[min(n,max(10,math.ceil(f*n))) for f in (.001,.002,.005,.01,.02,.05,.1,.2,.3,.5,.75,1)]}
        counts.update(r['M'] for r in old if tuple(r.get(k) for k in BUILD_KEYS)==key)
        for batch in cfg['batch_sizes']:
            all_points.extend({**{k:spec.get(k) for k in BUILD_KEYS}, 'batch':batch,'M':m,
                               'nprobe':None,'ef_search':None} for m in sorted(counts))
    if training and dataset in ('coco','coco_large'):
        trained = yaml.safe_load(Path(training).read_text())
        domain_inputs.append(input_identity(Path(training)))
        for seed, run in trained['training_runs'].items():
            checkpoint = runs_root()/run/'checkpoints'/f'hashclip_b128_seed{seed}_reselect.pt'
            for direction in ('t2i','i2t'):
                original = next(s for s in templates.values() if s['method']=='hashclip' and s['seed']==seed and s['direction']==direction)
                spec = {**original, 'method':'hashclip_coco', 'checkpoint':str(checkpoint)}
                key = tuple(spec.get(k) for k in BUILD_KEYS)
                templates[key] = spec
                n = next(r['gallery_size'] for r in old if r['direction']==direction)
                # Include same-domain minimal validation targets measured in
                # the full-query quality job; never consult its test results.
                if domain_quality is None:
                    raise ValueError('Supply the same-domain validation-quality directory')
                quality = Path(domain_quality)/dataset
                from .numpy_adapter import numpy_checkpoint_path
                identity = input_identity(numpy_checkpoint_path(checkpoint))
                quality_meta = json.loads((quality/'config.json').read_text())
                expected = next(p for p in quality_meta['checkpoints'] if Path(p['path']).name == numpy_checkpoint_path(checkpoint).name)
                if (identity['sha256'],identity['bytes']) != (expected['sha256'],expected['bytes']):
                    raise ValueError('Same-domain checkpoint changed since quality evaluation')
                if direction == 't2i':
                    domain_inputs.append(identity)
                if seed == 0 and direction == 't2i':
                    domain_inputs.extend(input_identity(quality/n) for n in ('config.json','selected.json'))
                selected = json.loads((quality/'selected.json').read_text())[f'{direction}/hashclip/{seed}']
                counts = {10,*selected.values(),*[min(n,max(10,math.ceil(f*n))) for f in (.001,.002,.005,.01,.02,.05,.1,.2,.3,.5,.75,1)]}
                for batch in cfg['batch_sizes']:
                    all_points.extend({**{k:spec.get(k) for k in BUILD_KEYS},'batch':batch,'M':m,
                                       'nprobe':None,'ef_search':None} for m in sorted(counts))
    # Freeze the entire curve configuration set before inspecting any final test.
    manifest = {'source':input_identity(source/'config.json'),'config':cfg,'dataset':dataset,
                'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'validation_selection':'union of latency/retention and latency/complete frontiers per method; binary M grid',
                'frozen_curve_points':all_points,'in_domain_training':str(training) if training else None,
                'same_domain_inputs':domain_inputs,
                'same_domain_large_gallery_contains_training_images':dataset=='coco_large' and bool(training)}
    (output/'config.json').write_text(json.dumps(manifest,indent=2)+'\n')

    def spec_for(group, split):
        key = tuple(group.get(k) for k in BUILD_KEYS)
        spec = deepcopy(templates[key])
        spec.pop('budgets',None)
        spec.update(group,config=cfg,split=split)
        paths = meta['dataset_paths']
        spec.update(features=str(Path(paths['queries'])/f'backbone_{split}.npz'),
                    gallery_features=str(Path(paths['gallery'])/f'backbone_{split}.npz'),
                    reference=str(Path(paths['references'])/f'{split}_reference.npz'))
        return spec

    val = []
    groups = selected_groups([{**p,'target_met_on_validation':True} for p in all_points])
    for group in groups:
        if enforce_quiet:
            quiet()
        spec = spec_for(group,'val')
        if spec['method'] not in ('hashclip','hashclip_coco','lsh'):
            spec['quality_rows'] = [r for r in natives if tuple(r.get(k) for k in BUILD_KEYS)==tuple(group.get(k) for k in BUILD_KEYS)]
        print('final validation',dataset,{k:v for k,v in group.items() if k!='points'},flush=True)
        val.extend(execute(spec,output/'workers.jsonl')['rows'])
        write_csv(output/'val.csv',val)
    chosen = select_points(val,cfg['quality_targets'])
    (output/'selected.json').write_text(json.dumps(chosen,indent=2)+'\n')
    costs = cost_points(chosen,val)
    (output/'cost_points.json').write_text(json.dumps(costs,indent=2)+'\n')
    test = []
    for group in groups:
        if enforce_quiet:
            quiet()
        print('final test curve',dataset,{k:v for k,v in group.items() if k!='points'},flush=True)
        test.extend(execute(spec_for(group,'test'),output/'workers.jsonl')['rows'])
        write_csv(output/'test.csv',test)
    write_csv(output/'selected_test.csv',report_selected(chosen,test))
    memory, seen = [],set()
    for point in costs:
        key = point_key(point)
        if key in seen:
            continue
        seen.add(key)
        if enforce_quiet:
            quiet()
        spec = spec_for(point,'test')
        spec.update(mode='memory',point=point)
        memory.append({**point,**execute(spec,output/'memory_workers.jsonl')})
        (output/'memory.json').write_text(json.dumps(memory,indent=2)+'\n')
    test_lookup = {point_key(r):r for r in test}
    memory_lookup = {point_key(r):r for r in memory}
    write_csv(output/'costs.csv', [{**test_lookup[point_key(p)], **p,
        'rss_after_query_bytes':memory_lookup[point_key(p)]['rss_after_query_bytes']}
        for p in costs])
    print('Completed final',dataset,len(val),len(test),len(memory),flush=True)
    if dataset != 'fixture':
        from .plot_cpu import plot_systems
        plot_systems(test,report_selected(chosen,test),output/'plots',dataset)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--training',type=Path,default=ROOT/'configs/icde_coco_reselect.yaml')
    p.add_argument('--domain-quality',type=Path,required=True)
    p.add_argument('--datasets',nargs='+',default=['coco','flickr_large','coco_large'])
    a=p.parse_args()
    for name in a.datasets:
        run_dataset(name,a.source/name,a.output/name,a.training,domain_quality=a.domain_quality)


if __name__=='__main__':
    main()

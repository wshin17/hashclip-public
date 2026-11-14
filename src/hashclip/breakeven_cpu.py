"""Validation-selected minimum total cost and fixed test supplements.

Keep early cheap-build/slower-query configurations: the strict lower envelope
can beat HashCLIP at Q=0, then lose, then win again. Report both endpoints of
HashCLIP's advantage interval as well as the faster-query catch-up point.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
from statistics import mean
import time

from .evaluate_cpu import input_identity, write_csv
from .final_cpu import point_key, quiet
from .opq_timing import read_json, save_json
from .paths import ROOT, runs_root
from .refine_ivf import read_rows
from .systems_cpu import BUILD_KEYS, POINT_KEYS, execute, resolve_path, selected_groups


def interval(hbuild, hquery, candidates):
    """Strict HC advantage over every line, with ties at interval endpoints."""
    lower, upper = 0., math.inf
    entry = exit_point = blocker = None
    initial = min(candidates,key=lambda r:(r['build_ms'],r['batch_p50_ms'])) if candidates else None
    for row in candidates:
        intercept = row['build_ms']-hbuild
        slope = hquery-row['batch_p50_ms']
        if slope > 0:
            crossing = intercept/slope
            if crossing < upper:
                upper, exit_point = crossing, row
        elif slope < 0:
            crossing = intercept/slope
            if crossing > lower:
                lower, entry = crossing, row
        elif intercept <= 0:
            blocker = row
    return {'lower_queries':lower, 'upper_queries':None if math.isinf(upper) else max(0.,upper),
            'has_advantage_interval':blocker is None and upper > lower,
            'strict_first_not_cheaper_queries':0. if initial and initial['build_ms'] <= hbuild
                    else None if math.isinf(upper) else max(0.,upper),
            'entry':entry,'exit':exit_point,'blocker':blocker,'initial':initial}


def load_rows(path, tag):
    return [{**r,'source':str(path),'source_round':tag} for r in read_rows(path)] if path.exists() else []


def training_records():
    import yaml
    records = []
    for dataset, file in (('flickr_large','icde_reselect.yaml'),('coco_large','icde_coco_reselect.yaml')):
        config = yaml.safe_load((ROOT/'configs'/file).read_text())
        for seed, run in config['training_runs'].items():
            folder = runs_root()/run
            result, meta = read_json(folder/'result.json'), read_json(folder/'config.json')
            records.append({'dataset':dataset,'seed':seed,'device':meta['device'],
                'training_seconds':sum(h['train']['time_sec'] for h in result['history']),
                'train_and_validation_seconds':result.get('train_and_validation_seconds'),
                'source':input_identity(folder/'result.json'),
                'policy':'full 15-epoch trajectory; training only; no shared encoder extraction or validation sweep'})
    return records


def make_plan(config, output):
    """Only validation numbers choose configurations; test values are not read."""
    if (output/'plan.json').exists():
        plan = read_json(output/'plan.json')
        if not plan.get('cost_interval') or any(g['basis']!='reusable' for g in plan['groups']):
            raise ValueError('Outdated cost plan; remove plan.json and rerun')
        return plan
    train = training_records()
    groups, sources, grids = [], [], []
    first, final = resolve_path(config['source_systems']), resolve_path(config['final_systems'])
    for dataset, options in config['datasets'].items():
        files = [(first/dataset/'val.csv','original_grid'),(final/dataset/'val.csv','final_retiming')]
        validation = {}
        for file, tag in files:
            sources.append(input_identity(file))
            validation.update({point_key(r):r for r in load_rows(file,tag)})
        selection_file = final/dataset/'selected.json'
        # The final run stores frozen target choices in its configuration snapshot.
        meta = read_json(final/dataset/'config.json')
        selected = meta['selected'] if 'selected' in meta else None
        if selected is None:
            # Target files contain test results, but ONLY the frozen validation
            # fields and parameter identities are used here, never test costs.
            selection_file = final/dataset/options['anchor_selection']
            sources.append(input_identity(selection_file))
            selected = [{k:r.get(k) for k in POINT_KEYS+('target_mean_recall_at_10',
                        'target_met_on_validation','validation_recall_at_10')}
                        for r in read_rows(selection_file)]
        hmethod = options['anchor_methods'][0]
        hselected = [r for r in selected if r['method']==hmethod and r['batch']==1]
        old_file = final/dataset/'all_targets_selected_test.csv'
        sources.append(input_identity(old_file))
        manuscript = [{k:r.get(k) for k in POINT_KEYS+('target_mean_recall_at_10',
                        'target_met_on_validation','validation_recall_at_10')}
                      for r in read_rows(old_file)]
        for direction in ('t2i','i2t'):
            for target in (.95,.99):
                hp = [r for r in hselected if r['direction']==direction and r['target_mean_recall_at_10']==target]
                if {r['seed'] for r in hp} != {0,1,42}:
                    raise ValueError('Missing same-domain HashCLIP validation targets')
                hrows = [validation[point_key(p)] for p in hp]
                hbuild = mean(r['build_ms'] for r in hrows)
                hquery = mean(r['batch_p50_ms'] for r in hrows)
                for method in ('hnsw','ivf_flat'):
                    eligible = [r for r in validation.values() if r['method']==method and r['direction']==direction
                                and r['batch']==1 and r['final_recall_at_10']>=target]
                    if not eligible:
                        raise ValueError('No eligible validation configuration')
                    old = next(r for r in manuscript if r['method']==method and r['direction']==direction
                               and r['batch']==1 and r['target_mean_recall_at_10']==target)
                    bounds = interval(hbuild,hquery,eligible)
                    roles = {k:bounds[k] for k in ('entry','exit','blocker') if bounds[k] is not None}
                    roles['manuscript'] = validation[point_key(old)]
                    groups.append({'dataset':dataset,'direction':direction,'target':target,'method':method,
                        'basis':'reusable','hashclip_validation_build_ms':hbuild,
                        'hashclip_validation_query_ms':hquery,'hashclip_gpu_train_ms':0.,
                        'hashclip_points':hp,'eligible_count':len(eligible),
                        'validation_interval':{k:v for k,v in bounds.items() if k not in ('entry','exit','blocker','initial')},
                        'roles':roles})
                    grids.extend({**{k:r.get(k) for k in POINT_KEYS},'dataset':dataset,'target':target,
                                  'build_ms':r['build_ms'],'batch_p50_ms':r['batch_p50_ms'],
                                  'validation_recall_at_10':r['final_recall_at_10'],
                                  'source':r['source'],'source_round':r['source_round']} for r in eligible)
    plan = {'cost_interval':'reusable-model interval only; both endpoints frozen on validation',
            'created_utc':time.time(),'inputs':sources,'groups':groups,
            'adapter_training':train,
            'selection_policy':'validation-qualified full grids; final validation retiming overrides identical points; test never chooses parameters',
            'first_deployment_policy':'report GPU training and multithread CPU fitting with hardware separately; no time/resource conversion and no first-deployment interval',
            'offline_policy':'full adapter training trajectory recorded; cached backbone features; excludes encoder extraction and model-selection validation'}
    save_json(output/'plan.json',plan)
    write_csv(output/'validation_costs.csv',grids)
    return plan


def test_sources(config, dataset, output):
    first, final = resolve_path(config['source_systems']), resolve_path(config['final_systems'])
    rows = {}
    for file, tag in ((first/dataset/'test.csv','original_test'),
                      (final/dataset/'test.csv','final_test'),(output/dataset/'test.csv','009_supplement')):
        rows.update({point_key(r):r for r in load_rows(file,tag)})
    return rows


def supplement(config, plan, output, enforce_quiet=True):
    for dataset in config['datasets']:
        folder = output/dataset
        folder.mkdir(exist_ok=True)
        requested = {point_key(r):r for g in plan['groups'] if g['dataset']==dataset for r in g['roles'].values()}
        existing = test_sources(config,dataset,output)
        missing = [{**r,'target_met_on_validation':True} for k,r in requested.items() if k not in existing]
        source = resolve_path(config['source_systems'])/dataset
        final = resolve_path(config['final_systems'])/dataset
        cfg = read_json(final/'config.json')['config']
        workers = [json.loads(line)['spec'] for line in (source/'workers.jsonl').read_text().splitlines()]
        templates = {tuple(s.get(k) for k in BUILD_KEYS):s for s in workers if s['split']=='val'}
        paths = read_json(source/'config.json')['dataset_paths']
        rows = read_rows(folder/'test.csv') if (folder/'test.csv').exists() else []
        for group in selected_groups(missing):
            if config.get('deadline') and time.time() >= datetime.fromisoformat(config['deadline']).timestamp():
                raise TimeoutError('Deadline reached')
            if enforce_quiet:
                quiet()
            template = templates[tuple(group.get(k) for k in BUILD_KEYS)]
            spec = {**template,**group,'config':cfg,'split':'test',
                    'features':str(Path(paths['queries'])/'backbone_test.npz'),
                    'gallery_features':str(Path(paths['gallery'])/'backbone_test.npz'),
                    'reference':str(Path(paths['references'])/'test_reference.npz'),
                    'insertion_features':str(Path(paths['queries'])/'backbone_val.npz')}
            spec.pop('quality_rows',None)
            print('Supplement total-cost test',dataset,{k:group.get(k) for k in BUILD_KEYS},flush=True)
            rows.extend(execute(spec,folder/'workers.jsonl')['rows'])
            write_csv(folder/'test.csv',rows)


def report(config, plan, output):
    """Evaluate frozen validation endpoint choices, not a new test optimum."""
    results, selected_test = [], []
    for g in plan['groups']:
        lookup = test_sources(config,g['dataset'],output)
        hc = [lookup[point_key(p)] for p in g['hashclip_points']]
        hb = mean(r['build_ms'] for r in hc)+g['hashclip_gpu_train_ms']
        ht = mean(r['batch_p50_ms'] for r in hc)
        roles = {k:lookup[point_key(r)] for k,r in g['roles'].items()}
        # Evaluate only the active constraints frozen on validation. The
        # old/fastest configurations are comparison columns, not candidates
        # for choosing a new test-optimal configuration.
        selected = [roles[k] for k in ('entry','exit','blocker') if k in roles]
        bounds = interval(hb,ht,selected)
        out = {k:g[k] for k in ('dataset','direction','target','method','basis','eligible_count')}
        out.update(hashclip_cpu_build_ms=hb-g['hashclip_gpu_train_ms'],hashclip_gpu_training_ms=g['hashclip_gpu_train_ms'],
                   hashclip_query_ms=ht,hashclip_test_retention_mean=mean(r['final_recall_at_10'] for r in hc),
                   hashclip_test_target_met=sum(r['final_recall_at_10']>=g['target'] for r in hc),
                   hashclip_test_source=hc[0]['source'],
                   **{k:v for k,v in bounds.items() if k not in ('entry','exit','blocker','initial')},
                   scope='test costs at validation-frozen active/old configurations; not re-optimized on test')
        for role,row in roles.items():
            b = interval(hb,ht,[row])
            out[role+'_query_faster_than_hashclip'] = row['batch_p50_ms']<ht
            out[role+'_build_cheaper_than_hashclip'] = row['build_ms']<hb
            out[role+'_crossing_queries'] = b['upper_queries'] if row['batch_p50_ms']<ht else b['lower_queries'] if row['build_ms']<hb else None
            selected_test.append({**{k:g[k] for k in ('dataset','direction','target','method','basis')},'role':role,
                                 **row,'test_target_met':row['final_recall_at_10']>=g['target'],
                                 'validation_source':g['roles'][role]['source']})
        results.append(out)
    fastscan_checks = []
    for dataset in config['datasets']:
        folder = resolve_path(config['output'])/dataset
        fs = read_rows(folder/'selected_test.csv')
        anchors = {point_key(r):r for r in read_rows(folder/'anchors/test.csv')}
        for direction in ('t2i','i2t'):
            for target in (.95,.99):
                points = next(g['hashclip_points'] for g in plan['groups'] if g['dataset']==dataset
                              and g['direction']==direction and g['target']==target)
                h = [anchors[point_key(p)] for p in points]
                f = [r for r in fs if r['direction']==direction and r['batch']==1 and r['target_mean_recall_at_10']==target]
                if {r['seed'] for r in f} != {0,1,42}:
                    raise ValueError('Missing three-seed FastScan fixed targets')
                hq,fq = mean(r['batch_p50_ms'] for r in h),mean(r['batch_p50_ms'] for r in f)
                fastscan_checks.append({'dataset':dataset,'direction':direction,'target':target,
                    'fastscan_query_ms':fq,'hashclip_anchor_query_ms':hq,'ratio_fastscan_over_hashclip':fq/hq,
                    'fastscan_faster':fq<hq,'source':str(folder),'selection':'same M fixed on validation before test'})
                if fq>=hq:
                    continue
                hb,fb = mean(r['build_ms'] for r in h),mean(r['local_build_ms'] for r in f)
                bounds = interval(hb,hq,[{'build_ms':fb,'batch_p50_ms':fq}])
                results.append({'dataset':dataset,'direction':direction,'target':target,'method':'fastscan','basis':'reusable',
                    'eligible_count':1,'hashclip_cpu_build_ms':hb,'hashclip_gpu_training_ms':0.,'hashclip_query_ms':hq,
                    'baseline_cpu_build_ms':fb,'baseline_multithread_fit_ms':0.,
                    'baseline_query_ms':fq,'baseline_test_retention_mean':mean(r['final_recall_at_10'] for r in f),
                    'baseline_test_target_met':sum(r['target_met_on_test'] for r in f),
                    **{k:v for k,v in bounds.items() if k not in ('entry','exit','blocker','initial')},
                    'scope':'three-seed means at validation-frozen M; same-session HashCLIP anchors; no test parameter selection'})
    write_csv(output/'fastscan_checks.csv',fastscan_checks)
    write_csv(output/'breakeven.csv',results)
    write_csv(output/'selected_test.csv',selected_test)
    offline = [{k:v for k,v in r.items() if k!='source'} | {
        'method':'hashclip','modality':'paired adapters','threads':None,
        'cost_type':'GPU training elapsed seconds','source':r['source']['path']}
        for r in plan['adapter_training']]
    for dataset,options in config['datasets'].items():
        for method,manifest,host in (
            ('fastscan',resolve_path(options['models'])/'manifest.json','workstation'),
            ('opq',resolve_path(options['training_source']),'fitting-server')):
            meta = read_json(manifest)
            offline.extend({'dataset':dataset,'method':method,'modality':r['modality'],'seed':r['seed'],
                            'training_seconds':r['training_seconds'],'device':meta.get('host',host),
                            'threads':meta['threads'],'cost_type':'CPU fitting elapsed seconds; not CPU-thread seconds',
                            'source':str(manifest)} for r in meta['models'] if r['method']==method)
    write_csv(output/'offline_costs.csv',offline)
    save_json(output/'complete.json',{'groups':len(results),'completed_utc':time.time()})
    archive = ROOT/'outputs/tables/icde_systems/breakeven'
    archive.mkdir(parents=True,exist_ok=True)
    for name in ('plan.json','validation_costs.csv','breakeven.csv','selected_test.csv',
                 'fastscan_checks.csv','offline_costs.csv','complete.json'):
        shutil.copyfile(output/name,archive/name)


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/icde_fastscan.yaml')
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--wait-fastscan',action='store_true')
    parser.add_argument('--delete-after',action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    root = resolve_path(config['output'])
    output = root/'breakeven'
    output.mkdir(parents=True,exist_ok=True)
    plan = make_plan(config,output)
    if args.plan_only:
        for dataset in config['datasets']:
            requested = {point_key(r):r for g in plan['groups'] if g['dataset']==dataset for r in g['roles'].values()}
            have = test_sources(config,dataset,output)
            print(dataset,'missing test points',sum(k not in have for k in requested),flush=True)
        return
    if args.wait_fastscan:
        while not (root/'fastscan_complete.json').exists():
            if config.get('deadline') and time.time() >= datetime.fromisoformat(config['deadline']).timestamp():
                raise TimeoutError('Deadline reached while waiting for FastScan timing')
            time.sleep(30)
    supplement(config,plan,output)
    report(config,plan,output)
    if args.delete_after:
        from .code_deletion_cpu import run
        run(config)


if __name__ == '__main__':
    main()

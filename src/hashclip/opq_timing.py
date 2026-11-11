"""D-011: fixed OPQ operating points, full-query quality, timing and RSS."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import time

from .evaluate_cpu import input_identity, write_csv
from .final_cpu import point_key, quiet
from .paths import ROOT
from .refine_ivf import read_rows
from .systems_cpu import BUILD_KEYS, POINT_KEYS, execute, report_selected, resolve_path, selected_groups


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def check_identity(path, expected):
    identity = input_identity(Path(path))
    if (identity['sha256'],identity['bytes']) != (expected['sha256'],expected['bytes']):
        raise ValueError(f'Input identity differs: {path}')
    return identity


def summarize(selected, costs):
    grouped = defaultdict(list)
    cost_lookup = {point_key(r):r for r in costs}
    for row in selected:
        grouped[row['direction'],row['batch'],row['target_mean_recall_at_10']].append(row)
    metrics = ('M','final_recall_at_10','final_complete_at_10','batch_p50_ms','batch_p95_ms',
               'amortized_p50_ms','throughput_qps','build_ms','recorded_fit_ms','local_build_ms',
               'gallery_prepare_clone_ms','gallery_encode_write_ms','reranker_prepare_ms',
               'profile_query_normalize_ms','profile_query_rotation_ms','profile_query_prepare_ms',
               'profile_search_ms','profile_rerank_ms','insert_p50_ms','insert_p95_ms',
               'insert_mean_ms','insert_capacity_prepare_ms','insert_amortized_total_ms')
    result = []
    for (direction,batch,target),rows in grouped.items():
        out = {'method':rows[0]['method'],'direction':direction,'batch':batch,'target':target,'seeds':len(rows),
               'test_target_met':sum(r['target_met_on_test'] for r in rows)}
        out.update({k+'_mean':sum(r[k] for r in rows)/len(rows) for k in metrics})
        for key in ('boundary_tie_query_rate','candidate_set_mismatch_batch_1_32',
                    'candidate_order_mismatch_batch_1_32'):
            if key in rows[0]:
                out[key+'_mean'] = sum(r[key] for r in rows)/len(rows)
        rss = [cost_lookup[point_key(r)]['rss_after_query_bytes']/2**20 for r in rows]
        out.update(rss_mib_mean=sum(rss)/len(rss),rss_mib_min=min(rss),rss_mib_max=max(rss))
        result.append(out)
    return result


def run_dataset(name, options, config, output, *, enforce_quiet=True):
    method = config.get('method','opq')
    if method not in ('opq','fastscan'):
        raise ValueError(method)
    output = Path(output)/name
    output.mkdir(parents=True,exist_ok=True)
    if (output/'complete.json').exists():
        return read_json(output/'complete.json')
    start = time.time()
    source = resolve_path(config['source_systems'])/name
    final_source = resolve_path(config['final_systems'])/name
    meta = read_json(source/'config.json')
    cfg = read_json(final_source/'config.json')['config']
    if cfg != {**meta['config'],'record_load':True,'quality_workers':8}:
        raise ValueError('OPQ and final timing settings differ')
    if cfg['threads'] != 1 or cfg['batch_sizes'] != [1,32]:
        raise ValueError('D-011 uses one CPU thread and batches 1/32')
    models = resolve_path(options['models'])
    if method == 'opq':
        representation = read_json(resolve_path(options['representation_source']))
        models_identity = check_identity(models/'manifest.json',representation['training_models'])
    else:
        models_identity = input_identity(models/'manifest.json')
    model_manifest = read_json(models/'manifest.json')
    trained = {}
    model_identities = []
    for row in model_manifest['models']:
        if row['method'] != method:
            continue
        model_file = models/Path(row['artifact']['path']).name
        model_identities.append(check_identity(model_file,row['artifact']))
        trained[row['seed'],row['modality']] = {**row,'path':str(model_file)}
    if set(trained) != {(s,m) for s in (0,1,42) for m in ('img','txt')}:
        raise ValueError('Expected all six frozen candidate templates')
    checked = set()
    for item in meta['inputs']:
        path = resolve_path(item['path'])
        if str(path.resolve()) not in checked:
            check_identity(path,item)
            checked.add(str(path.resolve()))
    identity = {'task':'008' if method=='opq' else '009','method':method,'dataset':name,'source':input_identity(source/'config.json'),
                'final_source':input_identity(final_source/'config.json'),'config':cfg,
                'models_manifest':models_identity,'models':model_identities,
                'fit_host':model_manifest.get('host',config.get('fit_host')),
                'fit_threads':model_manifest['threads'],'training_policy':model_manifest['training_policy'],
                'selection':'fixed Flickr validation budgets' if options.get('fixed_selection') else
                            'minimum full-validation top-10 retention M; fixed before test',
                'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'large_gallery_contains_training_images':True}
    if options.get('fixed_selection'):
        identity['fixed_selection_source'] = input_identity(resolve_path(options['fixed_selection']))
    if (output/'config.json').exists():
        old = read_json(output/'config.json')
        for k in ('source','final_source','config','models_manifest','models','fixed_selection_source'):
            if old.get(k) != identity.get(k):
                raise ValueError(f'Cannot resume changed experiment input: {k}')
    else:
        save_json(output/'config.json',identity)

    def before_worker():
        if time.time() >= datetime.fromisoformat(config['deadline']).timestamp():
            raise TimeoutError('Execution deadline reached; preserve completed work')
        if enforce_quiet:
            quiet()

    def base_spec(seed,direction,split):
        paths = meta['dataset_paths']
        model = trained[seed,'img' if direction=='t2i' else 'txt']
        return {'method':method,'seed':seed,'direction':direction,
                'pq_m':32 if method=='fastscan' else 16,'pq_bits':4 if method=='fastscan' else 8,
                'nlist':None,'hnsw_m':None,'ef_construction':None,'config':cfg,'split':split,
                'features':str(Path(paths['queries'])/f'backbone_{split}.npz'),
                'gallery_features':str(Path(paths['gallery'])/f'backbone_{split}.npz'),
                'reference':str(Path(paths['references'])/f'{split}_reference.npz'),
                'insertion_features':str(Path(paths['queries'])/'backbone_val.npz'),
                method+'_model':model['path'], method+'_training':{'training_seconds':model['training_seconds'],
                    'host':identity['fit_host'],'threads':model_manifest['threads']}}

    if (output/'selected.json').exists():
        chosen = read_json(output/'selected.json')
    else:
        selected_by_group = read_json(output/'selection_groups.json') if (output/'selection_groups.json').exists() else {}
        prior = read_rows(resolve_path(options['fixed_selection'])) if options.get('fixed_selection') else []
        chosen = []
        for direction in ('t2i','i2t'):
            for seed in (0,1,42):
                spec = base_spec(seed,direction,'val')
                if prior:
                    values = {str(r['target']):r['M'] for r in prior
                              if r['method']=='opq' and r['seed']==seed and r['direction']==direction}
                    evidence = {'selected':values,'source':'fixed prior validation selection'}
                else:
                    group_key = f'{direction}/{seed}'
                    if group_key not in selected_by_group:
                        before_worker()
                        spec.update(mode='select_'+method,positions_output=str(output/f'val_{direction}_seed{seed}_positions.npy'))
                        print('Selecting budgets',method,name,direction,seed,flush=True)
                        selected_by_group[group_key] = execute(spec,output/'selection_workers.jsonl')
                        save_json(output/'selection_groups.json',selected_by_group)
                    evidence = selected_by_group[group_key]
                    values = evidence['selected']
                for batch in cfg['batch_sizes']:
                    for target in cfg['quality_targets']:
                        m = int(values[str(target)])
                        metrics = evidence.get('validation_metrics',{}).get(str(target),{})
                        if method == 'fastscan':
                            metrics = metrics[str(batch)]
                        chosen.append({**{k:spec.get(k) for k in BUILD_KEYS},'batch':batch,'M':m,
                            'nprobe':None,'ef_search':None,'target_mean_recall_at_10':target,
                            'target_met_on_validation':True,
                            'validation_recall_at_10':metrics.get('final_recall_at_10')})
        save_json(output/'selected.json',chosen)
    rows = read_rows(output/'test.csv') if (output/'test.csv').exists() else []
    have = {point_key(r) for r in rows}
    for group in selected_groups([p for p in chosen if point_key(p) not in have]):
        before_worker()
        spec = {**base_spec(group['seed'],group['direction'],'test'),**group}
        print('Timing',method,name,group['direction'],group['seed'],flush=True)
        rows.extend(execute(spec,output/'workers.jsonl')['rows'])
        write_csv(output/'test.csv',rows)
    selected = report_selected(chosen,rows)
    write_csv(output/'selected_test.csv',selected)
    if options.get('fixed_selection'):
        expected = {(r['direction'],r['seed'],r['target']):r
                    for r in read_rows(resolve_path(options['fixed_selection'])) if r['method']=='opq'}
        differences = []
        for row in selected:
            previous = expected[row['direction'],row['seed'],row['target_mean_recall_at_10']]
            delta = row['final_recall_at_10']-previous['candidate_recall_at_10']
            full_delta = row['final_complete_at_10']-previous['candidate_complete_at_10']
            differences.append({**{k:row.get(k) for k in POINT_KEYS},'retention_difference':delta,
                                'complete_difference':full_delta,'prior_batch':32})
        write_csv(output/'prior_quality_comparison.csv',differences)
        if any(abs(r['retention_difference'])>1e-12 or abs(r['complete_difference'])>1e-12
               for r in differences if r['batch']==32):
            raise ValueError('Flickr OPQ batch-32 quality differs from the prior result; inspect before continuing')
    memory = read_json(output/'memory.json') if (output/'memory.json').exists() else []
    done = {point_key(r) for r in memory}
    for point in {point_key(p):p for p in chosen}.values():
        if point_key(point) in done:
            continue
        before_worker()
        spec = base_spec(point['seed'],point['direction'],'test')
        spec.update(mode='memory',point=point)
        memory.append({**point,**execute(spec,output/'memory_workers.jsonl')})
        save_json(output/'memory.json',memory)
    mem = {point_key(r):r for r in memory}
    costs = [{**r,'rss_after_query_bytes':mem[point_key(r)]['rss_after_query_bytes']} for r in selected]
    write_csv(output/'costs.csv',costs)
    write_csv(output/'summary.csv',summarize(selected,costs))

    if options.get('anchors'):
        anchor_out = output/'anchors'
        anchor_out.mkdir(exist_ok=True)
        selected_file = options.get('anchor_selection','selected_test.csv')
        anchor_methods = options.get('anchor_methods',['hashclip','flat_ip'])
        previous = [r for r in read_rows(final_source/selected_file) if r['method'] in anchor_methods]
        unique = list({point_key(r):r for r in previous}.values())
        worker_source = final_source if method=='fastscan' else source
        workers = [json.loads(line) for line in (worker_source/'workers.jsonl').read_text().splitlines()]
        templates = {tuple(w['spec'].get(k) for k in BUILD_KEYS):w['spec'] for w in workers if w['spec']['split']=='test'}
        anchor_rows = read_rows(anchor_out/'test.csv') if (anchor_out/'test.csv').exists() else []
        have = {point_key(r) for r in anchor_rows}
        for group in selected_groups([{**r,'target_met_on_validation':True} for r in unique if point_key(r) not in have]):
            before_worker()
            spec = {**templates[tuple(group.get(k) for k in BUILD_KEYS)],**group,'config':cfg}
            spec['quality_rows'] = [r for r in unique if tuple(r.get(k) for k in BUILD_KEYS)==tuple(group.get(k) for k in BUILD_KEYS)]
            spec['quality_source'] = 'frozen final-test quality; identical inputs and query path; anchor retiming'
            print('Timing anchor',group['method'],group['direction'],group['seed'],flush=True)
            anchor_rows.extend(execute(spec,anchor_out/'workers.jsonl')['rows'])
            write_csv(anchor_out/'test.csv',anchor_rows)
        old = {point_key(r):r for r in unique}
        comparison = []
        for row in anchor_rows:
            original = old[point_key(row)]
            comparison.append({**{k:row.get(k) for k in POINT_KEYS},
                'original_batch_p50_ms':original['batch_p50_ms'],'new_batch_p50_ms':row['batch_p50_ms'],
                'p50_ratio_new_over_final':row['batch_p50_ms']/original['batch_p50_ms'],
                'original_qps':original['throughput_qps'],'new_qps':row['throughput_qps'],
                'qps_ratio_new_over_final':row['throughput_qps']/original['throughput_qps']})
        write_csv(anchor_out/'comparison.csv',comparison)

    completion = {'dataset':name,method+'_test_points':len(rows),method+'_target_records':len(selected),
                  method+'_test_target_met':sum(r['target_met_on_test'] for r in selected),'rss_points':len(memory),
                  'seconds_this_invocation':time.time()-start,'completed_utc':time.time()}
    save_json(output/'complete.json',completion)
    archive = ROOT/'outputs/tables/icde_systems'/method/name
    archive.mkdir(parents=True,exist_ok=True)
    for file in output.iterdir():
        if file.suffix in ('.csv','.json'):
            shutil.copyfile(file,archive/file.name)
    if (output/'anchors').exists():
        (archive/'anchors').mkdir(exist_ok=True)
        for file in (output/'anchors').glob('*.csv'):
            shutil.copyfile(file,archive/'anchors'/file.name)
    print('COMPLETED',method,json.dumps(completion),flush=True)
    return completion


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/icde_opq.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = resolve_path(config['output'])
    output.mkdir(parents=True,exist_ok=True)
    results = []
    for name,options in config['datasets'].items():
        results.append(run_dataset(name,options,config,output))
    save_json(output/'complete.json',{'datasets':results,'completed_utc':time.time()})


if __name__ == '__main__':
    main()

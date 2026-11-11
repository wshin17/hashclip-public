"""D-013: remote OPQ32x4 fitting and a serial, resumable post-009 queue."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import shlex
import signal
import shutil
import subprocess
import time

from .evaluate_cpu import configure, input_identity, write_csv
from .final_cpu import point_key, quiet
from .opq_timing import check_identity, read_json, save_json, summarize
from .paths import ROOT
from .refine_ivf import read_rows
from .systems_cpu import BUILD_KEYS, POINT_KEYS, execute, report_selected, resolve_path, selected_groups


def deadline(config):
    value = datetime.fromisoformat(config['deadline']).timestamp()
    if time.time() >= value:
        raise TimeoutError('D-013 hard deadline reached; preserve completed groups')
    return value


def pending_dependencies(config):
    pending = [str(p) for p in config.get('wait_for', []) if not resolve_path(p).exists()]
    for path in config.get('wait_processes', []):
        pidfile = resolve_path(path)
        if pidfile.exists():
            command = Path('/proc')/pidfile.read_text().strip()/'cmdline'
            try:
                if any(m in command.read_bytes() for m in
                       (b'hashclip.fastscan_timing', b'hashclip.breakeven_cpu')):
                    pending.append(path)
            except FileNotFoundError:
                pass
    return pending


def pull_models(options, host):
    """Copy only a completed remote dataset's models, before its local timing."""
    relative = Path(options['models'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Remote model path must stay under the project')
    remote = Path('Projects')/ROOT.name/relative
    result = subprocess.run(['ssh','-o','ConnectTimeout=10',host,
                             'test -f '+shlex.quote(str(remote/'fit_complete.json'))],timeout=20)
    if result.returncode==1:
        return False
    if result.returncode:
        raise RuntimeError(f'Remote model readiness check failed: {result.returncode}')
    local = resolve_path(relative)
    local.mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-rt','--partial','--exclude=fit_complete.json',
                    f'{host}:{remote}/',str(local)+'/'],check=True)
    manifest = read_json(local/'manifest.json')
    if {(r['modality'],r['seed']) for r in manifest['models']}!={(m,s) for m in ('img','txt') for s in (0,1,42)}:
        raise ValueError('Remote fitted model set is incomplete')
    for row in manifest['models']:
        check_identity(local/Path(row['artifact']['path']).name,row['artifact'])
    save_json(local/'fit_complete.json',{'models':len(manifest['models']),
             'copied_from_host':host,'verified_copy_utc':time.time()})
    return True


def fit(config, datasets, threads=8):
    """Identical OPQ training flow to OPQ16, with 32x4 training/final PQ."""
    configure({'threads':threads, 'blas_core':'Haswell'})
    import faiss
    import numpy as np
    from threadpoolctl import threadpool_limits
    from .representations import train_representation

    faiss.omp_set_num_threads(threads)
    for name in datasets:
        deadline(config)
        options = config['datasets'][name]
        features = resolve_path(options['training_features'])
        identity = check_identity(features, options['training_identity'])
        output = resolve_path(options['models'])
        output.mkdir(parents=True, exist_ok=True)
        manifest = {'method':'opq_fastscan', 'training_features':identity,
                    'seeds':[0,1,42], 'bits':128, 'threads':threads,
                    'host':platform.node(), 'cpu_affinity':sorted(os.sched_getaffinity(0)),
                    'faiss':faiss.__version__, 'git_commit':subprocess.check_output(
                        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    'training_policy':'gallery-side training vectors, unit normalized; standard Faiss iteration/sample caps',
                    'models':[]}
        path = output/'manifest.json'
        if path.exists():
            previous = read_json(path)
            for key in ('method','training_features','seeds','bits','threads','host','cpu_affinity','faiss','training_policy'):
                if previous[key] != manifest[key]:
                    raise ValueError(f'Cannot resume changed fitting: {key}')
            manifest = previous
            for row in manifest['models']:
                check_identity(output/Path(row['artifact']['path']).name,row['artifact'])
        save_json(path,manifest)
        with np.load(features,allow_pickle=True) as archive:
            arrays = {key:archive[key] for key in ('img_feat','txt_feat','img_ids')}
        if len(set(map(str,arrays['img_ids']))) != len(arrays['img_feat']):
            raise ValueError('Training image IDs are not unique')
        with threadpool_limits(limits=threads):
            for modality in ('img','txt'):
                for seed in (0,1,42):
                    if any((r['modality'],r['seed']) == (modality,seed) for r in manifest['models']):
                        continue
                    deadline(config)
                    print('Fitting OPQ32x4',name,modality,seed,flush=True)
                    start = time.perf_counter()
                    model = train_representation('opq_fastscan',arrays[modality+'_feat'],seed)
                    elapsed = time.perf_counter()-start
                    rotation = faiss.downcast_VectorTransform(model.chain.at(0))
                    pq = faiss.downcast_index(model.index)
                    model_path = output/f'opq_fastscan_{modality}_seed{seed}.faiss'
                    faiss.write_index(model,str(model_path))
                    manifest['models'].append({'method':'opq_fastscan','modality':modality,'seed':seed,
                        'training_rows':len(arrays[modality+'_feat']),'training_seconds':elapsed,
                        'artifact':input_identity(model_path),
                        'opq_settings':{k:int(getattr(rotation,k)) for k in
                                        ('M','niter','niter_pq','niter_pq_0','max_train_points')},
                        'search_settings':{k:int(getattr(pq,k)) for k in
                                           ('M','M2','nbits','code_size','bbs','implem','qbs')},
                        'pq_iterations':int(pq.pq.cp.niter),
                        'pq_max_points_per_centroid':int(pq.pq.cp.max_points_per_centroid)})
                    save_json(path,manifest)
                    print('Saved',model_path.name,'fit_seconds',elapsed,flush=True)
                    del model
        save_json(output/'fit_complete.json',{'models':len(manifest['models']),'completed_utc':time.time()})


def archive_results(output, name):
    destination = ROOT/'outputs/tables/icde_systems/opq_fastscan'/name
    destination.mkdir(parents=True,exist_ok=True)
    for path in output.iterdir():
        if path.suffix in ('.csv','.json'):
            shutil.copyfile(path,destination/path.name)
    if (output/'anchors').exists():
        (destination/'anchors').mkdir(exist_ok=True)
        for path in (output/'anchors').glob('*.csv'):
            shutil.copyfile(path,destination/'anchors'/path.name)


def cost_comparison(selected, anchors, old_anchors):
    """Only fixed validation-selected targets; never optimize on test latency."""
    from .breakeven_cpu import interval
    lookup = {point_key(r):r for r in anchors}
    result = []
    for direction in ('t2i','i2t'):
        for target in (.95,.99):
            rows = [r for r in selected if r['direction']==direction and r['batch']==1
                    and r['target_mean_recall_at_10']==target]
            old = [r for r in old_anchors if r['direction']==direction and r['batch']==1
                   and r['target_mean_recall_at_10']==target]
            if len(rows)!=3 or len(old)!=3 or not all(point_key(r) in lookup for r in old):
                continue
            hc = [lookup[point_key(r)] for r in old]
            avg = lambda items,key: sum(r[key] for r in items)/len(items)
            candidate = {'build_ms':avg(rows,'local_build_ms'), 'batch_p50_ms':avg(rows,'batch_p50_ms')}
            hbuild, hquery = avg(hc,'build_ms'), avg(hc,'batch_p50_ms')
            faster = candidate['batch_p50_ms'] < hquery
            row = {'direction':direction,'target':target,'seeds':3,
                   'hashclip_p50_ms':hquery,'opq_fastscan_p50_ms':candidate['batch_p50_ms'],
                   'hashclip_build_ms':hbuild,'opq_fastscan_build_ms':candidate['build_ms'],
                   'opq_fastscan_faster':faster,'cost_basis':'model reusable; fitting excluded',
                   'scope':'fixed validation-target configurations, three-seed means, same-session test times',
                   'opq_fastscan_test_target_met':sum(r['target_met_on_test'] for r in rows),
                   'hashclip_test_target_met':sum(r['final_recall_at_10']>=target for r in hc)}
            if faster:
                bounds = interval(hbuild,hquery,[candidate])
                row.update(Q_lo=bounds['lower_queries'],Q_hi=bounds['upper_queries'],
                           has_advantage_interval=bounds['has_advantage_interval'],
                           endpoint_parameters=json.dumps({r['seed']:r['M'] for r in rows},sort_keys=True))
            result.append(row)
    return result


def run_dataset(name, options, config, *, enforce_quiet=True):
    output = resolve_path(config['output'])/name
    output.mkdir(parents=True,exist_ok=True)
    if (output/'complete.json').exists():
        return read_json(output/'complete.json')
    source = resolve_path(config['source_systems'])/name
    final = resolve_path(config['final_systems'])/name
    meta = read_json(source/'config.json')
    cfg = read_json(final/'config.json')['config']
    if cfg != {**meta['config'],'record_load':True,'quality_workers':8}:
        raise ValueError('Timing settings differ from the final round')
    if cfg['threads']!=1 or cfg['batch_sizes']!=[1,32]:
        raise ValueError('D-013 fixes CPU threads=1 and batches=1/32')
    models = resolve_path(options['models'])
    manifest = read_json(models/'manifest.json')
    if any(manifest['training_features'][k]!=options['training_identity'][k] for k in ('bytes','sha256')):
        raise ValueError('Training cache differs from 009')
    trained, identities = {}, []
    for row in manifest['models']:
        if row['method']!='opq_fastscan':
            raise ValueError('Unexpected model type')
        path = models/Path(row['artifact']['path']).name
        identities.append(check_identity(path,row['artifact']))
        trained[row['seed'],row['modality']] = {**row,'path':str(path)}
    if set(trained)!={(s,m) for s in (0,1,42) for m in ('img','txt')}:
        raise ValueError('Expected six frozen OPQ32x4 templates')
    checked = set()
    for item in meta['inputs']:
        path = resolve_path(item['path'])
        if str(path.resolve()) not in checked:
            check_identity(path,item)
            checked.add(str(path.resolve()))
    identity = {'task':'010','method':'opq_fastscan','dataset':name,'config':cfg,
        'source':input_identity(source/'config.json'),'final_source':input_identity(final/'config.json'),
        'models_manifest':input_identity(models/'manifest.json'),'models':identities,
        'fit_host':manifest['host'],'fit_threads':manifest['threads'],
        'training_policy':manifest['training_policy'],
        'selection':'minimum full-val native M for both batches; each group frozen before its test',
        'large_gallery_contains_training_images':True,
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    if (output/'config.json').exists():
        previous = read_json(output/'config.json')
        for key in ('config','source','final_source','models_manifest','models'):
            if previous[key]!=identity[key]:
                raise ValueError(f'Cannot resume changed inputs: {key}')
    else:
        save_json(output/'config.json',identity)

    def work(spec, log):
        deadline(config)
        if enforce_quiet:
            quiet()
        return execute({**spec,'execution_deadline':deadline(config)},log)

    def spec_for(seed,direction,split):
        paths = meta['dataset_paths']
        model = trained[seed,'img' if direction=='t2i' else 'txt']
        return {**dict.fromkeys(BUILD_KEYS),'method':'opq_fastscan','seed':seed,'direction':direction,
            'pq_m':32,'pq_bits':4,'config':cfg,'split':split,
            'features':str(Path(paths['queries'])/f'backbone_{split}.npz'),
            'gallery_features':str(Path(paths['gallery'])/f'backbone_{split}.npz'),
            'reference':str(Path(paths['references'])/f'{split}_reference.npz'),
            'insertion_features':str(Path(paths['queries'])/'backbone_val.npz'),
            'opq_fastscan_model':model['path'],'opq_fastscan_training':{
                'training_seconds':model['training_seconds'],'host':manifest['host'],'threads':manifest['threads']}}

    old_anchors = [r for r in read_rows(final/options['anchor_selection']) if r['method']==options['anchor_method']]
    templates = {}
    for line in (final/'workers.jsonl').read_text().splitlines():
        spec = json.loads(line)['spec']
        if spec['split']=='test' and spec['method']==options['anchor_method']:
            templates[spec['direction'],spec['seed']] = spec
    anchor_out = output/'anchors'
    anchor_out.mkdir(exist_ok=True)
    evidence = read_json(output/'selection_groups.json') if (output/'selection_groups.json').exists() else {}
    chosen = read_json(output/'selected.json') if (output/'selected.json').exists() else []
    rows = read_rows(output/'test.csv') if (output/'test.csv').exists() else []
    memory = read_json(output/'memory.json') if (output/'memory.json').exists() else []
    anchors = read_rows(anchor_out/'test.csv') if (anchor_out/'test.csv').exists() else []
    completed = read_json(output/'completed_groups.json') if (output/'completed_groups.json').exists() else []
    try:
        for direction in ('t2i','i2t'):
            for seed in (0,1,42):
                key = f'{direction}/{seed}'
                if key in completed:
                    continue
                if key not in evidence:
                    print('Selecting OPQ FastScan',name,key,flush=True)
                    evidence[key] = work({**spec_for(seed,direction,'val'),'mode':'select_fastscan'},output/'selection_workers.jsonl')
                    save_json(output/'selection_groups.json',evidence)
                spec = spec_for(seed,direction,'test')
                points = []
                for batch in cfg['batch_sizes']:
                    for target in cfg['quality_targets']:
                        m = int(evidence[key]['selected'][str(target)])
                        metrics = evidence[key]['validation_metrics'][str(target)][str(batch)]
                        if metrics['final_recall_at_10']<target:
                            raise ValueError('Validation target not met')
                        points.append({**{k:spec.get(k) for k in BUILD_KEYS},'batch':batch,'M':m,
                            'nprobe':None,'ef_search':None,'target_mean_recall_at_10':target,
                            'target_met_on_validation':True,'validation_recall_at_10':metrics['final_recall_at_10']})
                previous = [p for p in chosen if p['seed']==seed and p['direction']==direction]
                if previous and previous!=points:
                    raise ValueError('Attempt to change previously frozen validation budgets')
                if not previous:
                    chosen.extend(points)
                    save_json(output/'selected.json',chosen)
                have = {point_key(r) for r in rows}
                missing = [p for p in points if point_key(p) not in have]
                if missing:
                    group, = selected_groups(missing)
                    print('Testing OPQ FastScan',name,key,flush=True)
                    rows.extend(work({**spec,**group},output/'workers.jsonl')['rows'])
                    write_csv(output/'test.csv',rows)
                done = {point_key(r) for r in memory}
                for point in {point_key(p):p for p in points}.values():
                    if point_key(point) not in done:
                        memory.append({**point,**work({**spec,'mode':'memory','point':point},output/'memory_workers.jsonl')})
                        save_json(output/'memory.json',memory)
                        done.add(point_key(point))
                unique = list({point_key(r):r for r in old_anchors if r['direction']==direction and r['seed']==seed}.values())
                have = {point_key(r) for r in anchors}
                for group in selected_groups([r for r in unique if point_key(r) not in have]):
                    anchor_spec = {**templates[direction,seed],**group,'config':cfg,'quality_rows':unique,
                        'quality_source':'frozen final-test quality; identical inputs and query path; anchor retiming'}
                    print('Anchor',name,key,flush=True)
                    anchors.extend(work(anchor_spec,anchor_out/'workers.jsonl')['rows'])
                    write_csv(anchor_out/'test.csv',anchors)
                completed.append(key)
                save_json(output/'completed_groups.json',completed)
                publish(output,name,chosen,rows,memory,anchors,old_anchors)
                print('Completed group',name,key,flush=True)
    except TimeoutError:
        save_json(output/'incomplete.json',{'reason':'hard deadline','completed_groups':completed,
                 'validation_groups':list(evidence),'recorded_utc':time.time()})
        publish(output,name,chosen,rows,memory,anchors,old_anchors)
        raise
    completion = {'dataset':name,'groups':len(completed),'test_target_records':len(chosen),
                  'test_points':len(rows),'rss_points':len(memory),'anchor_points':len(anchors),'completed_utc':time.time()}
    save_json(output/'complete.json',completion)
    archive_results(output,name)
    return completion


def publish(output,name,chosen,rows,memory,anchors,old_anchors):
    have = {point_key(r) for r in rows}
    selected = report_selected([p for p in chosen if point_key(p) in have],rows)
    if selected:
        write_csv(output/'selected_test.csv',selected)
    mem = {point_key(r):r for r in memory}
    costs = [{**r,'rss_after_query_bytes':mem[point_key(r)]['rss_after_query_bytes']}
             for r in selected if point_key(r) in mem]
    if costs:
        write_csv(output/'costs.csv',costs)
        write_csv(output/'summary.csv',summarize(costs,costs))
    old = {point_key(r):r for r in old_anchors}
    comparison = []
    for row in anchors:
        original = old[point_key(row)]
        comparison.append({**{k:row.get(k) for k in POINT_KEYS},
            'original_batch_p50_ms':original['batch_p50_ms'],'new_batch_p50_ms':row['batch_p50_ms'],
            'p50_ratio_new_over_final':row['batch_p50_ms']/original['batch_p50_ms'],
            'original_qps':original['throughput_qps'],'new_qps':row['throughput_qps'],
            'qps_ratio_new_over_final':row['throughput_qps']/original['throughput_qps']})
    if comparison:
        write_csv(output/'anchors/comparison.csv',comparison)
    checks = cost_comparison(selected,anchors,old_anchors)
    if checks:
        write_csv(output/'cost_comparison.csv',checks)
    archive_results(output,name)


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/icde_opq_fastscan.yaml')
    parser.add_argument('--fit',action='store_true')
    parser.add_argument('--datasets',nargs='+',choices=['flickr_large','coco_large'])
    parser.add_argument('--models-host',help='Authorized fitting host to fetch completed models from')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    names = args.datasets or list(config['datasets'])
    if args.fit:
        fit(config,names)
        return
    output = resolve_path(config['output'])
    output.mkdir(parents=True,exist_ok=True)
    def stop_at_deadline(signum, frame):
        raise TimeoutError('D-013 hard deadline reached')
    signal.signal(signal.SIGALRM,stop_at_deadline)
    signal.setitimer(signal.ITIMER_REAL,max(.001,datetime.fromisoformat(config['deadline']).timestamp()-time.time()))
    try:
        pending = pending_dependencies(config)
        while pending:
            deadline(config)
            print('Waiting for 009 completion',pending,flush=True)
            time.sleep(30)
            pending = pending_dependencies(config)
        results = []
        for name in names:
            options = config['datasets'][name]
            while not (resolve_path(options['models'])/'fit_complete.json').exists():
                deadline(config)
                if args.models_host and pull_models(options,args.models_host):
                    break
                print('Waiting for remote fitted models',name,flush=True)
                time.sleep(30)
            results.append(run_dataset(name,options,config))
        save_json(output/'complete.json',{'datasets':results,'completed_utc':time.time()})
    except TimeoutError as exc:
        save_json(output/'incomplete.json',{'reason':str(exc),'recorded_utc':time.time()})
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)


if __name__ == '__main__':
    main()

"""PI-ordered final timing: same-domain targets, all targets, bounded curves.

Existing final measurements are preserved. Parameters come only from validation;
each new worker uses the original query timer, build/insertion and fresh RSS.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import time

from .evaluate_cpu import input_identity, write_csv
from .final_cpu import cost_points, frontiers, point_key, quiet
from .paths import ROOT, runs_root
from .refine_ivf import read_rows
from .systems_cpu import BUILD_KEYS, POINT_KEYS, execute, report_selected, select_points, selected_groups

BINARY = ('hashclip', 'hashclip_coco', 'lsh')
GROUP_KEYS = ('method', 'seed', 'direction', 'batch')


def groups(points):
    return selected_groups([{**p, 'target_met_on_validation': True} for p in points])


def deduplicate(rows):
    return list({point_key(r): r for r in rows}.values())


def added_curve_points(validation, selected, dataset, maximum=8):
    """At most eight *additional* parameter settings per seed/direction.

    Binary counts are shared across seeds. Native points spread along the
    validation retention/complete frontiers. Selected points are exempt from
    both the extra-point limit and the 10% candidate cap.
    """
    selected_keys = {point_key(p) for p in selected}
    by_group = defaultdict(list)
    for row in validation:
        by_group[tuple(row[k] for k in ('method', 'seed', 'direction'))].append(row)
    extra = []
    for (method, seed, direction), values in by_group.items():
        if dataset == 'coco_large' and method == 'lsh':
            continue
        n = values[0]['gallery_size']
        ceiling = n // 10
        if method in BINARY:
            template = {k: values[0].get(k) for k in POINT_KEYS}
            candidates = [{**template, 'M':m} for m in sorted({
                max(10, math.floor(f * n)) for f in (.001, .002, .005, .01, .02, .03, .05, .1)})
                if m <= ceiling]
        else:
            candidates = frontiers(values)
            candidates = [p for p in candidates if p.get('M') is None or p['M'] <= ceiling]
            candidates = list({tuple(p.get(k) for k in POINT_KEYS if k != 'batch'):p
                               for p in candidates}.values())
            candidates.sort(key=lambda p: (p['final_recall_at_10'], p['final_complete_at_10'],
                                          p['amortized_p50_ms']))
        batches = sorted({r['batch'] for r in values})
        candidates = [p for p in candidates if any(point_key({**p,'batch':b}) not in selected_keys for b in batches)]
        if len(candidates) > maximum:
            ids = sorted({round(i * (len(candidates) - 1) / (maximum - 1)) for i in range(maximum)})
            candidates = [candidates[i] for i in ids]
        extra.extend({**{k:p.get(k) for k in POINT_KEYS}, 'batch':batch}
                     for p in candidates for batch in batches
                     if point_key({**p,'batch':batch}) not in selected_keys)
    return extra


def load_dataset(name, source, previous, domain_quality, training):
    import yaml
    from .numpy_adapter import numpy_checkpoint_path

    source = Path(source) / name
    meta = json.loads((source / 'config.json').read_text())
    workers = [json.loads(line) for line in (source / 'workers.jsonl').read_text().splitlines()]
    templates = {tuple(w['spec'].get(k) for k in BUILD_KEYS): w['spec']
                 for w in workers if w['spec']['split'] == 'val'}
    old_val = read_rows(source / 'val.csv')
    previous = Path(previous) / name
    saved = {split:read_rows(previous / f'{split}.csv') if (previous / f'{split}.csv').exists() else []
             for split in ('val', 'test')}
    if any(saved.values()):
        prior_meta = json.loads((previous / 'config.json').read_text())
        if prior_meta['source'] != input_identity(source / 'config.json'):
            raise ValueError('Previous final rows refer to a different first-round source')
        if prior_meta['config'] != {**meta['config'], 'record_load':True, 'quality_workers':8}:
            raise ValueError('Previous final timing configuration differs')
    # Prefer the already completed quiet validation for COCO, retaining every
    # saved row in its original directory even when not part of the new plan.
    validation = deduplicate(old_val + saved['val'])
    selected = select_points(validation, meta['config']['quality_targets'])
    caches, cache_sources, domain_rows = {}, [], []
    if name in ('coco', 'coco_large'):
        training_config = yaml.safe_load(Path(training).read_text())
        qdir = Path(domain_quality) / name
        qmeta = json.loads((qdir / 'config.json').read_text())
        budgets = json.loads((qdir / 'selected.json').read_text())
        quality_val = [r for r in read_rows(qdir / 'curves.csv') if r['split'] == 'val']
        cache_sources = [input_identity(qdir / n) for n in ('config.json', 'selected.json')]
        cache_sources.append(input_identity(Path(training)))
        for seed, run in training_config['training_runs'].items():
            checkpoint = runs_root() / run / 'checkpoints' / f'hashclip_b128_seed{seed}_reselect.pt'
            actual = input_identity(numpy_checkpoint_path(checkpoint))
            expected = next(c for c in qmeta['checkpoints'] if Path(c['path']).name == Path(actual['path']).name)
            if (actual['bytes'], actual['sha256']) != (expected['bytes'], expected['sha256']):
                raise ValueError('Same-domain checkpoint differs from cached quality')
            cache_sources.append(actual)
            for direction in ('t2i', 'i2t'):
                original = next(s for s in templates.values() if s['method'] == 'hashclip'
                                and s['seed'] == seed and s['direction'] == direction)
                spec = {**original, 'method':'hashclip_coco', 'checkpoint':str(checkpoint)}
                templates[tuple(spec.get(k) for k in BUILD_KEYS)] = spec
                for split in ('val', 'test'):
                    cache_file = qdir / f'{split}_{direction}_hashclip_seed{seed}_positions.npy'
                    caches[('hashclip_coco', seed, direction, split)] = {
                        'path':str(cache_file), 'threads':qmeta['threads'], 'batch':qmeta['batch'],
                        'identity':input_identity(cache_file)}
                    cache_sources.append(caches[('hashclip_coco', seed, direction, split)]['identity'])
                for batch in meta['config']['batch_sizes']:
                    for target, m in budgets[f'{direction}/hashclip/{seed}'].items():
                        row = next(r for r in quality_val if r['direction'] == direction
                                   and r['seed'] == seed and r['M'] == m)
                        selected.append({**{k:spec.get(k) for k in BUILD_KEYS}, 'batch':batch,
                            'M':m, 'nprobe':None, 'ef_search':None,
                            'validation_recall_at_10':row['candidate_recall_at_10'],
                            'target_mean_recall_at_10':float(target), 'target_met_on_validation':True})
                    # Only validation metadata seeds the additional M grid.
                    domain_rows.append({**{k:spec.get(k) for k in BUILD_KEYS}, 'batch':batch,
                        'gallery_size':row['gallery_size']})
        # Match the cached query/gallery/reference identities to the measured
        # dataset; filenames alone are insufficient to authorize rank reuse.
        identities = {(Path(i['path']).resolve(), i['bytes'], i['sha256']) for i in meta['inputs']}
        for item in qmeta['inputs']:
            if (Path(item['path']).resolve(), item['bytes'], item['sha256']) not in identities:
                raise ValueError('Domain quality input differs from timing input')
    targets = cost_points(selected, validation)
    extra = added_curve_points(validation + domain_rows, targets, name)
    return {'dataset':name, 'source':str(source), 'previous':str(previous), 'meta':meta,
            'config':{**meta['config'], 'record_load':True, 'quality_workers':8},
            'templates':templates, 'selected':selected, 'targets':deduplicate(targets), 'extra':extra,
            'validation':validation, 'saved':saved, 'caches':caches, 'cache_sources':cache_sources,
            'old_quality':{'val':old_val, 'test':read_rows(source / 'test.csv')}}


def row_estimate(context, point):
    method = 'hashclip' if point['method'] == 'hashclip_coco' else point['method']
    candidates = [r for r in context['validation'] if r['method'] == method
                  and r['direction'] == point['direction'] and r['batch'] == point['batch']]
    exact = [r for r in candidates if all(r.get(k) == point.get(k) for k in POINT_KEYS if k != 'method')]
    if exact:
        return exact[0]
    same_build = [r for r in candidates if all(r.get(k) == point.get(k) for k in BUILD_KEYS if k != 'method')]
    candidates = same_build or candidates
    return min(candidates, key=lambda r: abs((r.get('M') or 0) - (point.get('M') or 0)))


def estimate(context, points, *, rss):
    """First-round costs, timed-call counts, and missing quality evaluations.

    Excludes unknown quiet-window waits. Uses four effective workers for
    native quality, serial PQ, and a conservative binary rank-scan allowance.
    """
    seconds, build_seconds = 0., 0.
    for split in ('val', 'test'):
        done = {point_key(r) for r in context['saved'][split]}
        missing = [p for p in points if point_key(p) not in done]
        quality_keys = {point_key(r) for r in context['old_quality'][split]}
        for group in groups(missing):
            sample = row_estimate(context, group['points'][0])
            build = sample['build_ms'] / 1000
            build_seconds += build
            seconds += build + (12 if context['dataset'] == 'coco_large' else 3)
            binary_missing = False
            for point in group['points']:
                row = row_estimate(context, point)
                cfg = context['config']
                calls = (cfg['repeats'] + 1) * cfg['timing_queries'] / point['batch'] + cfg['warmup']
                seconds += row['batch_p50_ms'] * calls / 1000
                if point_key(point) not in quality_keys:
                    if point['method'] in BINARY:
                        binary_missing = True
                    elif split == 'test':
                        parallel = 1 if point['method'] == 'ivf_pq' else 4
                        seconds += row['queries'] * row['amortized_p50_ms'] / 1000 / parallel
            if binary_missing:
                seconds += 30 if group['method'] == 'hashclip_coco' else (240 if context['dataset'] == 'coco_large' else 10)
    if rss:
        for point in deduplicate(points):
            row = row_estimate(context, point)
            build = row['build_ms'] / 1000
            build_seconds += build
            seconds += build + 12 + 5 * row['batch_p50_ms'] / 1000
    return {'central_seconds':round(seconds), 'build_seconds':round(build_seconds),
            'range_hours':[round(seconds * .75 / 3600, 2), round(seconds * 1.5 / 3600, 2)]}


class Measurement:
    def __init__(self, context, output, *, enforce_quiet=True):
        self.ctx, self.out, self.enforce_quiet = context, Path(output), enforce_quiet
        self.out.mkdir(parents=True, exist_ok=True)
        config_file = self.out / 'config.json'
        if not config_file.exists():
            config_file.write_text(json.dumps({
                'dataset':context['dataset'], 'config':context['config'],
                'source':input_identity(Path(context['source'])/'config.json'),
                'previous_final':context['previous'], 'same_domain_inputs':context['cache_sources'],
                'frozen_selected_points':context['selected'], 'frozen_extra_points':context['extra'],
                'same_domain_large_gallery_contains_training_images':context['dataset']=='coco_large',
                'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
            }, indent=2)+'\n')
        self.rows = {}
        for split in ('val', 'test'):
            file = self.out / f'{split}.csv'
            self.rows[split] = read_rows(file) if file.exists() else list(context['saved'][split])
        f = self.out / 'memory.json'
        self.memory = json.loads(f.read_text()) if f.exists() else []
        self.quality = {split:deduplicate(context['old_quality'][split] + self.rows[split]) for split in self.rows}

    def spec(self, group, split):
        c = self.ctx
        spec = deepcopy(c['templates'][tuple(group.get(k) for k in BUILD_KEYS)])
        spec.pop('budgets', None)
        spec.update(group, split=split, config=c['config'])
        paths = c['meta']['dataset_paths']
        spec.update(features=str(Path(paths['queries']) / f'backbone_{split}.npz'),
                    gallery_features=str(Path(paths['gallery']) / f'backbone_{split}.npz'),
                    reference=str(Path(paths['references']) / f'{split}_reference.npz'))
        return spec

    def measure(self, points, split):
        have = {point_key(r) for r in self.rows[split]}
        for group in groups([p for p in points if point_key(p) not in have]):
            if self.enforce_quiet:
                quiet()
            spec = self.spec(group, split)
            lookup = {point_key(r):r for r in self.quality[split]}
            if all(point_key(p) in lookup for p in group['points']):
                spec['quality_rows'] = [lookup[point_key(p)] for p in group['points']]
                spec['quality_source'] = f'frozen {split} quality; identical batch/index/inputs; timing rerun'
            else:
                key = (group['method'], group['seed'], group['direction'], split)
                if key in self.ctx['caches']:
                    spec['position_cache'] = self.ctx['caches'][key]
            print('priority measurement', self.ctx['dataset'], split,
                  {k:v for k,v in group.items() if k != 'points'}, 'points',len(group['points']),flush=True)
            result = execute(spec, self.out / 'workers.jsonl')
            self.rows[split] = deduplicate(self.rows[split] + result['rows'])
            self.quality[split] = deduplicate(self.quality[split] + result['rows'])
            write_csv(self.out / f'{split}.csv', self.rows[split])

    def rss(self, points):
        have = {point_key(r) for r in self.memory}
        for point in deduplicate(points):
            if point_key(point) in have:
                continue
            if self.enforce_quiet:
                quiet()
            spec = self.spec(point, 'test')
            spec.update(mode='memory', point=point)
            self.memory.append({**point, **execute(spec, self.out / 'memory_workers.jsonl')})
            (self.out / 'memory.json').write_text(json.dumps(self.memory, indent=2) + '\n')

    def targets(self, points, label):
        start = time.time()
        self.measure(points, 'val')
        self.measure(points, 'test')
        self.rss(points)
        keys = {point_key(p) for p in points}
        covered = {tuple(p[k] for k in GROUP_KEYS) for p in points}
        chosen = [p for p in self.ctx['selected'] if tuple(p[k] for k in GROUP_KEYS) in covered
                  and (not p['target_met_on_validation'] or point_key(p) in keys)]
        selected_rows = report_selected(chosen, self.rows['test'])
        val = {point_key(r):r for r in self.rows['val']}
        for row in selected_rows:
            if row['target_met_on_validation']:
                value = val[point_key(row)]['final_recall_at_10']
                row['retimed_validation_recall_at_10'] = value
                row['retimed_target_met_on_validation'] = value >= row['target_mean_recall_at_10']
        write_csv(self.out / f'{label}_selected_test.csv', selected_rows)
        test = {point_key(r):r for r in self.rows['test']}
        memory = {point_key(r):r for r in self.memory}
        write_csv(self.out / f'{label}_costs.csv', [{**test[point_key(p)], **p,
            'rss_after_query_bytes':memory[point_key(p)]['rss_after_query_bytes']} for p in points])
        completion = {'dataset':self.ctx['dataset'], 'label':label, 'seconds':time.time()-start,
                      'targets':len(selected_rows), 'measured_points':len(keys),
                      'test_target_met':sum(r.get('target_met_on_test',False) for r in selected_rows)}
        (self.out / f'{label}_complete.json').write_text(json.dumps(completion, indent=2) + '\n')
        print('COMPLETED PRIORITY TARGETS', json.dumps(completion), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--domain-quality', type=Path, required=True)
    parser.add_argument('--training', type=Path, default=ROOT / 'configs/icde_coco_reselect.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    contexts = {n:load_dataset(n, args.source, args.previous, args.domain_quality, args.training)
                for n in ('coco', 'flickr_large', 'coco_large')}
    stages = []
    for name in ('coco', 'coco_large'):
        c = contexts[name]
        points = [p for p in c['targets'] if p['method'] == 'hashclip_coco']
        stages.append({'priority':1, 'dataset':name, 'label':'domain_targets', 'points':points,
                       'estimate':estimate(c, points, rss=True)})
    for name in ('flickr_large', 'coco_large'):
        c = contexts[name]
        points = [p for p in c['targets'] if p['method'] != 'hashclip_coco']
        stages.append({'priority':2, 'dataset':name, 'label':'all_targets', 'points':points,
                       'estimate':estimate(c, points, rss=True)})
    # Unfinished COCO original test/RSS stays behind the requested large-gallery
    # targets, while all already measured COCO rows remain available for reuse.
    c = contexts['coco']
    points = [p for p in c['targets'] if p['method'] != 'hashclip_coco']
    stages.append({'priority':3, 'dataset':'coco', 'label':'all_targets', 'points':points,
                   'estimate':estimate(c, points, rss=True)})
    for name,c in contexts.items():
        stages.append({'priority':3, 'dataset':name, 'label':'curves', 'points':c['extra'],
                       'estimate':estimate(c, c['extra'], rss=False)})
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'selection':'frozen validation-selected 95%/99% points; no test tuning',
            'additional_curves':'up to 8 parameter settings per method/seed/direction, measured at both batches; M <= 10%; no large COCO LSH extras',
            'selected_points_exempt_from_caps':True,
            'existing_coco_rows_preserved':{s:len(r) for s,r in contexts['coco']['saved'].items()},
            'estimate_excludes_quiet_waiting':True, 'stages':stages}
    plan_file = args.output / 'plan.json'
    if plan_file.exists():
        previous_plan = json.loads(plan_file.read_text())
        if previous_plan['stages'] != stages:
            raise ValueError('Frozen priority plan changed; use a new output directory')
    else:
        plan_file.write_text(json.dumps(plan,indent=2) + '\n')
    print('PLAN', json.dumps([{k:v for k,v in s.items() if k != 'points'} | {'points':len(s['points'])}
                             for s in stages]),flush=True)
    if args.plan_only:
        return
    # Verify each distinct current input once before consuming frozen quality.
    seen = set()
    for c in contexts.values():
        for item in c['meta']['inputs'] + c['cache_sources']:
            key = (str(Path(item['path']).resolve()), item['sha256'])
            if key not in seen:
                if input_identity(Path(item['path'])) != item:
                    raise ValueError(f"Input changed: {item['path']}")
                seen.add(key)
    measurements = {n:Measurement(c, args.output/n) for n,c in contexts.items()}
    for stage in stages:
        measurement = measurements[stage['dataset']]
        complete = measurement.out / f"{stage['label']}_complete.json"
        if complete.exists():
            continue
        print('START PRIORITY',stage['priority'],stage['dataset'],stage['label'],flush=True)
        if stage['label'] != 'curves':
            measurement.targets(stage['points'], stage['label'])
        else:
            measurement.measure(stage['points'], 'val')
            measurement.measure(stage['points'], 'test')
            complete.write_text(json.dumps({'additional_points':len(stage['points'])}) + '\n')
        if stage['priority'] == 1 and stage['dataset'] == 'coco_large':
            (args.output/'priority1_complete.json').write_text(json.dumps({'completed_utc':time.time()})+'\n')
            print('PRIORITY 1 COMPLETE: same-domain COCO original and large selected points',flush=True)
    from .plot_cpu import plot_systems
    for name, measurement in measurements.items():
        c = contexts[name]
        keys = {point_key(p) for p in c['targets'] + c['extra']}
        rows = [r for r in measurement.rows['test'] if point_key(r) in keys]
        selected = report_selected(c['selected'], rows)
        write_csv(measurement.out / 'selected_test.csv', selected)
        write_csv(measurement.out / 'curve_test.csv', rows)
        plot_systems(rows, selected, measurement.out / 'plots', name)
    (args.output/'complete.json').write_text(json.dumps({'completed_utc':time.time()})+'\n')


if __name__ == '__main__':
    main()

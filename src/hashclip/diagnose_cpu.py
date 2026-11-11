"""Replay flagged, frozen COCO test settings after load isolation; no tuning."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

from .evaluate_cpu import configure, load_features, retrieval_inputs, timing
from .systems_cpu import build_index, query_budget, row_parameters


def run(source, output):
    # Replay the exact old test configurations for diagnosis only. Selection
    # for the final experiment is still performed on validation data.
    old = [json.loads(line) for line in (source / 'workers.jsonl').read_text().splitlines()]
    specs = [r for r in old if r['spec']['split'] == 'test' and r['spec']['method'] in ('flat_ip', 'ivf_pq')]
    configure(specs[0]['spec']['config'])
    import faiss
    import numpy as np
    from threadpoolctl import threadpool_limits
    from .cpu_load import snapshot, difference
    faiss.omp_set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for old_worker in specs:
        spec = old_worker['spec']
        arrays = load_features(spec['features'], captions_per_image=None)
        galleries = load_features(spec['gallery_features'], captions_per_image=None)
        gallery, queries, _, _ = retrieval_inputs(arrays, galleries, spec['direction'])
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        with threadpool_limits(limits=1):
            index = build_index(spec, gallery, spec['config'])
            for point in spec['points']:
                batch, budget = point['batch'], query_budget(point)
                before = snapshot()
                metrics, samples = timing(index, queries, batch, budget, spec['config'])
                load = difference(before, snapshot())
                # Wall vs process CPU time and native top-K widths distinguish
                # scheduling stalls from boundary expansion or padding.
                calls, wall, cpu, padding = [], [], [], []
                original_search = index.index.search
                def traced_search(q, width):
                    calls.append((len(q), width))
                    return original_search(q, width)
                index.index.search = traced_search
                for start in range(0, min(256, len(queries)), batch):
                    q = queries[start:start + batch]
                    a, b = time.perf_counter_ns(), time.process_time_ns()
                    result = index.query(q, budget=budget)
                    cpu.append((time.process_time_ns() - b) / 1e6)
                    wall.append((time.perf_counter_ns() - a) / 1e6)
                    if result.candidates is not None:
                        padding.append(int(np.sum(np.any(result.candidates < 0, axis=1))))
                index.index.search = original_search
                params = row_parameters(spec, batch, budget)
                previous = next(r for r in old_worker['result']['rows'] if all(r.get(k) == v for k, v in params.items()))
                record = {**params, 'original_batch_p50_ms': previous['batch_p50_ms'], **metrics,
                          'load': load, 'trace_wall_p50_ms': float(np.median(wall)),
                          'trace_cpu_p50_ms': float(np.median(cpu)), 'native_search_calls': calls,
                          'padding_rows': padding, 'raw_latency_ms': samples}
                records.append(record)
                (output / 'results.json').write_text(json.dumps(records, indent=2) + '\n')
                print('diagnostic', spec['method'], spec['direction'], batch,
                      'old/new ms', previous['batch_p50_ms'], metrics['batch_p50_ms'], flush=True)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.source, a.output)


if __name__ == '__main__':
    main()

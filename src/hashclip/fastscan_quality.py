"""Validation-only budgets for native quantized FastScan ties (D-012).

Native top-M sets need not be nested. Bound each reference neighbor's rank
by its equal-distance interval, then test every ambiguous integer M in order.
Every search retains the original batch shape; the selected points are also
verified through complete online retrieval, including the common reranker.
"""
from __future__ import annotations

from fractions import Fraction
import sys

import numpy as np

from .representation_quality import minimum_budget
from .retrieve import prepare_vectors


def rank_bounds(index, queries, reference, batch, maximum):
    infinity = np.iinfo(np.int64).max
    lower = np.empty(reference.shape, dtype=np.int64)
    upper = np.full(reference.shape, infinity, dtype=np.int64)
    # Reused ID lookup avoids an O(M log M) sort or an M x 10 allocation.
    by_id = np.full(index.index.ntotal, np.nan, dtype=np.float32)
    for start in range(0, len(queries), batch):
        d, ids = index.index.search(prepare_vectors(queries[start:start+batch]), maximum)
        if (ids < 0).any() or np.any(d[:, 1:] < d[:, :-1]):
            raise ValueError('FastScan omitted items or returned unsorted distances')
        for offset, (distances, labels) in enumerate(zip(d, ids)):
            by_id[labels] = distances
            values = by_id[reference[start+offset]]
            found = ~np.isnan(values)
            lo = np.searchsorted(distances, values, side='left')+1
            hi = np.searchsorted(distances, values, side='right')
            lo[~found] = np.searchsorted(distances, distances[-1], side='left')+1
            hi[~found | ((values == distances[-1]) & (maximum < index.index.ntotal))] = infinity
            lower[start+offset], upper[start+offset] = lo, hi
            by_id[labels] = np.nan
    return lower, upper


def bounded_hits(index, queries, reference, batch, budget, bounds):
    lower, upper = bounds
    definite = upper <= budget
    ambiguous = (lower <= budget) & ~definite
    hits = int(definite.sum())
    calls = 0
    flags = np.zeros(index.index.ntotal, dtype=bool)
    for start in range(0, len(queries), batch):
        mask = ambiguous[start:start+batch]
        if not mask.any():
            continue
        # Do not shrink the batch to just ambiguous rows: LUT numerics and
        # native tie resolution must see the original batch of queries.
        _, ids = index.index.search(prepare_vectors(queries[start:start+batch]), budget)
        calls += 1
        for offset, labels in enumerate(ids):
            if not mask[offset].any():
                continue
            flags[labels] = True
            hits += int((flags[reference[start+offset]] & mask[offset]).sum())
            flags[labels] = False
    return hits, calls


def online_quality(index, queries, reference, budgets, batches=(1, 32)):
    """Full actual-query quality, boundary tie rate and cross-batch set mismatch.

    The M+1 diagnostic is outside the measured query and never changes its
    native M candidates. Batches are paired in memory, not hashed or stored.
    """
    if tuple(batches) != (1, 32):
        raise ValueError('D-012 compares batch 1 and 32')
    n = index.index.ntotal
    flags = np.zeros(n, dtype=bool)
    records = {}
    for budget in sorted(set(budgets)):
        m = min(int(budget), n)
        accum = {b: {} for b in batches}
        ties = {b: 0 for b in batches}
        mismatch = ordered_mismatch = 0
        for start in range(0, len(queries), 32):
            chunk = queries[start:start+32]
            ref = reference[start:start+len(chunk)]
            single = [index.query(q[None], budget=m) for q in chunk]
            batched = index.query(chunk, budget=m)
            candidates = {1:np.concatenate([x.candidates for x in single]), 32:batched.candidates}
            final = {1:np.concatenate([x.final for x in single]), 32:batched.final}
            for labels1, labels32 in zip(candidates[1], candidates[32]):
                flags[labels1] = True
                mismatch += int(not flags[labels32].all())
                ordered_mismatch += int(not np.array_equal(labels1, labels32))
                flags[labels1] = False
            for b in batches:
                candidate_hits = np.empty(ref.shape, dtype=bool)
                for i, labels in enumerate(candidates[b]):
                    flags[labels] = True
                    candidate_hits[i] = flags[ref[i]]
                    flags[labels] = False
                for k in (1, 10):
                    final_hits = (ref[:, :k, None] == final[b][:, None, :k]).any(axis=2)
                    ch = candidate_hits[:, :k]
                    counts = {f'candidate_recall_at_{k}':int(ch.sum()),
                              f'candidate_complete_at_{k}':int(ch.all(axis=1).sum()),
                              f'final_recall_at_{k}':int(final_hits.sum()),
                              f'final_complete_at_{k}':int(final_hits.all(axis=1).sum()),
                              f'ordered_recovery_at_{k}':int((ref[:, :k] == final[b][:, :k]).all(axis=1).sum())}
                    for key, value in counts.items():
                        accum[b][key] = accum[b].get(key, 0)+value
                if m < n:
                    for offset in range(0, len(chunk), b):
                        distances, _ = index.index.search(prepare_vectors(chunk[offset:offset+b]), m+1)
                        ties[b] += int((distances[:, m-1] == distances[:, m]).sum())
        for b in batches:
            row = {k:v/(len(queries)*(10 if k.endswith('recall_at_10') else 1))
                   for k,v in accum[b].items()}
            if accum[b]['candidate_recall_at_10'] != accum[b]['final_recall_at_10']:
                raise ValueError('Shared refinement did not recover retained reference neighbors')
            row.update(boundary_tie_query_rate=ties[b]/len(queries),
                       boundary_tie_query_count=ties[b], boundary_tie_applicable=m < n,
                       candidate_set_mismatch_batch_1_32=mismatch/len(queries),
                       candidate_order_mismatch_batch_1_32=ordered_mismatch/len(queries),
                       candidate_hits_at_10=accum[b]['candidate_recall_at_10'])
            records[b, budget] = row
    return records


def select_budgets(index, queries, reference, targets=(.95, .99)):
    n = index.index.ntotal
    maximum = min(n, max(10, int(np.ceil(n*.02))))
    batches = (1, 32)
    while True:
        print('FastScan validation rank bounds', maximum, flush=True, file=sys.stderr)
        bounds = {b:rank_bounds(index, queries, reference, b, maximum) for b in batches}
        if all(minimum_budget(bounds[b][1], max(targets)) <= maximum for b in batches):
            break
        if maximum == n:
            raise ValueError('Full native FastScan failed to cover all reference IDs')
        maximum = min(n, maximum*2)
    selected, evidence, cache = {}, {}, {}
    for target in sorted(targets):
        lower = max(10, *(minimum_budget(bounds[b][0], target) for b in batches))
        upper = max(10, *(minimum_budget(bounds[b][1], target) for b in batches))
        fraction = Fraction(str(target))*reference.size
        needed = (fraction.numerator+fraction.denominator-1)//fraction.denominator
        counts, calls = {}, 0
        print('FastScan minimum-M interval', target, lower, upper, flush=True, file=sys.stderr)
        for m in range(lower, upper+1):
            for b in batches:
                if (b,m) not in cache:
                    cache[b,m] = bounded_hits(index,queries,reference,b,m,bounds[b])
                    calls += cache[b,m][1]
                counts[b] = cache[b,m][0]
                if counts[b] < needed:
                    break
            else:
                selected[str(target)] = m
                evidence[str(target)] = {'lower_M':lower, 'upper_M':upper, 'checked_through_M':m,
                    'native_search_calls':calls, 'hits':dict(counts), 'required_hits':needed}
                break
        else:
            raise ValueError('FastScan tie bounds disagree with native searches')
    verified = online_quality(index,queries,reference,selected.values())
    for target, m in selected.items():
        for b in batches:
            actual = verified[b,m]
            if actual['candidate_hits_at_10'] != evidence[target]['hits'][b]:
                raise ValueError('Bounded selection disagrees with full online validation')
            if actual['candidate_hits_at_10'] < evidence[target]['required_hits']:
                raise ValueError('Selected FastScan M failed complete online validation')
    return {'selected':selected, 'prefix_maximum':maximum, 'selection_evidence':evidence,
            'validation_metrics':{str(t):{str(b):verified[b,selected[str(t)]] for b in batches} for t in targets},
            'selection_policy':'smallest native M meeting full-validation target for both batch 1 and 32; exhaustive within tie bounds'}

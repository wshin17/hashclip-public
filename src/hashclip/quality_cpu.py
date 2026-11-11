"""Reuse canonical continuous scores for offline binary-quality grid sweeps.

The timed index.query path is untouched. Exact Hamming ordering and the shared
continuous scorer define the same candidates/results for every budget. Every
budget is checked against online queries, and any disagreement aborts the run.
"""

from pathlib import Path

import numpy as np

from .metrics import retrieval_batch_metrics
from .retrieve import prepare_vectors, score_rows, stable_topk


def reference_with_scores(gallery, queries, destination: Path, k=10):
    gallery = prepare_vectors(gallery)
    queries = prepare_vectors(queries)
    scores = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float32,
                                        shape=(len(queries), len(gallery)))
    reference = np.empty((len(queries), min(k, len(gallery))), dtype=np.int64)
    ids = np.arange(len(gallery))
    for i, query in enumerate(queries):
        values = score_rows(query, gallery)
        scores[i] = values
        reference[i] = stable_topk(values, ids, k, largest=True)
    scores.flush()
    return reference


def binary_quality_grid(index, queries, reference, scores_path, mapping, direction,
                        batch, budgets, query_image_rows, audit_queries=64, reuse=None,
                        *, report_task_recall=True):
    """Evaluate all queries; optional reuse requires caller-verified code equality.

    A full-radius BinaryFlat range search obtains exact Hamming distances once.
    Stable sorting gives every budget's prefix. Cached float32 scores replace
    repeated gallery gathers / dot products, with identical score/ID tie rules.
    """
    if index.backend != "faiss":
        raise ValueError("Cached quality requires the exact Faiss binary backend")
    scores = np.load(scores_path, mmap_mode="r")
    if scores.shape != (len(queries), len(index.gallery)) or scores.dtype != np.float32:
        raise ValueError("Score cache does not match query/gallery arrays")
    if not budgets or min(budgets) < 10:
        raise ValueError("Binary budgets must include at least top-10")
    sums = {budget: {} for budget in budgets}
    starts = list(range(0, len(queries), batch))
    count = min(len(starts), max(1, (audit_queries + batch - 1) // batch))
    audit_starts = {starts[i] for i in np.linspace(0, len(starts) - 1, count, dtype=int)}
    measured = 0
    for start in (starts if reuse is None else sorted(audit_starts)):
        q = queries[start:start + batch]
        # This is the same encoder, device, input dtype and batch shape used by
        # BinaryRerank.query. No archived/precomputed query codes are substituted.
        codes = index.encoder.encode(q)
        limits, distances, labels = index.index.range_search(codes, index.index.d + 1)
        width = min(max(budgets), len(index.gallery))
        candidates = np.empty((len(q), width), dtype=np.int64)
        for i in range(len(q)):
            lo, hi = int(limits[i]), int(limits[i + 1])
            if hi - lo != len(index.gallery):
                raise ValueError("Full-radius Hamming search omitted gallery items")
            candidates[i] = stable_topk(distances[lo:hi], labels[lo:hi], width)
        for budget in budgets:
            selected = candidates[:, :min(budget, len(index.gallery))]
            final = np.stack([stable_topk(scores[start + i, ids], ids, 10, largest=True)
                              for i, ids in enumerate(selected)])
            if start in audit_starts:
                actual = index.query(q, budget=budget)
                if not np.array_equal(actual.candidates, selected) or not np.array_equal(actual.final, final):
                    raise ValueError(f"Cached quality disagrees with online path: {direction}, batch={batch}, M={budget}, query={start}")
            if reuse is None:
                values = retrieval_batch_metrics(reference[start:start + len(q)], final, selected,
                                                  mapping, direction, start, query_image_rows,
                                                  report_task_recall=report_task_recall)
                for key, value in values.items():
                    sums[budget][key] = sums[budget].get(key, 0.0) + value * len(q)
        if start in audit_starts:
            measured += len(q)
    metrics = ({budget: {key: value / len(queries) for key, value in sums[budget].items()} for budget in budgets}
               if reuse is None else {budget: dict(reuse[budget]) for budget in budgets})
    return metrics, measured

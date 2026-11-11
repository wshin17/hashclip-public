"""Continuous-neighbor recovery and task recall (fractions, not percentages)."""

from __future__ import annotations

import numpy as np


def neighbor_metrics(reference: np.ndarray, final: np.ndarray,
                     candidates: np.ndarray | None = None) -> dict[str, float]:
    """Keep average overlap, complete-set recovery and exact order separate."""
    out = {}
    for requested_k in (1, 10):
        k = min(requested_k, reference.shape[1])
        ref = reference[:, :k]
        for name, found in (("final", final[:, :k]), ("candidate", candidates)):
            if found is None:
                continue
            retained = np.array([
                np.isin(r, row[row >= 0]).sum() for r, row in zip(ref, found)
            ])
            out[f"{name}_recall_at_{requested_k}"] = float(np.mean(retained / k))
            out[f"{name}_complete_at_{requested_k}"] = float(np.mean(retained == k))
        exact_order = (final[:, :k] == ref).all(axis=1) if final.shape[1] >= k else np.zeros(len(ref), bool)
        out[f"ordered_recovery_at_{requested_k}"] = float(exact_order.mean())
    return out


def task_recall(final: np.ndarray, txt_to_img: np.ndarray, direction: str) -> dict[str, float]:
    out = {}
    for k in (1, 5, 10):
        found = final[:, :k]
        if direction == "t2i":
            hits = (found == txt_to_img[:, None]).any(axis=1)
        elif direction == "i2t":
            valid = found >= 0
            owners = txt_to_img[np.maximum(found, 0)]
            hits = (valid & (owners == np.arange(len(found))[:, None])).any(axis=1)
        else:
            raise ValueError(direction)
        out[f"task_recall_at_{k}"] = float(hits.mean())
    return out


def retrieval_batch_metrics(reference, final, candidates, mapping, direction, start, query_image_rows=None,
                            *, report_task_recall=True):
    """Shared metric computation for online and cached-score quality passes."""
    values = neighbor_metrics(reference, final, candidates)
    if not report_task_recall:
        return values
    if direction == "t2i":
        values.update(task_recall(final, mapping[start:start + len(final)], direction))
    else:
        image_rows = (np.arange(start, start + len(final)) if query_image_rows is None
                      else query_image_rows[start:start + len(final)])
        for k in (1, 5, 10):
            found = final[:, :k]
            hits = ((found >= 0) & (mapping[np.maximum(found, 0)] == image_rows[:, None])).any(axis=1)
            values[f"task_recall_at_{k}"] = float(hits.mean())
    return values

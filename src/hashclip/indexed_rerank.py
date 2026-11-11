"""Indexed float32 scoring with canonical re-scoring around the top-K boundary.

Faiss reads candidate vectors in place. Its float32 reduction can differ from
NumPy's row scorer, so it only screens candidates: the final ordering always
uses the original scorer and global-ID tie rule.
"""

from __future__ import annotations

import faiss
import numpy as np

from .retrieve import rerank_prepared


def norm_upper(x):
    """Conservative float64 upper bounds on the norms of float32 rows."""
    d = x.shape[1]
    values = np.sqrt(np.einsum("ij,ij->i", x, x, dtype=np.float64, optimize=False))
    return np.nextafter(values * (1 + 4 * d * np.finfo(np.float64).eps), np.inf)


class IndexedReranker:
    def __init__(self, gallery):
        if gallery.dtype != np.float32 or not gallery.flags.c_contiguous:
            raise ValueError("Indexed scorer requires a contiguous float32 gallery")
        self.gallery = gallery
        self.max_norm = max(float(norm_upper(gallery[start:start + 4096]).max())
                            for start in range(0, len(gallery), 4096))

    def screen(self, query, ids, k):
        """Return every possible canonical top-K member, then re-score them.

        For a d-term float32 dot product, use E = gamma(2d) * ||q|| ||x||
        as an error bound relative to real arithmetic (u = eps32/2). Each of
        the two reductions has error at most E, hence |Faiss - canonical|
        <= 2E. An item more than 4E below the kth Faiss score cannot enter the
        canonical top-K. Include equality, and retain the original ID tie rule.
        The tiny additive term covers underflow/flush-to-zero in the reductions.
        This bound is deliberately independent of observed validation errors.
        """
        d = self.gallery.shape[1]
        unit = np.finfo(np.float32).eps / 2
        if 2 * d * unit >= 1:
            return ids, None
        scores = np.empty(len(ids), dtype=np.float32)
        faiss.fvec_inner_products_by_idx(
            faiss.swig_ptr(scores), faiss.swig_ptr(query), faiss.swig_ptr(self.gallery),
            faiss.swig_ptr(ids), d, 1, len(ids))
        if not np.isfinite(scores).all():
            return ids, scores
        gamma = (2 * d * unit) / (1 - 2 * d * unit)
        error = gamma * float(norm_upper(query[None])[0]) * self.max_norm
        guard = np.nextafter(4 * error + 8 * d * np.finfo(np.float32).tiny, np.inf)
        boundary = float(np.partition(scores, len(ids) - k)[len(ids) - k])
        selected = ids[scores.astype(np.float64) >= boundary - guard]
        return selected, scores

    def __call__(self, queries, candidates, k=10):
        k = min(k, candidates.shape[1])
        final = np.empty((len(queries), k), dtype=np.int64)
        for i, candidates_i in enumerate(candidates):
            # Read gallery rows in physical order; membership is unchanged.
            ids = np.sort(candidates_i)
            selected, _ = self.screen(queries[i], ids, k)
            final[i] = rerank_prepared(queries[i:i + 1], self.gallery, selected[None], k,
                                       block_size=256)[0]
        return final

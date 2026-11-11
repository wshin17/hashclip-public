"""Retrieval primitives — Hamming search, cosine re-ranking, FAISS / LSH baselines.

All functions are NumPy + CPU; the matrices we deal with on Flickr30k test
(1k images, 5k captions) are tiny, so a CPU FAISS index is more than fast
enough and avoids the GPU-wheel architecture headaches.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


# Pre-computed Hamming popcount table for fast XOR-byte counting.
_POPCOUNT_8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Hamming search
# ---------------------------------------------------------------------------


def hamming_distance_matrix(q_packed: np.ndarray, db_packed: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances between packed queries and database."""
    if q_packed.dtype != np.uint8 or db_packed.dtype != np.uint8:
        raise ValueError("Inputs must be uint8-packed bit arrays.")
    if q_packed.ndim != 2 or db_packed.ndim != 2:
        raise ValueError("Expected 2-D arrays.")
    if q_packed.shape[1] != db_packed.shape[1]:
        raise ValueError(f"Hash-byte-dim mismatch: {q_packed.shape[1]} vs {db_packed.shape[1]}")

    # Keep the byte-sized intermediate bounded; the requested distance matrix
    # itself still has Q*N elements. hamming_topk works one query at a time.
    dist = np.empty((len(q_packed), len(db_packed)), dtype=np.int32)
    for qi, query in enumerate(q_packed):
        for start in range(0, len(db_packed), 4096):
            xored = query ^ db_packed[start:start + 4096]
            dist[qi, start:start + 4096] = _POPCOUNT_8[xored].sum(axis=-1, dtype=np.int32)
    return dist


def hamming_topk(q_packed: np.ndarray, db_packed: np.ndarray, k: int) -> np.ndarray:
    """Return the indices of the k smallest-Hamming-distance db rows per query.

    Ties, including the selection boundary, use ascending gallery row ID.
    """
    if k < 1 or len(db_packed) == 0:
        raise ValueError("k and gallery size must be positive")
    k = min(int(k), len(db_packed))
    out = np.empty((len(q_packed), k), dtype=np.int64)
    ids = np.arange(len(db_packed))
    for qi in range(len(q_packed)):
        dist = hamming_distance_matrix(q_packed[qi:qi + 1], db_packed)[0]
        out[qi] = stable_topk(dist, ids, k)
    return out


# ---------------------------------------------------------------------------
# Cosine re-ranking
# ---------------------------------------------------------------------------


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def prepare_vectors(x: np.ndarray) -> np.ndarray:
    """Normalize once into the CPU float32 representation used by the scorer."""
    x = np.asarray(x, dtype=np.float32, order="C")
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected a finite 2-D feature matrix")
    return np.ascontiguousarray(_l2_normalize(x))


def stable_topk(values: np.ndarray, ids: np.ndarray, k: int,
                *, largest: bool = False) -> np.ndarray:
    """Select by (value, global ID), fixing ties before truncation."""
    k = min(int(k), len(ids))
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    key = -values if largest else values
    if k < len(ids):
        boundary = np.partition(key, k - 1)[k - 1]
        selected = np.flatnonzero(key <= boundary)
    else:
        selected = np.arange(len(ids))
    order = np.lexsort((ids[selected], key[selected]))[:k]
    return ids[selected[order]]


def score_rows(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Canonical float32 row-wise cosine scorer for reference and re-ranking.

    Both inputs are pre-normalized. Using the same per-row reduction avoids
    switching GEMM/GEMV algorithms when the candidate count changes.
    """
    return np.einsum("ij,j->i", gallery, query, optimize=False)


def rerank_prepared(q: np.ndarray, db: np.ndarray,
                    candidates: np.ndarray, k: int, *, block_size: int = 0) -> np.ndarray:
    """Canonical re-ranking, optionally gathering cache-sized blocks.

    Sorting candidate IDs improves gallery access locality without changing
    membership or score/ID ordering. Each dot product still uses score_rows;
    blocks only bound the temporary copied vectors, not the candidate budget.
    A zero block size retains the original implementation for comparisons.
    """
    if block_size < 0:
        raise ValueError("block_size must be nonnegative")
    out = np.empty((len(q), min(k, candidates.shape[1])), dtype=np.int64)
    for i, cand in enumerate(candidates):
        if block_size:
            ids = np.sort(cand)
            scores = np.empty(len(ids), dtype=np.float32)
            for start in range(0, len(ids), block_size):
                rows = ids[start:start + block_size]
                scores[start:start + len(rows)] = score_rows(q[i], db[rows])
        else:
            ids = cand
            scores = score_rows(q[i], db[cand])
        out[i] = stable_topk(scores, ids, k, largest=True)
    return out


def rerank_cosine(q_cont: np.ndarray, db_cont: np.ndarray,
                  candidate_idx: np.ndarray, *, k: int) -> np.ndarray:
    """Re-rank the stage-1 candidate pool by cosine similarity."""
    return rerank_prepared(prepare_vectors(q_cont), prepare_vectors(db_cont), candidate_idx, k)


def cosine_exact_topk(q_cont: np.ndarray, db_cont: np.ndarray, k: int) -> np.ndarray:
    q, db = prepare_vectors(q_cont), prepare_vectors(db_cont)
    ids = np.arange(len(db))
    out = np.empty((len(q), min(k, len(db))), dtype=np.int64)
    for i, query in enumerate(q):
        out[i] = stable_topk(score_rows(query, db), ids, k, largest=True)
    return out


# ---------------------------------------------------------------------------
# FAISS IVF baseline (CPU build; the test set is small enough that GPU is overkill)
# ---------------------------------------------------------------------------


def faiss_ivf_topk(q_cont: np.ndarray, db_cont: np.ndarray, k: int,
                   *, nlist: int = 1024, nprobe: int = 16) -> np.ndarray:
    import faiss  # local import

    db = _l2_normalize(db_cont.astype(np.float32))
    q = _l2_normalize(q_cont.astype(np.float32))
    d = db.shape[1]
    # Cap nlist to a sane value for tiny test sets (Flickr30k images = 1000).
    nlist = int(max(1, min(nlist, db.shape[0])))
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(db)
    index.add(db)
    index.nprobe = int(max(1, min(nprobe, nlist)))
    _, idx = index.search(q, int(min(k, db.shape[0])))
    return idx.astype(np.int64)


# ---------------------------------------------------------------------------
# Random Sign LSH baseline
# ---------------------------------------------------------------------------


def random_lsh_codes(cont: np.ndarray, bits: int, *, seed: int = 0) -> np.ndarray:
    """Project continuous features with a random Gaussian matrix and sign."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((cont.shape[1], bits)).astype(np.float32)
    proj = cont.astype(np.float32) @ W
    return (proj > 0).astype(np.uint8)


def random_lsh_pack(cont: np.ndarray, bits: int, *, seed: int = 0) -> np.ndarray:
    codes01 = random_lsh_codes(cont, bits, seed=seed)
    return np.packbits(codes01, axis=-1)


# ---------------------------------------------------------------------------
# Latency timing helpers
# ---------------------------------------------------------------------------


@dataclass
class TimedResult:
    final: np.ndarray            # (Q, K) int64 — final top-K db indices
    stage1_ms: float             # ms per query
    stage2_ms: float
    total_ms: float


def asymmetric_search_timed(q_packed: np.ndarray, q_cont: np.ndarray,
                            db_packed: np.ndarray, db_cont: np.ndarray,
                            *, M: int, K: int, warmup: int = 5) -> TimedResult:
    """Full HashCLIP search with stage-1 / stage-2 latencies measured."""
    Q = q_packed.shape[0]

    # ---- warmup -----------------------------------------------------------
    if warmup > 0:
        w = min(warmup, Q)
        cand_w = hamming_topk(q_packed[:w], db_packed, k=M)
        _ = rerank_cosine(q_cont[:w], db_cont, cand_w, k=K)

    # ---- stage 1 ----------------------------------------------------------
    t0 = time.perf_counter()
    cand = hamming_topk(q_packed, db_packed, k=M)
    t1 = time.perf_counter()

    # ---- stage 2 ----------------------------------------------------------
    final = rerank_cosine(q_cont, db_cont, cand, k=K)
    t2 = time.perf_counter()

    stage1_ms = (t1 - t0) * 1000.0 / max(1, Q)
    stage2_ms = (t2 - t1) * 1000.0 / max(1, Q)
    return TimedResult(final=final, stage1_ms=stage1_ms, stage2_ms=stage2_ms,
                       total_ms=stage1_ms + stage2_ms)


def hamming_only_timed(q_packed: np.ndarray, db_packed: np.ndarray,
                       *, K: int, warmup: int = 5) -> TimedResult:
    Q = q_packed.shape[0]
    if warmup > 0:
        w = min(warmup, Q)
        _ = hamming_topk(q_packed[:w], db_packed, k=K)
    t0 = time.perf_counter()
    final = hamming_topk(q_packed, db_packed, k=K)
    stage1_ms = (time.perf_counter() - t0) * 1000.0 / max(1, Q)
    return TimedResult(final=final, stage1_ms=stage1_ms, stage2_ms=0.0,
                       total_ms=stage1_ms)


def cosine_exact_timed(q_cont: np.ndarray, db_cont: np.ndarray,
                       *, K: int, warmup: int = 5) -> TimedResult:
    Q = q_cont.shape[0]
    if warmup > 0:
        w = min(warmup, Q)
        _ = cosine_exact_topk(q_cont[:w], db_cont, k=K)
    t0 = time.perf_counter()
    final = cosine_exact_topk(q_cont, db_cont, k=K)
    stage2_ms = (time.perf_counter() - t0) * 1000.0 / max(1, Q)
    return TimedResult(final=final, stage1_ms=0.0, stage2_ms=stage2_ms,
                       total_ms=stage2_ms)


def faiss_ivf_timed(q_cont: np.ndarray, db_cont: np.ndarray, *, K: int,
                    nlist: int = 1024, nprobe: int = 16,
                    warmup: int = 5) -> TimedResult:
    Q = q_cont.shape[0]
    # Build/index time is amortized — count only the queries.
    import faiss

    db = _l2_normalize(db_cont.astype(np.float32))
    q = _l2_normalize(q_cont.astype(np.float32))
    d = db.shape[1]
    nl = int(max(1, min(nlist, db.shape[0])))
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nl, faiss.METRIC_INNER_PRODUCT)
    index.train(db)
    index.add(db)
    index.nprobe = int(max(1, min(nprobe, nl)))
    if warmup > 0:
        _ = index.search(q[:min(warmup, Q)], int(min(K, db.shape[0])))
    t0 = time.perf_counter()
    _, idx = index.search(q, int(min(K, db.shape[0])))
    stage2_ms = (time.perf_counter() - t0) * 1000.0 / max(1, Q)
    return TimedResult(final=idx.astype(np.int64), stage1_ms=0.0,
                       stage2_ms=stage2_ms, total_ms=stage2_ms)

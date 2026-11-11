"""CPU indexes built once, queried from continuous float32 embeddings.

Gallery row IDs are the stable tie-break key. HashCLIP and LSH use identical
candidate selection and canonical re-ranking. Faiss float indexes keep their
optimized inner-product kernels; their numerical agreement with the NumPy
reference is measured, rather than silently assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import faiss
import numpy as np

from .retrieve import (hamming_topk, prepare_vectors, rerank_prepared,
                       score_rows, stable_topk)


@dataclass
class SearchResult:
    final: np.ndarray
    candidates: np.ndarray | None = None
    stages_ms: dict[str, float] = field(default_factory=dict)


def faiss_topk(index, queries: np.ndarray, k: int, *, binary: bool = False,
               largest: bool | None = None) -> np.ndarray:
    """Include all boundary ties before selecting by global ID.

    Requesting K+1 detects a truncated tie. Only affected queries need a range
    search for integer Hamming radii. Float indexes widen the top-K search on
    the same query batch: a one-query range search could switch Faiss's floating
    reduction kernel and disagree numerically with a batched boundary score.
    """
    k = min(int(k), index.ntotal)
    if k < 1:
        raise ValueError("Positive k and nonempty index required")
    width = min(k + 1, index.ntotal)
    scores, labels = index.search(queries, width)
    if not binary:
        while width < index.ntotal and np.any(
            (labels[:, -1] >= 0) & (scores[:, k - 1] == scores[:, -1])
        ):
            width = min(index.ntotal, width * 2)
            scores, labels = index.search(queries, width)
    out = np.full((len(queries), k), -1, dtype=np.int64)
    for i in range(len(queries)):
        valid = labels[i] >= 0
        ids, values = labels[i, valid], scores[i, valid]
        if binary and len(ids) > k and values[k - 1] == values[k]:
            radius = int(values[k - 1]) + 1
            _, values, ids = index.range_search(queries[i:i + 1], radius)
        selected = stable_topk(values, ids, k, largest=not binary if largest is None else largest)
        out[i, :len(selected)] = selected
    return out


class NumpyExact:
    """Canonical continuous scorer, retained for correctness cross-checks."""

    def __init__(self, gallery: np.ndarray):
        self.gallery = prepare_vectors(gallery)
        self.storage = {"continuous_bytes": self.gallery.nbytes, "binary_bytes": 0,
                        "query_parameters_bytes": 0}

    def query(self, queries, k=10, *, budget=None, profile=False):
        t0 = time.perf_counter() if profile else 0
        q = prepare_vectors(queries)
        t1 = time.perf_counter() if profile else 0
        ids = np.arange(len(self.gallery))
        out = np.empty((len(q), min(k, len(ids))), dtype=np.int64)
        for i, x in enumerate(q):
            out[i] = stable_topk(score_rows(x, self.gallery), ids, k, largest=True)
        stages = ({"query_prepare_ms": (t1 - t0) * 1000,
                   "search_ms": (time.perf_counter() - t1) * 1000} if profile else {})
        return SearchResult(out, stages_ms=stages)


class FaissFloat:
    def __init__(self, gallery: np.ndarray, *, nlist: int | None = None, seed: int = 42):
        vectors = prepare_vectors(gallery)
        n, d = vectors.shape
        if not n:
            raise ValueError("Empty gallery")
        if nlist is None:
            self.index = faiss.IndexFlatIP(d)
        else:
            if not 1 <= nlist <= n:
                raise ValueError("nlist must be within gallery size")
            self.index = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.cp.seed = int(seed)
            self.index.train(vectors)
        self.index.add(vectors)
        # Faiss owns its vector copy; do not retain a redundant NumPy gallery.
        self.storage = {"continuous_bytes": n * d * 4, "binary_bytes": 0,
                        "query_parameters_bytes": 0,
                        "ivf_ids_and_centroids_bytes": 0 if nlist is None else n * 8 + nlist * d * 4}

    def query(self, queries, k=10, *, budget=None, profile=False):
        if hasattr(self.index, "nprobe"):
            if budget is None or not 1 <= budget <= self.index.nlist:
                raise ValueError("IVF budget must be a valid nprobe")
            self.index.nprobe = int(budget)
        t0 = time.perf_counter() if profile else 0
        q = prepare_vectors(queries)
        t1 = time.perf_counter() if profile else 0
        out = faiss_topk(self.index, q, k)
        stages = ({"query_prepare_ms": (t1 - t0) * 1000,
                   "search_ms": (time.perf_counter() - t1) * 1000} if profile else {})
        return SearchResult(out, stages_ms=stages)


class LSHEncoder:
    def __init__(self, dimension: int, bits: int, seed: int):
        rng = np.random.default_rng(seed)
        self.weights = rng.standard_normal((dimension, bits)).astype(np.float32)
        self.bits = bits
        self.nbytes = self.weights.nbytes

    def encode(self, raw: np.ndarray) -> np.ndarray:
        projected = prepare_vectors(raw) @ self.weights
        return np.packbits(projected > 0, axis=1)


class AdapterEncoder:
    """The existing adapter on CPU, with its original normalization and sign."""

    def __init__(self, adapter):
        self.adapter = adapter.cpu().eval()
        self.bits = adapter.bits
        self.nbytes = sum(p.numel() * p.element_size() for p in adapter.parameters())

    def encode(self, raw: np.ndarray) -> np.ndarray:
        import torch

        x = np.asarray(raw, dtype=np.float32, order="C")
        with torch.inference_mode():
            output = self.adapter(torch.from_numpy(x)).numpy()
        return np.packbits(output > 0, axis=1)


def encode_chunks(encoder, raw: np.ndarray, batch: int = 256) -> np.ndarray:
    return np.concatenate([encoder.encode(raw[i:i + batch]) for i in range(0, len(raw), batch)])


class BinaryRerank:
    def __init__(self, gallery: np.ndarray, gallery_codes: np.ndarray,
                 query_encoder, *, backend: str = "faiss", use_heap: bool = True,
                 rerank_block: int = 0, rerank_mode: str = "canonical"):
        self.gallery = prepare_vectors(gallery)
        codes = np.ascontiguousarray(gallery_codes, dtype=np.uint8)
        if len(codes) != len(self.gallery) or codes.ndim != 2:
            raise ValueError("Codes must match gallery rows")
        self.encoder = query_encoder
        self.backend = backend
        self.rerank_block = int(rerank_block)
        if self.rerank_block < 0:
            raise ValueError("rerank_block must be nonnegative")
        if rerank_mode not in ("canonical", "indexed"):
            raise ValueError(rerank_mode)
        self.rerank_mode = rerank_mode
        self.indexed_reranker = None
        if rerank_mode == "indexed":
            from .indexed_rerank import IndexedReranker
            self.indexed_reranker = IndexedReranker(self.gallery)
        if backend == "faiss":
            self.index = faiss.IndexBinaryFlat(codes.shape[1] * 8)
            self.index.use_heap = bool(use_heap)
            self.index.add(codes)
        elif backend == "numpy":
            self.codes = codes
        else:
            raise ValueError(backend)
        self.storage = {"continuous_bytes": self.gallery.nbytes,
                        "binary_bytes": codes.nbytes,
                        "query_parameters_bytes": query_encoder.nbytes}

    def query(self, queries, k=10, *, budget=None, profile=False):
        if budget is None or budget < k:
            raise ValueError("Candidate budget must be at least k")
        m = min(int(budget), len(self.gallery))
        t0 = time.perf_counter() if profile else 0
        q = prepare_vectors(queries)
        codes = self.encoder.encode(queries)
        t1 = time.perf_counter() if profile else 0
        if self.backend == "faiss":
            candidates = faiss_topk(self.index, codes, m, binary=True)
        else:
            candidates = hamming_topk(codes, self.codes, m)
        t2 = time.perf_counter() if profile else 0
        if self.rerank_mode == "indexed":
            final = self.indexed_reranker(q, candidates, k)
        else:
            final = rerank_prepared(q, self.gallery, candidates, k, block_size=self.rerank_block)
        stages = ({"query_prepare_ms": (t1 - t0) * 1000,
                   "search_ms": (t2 - t1) * 1000,
                   "rerank_ms": (time.perf_counter() - t2) * 1000} if profile else {})
        return SearchResult(final, candidates, stages)

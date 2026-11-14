"""CPU ANN comparators and incremental insertion measurements."""

from __future__ import annotations

import time

import faiss
import numpy as np

from .indexed_rerank import IndexedReranker, norm_upper
from .retrieve import prepare_vectors
from .search import FaissFloat, SearchResult, faiss_topk


class HNSW(FaissFloat):
    def __init__(self, gallery, *, links=32, ef_construction=200, seed=42):
        if links < 2 or ef_construction < 1:
            raise ValueError("Positive HNSW build parameters are required")
        vectors = prepare_vectors(gallery)
        n, d = vectors.shape
        self.index = faiss.IndexHNSWFlat(d, links, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = int(ef_construction)
        self._rng = faiss.RandomGenerator(int(seed))
        self.index.hnsw.rng = self._rng
        self.index.add(vectors)
        self.storage = {"continuous_bytes": n * d * 4,
                        "graph_neighbors_bytes": faiss.vector_to_array(self.index.hnsw.neighbors).nbytes,
                        "graph_offsets_bytes": faiss.vector_to_array(self.index.hnsw.offsets).nbytes,
                        "graph_levels_bytes": faiss.vector_to_array(self.index.hnsw.levels).nbytes}

    def query(self, queries, k=10, *, budget=None, profile=False):
        if budget is None or budget < k:
            raise ValueError("efSearch must be at least K")
        self.index.hnsw.efSearch = int(budget)
        return super().query(queries, k, profile=profile)


class IVFPQRerank:
    """Standard residual PQ ADC, followed by the existing canonical refinement."""

    def __init__(self, gallery, training, *, nlist, seed=42, pq_m=16, pq_bits=8):
        self.gallery = prepare_vectors(gallery)
        train = prepare_vectors(training)
        n, d = self.gallery.shape
        if train.shape[1] != d or d % pq_m or not 1 <= nlist <= len(train):
            raise ValueError("Incompatible training vectors or IVF-PQ parameters")
        self.index = faiss.IndexIVFPQ(faiss.IndexFlatL2(d), d, nlist, pq_m, pq_bits, faiss.METRIC_L2)
        self.index.cp.seed = int(seed)
        self.index.pq.cp.seed = int(seed)
        self.index.train(train)
        self.index.add(self.gallery)
        self.reranker = IndexedReranker(self.gallery)
        self.storage = {"continuous_bytes": self.gallery.nbytes,
                        "candidate_code_bytes": n * self.index.code_size,
                        "candidate_ids_bytes": n * 8, "coarse_centroid_bytes": nlist * d * 4,
                        "codebook_bytes": faiss.vector_to_array(self.index.pq.centroids).nbytes}

    def candidates(self, prepared_queries, budget):
        if not 1 <= int(budget["nprobe"]) <= self.index.nlist or int(budget["M"]) < 10:
            raise ValueError("Invalid IVF-PQ query parameters")
        self.index.nprobe = int(budget["nprobe"])
        return faiss_topk(self.index, prepared_queries, int(budget["M"]), largest=False)

    def query(self, queries, k=10, *, budget=None, profile=False):
        if budget is None or int(budget["M"]) < k:
            raise ValueError("A candidate budget of at least K is required")
        t0 = time.perf_counter() if profile else 0
        q = prepare_vectors(queries)
        t1 = time.perf_counter() if profile else 0
        candidates = self.candidates(q, budget)
        t2 = time.perf_counter() if profile else 0
        if (candidates >= 0).all():
            final = self.reranker(q, candidates, k)
        else:
            # A small nprobe can expose fewer than M (or K) entries. Padding is
            # not a vector ID and must never reach the continuous scorer.
            final = np.full((len(q), min(k, self.index.ntotal)), -1, dtype=np.int64)
            for i, ids in enumerate(candidates):
                ids = ids[ids >= 0]
                if len(ids):
                    values = self.reranker(q[i:i + 1], ids[None], k)[0]
                    final[i, :len(values)] = values
        stages = ({"query_prepare_ms": (t1 - t0) * 1000, "search_ms": (t2 - t1) * 1000,
                   "rerank_ms": (time.perf_counter() - t2) * 1000} if profile else {})
        return SearchResult(final, candidates, stages)


class Insertions:
    """Append vectors with existing trained parameters; no retraining or deletion.

    Continuous refinement arrays reserve one measured insertion block. Its
    allocation/copy cost is reported separately and included in the amortized
    total. Native Faiss allocation remains inside each native add call.
    """

    def __init__(self, index, count):
        if count < 1:
            raise ValueError("Insertion count must be positive")
        self.index, self.remaining = index, count
        self.size = index.index.ntotal
        self.buffer = None
        if hasattr(index, "gallery"):
            self.buffer = np.empty((self.size + count, index.gallery.shape[1]), dtype=np.float32)
            self.buffer[:self.size] = index.gallery
            index.gallery = self.buffer[:self.size]
            self.reranker = getattr(index, "reranker", getattr(index, "indexed_reranker", None))
            if self.reranker is not None:
                self.reranker.gallery = index.gallery

    def add(self, vector):
        if self.remaining < 1 or len(vector) != 1:
            raise ValueError("Insert exactly one vector within the reserved block")
        normalized = prepare_vectors(vector)
        if self.buffer is not None:
            self.buffer[self.size] = normalized[0]
            self.index.gallery = self.buffer[:self.size + 1]
            if self.reranker is not None:
                self.reranker.gallery = self.index.gallery
                self.reranker.max_norm = max(self.reranker.max_norm, float(norm_upper(normalized)[0]))
        if hasattr(self.index, "gallery_encoder"):
            self.index.index.add(self.index.gallery_encoder.encode(vector))
        else:
            self.index.index.add(normalized)
        self.size += 1
        self.remaining -= 1


def measure_insertions(index, vectors):
    start = time.perf_counter_ns()
    additions = Insertions(index, len(vectors))
    prepare_ms = (time.perf_counter_ns() - start) / 1e6
    samples = []
    for vector in vectors:
        start = time.perf_counter_ns()
        additions.add(vector[None])
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {"insert_count": len(samples), "insert_p50_ms": float(np.median(samples)),
            "insert_p95_ms": float(np.percentile(samples, 95)), "insert_mean_ms": float(np.mean(samples)),
            "insert_capacity_prepare_ms": prepare_ms,
            "insert_amortized_total_ms": (prepare_ms + sum(samples)) / len(samples),
            "insert_samples_ms": samples,
            "insert_policy": "one float32 vector per add; frozen codebooks/adapter; storage preparation reported"}

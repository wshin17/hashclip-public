"""Equal-size full-scan candidate representations with shared exact refinement.

PQ/OPQ use normal asymmetric squared-L2 distance to reconstructed gallery
vectors, not Hamming distance between code bytes. Inputs are unit-normalized.
ITQ uses Faiss's PCA/ITQ transform trained on the gallery-side training modality.
"""

from __future__ import annotations

import time

import faiss
import numpy as np

from .indexed_rerank import IndexedReranker
from .retrieve import prepare_vectors, stable_topk
from .search import SearchResult


def train_representation(method, train, seed=42, bits=128):
    x = prepare_vectors(train)
    d = x.shape[1]
    if bits != 128:
        raise ValueError("The representation comparison fixes the code at 128 bits")
    if method == "itq":
        transform = faiss.ITQTransform(d, bits, True)
        transform.itq.seed = int(seed)
        transform.train(x)
        return transform
    if method == "fastscan":
        index = faiss.IndexPQFastScan(d, 32, 4, faiss.METRIC_L2)
        index.pq.cp.seed = int(seed)
        index.train(x)
        return index
    if method not in ("pq", "opq", "opq_fastscan"):
        raise ValueError(method)
    rotated_fastscan = method == 'opq_fastscan'
    parts, width = (32, 4) if rotated_fastscan else (16, 8)
    index = (faiss.IndexPQFastScan(d, parts, width, faiss.METRIC_L2) if rotated_fastscan
             else faiss.IndexPQ(d, parts, width, faiss.METRIC_L2))
    index.pq.cp.seed = int(seed)
    if method in ("opq", "opq_fastscan"):
        opq = faiss.OPQMatrix(d, parts)
        rotation = faiss.RandomRotationMatrix(d, d)
        rotation.init(int(seed))
        faiss.copy_array_to_vector(faiss.vector_to_array(rotation.A), opq.A)
        # Hold this object alive through training; OPQ does not own its pq pointer.
        training_pq = faiss.ProductQuantizer(d, parts, width)
        training_pq.cp.seed = int(seed)
        opq.pq = training_pq
        index = faiss.IndexPreTransform(opq, index)
    index.train(x)
    if method in ("opq", "opq_fastscan"):
        # Search uses the final IndexPQ; no pointer to an expired training helper.
        opq.pq = None
    return index


class ITQEncoder:
    def __init__(self, transform):
        # SWIG downcast returns a borrowed pointer. Keep the read_VectorTransform
        # owner alive even when the caller passes the loaded object inline.
        self._owner = transform
        transform = faiss.downcast_VectorTransform(transform)
        if not transform.is_trained:
            raise ValueError("ITQ must be fitted on training data first")
        self.transform = transform
        self.bits = transform.d_out
        self.nbytes = (faiss.vector_to_array(transform.mean).nbytes
                       + faiss.vector_to_array(transform.pca_then_itq.A).nbytes)

    def encode(self, raw):
        return np.packbits(self.transform.apply_py(prepare_vectors(raw)) > 0, axis=1)


def pq_topk(index, queries, k, *, search=None):
    """Stable ADC distance/ID order, including all ties at the M boundary."""
    k = min(int(k), index.ntotal)
    if k < 1:
        raise ValueError("Positive budget and nonempty PQ index required")
    width = min(k + 1, index.ntotal)
    search = index.search if search is None else search
    distances, labels = search(queries, width)
    while width < index.ntotal and np.any(distances[:, k - 1] == distances[:, -1]):
        width = min(index.ntotal, width * 2)
        distances, labels = search(queries, width)
    if (labels < 0).any():
        raise ValueError("Full-scan PQ omitted valid gallery items")
    return np.stack([stable_topk(ds, ids, k) for ds, ids in zip(distances, labels)])


class PQRerank:
    def __init__(self, gallery, trained_index):
        start = time.perf_counter()
        if not trained_index.is_trained or trained_index.ntotal:
            raise ValueError("Expected a trained empty PQ/OPQ template")
        self.gallery = prepare_vectors(gallery)
        self.index = faiss.clone_index(trained_index)
        prepared = time.perf_counter()
        self.index.add(self.gallery)
        encoded = time.perf_counter()
        self.reranker = IndexedReranker(self.gallery)
        self.build_stages_ms = {'gallery_prepare_clone_ms':(prepared-start)*1000,
                                'gallery_encode_write_ms':(encoded-prepared)*1000,
                                'reranker_prepare_ms':(time.perf_counter()-encoded)*1000}
        pq = faiss.downcast_index(self.index.index) if isinstance(self.index, faiss.IndexPreTransform) else self.index
        if not isinstance(pq, faiss.IndexPQ) or pq.code_size != 16 or pq.search_type != faiss.IndexPQ.ST_PQ:
            raise ValueError("Expected normal 16-byte PQ ADC")
        self.storage = {"continuous_bytes": self.gallery.nbytes,
                        "candidate_code_bytes": len(self.gallery) * pq.code_size,
                        "codebook_bytes": faiss.vector_to_array(pq.pq.centroids).nbytes}
        self.pq_index = pq
        self.transforms = ([faiss.downcast_VectorTransform(self.index.chain.at(i))
                            for i in range(self.index.chain.size())]
                           if isinstance(self.index, faiss.IndexPreTransform) else [])
        self.storage['rotation_bytes'] = sum(faiss.vector_to_array(t.A).nbytes + faiss.vector_to_array(t.b).nbytes
                                             for t in self.transforms)

    def query(self, queries, k=10, *, budget=None, profile=False):
        if budget is None or budget < k:
            raise ValueError("Candidate budget must include at least K items")
        t0 = time.perf_counter() if profile else 0
        q = prepare_vectors(queries)
        t1 = time.perf_counter() if profile else 0
        rotation_seconds = 0
        if profile:
            def search_with_rotation(queries, width):
                # Boundary-tie retries repeat the wrapper's rotation too.
                nonlocal rotation_seconds
                start = time.perf_counter()
                rotated = queries
                for transform in self.transforms:
                    rotated = transform.apply_py(rotated)
                rotation_seconds += time.perf_counter()-start
                return self.pq_index.search(rotated,width)
            candidates = pq_topk(self.index, q, budget, search=search_with_rotation)
        else:
            # Preserve the existing timed IndexPreTransform search path.
            candidates = pq_topk(self.index, q, budget)
        t2 = time.perf_counter() if profile else 0
        final = self.reranker(q, candidates, k)
        stages = ({"query_prepare_ms": (t1-t0+rotation_seconds) * 1000,
                   "query_normalize_ms": (t1 - t0) * 1000,
                   "query_rotation_ms": rotation_seconds * 1000,
                   "search_ms": (t2-t1-rotation_seconds) * 1000,
                   "rerank_ms": (time.perf_counter() - t2) * 1000} if profile else {})
        return SearchResult(final, candidates, stages)


class FastScanRerank:
    """PQ32x4 with native Faiss top-M and the same continuous refinement.

    Quantized-distance ties follow Faiss. Do not extend the boundary or
    impose a canonical ID order on the candidate search.
    """

    def __init__(self, gallery, trained_index, *, rotation=False):
        start = time.perf_counter()
        wrapped = isinstance(trained_index, faiss.IndexPreTransform)
        if wrapped != rotation:
            raise ValueError('FastScan rotation does not match the requested method')
        pq = faiss.downcast_index(trained_index.index) if wrapped else trained_index
        if not isinstance(pq, faiss.IndexPQFastScan):
            raise ValueError('Expected a PQ FastScan template')
        # Faiss serializes a fitted OPQMatrix as its LinearTransform; the
        # training manifest retains the OPQ fitting parameters and provenance.
        if wrapped:
            transform = faiss.downcast_VectorTransform(trained_index.chain.at(0))
            if (trained_index.chain.size() != 1 or not isinstance(transform, faiss.LinearTransform)
                    or transform.d_in != trained_index.d or transform.d_out != pq.d):
                raise ValueError('OPQ FastScan requires one full-dimensional trained rotation')
        if not trained_index.is_trained or trained_index.ntotal:
            raise ValueError('Expected a trained empty FastScan template')
        if (pq.M, pq.nbits, pq.code_size) != (32,4,16):
            raise ValueError('FastScan requires PQ32x4 and 16 bytes per item')
        self.gallery = prepare_vectors(gallery)
        self.index = faiss.clone_index(trained_index)
        prepared = time.perf_counter()
        self.index.add(self.gallery)
        encoded = time.perf_counter()
        self.reranker = IndexedReranker(self.gallery)
        self.build_stages_ms = {'gallery_prepare_clone_ms':(prepared-start)*1000,
                                'gallery_encode_write_ms':(encoded-prepared)*1000,
                                'reranker_prepare_ms':(time.perf_counter()-encoded)*1000}
        self.pq_index = faiss.downcast_index(self.index.index) if wrapped else self.index
        self.transforms = ([faiss.downcast_VectorTransform(self.index.chain.at(0))] if wrapped else [])
        self.storage = {'continuous_bytes':self.gallery.nbytes,
                        'candidate_code_bytes':self.pq_index.ntotal2*self.pq_index.code_size,
                        'codebook_bytes':faiss.vector_to_array(self.pq_index.pq.centroids).nbytes,
                        'rotation_bytes':sum(faiss.vector_to_array(t.A).nbytes + faiss.vector_to_array(t.b).nbytes
                                             for t in self.transforms)}
        self.search_settings = {k:int(getattr(self.pq_index,k)) for k in
                                ('M','M2','nbits','code_size','bbs','implem','qbs')}

    def query(self, queries, k=10, *, budget=None, profile=False):
        if budget is None or budget < k:
            raise ValueError('Candidate budget must cover top-K')
        t0 = time.perf_counter() if profile else 0
        q = prepare_vectors(queries)
        t1 = time.perf_counter() if profile else 0
        width = min(int(budget),len(self.gallery))
        rotation_seconds = 0
        if profile and self.transforms:
            rotated = q
            for transform in self.transforms:
                rotated = transform.apply_py(rotated)
            rotation_seconds = time.perf_counter()-t1
            _, candidates = self.pq_index.search(rotated, width)
        else:
            _, candidates = self.index.search(q, width)
        t2 = time.perf_counter() if profile else 0
        final = self.reranker(q,candidates,k)
        stages = ({'query_prepare_ms':(t1-t0)*1000,'query_normalize_ms':(t1-t0)*1000,
                   'query_rotation_ms':rotation_seconds*1000,'search_ms':(t2-t1-rotation_seconds)*1000,
                   'rerank_ms':(time.perf_counter()-t2)*1000} if profile else {})
        return SearchResult(final,candidates,stages)


def pq_prefix_positions(index, queries, reference, *, batch, maximum, audit_budgets=()):
    """Exact ranks inside a stable ADC prefix; unknown ranks stay at infinity.

    Once a prefix retains at least 99% of reference members over all validation
    queries, the 95% and 99% order statistics equal those of the full ranking.
    Query batching, ADC and ID tie-breaking are the unchanged retrieval path.
    """
    from .metrics import neighbor_metrics
    from .representation_quality import position_metrics
    positions = np.full(reference.shape, np.iinfo(np.int64).max, dtype=np.int64)
    audit_starts = {0, ((len(queries)-1)//batch)*batch}
    for start in range(0, len(queries), batch):
        q = queries[start:start+batch]
        ref = reference[start:start+len(q)]
        candidates = pq_topk(index.index, prepare_vectors(q), maximum)
        qi, rank, ri = np.where(candidates[:, :, None] == ref[:, None, :])
        positions[start+qi, ri] = rank+1
        if start in audit_starts:
            for budget in audit_budgets:
                result = index.query(q, budget=budget)
                np.testing.assert_array_equal(result.candidates, candidates[:, :min(budget,len(index.gallery))])
                expected = position_metrics(positions[start:start+len(q)], budget)
                actual = neighbor_metrics(ref, result.final, result.candidates)
                for name in ('candidate_recall_at_10','candidate_complete_at_10','candidate_recall_at_1'):
                    if not np.isclose(expected[name],actual[name],atol=1e-12,rtol=0):
                        raise ValueError(f'OPQ prefix differs from online retrieval: {name}')
                if not np.isclose(expected['candidate_recall_at_10'], actual['final_recall_at_10'],atol=1e-12,rtol=0):
                    raise ValueError('OPQ refinement failed to recover retained reference members')
    return positions

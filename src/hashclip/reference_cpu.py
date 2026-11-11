"""Prepare canonical top-10 references without a query-by-gallery score cache.

Threads distribute independent queries; every gallery score still uses the
original float32 row scorer and stable global-ID tie rule. This is offline
reference preparation, never a retrieval latency measurement.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from .evaluate_cpu import configure, input_identity, load_features


def canonical_reference(gallery, queries, *, k=10, workers=1, batch=256, progress=None):
    import numpy as np
    from .retrieve import prepare_vectors, score_rows, stable_topk

    if workers < 1 or batch < 1 or k < 1 or not len(gallery):
        raise ValueError("Positive workers, batch, k and a nonempty gallery are required")
    g, q = prepare_vectors(gallery), prepare_vectors(queries)
    ids = np.arange(len(g))
    result = np.empty((len(q), min(k, len(g))), dtype=np.int64)

    def nearest(query):
        return stable_topk(score_rows(query, g), ids, k, largest=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(q), batch):
            stop = min(start + batch, len(q))
            result[start:stop] = list(pool.map(nearest, q[start:stop]))
            if progress is not None:
                progress(stop, len(q))
    return result


def run(args):
    # The scorer itself is single-threaded; parallelism is across queries.
    configure({"threads": 1, "blas_core": "Haswell"})
    import numpy as np
    from threadpoolctl import threadpool_limits
    from .utils import git_rev

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new reference output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"git_commit": git_rev(), "workers": args.workers, "k": 10,
                "quality_only": True, "score_cache": False,
                "scorer": "original float32 row scorer; score descending then gallery ID ascending",
                "allow_unlabeled_gallery": args.allow_unlabeled_gallery, "splits": {}}
    for split in ("val", "test"):
        qpath = args.queries / f"backbone_{split}.npz"
        gpath = args.gallery / f"backbone_{split}.npz"
        q = load_features(qpath, captions_per_image=None)
        g = load_features(gpath, captions_per_image=None, allow_unlabeled=args.allow_unlabeled_gallery)
        refs, timings = {}, {}
        with threadpool_limits(limits=1):
            for direction, qm, gm in (("t2i", "txt", "img"), ("i2t", "img", "txt")):
                start = time.perf_counter()
                refs[direction] = canonical_reference(g[gm + "_feat"], q[qm + "_feat"], workers=args.workers,
                    progress=lambda done, total: print(f"{split} {direction} reference {done}/{total}", flush=True))
                timings[direction] = time.perf_counter() - start
        target = args.output / f"{split}_reference.npz"
        temporary = target.with_suffix(".partial.npz")
        np.savez(temporary, **refs)
        temporary.replace(target)
        manifest["splits"][split] = {"queries": input_identity(qpath), "gallery": input_identity(gpath),
                                      "reference": input_identity(target), "preparation_seconds": timings}
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Completed {split} canonical references", flush=True)
        del q, g, refs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("queries", "gallery", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--allow-unlabeled-gallery", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    run(args)


if __name__ == "__main__":
    main()

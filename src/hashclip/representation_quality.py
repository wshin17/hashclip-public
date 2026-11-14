"""Full-query candidate curves and validation-selected M.

This is a quality-only job, not a latency benchmark. All methods scan every
gallery code. Query batch is 32. Continuous refinement uses the shared scorer;
the candidate ranks determine mean retention and complete top-10 recovery.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess

from .evaluate_cpu import configure, input_identity, load_features, make_index, write_csv
from .paths import ROOT, runs_root


def binary_reference_positions(distances, reference):
    """One-based stable Hamming ranks of reference IDs; distances are ID-ordered."""
    import numpy as np

    before = np.r_[0, np.cumsum(np.bincount(distances))]
    values = distances[reference]
    positions = np.empty(len(reference), dtype=np.int64)
    for value in np.unique(values):
        selected = values == value
        tied_ids = np.flatnonzero(distances == value)
        positions[selected] = before[value] + np.searchsorted(tied_ids, reference[selected]) + 1
    return positions


def minimum_budget(positions, target):
    import numpy as np

    fraction = Fraction(str(target)) * positions.size
    count = (fraction.numerator + fraction.denominator - 1) // fraction.denominator
    if count < 1 or count > positions.size:
        raise ValueError("Retention target must be in (0,1]")
    return int(np.partition(positions.ravel(), count - 1)[count - 1])


def position_metrics(positions, budget):
    import numpy as np

    hits = positions <= budget
    return {"candidate_recall_at_10": float(hits.mean()),
            "candidate_complete_at_10": float(hits.all(axis=1).mean()),
            "candidate_recall_at_1": float(hits[:, 0].mean())}


def measure_positions(index, queries, reference, batch=32):
    import numpy as np
    from .representations import PQRerank, pq_topk
    from .retrieve import prepare_vectors, rerank_prepared

    n = len(index.gallery)
    positions = np.empty(reference.shape, dtype=np.int64)
    for start in range(0, len(queries), batch):
        q = queries[start:start + batch]
        if isinstance(index, PQRerank):
            order = pq_topk(index.index, prepare_vectors(q), n)
            for i, ids in enumerate(order):
                ranks = np.empty(n, dtype=np.int64)
                ranks[ids] = np.arange(1, n + 1)
                positions[start + i] = ranks[reference[start + i]]
        else:
            codes = index.encoder.encode(q)
            limits, distances, labels = index.index.range_search(codes, index.index.d + 1)
            for i in range(len(q)):
                lo, hi = int(limits[i]), int(limits[i + 1])
                if hi - lo != n:
                    raise ValueError("Full Hamming scan omitted gallery items")
                by_id = np.empty(n, dtype=np.int32)
                by_id[labels[lo:hi]] = distances[lo:hi]
                positions[start + i] = binary_reference_positions(by_id, reference[start + i])
    # Audit the candidate-rank shortcut against the actual shared refinement.
    # Full batch inputs preserve the same projection / ADC numeric path.
    starts = sorted({0, max(0, ((len(queries) - 1) // batch) * batch)})
    for start in starts:
        q = queries[start:start + batch]
        for budget in sorted({10, max(10, n // 100), max(10, n // 10), n}):
            result = index.query(q, budget=budget)
            canonical = rerank_prepared(prepare_vectors(q), index.gallery, result.candidates, 10, block_size=256)
            np.testing.assert_array_equal(result.final, canonical)
            if budget == n:
                np.testing.assert_array_equal(result.final, reference[start:start + len(q)])
            retained = np.array([[r in ids for r in ref] for ref, ids in
                                 zip(reference[start:start + len(q)], result.candidates)])
            np.testing.assert_array_equal(retained, positions[start:start + len(q)] <= budget)
    return positions


def run(args):
    configure({"threads": args.threads, "blas_core": "Haswell"})
    import faiss
    import numpy as np
    import yaml
    from threadpoolctl import threadpool_limits
    from .numpy_adapter import numpy_checkpoint_path
    from .representations import ITQEncoder, PQRerank
    from .search import BinaryRerank, encode_chunks

    faiss.omp_set_num_threads(args.threads)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    models = json.loads((args.models / "manifest.json").read_text())
    if input_identity(args.train)["sha256"] != models["training_features"]["sha256"]:
        raise ValueError("Baseline training features changed")
    with np.load(args.train, allow_pickle=True) as archive:
        training_ids = set(map(str, archive["img_ids"]))
    training = yaml.safe_load(args.training.read_text())
    cfg = {"bits": 128, "threads": args.threads, "binary_use_heap": False,
           "rerank_block": 256, "rerank_mode": "indexed", "adapter_backend": "numpy"}
    checkpoints = {}
    for seed, run in training["training_runs"].items():
        checkpoints[int(seed)] = runs_root() / run / "checkpoints" / f"hashclip_b128_seed{seed}_reselect.pt"
    manifest = {"code_bytes_per_item": 16, "batch": 32, "threads": args.threads,
                "methods": args.methods,
                "training_domain": training.get("training_domain", "flickr"),
                "large_gallery_contains_training_images": bool(args.allow_unlabeled_gallery and training.get("large_gallery_contains_training_images", False)),
                "quality_only": True, "training_models": input_identity(args.models / "manifest.json"),
                "checkpoints": [input_identity(numpy_checkpoint_path(p)) for p in checkpoints.values()],
                "targets": [.95, .99], "reference": "original canonical float32 scorer",
                "budget_selection": "minimum M on validation for mean top-10 retention; fixed before test",
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "inputs": []}
    rows, selected, test_selected = [], {}, []
    for split in ("val", "test"):
        qpath, gpath = args.queries / f"backbone_{split}.npz", args.gallery / f"backbone_{split}.npz"
        queries = load_features(qpath, captions_per_image=None)
        gallery = load_features(gpath, captions_per_image=None, allow_unlabeled=args.allow_unlabeled_gallery)
        if training_ids & set(map(str, queries["img_ids"])):
            raise ValueError("Baseline training set overlaps query validation/test images")
        ref_path = args.references / f"{split}_reference.npz"
        with np.load(ref_path) as f:
            references = {d: f[d] for d in ("t2i", "i2t")}
        manifest["inputs"].extend(input_identity(p) for p in (qpath, gpath, ref_path))
        (args.output / "config.json").write_text(json.dumps(manifest, indent=2) + "\n")
        with threadpool_limits(limits=args.threads):
            for direction, qm, gm in (("t2i", "txt", "img"), ("i2t", "img", "txt")):
                q, g, ref = queries[qm + "_feat"], gallery[gm + "_feat"], references[direction]
                if ref.shape != (len(q), 10):
                    raise ValueError("Reference/query shape mismatch")
                grid = sorted({10, *[max(10, min(len(g), int(np.ceil(f * len(g)))))
                                    for f in (.001, .002, .005, .01, .02, .05, .1, .2, .3, .5, .75, 1)]})
                for method in args.methods:
                    for seed in (0, 1, 42):
                        print(f"{split} {direction} {method} seed={seed}", flush=True)
                        if method in ("hashclip", "lsh"):
                            index = make_index({"method": method, "seed": seed, "direction": direction,
                                                "checkpoint": str(checkpoints[seed])}, g, cfg)
                        elif method == "itq":
                            encoder = ITQEncoder(faiss.read_VectorTransform(str(args.models / f"itq_{gm}_seed{seed}.faiss")))
                            index = BinaryRerank(g, encode_chunks(encoder, g), encoder,
                                                 use_heap=False, rerank_mode="indexed")
                        else:
                            trained = faiss.read_index(str(args.models / f"{method}_{gm}_seed{seed}.faiss"))
                            index = PQRerank(g, trained)
                        positions = measure_positions(index, q, ref)
                        np.save(args.output / f"{split}_{direction}_{method}_seed{seed}_positions.npy", positions)
                        key = f"{direction}/{method}/{seed}"
                        if split == "val":
                            selected[key] = {str(t): minimum_budget(positions, t) for t in (.95, .99)}
                        for budget in sorted(set(grid) | set(selected[key].values())):
                            row = {"split": split, "direction": direction, "method": method, "seed": seed,
                                   "batch": 32, "M": budget, "effective_M": min(budget, len(g)),
                                   "gallery_size": len(g), "queries": len(q), **position_metrics(positions, budget)}
                            rows.append(row)
                        if split == "test":
                            for target, budget in selected[key].items():
                                metrics = position_metrics(positions, budget)
                                test_selected.append({"direction": direction, "method": method, "seed": seed,
                                    "target": float(target), "M": budget, "effective_M": min(budget, len(g)),
                                    "target_met_on_test": metrics["candidate_recall_at_10"] >= float(target), **metrics})
                        write_csv(args.output / "curves.csv", rows)
                        if test_selected:
                            write_csv(args.output / "selected_test.csv", test_selected)
                        del index
        if split == "val":
            (args.output / "selected.json").write_text(json.dumps(selected, indent=2) + "\n")
    (args.output / "complete.json").write_text(json.dumps({"curve_records": len(rows),
        "test_targets": len(test_selected)}, indent=2) + "\n")
    print(f"Completed {len(rows)} curve points and {len(test_selected)} fixed-budget test targets", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("queries", "gallery", "references", "train", "models", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--training", type=Path, default=ROOT / "configs/icde_reselect.yaml")
    parser.add_argument("--methods", nargs="+", choices=("hashclip", "lsh", "itq", "pq", "opq"),
                        default=["hashclip", "lsh", "itq", "pq", "opq"])
    parser.add_argument("--allow-unlabeled-gallery", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

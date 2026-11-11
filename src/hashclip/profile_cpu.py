"""Validation-only breakdown of the existing binary retrieval implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .evaluate_cpu import configure, load_features, make_index, retrieval_inputs, write_csv
from .paths import ROOT


def source_context(source):
    """Resolve recorded input locations from an existing completed evaluation."""
    meta = json.loads((source / "config.json").read_text())
    inputs = (json.loads(Path(meta["baseline"]["path"]).read_text())
              if "baseline" in meta else meta)
    selected = json.loads((source / "selected.json").read_text())
    return meta, inputs, selected


def source_spec(source, group, split="val"):
    meta, inputs, _ = source_context(source)
    spec = {**group, "config": meta["config"], "split": split,
            "features": inputs["query_caches"][split],
            "gallery_features": inputs["gallery_caches"][split]}
    if group["method"] == "hashclip":
        name = f"hashclip_b{meta['config']['bits']}_seed{group['seed']}_main.pt"
        spec["checkpoint"] = next(x["path"] for x in meta["inputs"] if Path(x["path"]).name == name)
    return spec


def profile_query(index, raw, budget):
    """Same operations as the legacy online path, with separate internal timers."""
    import numpy as np
    from .retrieve import prepare_vectors, score_rows, stable_topk

    stats = {key: 0.0 for key in ("query_ms", "initial_search_ms", "tie_search_ms",
                                 "candidate_select_ms", "rerank_order_ms", "indexed_screen_ms",
                                 "gather_ms", "score_ms", "final_select_ms")}
    stats["canonical_scored_items"] = 0
    t = time.perf_counter()
    q = prepare_vectors(raw)
    codes = index.encoder.encode(raw)
    stats["query_ms"] = (time.perf_counter() - t) * 1000
    m = min(budget, len(index.gallery))
    t = time.perf_counter()
    distances, labels = index.index.search(codes, min(m + 1, len(index.gallery)))
    stats["initial_search_ms"] = (time.perf_counter() - t) * 1000
    candidates = np.empty((len(q), m), dtype=np.int64)
    tie_queries = tie_items = 0
    for i in range(len(q)):
        t = time.perf_counter()
        valid = labels[i] >= 0
        ids, values = labels[i, valid], distances[i, valid]
        tied = len(ids) > m and values[m - 1] == values[m]
        stats["candidate_select_ms"] += (time.perf_counter() - t) * 1000
        if tied:
            t = time.perf_counter()
            _, values, ids = index.index.range_search(codes[i:i + 1], int(values[m - 1]) + 1)
            stats["tie_search_ms"] += (time.perf_counter() - t) * 1000
            tie_queries += 1
            tie_items += len(ids)
        t = time.perf_counter()
        candidates[i] = stable_topk(values, ids, m)
        stats["candidate_select_ms"] += (time.perf_counter() - t) * 1000
    final = np.empty((len(q), 10), dtype=np.int64)
    for i, ids in enumerate(candidates):
        t = time.perf_counter()
        if index.rerank_block or index.rerank_mode == "indexed":
            ids = np.sort(ids)
        stats["rerank_order_ms"] += (time.perf_counter() - t) * 1000
        if index.rerank_mode == "indexed":
            t = time.perf_counter()
            ids, _ = index.indexed_reranker.screen(q[i], ids, 10)
            stats["indexed_screen_ms"] += (time.perf_counter() - t) * 1000
        stats["canonical_scored_items"] += len(ids)
        scores = np.empty(len(ids), dtype=np.float32)
        block = index.rerank_block or len(ids)
        for start in range(0, len(ids), block):
            t = time.perf_counter()
            vectors = index.gallery[ids[start:start + block]]
            stats["gather_ms"] += (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            scores[start:start + len(vectors)] = score_rows(q[i], vectors)
            stats["score_ms"] += (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        final[i] = stable_topk(scores, ids, 10, largest=True)
        stats["final_select_ms"] += (time.perf_counter() - t) * 1000
    stats.update(tie_queries=tie_queries, tie_items=tie_items)
    return candidates, final, stats


def worker(spec):
    configure(spec["config"])
    import faiss
    import numpy as np
    from threadpoolctl import threadpool_limits
    from .evaluate_cpu import timing

    cfg = spec["config"]
    faiss.omp_set_num_threads(cfg["threads"])
    captions = cfg.get("captions_per_image", 5)
    arrays = load_features(spec["features"], captions_per_image=captions)
    gallery_arrays = (arrays if spec["features"] == spec["gallery_features"] else
                      load_features(spec["gallery_features"], captions_per_image=captions))
    gallery, queries, _, _ = retrieval_inputs(arrays, gallery_arrays, spec["direction"])
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    with threadpool_limits(limits=cfg["threads"]):
        index = make_index(spec, gallery, cfg)
        if spec.get("indexed"):
            from .indexed_rerank import IndexedReranker
            index.indexed_reranker = IndexedReranker(index.gallery)
        rows, raw_samples = [], []
        for batch in cfg["batch_sizes"]:
            count = min(len(queries), cfg["timing_queries"])
            count -= count % batch
            sampled = queries[np.linspace(0, len(queries) - 1, count, dtype=int)]
            chunks = [sampled[i:i + batch] for i in range(0, count, batch)]
            for budget in spec["budgets"]:
                # A small implementation grid; never select with test data.
                variants = ([(True, 0), (False, 0), (False, 256), (False, 512), (False, 1024)]
                            if spec.get("variants") else [(True, 0), (False, 0)])
                variants = [(heap, block, "canonical") for heap, block in variants]
                if spec.get("indexed"):
                    variants += [(False, 256, "canonical"), (False, 256, "indexed")]
                index.index.use_heap = True
                index.rerank_block = 0
                index.rerank_mode = "canonical"
                expected = [index.query(q, budget=budget) for q in chunks]
                for heap, block, mode in variants:
                    index.index.use_heap = heap
                    index.rerank_block = block
                    index.rerank_mode = mode
                    times, samples = timing(index, queries, batch, budget, cfg)
                    sums = {}
                    for qi, q in enumerate(chunks):
                        candidates, final, stats = profile_query(index, q, budget)
                        actual = index.query(q, budget=budget)
                        np.testing.assert_array_equal(candidates, actual.candidates)
                        np.testing.assert_array_equal(final, actual.final)
                        np.testing.assert_array_equal(actual.candidates, expected[qi].candidates)
                        np.testing.assert_array_equal(actual.final, expected[qi].final)
                        for key, value in stats.items():
                            sums[key] = sums.get(key, 0.0) + value
                    row = {"method": spec["method"], "seed": spec["seed"], "direction": spec["direction"],
                           "split": spec["split"], "batch": batch, "M": budget,
                           "gallery_size": len(gallery), "sampled_queries": count, "heap": heap,
                           "rerank_block": block, "rerank_mode": mode,
                           **times, **{key: value / count for key, value in sums.items()}}
                    rows.append(row)
                    raw_samples.append({"batch": batch, "M": budget, "heap": heap, "rerank_block": block,
                                        "rerank_mode": mode,
                                        "samples_ms": samples})
        return {"rows": rows, "timings": raw_samples}


def run(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    meta, _, selected = source_context(args.source)
    manifest = {"source": str(args.source.resolve()), "config": meta["config"], "seed": args.seed,
                "variants": args.variants, "indexed": args.indexed,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "purpose": "validation-only bottleneck diagnosis; same original selected budgets"}
    (args.output / "config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    for direction in ("t2i", "i2t"):
        budgets = sorted({p["M"] for p in selected if p["method"] == "hashclip"
                          and p["seed"] == args.seed and p["direction"] == direction})
        spec = source_spec(args.source, {"method": "hashclip", "seed": args.seed,
                                        "direction": direction, "budgets": budgets,
                                        "variants": args.variants, "indexed": args.indexed})
        result = subprocess.run([sys.executable, "-m", "hashclip.profile_cpu", "--worker"],
                                input=json.dumps(spec), capture_output=True, text=True)
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        if result.returncode:
            raise RuntimeError(f"Profiling failed for {direction}")
        payload = json.loads(result.stdout)
        with (args.output / "workers.jsonl").open("a") as f:
            f.write(json.dumps({"spec": spec, "result": payload}) + "\n")
        rows.extend(payload["rows"])
        write_csv(args.output / "breakdown.csv", rows)
        print(f"{direction}: {len(payload['rows'])} profiles, online output checks passed", flush=True)


def main():
    if "--worker" in sys.argv:
        print(json.dumps(worker(json.load(sys.stdin))))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", action="store_true", help="Also compare three bounded re-ranking block sizes")
    parser.add_argument("--indexed", action="store_true", help="Compare indexed scoring with canonical boundary re-scoring")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

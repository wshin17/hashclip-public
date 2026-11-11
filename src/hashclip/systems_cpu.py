"""Validation-selected CPU systems comparison for D-008.

Uses the existing complete-query timer and canonical reference neighbors.
Memory runs are fresh processes without reference/quality allocations.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from .evaluate_cpu import (configure, input_identity, load_features, make_index,
                           process_memory, quality, retrieval_inputs, timing, write_csv)
from .paths import ROOT, data_root, runs_root

BUILD_KEYS = ("method", "seed", "direction", "nlist", "hnsw_m", "ef_construction", "pq_m", "pq_bits")
POINT_KEYS = BUILD_KEYS + ("batch", "M", "nprobe", "ef_search")


def native_quality(index, queries, reference, mapping, direction, batch, budget, query_rows, workers=1):
    """Parallelize independent quality queries, never timed retrieval.

    Each native call still uses one thread and exactly the original batch.
    Set search parameters once before concurrent read-only CPU searches.
    """
    if workers == 1:
        return quality(index, queries, reference, mapping, direction, batch, budget,
                       query_rows, report_task_recall=False)
    from concurrent.futures import ThreadPoolExecutor
    import faiss
    from .metrics import retrieval_batch_metrics
    from .retrieve import prepare_vectors
    from .search import faiss_topk
    if hasattr(index.index, "nprobe"):
        index.index.nprobe = int(budget)
    if hasattr(index.index, "hnsw"):
        index.index.hnsw.efSearch = int(budget)
    def one(start):
        q = queries[start:start + batch]
        final = faiss_topk(index.index, prepare_vectors(q), 10)
        values = retrieval_batch_metrics(reference[start:start + batch], final, None,
                    mapping, direction, start, query_rows, report_task_recall=False)
        return len(q), values
    sums = {}
    with ThreadPoolExecutor(max_workers=workers, initializer=lambda: faiss.omp_set_num_threads(1)) as pool:
        for count, values in pool.map(one, range(0, len(queries), batch)):
            for key, value in values.items():
                sums[key] = sums.get(key, 0.0) + value * count
    return {key: value / len(queries) for key, value in sums.items()}


def resolve_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts[0] in ("data", "runs"):
        root = data_root() if path.parts[0] == "data" else runs_root()
        return root.joinpath(*path.parts[1:])
    return ROOT / path


def build_index(spec, gallery, cfg):
    from .system_search import HNSW, IVFPQRerank
    if spec['method'] in ('fastscan', 'opq_fastscan'):
        import faiss
        from .representations import FastScanRerank
        return FastScanRerank(gallery, faiss.read_index(spec[spec['method']+'_model']),
                              rotation=spec['method']=='opq_fastscan')
    if spec['method'] == 'opq':
        import faiss
        from .representations import PQRerank
        trained = faiss.read_index(spec['opq_model'])
        if not isinstance(trained, faiss.IndexPreTransform):
            raise ValueError('OPQ requires a trained rotation and PQ template')
        return PQRerank(gallery, trained)
    if spec["method"] == "hnsw":
        return HNSW(gallery, links=spec["hnsw_m"], ef_construction=spec["ef_construction"], seed=spec["seed"])
    if spec["method"] == "ivf_pq":
        import numpy as np
        modality = "img" if spec["direction"] == "t2i" else "txt"
        with np.load(spec["training_features"], allow_pickle=True) as f:
            train = f[modality + "_feat"]
        return IVFPQRerank(gallery, train, nlist=spec["nlist"], seed=spec["seed"],
                           pq_m=spec["pq_m"], pq_bits=spec["pq_bits"])
    if spec["method"] == "hashclip_coco":
        spec = {**spec, "method": "hashclip"}
    return make_index(spec, gallery, cfg)


def query_budget(point):
    method = point["method"]
    if method in ("hashclip", "hashclip_coco", "lsh", "opq", "fastscan", "opq_fastscan"):
        return point["M"]
    if method == "ivf_pq":
        return {"M": point["M"], "nprobe": point["nprobe"]}
    if method == "ivf_flat":
        return point["nprobe"]
    if method == "hnsw":
        return point["ef_search"]
    return None


def row_parameters(spec, batch, budget):
    row = {k: spec.get(k) for k in BUILD_KEYS}
    row.update(batch=batch, M=None, nprobe=None, ef_search=None)
    method = spec["method"]
    if method in ("hashclip", "hashclip_coco", "lsh", "opq", "fastscan", "opq_fastscan"):
        row["M"] = int(budget)
    elif method == "ivf_pq":
        row.update(budget)
    elif method == "ivf_flat":
        row["nprobe"] = int(budget)
    elif method == "hnsw":
        row["ef_search"] = int(budget)
    return row


def positions_metrics(positions, budget):
    from .representation_quality import position_metrics
    metrics = position_metrics(positions, budget)
    for k in (1, 10):
        recall = metrics[f"candidate_recall_at_{k}"]
        complete = recall if k == 1 else metrics["candidate_complete_at_10"]
        metrics.update({f"final_recall_at_{k}": recall, f"final_complete_at_{k}": complete,
                        f"ordered_recovery_at_{k}": complete})
    return metrics


def binary_deletions(index, count=100):
    """Delete one code from a fresh full clone each time; clone is untimed."""
    import faiss
    import numpy as np
    if not isinstance(index,faiss.IndexBinaryFlat) or index.ntotal < 2 or count < 100:
        raise ValueError('Need a nonempty binary flat index and at least 100 trials')
    codes = faiss.vector_to_array(index.xb).reshape(index.ntotal,index.d//8)
    ids = np.random.default_rng(20261109).integers(0,index.ntotal,size=count+5)
    samples = []
    for trial,item in enumerate(ids):
        copy = faiss.IndexBinaryFlat(index.d)
        copy.add(codes)
        remove = np.array([item],dtype=np.int64)
        start = time.perf_counter_ns()
        removed = copy.remove_ids(remove)
        elapsed = (time.perf_counter_ns()-start)/1e6
        if removed != 1 or copy.ntotal != index.ntotal-1:
            raise ValueError('Single-code deletion did not remove exactly one item')
        if trial in (0,count+4):
            actual = faiss.vector_to_array(copy.xb).reshape(copy.ntotal,index.d//8)
            np.testing.assert_array_equal(actual,np.concatenate((codes[:item],codes[item+1:])))
        if trial >= 5:
            samples.append(elapsed)
    return {'delete_count':count,'delete_p50_ms':float(np.median(samples)),
            'delete_p95_ms':float(np.percentile(samples,95)),'delete_mean_ms':float(np.mean(samples)),
            'delete_samples_ms':samples,'delete_ids':ids[5:].tolist(),'gallery_size':index.ntotal,
            'code_bytes_per_item':index.d//8,'index_preserved':True,
            'delete_policy':'100 independent full binary-code index copies; 5 warmups; deterministic uniform ID; copy and verification excluded; native remove_ids wrapper included; embeddings untouched'}


def ivfpq_positions(index, queries, reference, *, batch, nprobe, budgets):
    """One ADC search supplies every candidate prefix at a fixed nprobe."""
    import numpy as np
    from .metrics import neighbor_metrics
    from .retrieve import prepare_vectors

    positions = np.full(reference.shape, np.iinfo(np.int64).max, dtype=np.int64)
    audit_starts = {0, ((len(queries) - 1) // batch) * batch}
    maximum = max(budgets)
    for start in range(0, len(queries), batch):
        q = queries[start:start + batch]
        ref = reference[start:start + len(q)]
        candidates = index.candidates(prepare_vectors(q), {"nprobe": nprobe, "M": maximum})
        qi, rank, ri = np.where(candidates[:, :, None] == ref[:, None, :])
        positions[start + qi, ri] = rank + 1
        if start in audit_starts:
            for budget in budgets:
                result = index.query(q, budget={"nprobe": nprobe, "M": budget})
                np.testing.assert_array_equal(result.candidates, candidates[:, :min(budget, index.index.ntotal)])
                expected = positions_metrics(positions[start:start + len(q)], budget)
                actual = neighbor_metrics(ref, result.final, result.candidates)
                for key, value in expected.items():
                    if not np.isclose(value, actual[key], atol=1e-12, rtol=0):
                        raise ValueError(f"IVF-PQ prefix quality disagrees with actual refinement: {key}")
    return positions


def cached_binary_positions(index, gallery, queries, reference, codes, batch, cache, budgets):
    """Reuse audited ranks only after checking the source encoding path.

    The controller verifies source inputs/checkpoints. Recreate source codes
    with its batch and BLAS thread count, compare every gallery code, and
    repair ranks for every query whose current online code differs.
    """
    import faiss
    import numpy as np
    from threadpoolctl import threadpool_limits
    from .representation_quality import binary_reference_positions, measure_positions
    from .search import encode_chunks

    positions = np.load(cache['path'], allow_pickle=False).copy()
    if positions.shape != reference.shape:
        raise ValueError('Cached binary ranks do not match reference shape')
    with threadpool_limits(limits=cache['threads']):
        source_gallery = encode_chunks(index.gallery_encoder, gallery)
        source_queries = encode_chunks(index.encoder, queries, batch=cache['batch'])
    current_gallery = faiss.vector_to_array(index.index.xb).reshape(source_gallery.shape)
    gallery_changed = int(np.count_nonzero(np.any(current_gallery != source_gallery, axis=1)))
    changed = np.flatnonzero(np.any(codes != source_queries, axis=1))
    if gallery_changed:
        positions = measure_positions(index, queries, reference, batch=batch)
    else:
        for row in changed:
            _, distances, labels = index.index.range_search(codes[row:row + 1], index.index.d + 1)
            by_id = np.empty(len(gallery), dtype=np.int32)
            by_id[labels] = distances
            positions[row] = binary_reference_positions(by_id, reference[row])
        # Keep actual refinement and current batch shape in the audit.
        for start in sorted({0, ((len(queries) - 1) // batch) * batch}):
            q = queries[start:start + batch]
            for budget in budgets:
                result = index.query(q, budget=budget)
                retained = np.array([[r in ids for r in ref] for ref, ids in
                                     zip(reference[start:start + len(q)], result.candidates)])
                np.testing.assert_array_equal(retained, positions[start:start + len(q)] <= budget)
    return positions, {'source':cache, 'batch':batch, 'gallery_code_mismatches':gallery_changed,
                       'query_code_mismatches':int(len(changed)),
                       'recomputed_all_ranks':bool(gallery_changed)}


def worker(spec):
    cfg = spec["config"]
    configure(cfg)
    import faiss
    import numpy as np
    from threadpoolctl import threadpool_info, threadpool_limits
    from .representation_quality import measure_positions, minimum_budget
    from .search import encode_chunks
    from .system_search import measure_insertions

    faiss.omp_set_num_threads(cfg["threads"])
    arrays = load_features(spec["features"], captions_per_image=None)
    gallery_arrays = load_features(spec["gallery_features"], captions_per_image=None,
                                    allow_unlabeled=cfg.get("allow_unlabeled_gallery", False))
    gallery, queries, mapping, query_rows = retrieval_inputs(arrays, gallery_arrays, spec["direction"])
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    rss_inputs = process_memory()["rss_bytes"]
    with threadpool_limits(limits=cfg["threads"]):
        start = time.perf_counter()
        index = build_index(spec, gallery, cfg)
        build_ms = (time.perf_counter() - start) * 1000
        build_details = {}
        if spec['method'] in ('opq','fastscan','opq_fastscan'):
            training = spec[spec['method']+'_training']
            build_details = {'local_build_ms':build_ms,
                             'recorded_fit_ms':1000*training['training_seconds'],
                             'fit_host':training['host'],
                             'fit_threads':training['threads'],
                             'build_environment':'separately recorded fit plus local single-thread initialization',
                             **index.build_stages_ms}
            if spec['method'] in ('fastscan','opq_fastscan'):
                build_details.update({'fastscan_'+k:v for k,v in index.search_settings.items()})
            if spec['method'] == 'opq_fastscan':
                # D-013: report only rotation/encoding/code writes as build;
                # normalization, template I/O and refinement preparation stay explicit.
                build_details['local_initialization_ms'] = build_ms
                build_ms = index.build_stages_ms['gallery_encode_write_ms']
                build_details['local_build_ms'] = build_ms
                build_details['build_environment'] = 'local single-thread rotation/encoding/write; fitting reported separately'
            else:
                build_ms += build_details['recorded_fit_ms']
        rss_ready = process_memory()["rss_bytes"]
        if spec.get("mode") == "memory":
            point = spec["point"]
            for _ in range(5):
                index.query(queries[:point["batch"]], budget=query_budget(point))
            memory = process_memory()
            return {"rss_with_inputs_bytes": rss_inputs, "rss_ready_bytes": rss_ready,
                    "rss_after_query_bytes": memory["rss_bytes"], "process_peak_rss_bytes": memory["peak_rss_bytes"],
                    "build_ms": build_ms, **build_details,
                    "numeric_buffers": index.storage, "torch_loaded": "torch" in sys.modules}
        if spec.get('mode') == 'delete':
            if spec['method'] not in ('hashclip','hashclip_coco'):
                raise ValueError('D-012 deletion is limited to the HashCLIP code index')
            from .cpu_load import snapshot, difference
            before = snapshot()
            deleted = binary_deletions(index.index)
            return {**deleted,'load':difference(before,snapshot()),'host':platform.node(),
                    'threads':cfg['threads'],'cpu_affinity':sorted(os.sched_getaffinity(0)),
                    'faiss':faiss.__version__,'torch_loaded':'torch' in sys.modules}
        with np.load(spec["reference"]) as f:
            reference = f[spec["direction"]]
        if reference.shape != (len(queries), 10) or (reference < 0).any() or (reference >= len(gallery)).any():
            raise ValueError("Reference top-10 does not match the query/gallery")
        if spec.get('mode') == 'select_fastscan':
            if spec['method'] not in ('fastscan','opq_fastscan') or spec['split'] != 'val':
                raise ValueError('FastScan candidate budgets are selected only on validation')
            from .fastscan_quality import select_budgets
            selection = select_budgets(index,queries,reference,cfg['quality_targets'])
            return {**selection,'gallery_size':len(gallery),'queries':len(queries),
                    'build_ms':build_ms,**build_details,'torch_loaded':'torch' in sys.modules}
        if spec.get('mode') == 'select_opq':
            if spec['method'] != 'opq' or spec['split'] != 'val':
                raise ValueError('OPQ candidate budgets are selected only on validation')
            from .representations import pq_prefix_positions
            maximum = min(len(gallery),max(10,int(np.ceil(len(gallery)*.02))))
            while True:
                print('OPQ validation prefix',spec['direction'],spec['seed'],maximum,flush=True,file=sys.stderr)
                positions = pq_prefix_positions(index,queries,reference,batch=32,maximum=maximum)
                if float(np.mean(positions<=maximum)) >= max(cfg['quality_targets']):
                    break
                if maximum == len(gallery):
                    raise ValueError('Full OPQ scan did not recover validation reference members')
                maximum = min(len(gallery),maximum*2)
            selected = {str(t):minimum_budget(positions,t) for t in cfg['quality_targets']}
            np.save(spec['positions_output'],positions)
            return {'selected':selected,'prefix_maximum':maximum,'gallery_size':len(gallery),
                    'queries':len(queries),'build_ms':build_ms,**build_details,
                    'validation_metrics':{t:positions_metrics(positions,m) for t,m in selected.items()},
                    'torch_loaded':'torch' in sys.modules}
        rows, samples, cache_audits = [], [], []
        fastscan_metrics = {}
        if spec['method'] in ('fastscan','opq_fastscan') and not spec.get('quality_rows'):
            from .fastscan_quality import online_quality
            budgets = {query_budget(p) for p in spec['points']}
            fastscan_metrics = online_quality(index,queries,reference,budgets)
        first_codes, first_positions = None, None
        for batch in cfg["batch_sizes"]:
            points = [p for p in spec.get("points", []) if p["batch"] == batch]
            if "points" in spec and not points:
                continue
            budgets = ([query_budget(p) for p in points] if points else list(spec.get("budgets", [])))
            positions, pq_positions, mismatch = None, {}, None
            if spec["method"] in ("hashclip", "hashclip_coco", "lsh") and not spec.get('quality_rows'):
                codes = encode_chunks(index.encoder, queries, batch=batch)
                mismatch = 0.0 if first_codes is None else float(np.mean(np.any(codes != first_codes, axis=1)))
                if first_codes is not None and mismatch == 0:
                    positions = first_positions
                elif spec.get('position_cache'):
                    positions, audit = cached_binary_positions(index, gallery, queries, reference,
                        codes, batch, spec['position_cache'], budgets)
                    cache_audits.append(audit)
                else:
                    positions = measure_positions(index, queries, reference, batch=batch)
                if first_codes is None:
                    first_codes, first_positions = codes, positions
                if "points" not in spec:
                    budgets = sorted({minimum_budget(positions, t) for t in cfg["quality_targets"]})
            elif spec['method'] == 'opq' and not spec.get('quality_rows'):
                from .representations import pq_prefix_positions
                positions = pq_prefix_positions(index,queries,reference,batch=batch,maximum=max(budgets),
                                                audit_budgets=budgets)
            elif spec["method"] == "ivf_pq" and not spec.get("quality_rows"):
                for nprobe in sorted({p["nprobe"] for p in budgets}):
                    print(f"quality ivf_pq {spec['direction']} batch={batch} nprobe={nprobe}", file=sys.stderr, flush=True)
                    counts = sorted({p["M"] for p in budgets if p["nprobe"] == nprobe})
                    pq_positions[nprobe] = ivfpq_positions(index, queries, reference, batch=batch,
                                                           nprobe=nprobe, budgets=counts)
            for budget in budgets:
                print(f"measure {spec['method']} {spec['direction']} batch={batch} budget={budget}", file=sys.stderr, flush=True)
                if cfg.get("record_load"):
                    from .cpu_load import snapshot, difference
                    before = snapshot()
                latency, raw = timing(index, queries, batch, budget, cfg)
                load = difference(before, snapshot()) if cfg.get("record_load") else None
                if spec.get("quality_rows"):
                    params = row_parameters(spec, batch, budget)
                    original = next(r for r in spec["quality_rows"] if all(r.get(k) == params.get(k) for k in POINT_KEYS))
                    metrics = {k: v for k, v in original.items() if k.startswith(("candidate_", "final_", "ordered_"))}
                elif positions is not None:
                    metrics = positions_metrics(positions, budget)
                elif spec["method"] == "ivf_pq":
                    metrics = positions_metrics(pq_positions[budget["nprobe"]], budget["M"])
                elif spec['method'] in ('fastscan','opq_fastscan'):
                    metrics = fastscan_metrics[batch,budget]
                else:
                    metrics = native_quality(index, queries, reference, mapping, spec["direction"], batch, budget,
                                             query_rows, workers=cfg.get("quality_workers", 1))
                params = row_parameters(spec, batch, budget)
                rows.append({"split": spec["split"], **params, "gallery_size": len(gallery),
                             "queries": len(queries), "build_ms": build_ms, **build_details,
                             "batch_code_mismatch_vs_first": mismatch,
                             "quality_source": (spec.get('quality_source', "frozen validation quality; identical index parameters/inputs") if spec.get("quality_rows")
                                else "candidate ranks + shared refinement audit" if positions is not None or pq_positions
                                else "online query path"), **metrics, **latency})
                samples.append({**params, "latency_ms": raw, "load": load})
        # Test-time updates use genuine unseen gallery-side validation vectors.
        # They run after query measurements and do not enter parameter selection
        # or the separate fresh RSS probe.
        inserts = {}
        if spec["split"] == "test":
            with np.load(spec["insertion_features"], allow_pickle=True) as f:
                if set(map(str, f["img_ids"])) & set(map(str, gallery_arrays["img_ids"])):
                    raise ValueError("Insertion images must be absent from the tested gallery")
                modality = "img" if spec["direction"] == "t2i" else "txt"
                insert_vectors = np.asarray(f[modality + "_feat"][:cfg["insertion_count"]], dtype=np.float32)
            inserts = measure_insertions(index, insert_vectors)
        for row in rows:
            row.update({k: v for k, v in inserts.items() if k != "insert_samples_ms"})
        return {"rows": rows, "raw_timings": samples, "insertion": inserts, 'position_cache_audits':cache_audits,
                "environment": {"host": platform.node(), "python": platform.python_version(),
                    "numpy": np.__version__, "faiss": faiss.__version__, "threads": cfg["threads"],
                    "cpu_affinity": sorted(os.sched_getaffinity(0)), "threadpools": threadpool_info(),
                    "quality_workers": cfg.get("quality_workers", 1),
                    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "torch_loaded": "torch" in sys.modules}}


def select_points(rows, targets):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in ("method", "seed", "direction", "batch")), []).append(row)
    result = []
    for key, values in groups.items():
        for target in targets:
            eligible = [r for r in values if r["final_recall_at_10"] >= target]
            base = dict(zip(("method", "seed", "direction", "batch"), key))
            if eligible:
                best = min(eligible, key=lambda r: r["batch_p50_ms"])
                base.update({k: best.get(k) for k in POINT_KEYS})
                base["validation_recall_at_10"] = best["final_recall_at_10"]
            result.append({**base, "target_mean_recall_at_10": target, "target_met_on_validation": bool(eligible)})
    return result


def selected_groups(selected):
    groups = {}
    for point in selected:
        if not point["target_met_on_validation"]:
            continue
        key = tuple(point.get(k) for k in BUILD_KEYS)
        group = groups.setdefault(key, {k: point.get(k) for k in BUILD_KEYS})
        points = group.setdefault("points", [])
        if not any(all(p.get(k) == point.get(k) for k in POINT_KEYS) for p in points):
            points.append({k: point.get(k) for k in POINT_KEYS})
    return list(groups.values())


def report_selected(selected, rows):
    lookup = {tuple(r.get(k) for k in POINT_KEYS): r for r in rows}
    result = []
    for point in selected:
        if not point["target_met_on_validation"]:
            result.append(point)
        else:
            measured = lookup[tuple(point.get(k) for k in POINT_KEYS)]
            result.append({**measured, **point,
                           "target_met_on_test": measured["final_recall_at_10"] >= point["target_mean_recall_at_10"]})
    return result


def execute(spec, log):
    timeout = None
    if 'execution_deadline' in spec:
        timeout = spec['execution_deadline'] - time.time()
        if timeout <= 0:
            raise TimeoutError('Execution deadline reached before worker')
    try:
        result = subprocess.run([sys.executable, "-m", "hashclip.systems_cpu", "--worker"],
                                input=json.dumps(spec), stdout=subprocess.PIPE, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError('Execution deadline reached; incomplete worker stopped') from exc
    if result.returncode:
        raise RuntimeError(f"System worker failed (exit {result.returncode}): {spec['method']} {spec['direction']}")
    payload = json.loads(result.stdout)
    with log.open("a") as f:
        f.write(json.dumps({"spec": spec, "result": payload}) + "\n")
    return payload


def run(args):
    import yaml
    config = yaml.safe_load(args.config.read_text())
    dataset = config.pop("datasets")[args.dataset]
    cfg = {**config, **dataset.get("execution", {}),
           "allow_unlabeled_gallery": dataset.get("allow_unlabeled_gallery", False)}
    configure(cfg)
    import numpy as np
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    if cfg.get("diagnostic_source"):
        while not resolve_path(cfg["wait_training_result"]).exists():
            print("Waiting for GPU training to finish before isolated timing diagnosis", flush=True)
            time.sleep(30)
        result = subprocess.run([sys.executable, "-m", "hashclip.diagnose_cpu", "--source",
                        str(resolve_path(cfg["diagnostic_source"])), "--output", str(args.output / "diagnostic")])
        if result.returncode:
            print(f"Timing diagnosis requires follow-up (exit {result.returncode}); continue the independent large-gallery grid", flush=True)
    paths = {k: resolve_path(dataset[k]) for k in ("queries", "gallery", "references", "train")}
    with np.load(paths["train"], allow_pickle=True) as f:
        train_ids = set(map(str, f["img_ids"]))
        train_sizes = {"t2i": len(f["img_ids"]), "i2t": len(f["txt_to_img"])}
    gallery_sizes = {}
    inputs = [input_identity(paths["train"])]
    for split in ("val", "test"):
        with np.load(paths["queries"] / f"backbone_{split}.npz", allow_pickle=True) as f:
            if train_ids & set(map(str, f["img_ids"])):
                raise ValueError("Training images overlap query val/test")
        inputs.extend(input_identity(paths[k] / f"backbone_{split}.npz") for k in ("queries", "gallery"))
        inputs.append(input_identity(paths["references"] / f"{split}_reference.npz"))
    with np.load(paths["gallery"] / "backbone_val.npz", allow_pickle=True) as f:
        gallery_sizes = {"t2i": len(f["img_ids"]), "i2t": len(f["txt_to_img"])}
    training_runs = yaml.safe_load((ROOT / "configs/icde_reselect.yaml").read_text())["training_runs"]
    checkpoints = {s: runs_root() / training_runs[s] / "checkpoints" / f"hashclip_b128_seed{s}_reselect.pt"
                   for s in cfg["seeds"]}
    from .numpy_adapter import numpy_checkpoint_path
    inputs.extend(input_identity(numpy_checkpoint_path(p)) for p in checkpoints.values())
    manifest = {"dataset": args.dataset, "config": cfg, "dataset_paths": {k: str(v) for k, v in paths.items()},
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "inputs": inputs, "selection": "validation only; fastest eligible complete-query median per target",
                "quality": "all queries; reference top-10; no task Recall",
                "timing_boundary": "continuous float32 query input through final top-10 output",
                "ivf_training": "IVF-Flat trains on its gallery; IVF-PQ on the training split's gallery modality",
                "insertion": f"{cfg['insertion_count']} held-out validation vectors of the gallery modality appended to the test index after quality/timing; capacity cost explicit"}
    (args.output / "config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    def spec_for(group, split):
        spec = {**group, "config": cfg, "split": split, "features": str(paths["queries"] / f"backbone_{split}.npz"),
                "gallery_features": str(paths["gallery"] / f"backbone_{split}.npz"),
                "reference": str(paths["references"] / f"{split}_reference.npz"), "training_features": str(paths["train"]),
                "insertion_features": str(paths["queries"] / "backbone_val.npz")}
        if spec["method"] == "hashclip":
            spec["checkpoint"] = str(checkpoints[spec["seed"]])
        return spec
    val = []
    for direction, size in gallery_sizes.items():
        groups = [{"method": "flat_ip", "seed": -1, "budgets": [None]}]
        groups += [{"method": m, "seed": s} for m in ("hashclip", "lsh") for s in cfg["seeds"]]
        for nl in cfg["ivf_nlist"]:
            if 39 * nl <= size:
                groups.append({"method": "ivf_flat", "seed": cfg["ivf_seed"], "nlist": nl,
                               "budgets": [p for p in cfg["ivf_nprobe"] if p <= nl]})
        groups += [{"method": "hnsw", "seed": cfg["ivf_seed"], "hnsw_m": m, "ef_construction": ec,
                    "budgets": cfg["hnsw_ef_search"]} for m in cfg["hnsw_m"] for ec in cfg["hnsw_ef_construction"]]
        counts = sorted({min(v, size) for v in cfg["ivfpq_candidates"]})
        for nl in cfg["ivfpq_nlist"]:
            if 39 * nl <= min(size, train_sizes[direction]):
                groups.append({"method": "ivf_pq", "seed": cfg["ivf_seed"], "nlist": nl,
                               "pq_m": cfg["pq_m"], "pq_bits": cfg["pq_bits"],
                               "budgets": [{"nprobe": p, "M": m} for p in cfg["ivfpq_nprobe"] if p <= nl for m in counts]})
        for group in groups:
            group["direction"] = direction
            print("val", {k: v for k, v in group.items() if k != "budgets"}, flush=True)
            val.extend(execute(spec_for(group, "val"), args.output / "workers.jsonl")["rows"])
            write_csv(args.output / "val.csv", val)
    selected = select_points(val, cfg["quality_targets"])
    (args.output / "selected.json").write_text(json.dumps(selected, indent=2) + "\n")
    test = []
    for group in selected_groups(selected):
        print("test", {k: v for k, v in group.items() if k != "points"}, flush=True)
        test.extend(execute(spec_for(group, "test"), args.output / "workers.jsonl")["rows"])
        write_csv(args.output / "test.csv", test)
    write_csv(args.output / "selected_test.csv", report_selected(selected, test))
    memory, seen = [], set()
    for point in selected:
        key = tuple(point.get(k) for k in POINT_KEYS)
        if not point["target_met_on_validation"] or key in seen:
            continue
        seen.add(key)
        spec = spec_for(point, "test")
        spec.update(mode="memory", point=point)
        print("memory", {k: point.get(k) for k in POINT_KEYS}, flush=True)
        memory.append({**point, **execute(spec, args.output / "memory_workers.jsonl")})
        (args.output / "memory.json").write_text(json.dumps(memory, indent=2) + "\n")
    (args.output / "complete.json").write_text(json.dumps({"validation": len(val),
        "test": len(test), "memory": len(memory)}, indent=2) + "\n")
    print(f"Completed {args.dataset}: {len(val)} validation, {len(test)} test, {len(memory)} memory records", flush=True)


def main():
    if "--worker" in sys.argv:
        print(json.dumps(worker(json.load(sys.stdin))))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/icde_systems.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

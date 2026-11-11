"""First-round evaluation of fixed checkpoints and CPU retrieval.

Run as ``python -m hashclip.evaluate_cpu --help``. Each method/direction is
evaluated in a fresh subprocess. Memory probes are separate fresh processes,
without reference computation or quality-diagnostic allocations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from .paths import ROOT, data_root, runs_root


def process_memory():
    """RSS and peak for this address space, excluding pre-exec parent peaks.

    Linux getrusage(RUSAGE_SELF).ru_maxrss can retain the launcher's high-water
    mark across exec. VmHWM resets with the address space and avoids counting
    the parent's reference / quality arrays in a fresh memory probe.
    """
    values = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "VmHWM"):
            values[key] = int(value.split()[0]) * 1024
    return {"rss_bytes": values["VmRSS"], "peak_rss_bytes": values["VmHWM"]}


def configure(cfg):
    if cfg.get("allow_unlabeled_gallery") and cfg.get("report_task_recall", True):
        raise ValueError("Unlabeled galleries report neighbor recovery only; disable task recall")
    threads = int(cfg["threads"])
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(threads)
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    if cfg.get("blas_core"):
        if cfg["blas_core"] == "Haswell" and "avx2" not in Path("/proc/cpuinfo").read_text():
            raise ValueError("Haswell BLAS requires AVX2; change blas_core for this host")
        os.environ["OPENBLAS_CORETYPE"] = cfg["blas_core"]
    if cfg.get("cpus"):
        os.sched_setaffinity(0, cfg["cpus"])


def load_features(path, expected_images=None, captions_per_image=5, *, allow_unlabeled=False):
    import numpy as np

    # Legacy caches store string IDs/captions as object arrays. Only use trusted
    # project caches; no checkpoint or cache is downloaded by this evaluator.
    with np.load(path, allow_pickle=True) as f:
        arrays = {k: f[k] for k in ("img_feat", "txt_feat", "img_ids", "txt_to_img")}
    img, txt = arrays["img_feat"], arrays["txt_feat"]
    mapping = arrays["txt_to_img"]
    if img.ndim != 2 or txt.ndim != 2 or img.shape[1] != txt.shape[1]:
        raise ValueError(f"Incompatible feature shapes in {path}")
    if not np.isfinite(img).all() or not np.isfinite(txt).all() or len(img) < 10:
        raise ValueError(f"Invalid or too-small feature cache: {path}")
    if mapping.shape != (len(txt),) or mapping.dtype.kind not in "iu" or (mapping < 0).any() or (mapping >= len(img)).any():
        raise ValueError(f"Invalid text/image mapping: {path}")
    if arrays["img_ids"].shape != (len(img),) or len(set(map(str, arrays["img_ids"]))) != len(img):
        raise ValueError(f"Missing or duplicate image IDs: {path}")
    counts = np.bincount(mapping, minlength=len(img))
    if not allow_unlabeled and (counts < 1).any():
        raise ValueError(f"Every image must have at least one caption: {path}")
    if captions_per_image is not None and not np.all(counts == captions_per_image):
        raise ValueError(f"Expected {captions_per_image} captions per image: {path}")
    if expected_images is not None and len(img) != expected_images:
        raise ValueError(f"{path}: expected {expected_images} images, got {len(img)}")
    return arrays


def make_index(spec, gallery, cfg):
    from .search import (AdapterEncoder, BinaryRerank, FaissFloat,
                         LSHEncoder, NumpyExact, encode_chunks)

    method = spec["method"]
    if method == "numpy_exact":
        return NumpyExact(gallery)
    if method == "flat_ip":
        return FaissFloat(gallery)
    if method == "ivf_flat":
        return FaissFloat(gallery, nlist=spec["nlist"], seed=cfg["ivf_seed"])
    if method == "lsh":
        query_encoder = gallery_encoder = LSHEncoder(gallery.shape[1], cfg["bits"], spec["seed"])
    elif method == "hashclip" and cfg.get("adapter_backend", "torch") == "numpy":
        from .numpy_adapter import load_adapters, numpy_checkpoint_path

        adapters, metadata = load_adapters(numpy_checkpoint_path(spec["checkpoint"]))
        if metadata["bits"] != cfg["bits"] or metadata["embed_dim"] != gallery.shape[1]:
            raise ValueError("Exported checkpoint dimensions differ from evaluation inputs")
        if metadata["seed"] != spec["seed"]:
            raise ValueError("Exported checkpoint seed differs from evaluation request")
        if spec["direction"] == "t2i":
            query_encoder, gallery_encoder = adapters["txt"], adapters["img"]
        else:
            query_encoder, gallery_encoder = adapters["img"], adapters["txt"]
        if "torch" in sys.modules:
            raise RuntimeError("NumPy query process unexpectedly loaded PyTorch")
    elif method == "hashclip":
        if cfg.get("adapter_backend", "torch") != "torch":
            raise ValueError("Unknown adapter backend")
        import torch
        from .model import RetrievalModel

        torch.set_num_threads(cfg["threads"])
        torch.set_num_interop_threads(1)
        model, checkpoint = RetrievalModel.from_checkpoint(spec["checkpoint"])
        if model.bits != cfg["bits"] or model.embed_dim != gallery.shape[1]:
            raise ValueError("Checkpoint dimensions differ from evaluation inputs")
        saved_seed = checkpoint.get("config", {}).get("train", {}).get("seed")
        if saved_seed != spec["seed"]:
            raise ValueError(f"Checkpoint seed {saved_seed} != requested {spec['seed']}")
        if spec["direction"] == "t2i":
            query_encoder, gallery_encoder = AdapterEncoder(model.txt_adapter), AdapterEncoder(model.img_adapter)
        else:
            query_encoder, gallery_encoder = AdapterEncoder(model.img_adapter), AdapterEncoder(model.txt_adapter)
    else:
        raise ValueError(method)
    codes = encode_chunks(gallery_encoder, gallery)
    result = BinaryRerank(gallery, codes, query_encoder,
                        use_heap=cfg.get("binary_use_heap", True),
                        rerank_block=cfg.get("rerank_block", 0),
                        rerank_mode=cfg.get("rerank_mode", "canonical"))
    if cfg.get("retain_gallery_encoder", False):
        result.gallery_encoder = gallery_encoder
        result.storage["gallery_parameters_bytes"] = 0 if gallery_encoder is query_encoder else gallery_encoder.nbytes
    return result


def retrieval_inputs(query_arrays, gallery_arrays, direction):
    """Join image IDs explicitly: gallery additions never become queries."""
    import numpy as np

    gallery_ids = {str(value): i for i, value in enumerate(gallery_arrays["img_ids"])}
    try:
        image_rows = np.array([gallery_ids[str(value)] for value in query_arrays["img_ids"]], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Query image {exc.args[0]} is absent from the gallery") from exc
    if direction == "t2i":
        return (gallery_arrays["img_feat"], query_arrays["txt_feat"],
                image_rows[query_arrays["txt_to_img"]], image_rows)
    if direction == "i2t":
        return (gallery_arrays["txt_feat"], query_arrays["img_feat"],
                gallery_arrays["txt_to_img"], image_rows)
    raise ValueError(direction)


def quality(index, queries, reference, mapping, direction, batch, budget, query_image_rows=None,
            *, report_task_recall=True):
    from .metrics import retrieval_batch_metrics

    sums = {}
    for start in range(0, len(queries), batch):
        q = queries[start:start + batch]
        result = index.query(q, budget=budget)
        values = retrieval_batch_metrics(reference[start:start + batch], result.final, result.candidates,
                                          mapping, direction, start, query_image_rows,
                                          report_task_recall=report_task_recall)
        for key, value in values.items():
            sums[key] = sums.get(key, 0.0) + value * len(q)
    return {key: value / len(queries) for key, value in sums.items()}


def timing(index, queries, batch, budget, cfg):
    import numpy as np

    count = min(len(queries), cfg["timing_queries"])
    count -= count % batch
    if count == 0:
        raise ValueError("Fewer timing queries than one full batch")
    # Uniform deterministic sample across the query split; excluded slicing time
    # is not part of the continuous-query-input to top-K boundary.
    selected = queries[np.linspace(0, len(queries) - 1, count, dtype=int)]
    chunks = [selected[i:i + batch] for i in range(0, count, batch)]
    for i in range(cfg["warmup"]):
        index.query(chunks[i % len(chunks)], budget=budget)
    samples = []
    for _ in range(cfg["repeats"]):
        for q in chunks:
            start = time.perf_counter_ns()
            index.query(q, budget=budget)
            samples.append((time.perf_counter_ns() - start) / 1e6)
    stages = {}
    # Profiling is a separate pass so internal timers do not inflate main times.
    for q in chunks:
        result = index.query(q, budget=budget, profile=True)
        for key, value in result.stages_ms.items():
            stages[key] = stages.get(key, 0.0) + value / len(chunks)
    return {
        "batch_p50_ms": float(np.median(samples)),
        "batch_p95_ms": float(np.percentile(samples, 95)),
        "amortized_p50_ms": float(np.median(samples) / batch),
        "throughput_qps": float(len(samples) * batch * 1000 / sum(samples)),
        "timed_queries_per_repeat": count,
        **{f"profile_{key}": value for key, value in stages.items()},
    }, samples


def worker(spec):
    cfg = spec["config"]
    configure(cfg)  # before loading BLAS / NumPy / PyTorch
    import faiss
    import numpy as np
    from threadpoolctl import threadpool_info, threadpool_limits

    faiss.omp_set_num_threads(cfg["threads"])
    captions_per_image = cfg.get("captions_per_image", 5)
    arrays = load_features(spec["features"], cfg["expected_images"].get(spec["split"]), captions_per_image)
    gallery_arrays = (load_features(spec["gallery_features"], cfg.get("expected_gallery_images", {}).get(spec["split"]), captions_per_image,
                                   allow_unlabeled=cfg.get("allow_unlabeled_gallery", False))
                      if spec.get("gallery_features") else arrays)
    gallery, queries, mapping, query_image_rows = retrieval_inputs(arrays, gallery_arrays, spec["direction"])
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    rss_inputs = process_memory()["rss_bytes"]
    with threadpool_limits(limits=cfg["threads"]):
        start = time.perf_counter()
        index = make_index(spec, gallery, cfg)
        build_ms = (time.perf_counter() - start) * 1000
        rss_ready = process_memory()["rss_bytes"]
        if spec.get("mode") == "memory":
            q = queries[:spec["batch"]]
            for _ in range(5):
                index.query(q, budget=spec["budget"])
            memory = process_memory()
            return {"rss_with_inputs_bytes": rss_inputs, "rss_ready_bytes": rss_ready,
                    "rss_after_query_bytes": memory["rss_bytes"],
                    "process_peak_rss_bytes": max(memory["peak_rss_bytes"], memory["rss_bytes"], rss_inputs, rss_ready),
                    "peak_rss_source": "Linux /proc/self/status VmHWM; current address space",
                    "build_ms": build_ms, "numeric_buffers": index.storage,
                    "torch_loaded": "torch" in sys.modules,
                    "adapter_backend": cfg.get("adapter_backend", "torch") if spec["method"] == "hashclip" else None}
        with np.load(spec["reference"]) as f:
            reference = f[spec["direction"]]
        rows, raw = [], []
        single_quality = None
        for batch in cfg["batch_sizes"]:
            # Same CPU encoding path and query batch size for quality and timing.
            mismatch = None
            if hasattr(index, "encoder"):
                from .search import encode_chunks
                a = encode_chunks(index.encoder, queries, batch=1)
                b = encode_chunks(index.encoder, queries, batch=batch)
                mismatch = float(np.mean(np.any(a != b, axis=1)))
            cached_quality, audited = None, 0
            reused = False
            if cfg.get("cache_quality_scores") and hasattr(index, "encoder"):
                from .quality_cpu import binary_quality_grid
                reused = single_quality is not None and mismatch == 0
                cached_quality, audited = binary_quality_grid(
                    index, queries, reference, spec["scores"], mapping, spec["direction"], batch,
                    spec["budgets"], query_image_rows, cfg.get("quality_audit_queries", 64),
                    reuse=single_quality if reused else None,
                    report_task_recall=cfg.get("report_task_recall", True))
                if batch == 1:
                    single_quality = cached_quality
            for budget in spec["budgets"]:
                latency, samples = timing(index, queries, batch, budget, cfg)
                legacy_latency, legacy_samples = {}, None
                if cfg.get("compare_legacy_binary") and hasattr(index, "encoder"):
                    settings = index.index.use_heap, index.rerank_block, index.rerank_mode
                    try:
                        index.index.use_heap, index.rerank_block, index.rerank_mode = True, 0, "canonical"
                        legacy_latency, legacy_samples = timing(index, queries, batch, budget, cfg)
                    finally:
                        index.index.use_heap, index.rerank_block, index.rerank_mode = settings
                metrics = (cached_quality[budget] if cached_quality is not None else
                           quality(index, queries, reference, mapping, spec["direction"], batch, budget, query_image_rows,
                                   report_task_recall=cfg.get("report_task_recall", True)))
                row = {"split": spec["split"], "direction": spec["direction"],
                       "method": spec["method"], "seed": spec["seed"], "batch": batch,
                       "gallery_size": len(gallery), "queries": len(queries),
                       "bits": cfg["bits"] if spec["method"] in ("lsh", "hashclip") else 0,
                       "M": budget if spec["method"] in ("lsh", "hashclip") else None,
                       "effective_M": min(budget, len(gallery)) if spec["method"] in ("lsh", "hashclip") else None,
                       "nlist": spec.get("nlist"),
                       "nprobe": budget if spec["method"] == "ivf_flat" else None,
                       "batch_code_mismatch_vs_single": mismatch,
                       "quality_source": "cached canonical scores + exact Hamming order" if cached_quality is not None else "online query path",
                       "quality_audit_queries": audited, "quality_reused_from_batch1": reused,
                       "build_ms": build_ms, **metrics, **latency,
                       **{f"legacy_{key}": value for key, value in legacy_latency.items()}}
                if hasattr(index, "encoder"):
                    row.update(binary_use_heap=index.index.use_heap, rerank_block=index.rerank_block,
                               rerank_mode=index.rerank_mode)
                if spec["method"] == "hashclip":
                    row["adapter_backend"] = cfg.get("adapter_backend", "torch")
                rows.append(row)
                raw.append({"batch": batch, "budget": budget, "latency_ms": samples,
                            "legacy_latency_ms": legacy_samples})
        environment = {"host": platform.node(), "python": platform.python_version(),
                       "numpy": np.__version__, "faiss": faiss.__version__,
                       "faiss_compile_options": faiss.get_compile_options(),
                       "cpu_affinity": sorted(os.sched_getaffinity(0)),
                       "threads": cfg["threads"], "threadpools": threadpool_info()}
        environment["torch_loaded"] = "torch" in sys.modules
        if spec["method"] == "hashclip" and cfg.get("adapter_backend", "torch") == "torch":
            import torch
            environment["torch"] = torch.__version__
        return {"rows": rows, "raw_timings": raw, "environment": environment}


def select_validation(rows, targets):
    """Choose retrieval budgets only; checkpoint choice is never changed."""
    groups = {}
    for row in rows:
        if row["method"] == "numpy_exact":
            continue
        key = tuple(row[k] for k in ("method", "seed", "direction", "batch"))
        groups.setdefault(key, []).append(row)
    selected = []
    for key, group in groups.items():
        for target in targets:
            eligible = [r for r in group if r["final_recall_at_10"] >= target]
            best = min(eligible, key=lambda r: r["batch_p50_ms"]) if eligible else None
            selected.append({"method": key[0], "seed": key[1], "direction": key[2],
                             "batch": key[3], "target_mean_recall_at_10": target,
                             "target_met_on_validation": best is not None,
                             "M": None if best is None else best["M"],
                             "nlist": None if best is None else best["nlist"],
                             "nprobe": None if best is None else best["nprobe"]})
    return selected


def selected_test_rows(selected, test_rows):
    keys = ("method", "seed", "direction", "batch", "M", "nlist", "nprobe")
    lookup = {tuple(r[k] for k in keys): r for r in test_rows}
    out = []
    for point in selected:
        if not point["target_met_on_validation"]:
            out.append({**point, "target_met_on_test": None})
            continue
        measured = lookup[tuple(point[k] for k in keys)]
        out.append({**measured, **point,
                    "target_met_on_test": measured["final_recall_at_10"] >= point["target_mean_recall_at_10"]})
    return out


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def input_identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def execute(spec, log):
    result = subprocess.run([sys.executable, "-m", "hashclip.evaluate_cpu", "--worker"],
                            input=json.dumps(spec), capture_output=True, text=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.returncode:
        raise RuntimeError(f"{spec['method']} / {spec['direction']} worker failed; see stderr")
    payload = json.loads(result.stdout)
    with log.open("a") as f:
        f.write(json.dumps({"spec": spec, "result": payload}) + "\n")
    return payload


def candidate_budgets(cfg, gallery_size):
    """Freeze absolute budgets using validation size, rounding fractions to 100."""
    budgets = list(cfg["candidate_budgets"])
    for fraction in cfg.get("candidate_budget_fractions", []):
        if not 0 < fraction <= 1:
            raise ValueError("Candidate fractions must be in (0, 1]")
        budgets.append(gallery_size if fraction == 1 else max(10, round(gallery_size * fraction / 100) * 100))
    return sorted({min(int(m), gallery_size) for m in budgets}) if cfg.get("candidate_budget_fractions") else budgets


def run(args):
    import yaml

    cfg = yaml.safe_load(args.config.read_text())
    configure(cfg)
    import numpy as np
    from .search import NumpyExact

    features = {s: args.features / f"backbone_{s}.npz" for s in ("val", "test")}
    gallery_dir = getattr(args, "gallery_features", None)
    gallery_features = ({s: gallery_dir / f"backbone_{s}.npz" for s in ("val", "test")}
                        if gallery_dir else {})
    checkpoints = {seed: args.checkpoints / f"hashclip_b{cfg['bits']}_seed{seed}_main.pt"
                   for seed in cfg["seeds"]}
    input_paths = (*features.values(), *gallery_features.values(), *checkpoints.values())
    missing = [str(p) for p in input_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required original assets are missing:\n" + "\n".join(missing))
    captions_per_image = cfg.get("captions_per_image", 5)
    arrays = {s: load_features(p, cfg["expected_images"].get(s), captions_per_image) for s, p in features.items()}
    overlap = set(map(str, arrays["val"]["img_ids"])) & set(map(str, arrays["test"]["img_ids"]))
    if overlap:
        raise ValueError("Validation and test image IDs overlap")
    galleries = ({s: load_features(p, cfg.get("expected_gallery_images", {}).get(s), captions_per_image,
                                  allow_unlabeled=cfg.get("allow_unlabeled_gallery", False))
                  for s, p in gallery_features.items()} if gallery_features else arrays.copy())
    for split, other in (("val", "test"), ("test", "val")):
        if set(map(str, galleries[split]["img_ids"])) & set(map(str, arrays[other]["img_ids"])):
            raise ValueError(f"{split} gallery contains {other} query images")
        retrieval_inputs(arrays[split], galleries[split], "t2i")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new output directory; existing results are not overwritten")
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = {"config": cfg, "inputs": [input_identity(p) for p in input_paths],
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "query_caches": {s: str(p.resolve()) for s, p in features.items()},
                "gallery_caches": {s: str(p.resolve()) for s, p in (gallery_features or features).items()},
                "timing_boundary": "continuous float32 query batch to final top-10; preprocessing/build excluded",
                "reference": "canonical float32 row scorer; score descending then gallery row ID ascending",
                "quality_units": "fractions in [0,1]; task recall separate from reference-neighbor recovery"}
    log = args.output / "workers.jsonl"
    all_rows = {}
    # Freeze the grid from validation gallery sizes before reading test outcomes.
    grid, budgets = {}, {}
    for direction in ("t2i", "i2t"):
        n = len(galleries["val"]["img_feat" if direction == "t2i" else "txt_feat"])
        grid[direction] = [v for v in cfg["ivf_nlist"] if 39 * v <= n]
        budgets[direction] = candidate_budgets(cfg, n)
        if not grid[direction]:
            raise ValueError("No IVF nlist has enough validation gallery vectors (39 per centroid)")
    snapshot["search_grid"] = {"ivf_nlist": grid, "candidate_budgets": budgets}
    (args.output / "config.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    for split in ("val", "test"):
        a = arrays.pop(split)
        g = galleries.pop(split)
        ref_file = args.output / f"{split}_reference.npz"
        score_files = {d: args.output / f"{split}_scores_{d}.npy" for d in ("t2i", "i2t")}
        if cfg.get("cache_quality_scores"):
            from .quality_cpu import reference_with_scores
            refs = {"t2i": reference_with_scores(g["img_feat"], a["txt_feat"], score_files["t2i"]),
                    "i2t": reference_with_scores(g["txt_feat"], a["img_feat"], score_files["i2t"])}
        else:
            refs = {"t2i": NumpyExact(g["img_feat"]).query(a["txt_feat"]).final,
                    "i2t": NumpyExact(g["txt_feat"]).query(a["img_feat"]).final}
        np.savez(ref_file, **refs)
        del refs, a, g
        rows = []
        for direction in ("t2i", "i2t"):
            groups = [{"method": m, "seed": -1, "budgets": [None]} for m in ("numpy_exact", "flat_ip")]
            groups += [{"method": "ivf_flat", "seed": cfg["ivf_seed"], "nlist": nl,
                        "budgets": [p for p in cfg["ivf_nprobe"] if p <= nl]} for nl in grid[direction]]
            if split == "test" and cfg.get("test_selected_ivf", False):
                # The complete validation grid decides the operating points.
                # Measure their union for both batch sizes; binary test curves
                # below still cover every frozen candidate budget.
                chosen_ivf = {(p["nlist"], p["nprobe"]) for p in selected
                              if p["method"] == "ivf_flat" and p["direction"] == direction
                              and p["target_met_on_validation"]}
                for group in groups:
                    if group["method"] == "ivf_flat":
                        group["budgets"] = [p for p in group["budgets"] if (group["nlist"], p) in chosen_ivf]
                groups = [group for group in groups if group["budgets"]]
            for method in ("lsh", "hashclip"):
                for seed in cfg["seeds"]:
                    groups.append({"method": method, "seed": seed,
                                   "checkpoint": str(checkpoints[seed].resolve()),
                                   "budgets": budgets[direction]})
            for group in groups:
                spec = {**group, "split": split, "direction": direction,
                        "features": str(features[split].resolve()), "reference": str(ref_file.resolve()), "config": cfg}
                if gallery_features:
                    spec["gallery_features"] = str(gallery_features[split].resolve())
                if cfg.get("cache_quality_scores"):
                    spec["scores"] = str(score_files[direction].resolve())
                print(f"{split} {direction} {group['method']} seed={group['seed']} nlist={group.get('nlist')}", flush=True)
                rows.extend(execute(spec, log)["rows"])
                write_csv(args.output / f"{split}.csv", rows)
        all_rows[split] = rows
        if split == "val":
            selected = select_validation(rows, cfg["quality_targets"])
            (args.output / "selected.json").write_text(json.dumps(selected, indent=2) + "\n")
    # Memory probes only for distinct validation-selected operating points.
    write_csv(args.output / "selected_test.csv", selected_test_rows(selected, all_rows["test"]))
    memory = []
    seen = set()
    for point in selected:
        if not point["target_met_on_validation"]:
            continue
        key = tuple(point[k] for k in ("method", "seed", "direction", "batch", "M", "nlist", "nprobe"))
        if key in seen:
            continue
        seen.add(key)
        seed = point["seed"]
        spec = {**point, "config": cfg, "mode": "memory", "split": "test",
                "features": str(features["test"].resolve()), "budget": point["M"] or point["nprobe"]}
        if gallery_features:
            spec["gallery_features"] = str(gallery_features["test"].resolve())
        if point["method"] == "hashclip":
            spec["checkpoint"] = str(checkpoints[seed].resolve())
        memory.append({**point, **execute(spec, log)})
    (args.output / "memory.json").write_text(json.dumps(memory, indent=2) + "\n")
    from .plot_cpu import plot_results
    plot_results(all_rows["test"], args.output)
    print(f"Results saved to {args.output}")


def main():
    if "--worker" in sys.argv:
        print(json.dumps(worker(json.load(sys.stdin))))
        return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/icde_eval.yaml")
    p.add_argument("--features", type=Path, default=data_root() / "features")
    p.add_argument("--gallery-features", type=Path, help="Separate gallery caches; queries always come from --features")
    p.add_argument("--checkpoints", type=Path, default=runs_root() / "checkpoints")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    try:
        run(args)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        p.exit(2, str(exc) + "\n")


if __name__ == "__main__":
    main()

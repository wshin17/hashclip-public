"""Fit full-scan baselines only on the training gallery modality."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from .evaluate_cpu import configure, input_identity
from .paths import ROOT


def run(args):
    configure({"threads": args.threads, "blas_core": "Haswell"})
    import faiss
    import numpy as np
    from threadpoolctl import threadpool_limits
    from .representations import train_representation

    faiss.omp_set_num_threads(args.threads)
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError("Use a new model output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.features, allow_pickle=True) as archive:
        arrays = {key: archive[key] for key in ("img_feat", "txt_feat", "img_ids")}
    if len(set(map(str, arrays["img_ids"]))) != len(arrays["img_feat"]):
        raise ValueError("Training image IDs are not unique")
    manifest = {"training_features": input_identity(args.features), "seeds": args.seeds,
                "bits": 128, "threads": args.threads, "faiss": faiss.__version__, "models": [],
                "host":platform.node(), "cpu_affinity":sorted(os.sched_getaffinity(0)),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "training_policy": "gallery-side training vectors, unit normalized; standard Faiss iteration/sample caps",
                "defaults": {"pq_iterations": 25, "pq_max_points_per_centroid": 256,
                             "opq_iterations": 50, "opq_pq_first_iterations": 40,
                             "opq_pq_iterations": 4, "opq_max_train_points": 65536,
                             "itq_iterations": 50, "itq_max_train_per_dimension": 10}}
    if args.resume and (args.output/'manifest.json').exists():
        previous = json.loads((args.output/'manifest.json').read_text())
        for key in ('training_features','seeds','bits','threads','faiss','host','cpu_affinity','training_policy','defaults'):
            if previous.get(key) != manifest.get(key):
                raise ValueError(f'Cannot resume changed model training: {key}')
        manifest = previous
        for row in manifest['models']:
            if input_identity(args.output/Path(row['artifact']['path']).name) != row['artifact']:
                raise ValueError('Previously fitted model changed')
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with threadpool_limits(limits=args.threads):
        for method in args.methods:
            for modality in ("img", "txt"):
                train = arrays[modality + "_feat"]
                for seed in args.seeds:
                    if any((r['method'],r['modality'],r['seed'])==(method,modality,seed) for r in manifest['models']):
                        continue
                    print(f"Training {method} {modality} seed={seed}, n={len(train)}", flush=True)
                    start = time.perf_counter()
                    model = train_representation(method, train, seed)
                    elapsed = time.perf_counter() - start
                    path = args.output / f"{method}_{modality}_seed{seed}.faiss"
                    if method == "itq":
                        faiss.write_VectorTransform(model, str(path))
                    else:
                        faiss.write_index(model, str(path))
                    details = ({'search_settings':{k:int(getattr(model,k)) for k in
                                ('M','M2','nbits','code_size','bbs','implem','qbs')},
                                'pq_iterations':int(model.pq.cp.niter),
                                'pq_max_points_per_centroid':int(model.pq.cp.max_points_per_centroid)}
                               if method=='fastscan' else {})
                    manifest["models"].append({"method": method, "modality": modality, "seed": seed,
                                               "training_rows": len(train), "training_seconds": elapsed,
                                               "artifact": input_identity(path),**details})
                    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                    print(f"Saved {path.name}, training {elapsed:.3f}s", flush=True)
                    del model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 42])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--methods", nargs='+', choices=['itq','pq','opq','fastscan'], default=['itq','pq','opq'])
    parser.add_argument('--resume',action='store_true')
    run(parser.parse_args())


if __name__ == "__main__":
    main()

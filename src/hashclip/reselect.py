"""Repeat original adapter training and compare two selectors on one trajectory."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import time

from .paths import ROOT, data_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Original checkpoint supplies the original configuration")
    parser.add_argument("--features", type=Path, default=data_root() / "features")
    parser.add_argument("--validation-features", type=Path, help="Separate validation cache, preserving the original training loop")
    parser.add_argument("--dataset-name", default="Flickr30k")
    parser.add_argument("--skip-original-evaluation", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["OPENBLAS_CORETYPE"] = "Haswell"
    import torch
    from .data import CachedFeatures
    from .evaluate_cpu import input_identity
    from .model import RetrievalModel
    from .selection import CandidateValidation
    from .train import train_one

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU training requested but CUDA is unavailable")
    old_model, old = RetrievalModel.from_checkpoint(args.checkpoint)
    cfg = deepcopy(old["config"])
    feature_dir = args.features
    if args.validation_features:
        feature_dir = args.output / "features"
        feature_dir.mkdir()
        (feature_dir / "backbone_train.npz").symlink_to((args.features / "backbone_train.npz").resolve())
        (feature_dir / "backbone_val.npz").symlink_to(args.validation_features.resolve())
    import numpy as np
    with np.load(feature_dir / "backbone_train.npz", allow_pickle=True) as f:
        train_ids = set(map(str, f["img_ids"]))
    with np.load(feature_dir / "backbone_val.npz", allow_pickle=True) as f:
        val_ids = set(map(str, f["img_ids"]))
    if train_ids & val_ids:
        raise ValueError("Adapter training and validation images overlap")
    cfg["paths"]["features"] = str(feature_dir.resolve())
    cfg["dataset"] = {"source": args.dataset_name, "expected_counts": {"train": len(train_ids), "val": len(val_ids)}}
    cfg["paths"]["checkpoints"] = str((args.output / "checkpoints").resolve())
    cfg["paths"]["logs"] = str((args.output / "logs").resolve())
    cfg["train"].update(checkpoint_selection="candidate_retention",
                         candidate_fractions=[.01, .02, .05, .1], selection_batch_size=32)
    record = {"config": cfg, "original_checkpoint": input_identity(args.checkpoint),
              "original_saved_epoch": old["epoch"], "torch": torch.__version__,
              "device": torch.cuda.get_device_name(),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "training": "fresh original architecture/loss/optimizer; paired selectors on same trajectory",
              "selection": f"{args.dataset_name} validation only; no test data",
              "cpu_affinity": sorted(os.sched_getaffinity(0)), "cpu_threads": torch.get_num_threads(),
              "inputs": [input_identity(feature_dir / f"backbone_{s}.npz") for s in ("train", "val")]}
    (args.output / "config.json").write_text(json.dumps(record, indent=2) + "\n")
    if not args.skip_original_evaluation:
        validator = CandidateValidation(old_model, CachedFeatures.load(feature_dir / "backbone_val.npz"))
        old_metrics = validator(old_model)
        (args.output / "old_validation.json").write_text(json.dumps(old_metrics, indent=2) + "\n")
        del validator
    del old_model
    start = time.perf_counter()
    result = train_one(cfg, bits=old["bits"], seed=cfg["train"]["seed"], tag="reselect", device="cuda")
    elapsed = time.perf_counter() - start
    (args.output / "result.json").write_text(json.dumps({"checkpoint": str(result.ckpt_path),
        "paired_r1_checkpoint": str(result.ckpt_path.with_stem(result.ckpt_path.stem + "_r1")),
        "best_validation": result.best_val, "history": result.history,
        "train_and_validation_seconds": elapsed,
        "training_epochs_seconds": sum(h["train"]["time_sec"] for h in result.history)}, indent=2) + "\n")
    from .numpy_adapter import export_checkpoint
    for checkpoint in (result.ckpt_path, result.ckpt_path.with_stem(result.ckpt_path.stem + "_r1")):
        export_checkpoint(checkpoint)
    print(f"Completed seed={cfg['train']['seed']} with both selectors saved", flush=True)


if __name__ == "__main__":
    main()

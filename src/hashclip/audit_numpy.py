"""Export immutable adapter weights and compare NumPy and torch on real caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from .evaluate_cpu import configure, input_identity, write_csv
from .paths import ROOT, data_root, runs_root


def run(output, *, training_path=None, include_original=True, inputs=None, keep_affinity=False):
    configure({"threads": 1, "blas_core": "Haswell", "cpus": None if keep_affinity else [0, 2, 4, 6, 8, 10, 12, 14]})
    import numpy as np
    import torch
    import yaml
    from .numpy_adapter import export_checkpoint, load_adapters

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    original = [runs_root() / f"wsdm-original/checkpoints/hashclip_b128_seed{s}_main.pt" for s in (0, 1, 42)]
    training = yaml.safe_load((training_path or ROOT / "configs/icde_reselect.yaml").read_text())
    checkpoints = list(original) if include_original else []
    for run in training["training_runs"].values():
        result = json.loads((runs_root() / run / "result.json").read_text())
        checkpoints.extend(Path(result[k]) for k in ("checkpoint", "paired_r1_checkpoint"))
    if inputs is None:
        inputs = [data_root() / "features" / f"backbone_{s}.npz" for s in ("val", "test", "train")]
        inputs += [data_root() / "features/coco" / f"backbone_{s}.npz" for s in ("val", "test")]
    models = []
    for checkpoint in checkpoints:
        model, _, export = export_checkpoint(checkpoint)
        adapters, metadata = load_adapters(export)
        models.append((checkpoint, model, adapters, metadata))
    manifest = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "inputs": [input_identity(p) for p in inputs],
                "checkpoints": [metadata for _, _, _, metadata in models],
                "scope": "all cached rows; gallery batch 256, validation/test query batches 1 and 32",
                "torch": torch.__version__, "numpy": np.__version__, "blas_core": "Haswell", "threads": 1}
    (output / "config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    for features in inputs:
        with np.load(features, allow_pickle=True) as archive:
            for modality in ("img", "txt"):
                x = np.ascontiguousarray(archive[modality + "_feat"], dtype=np.float32)
                batches = (256,) if features.stem == "backbone_train" else (1, 32, 256)
                for checkpoint, model, adapters, metadata in models:
                    for batch in batches:
                        max_error, bit_errors, row_errors = 0., 0, 0
                        for start in range(0, len(x), batch):
                            current = x[start:start + batch]
                            with torch.inference_mode():
                                expected = getattr(model, modality + "_adapter")(torch.from_numpy(current)).numpy()
                            actual = adapters[modality].forward(current)
                            difference = (actual > 0) != (expected > 0)
                            max_error = max(max_error, float(np.abs(actual - expected).max()))
                            bit_errors += int(difference.sum())
                            row_errors += int(difference.any(axis=1).sum())
                        row = {"features": str(features.resolve()), "checkpoint": str(checkpoint.resolve()),
                               "seed": metadata["seed"], "modality": modality, "batch": batch,
                               "rows": len(x), "bits": metadata["bits"], "max_absolute_output_error": max_error,
                               "different_bits": bit_errors, "different_codes": row_errors}
                        rows.append(row)
                        write_csv(output / "comparison.csv", rows)
                        print(json.dumps(row), flush=True)
                        if max_error > 1e-5:
                            raise ValueError("NumPy output deviates beyond float32 numerical tolerance")
    (output / "complete.json").write_text(json.dumps({"records": len(rows),
        "different_bits": sum(r["different_bits"] for r in rows),
        "compared_bits": sum(r["rows"] * r["bits"] for r in rows),
        "max_absolute_output_error": max(r["max_absolute_output_error"] for r in rows)}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--no-original", action="store_true")
    parser.add_argument("--features", type=Path, nargs="+")
    parser.add_argument("--keep-affinity", action="store_true")
    a = parser.parse_args()
    run(a.output, training_path=a.training, include_original=not a.no_original,
        inputs=a.features, keep_affinity=a.keep_affinity)


if __name__ == "__main__":
    main()

"""Validate retrieval budgets for new weights, then test them and paired controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from .evaluate_cpu import (candidate_budgets, execute, input_identity, select_validation,
                           selected_test_rows, write_csv)
from .paths import ROOT, runs_root
from .profile_cpu import source_context, source_spec
from .refine_ivf import read_rows, selected_groups


def run(args):
    import yaml

    meta, inputs, old_selected = source_context(args.source)
    implementation = yaml.safe_load(args.implementation.read_text())
    training = yaml.safe_load(args.training.read_text())
    if training["grid"] != "old_99_plus_one":
        raise ValueError("Unknown validation grid")
    cfg = {**meta["config"], **implementation, "compare_legacy_binary": False}
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints, paired = {}, {}
    for seed, run in training["training_runs"].items():
        result = json.loads((runs_root() / run / "result.json").read_text())
        checkpoints[int(seed)] = result["checkpoint"]
        paired[int(seed)] = result["paired_r1_checkpoint"]
    references = {s: (Path(meta["references"][s]["path"]) if "references" in meta
                      else args.source / f"{s}_reference.npz") for s in ("val", "test")}
    old_rows = read_rows(args.source / "val.csv")
    grids = {}
    for seed in cfg["seeds"]:
        for direction in ("t2i", "i2t"):
            size = next(r["gallery_size"] for r in old_rows if r["direction"] == direction)
            old_max = max(p["M"] for p in old_selected if p["method"] == "hashclip"
                          and p["seed"] == seed and p["direction"] == direction)
            grid = candidate_budgets(cfg, size)
            upper = next((v for v in grid if v > old_max), grid[-1])
            grids[f"{seed}/{direction}"] = [v for v in grid if v <= upper]
    manifest = {"config": cfg, "training": training, "validation_grids": grids,
                "inputs": meta["inputs"], "checkpoints": {s: input_identity(Path(p)) for s, p in checkpoints.items()},
                "paired_checkpoints": {s: input_identity(Path(p)) for s, p in paired.items()},
                "query_caches": inputs["query_caches"], "gallery_caches": inputs["gallery_caches"],
                "references": {s: input_identity(p) for s, p in references.items()},
                "source": input_identity(args.source / "config.json"),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "timing_boundary": meta["timing_boundary"],
                "paired_control": "same new trajectory, semantic-R1 checkpoint, evaluated at the candidate-selected weight's fixed budgets"}
    if cfg.get("adapter_backend") == "numpy":
        from .numpy_adapter import numpy_checkpoint_path
        manifest["numpy_exports"] = [input_identity(numpy_checkpoint_path(p))
                                     for p in (*checkpoints.values(), *paired.values())]
    (args.output / "config.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def measure(group, split, control=False):
        spec = source_spec(args.source, group, split)
        spec.update(config=cfg, checkpoint=(paired if control else checkpoints)[group["seed"]],
                    reference=str(references[split].resolve()),
                    scores=str((references[split].parent / f"{split}_scores_{group['direction']}.npy").resolve()))
        print(f"{split} {'paired_r1' if control else 'candidate'} {group['direction']} seed={group['seed']} M={group['budgets']}", flush=True)
        return execute(spec, args.output / "workers.jsonl")["rows"]

    validation = []
    for seed in cfg["seeds"]:
        for direction in ("t2i", "i2t"):
            group = {"method": "hashclip", "seed": seed, "direction": direction,
                     "budgets": grids[f"{seed}/{direction}"]}
            validation.extend(measure(group, "val"))
            write_csv(args.output / "val.csv", validation)
    selected = select_validation(validation, cfg["quality_targets"])
    # Both implementation and checkpoint choice are fixed before any test call.
    (args.output / "selected.json").write_text(json.dumps(selected, indent=2) + "\n")
    test, controls = [], []
    for group in selected_groups(selected):
        test.extend(measure(group, "test"))
        write_csv(args.output / "test.csv", test)
        controls.extend(measure(group, "test", control=True))
        write_csv(args.output / "paired_r1_test.csv", controls)
    write_csv(args.output / "selected_test.csv", selected_test_rows(selected, test))
    paired_rows = selected_test_rows(selected, controls)
    for row in paired_rows:
        row["target_met_on_validation"] = None
        row["budget_source"] = "candidate-selected checkpoint validation; paired control was not tuned"
    write_csv(args.output / "paired_r1_selected_test.csv", paired_rows)
    print(f"Completed {len(validation)} validation, {len(test)} test, {len(controls)} fixed-budget paired controls", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, default=ROOT / "configs/icde_optimized.yaml")
    parser.add_argument("--training", type=Path, default=ROOT / "configs/icde_reselect.yaml")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

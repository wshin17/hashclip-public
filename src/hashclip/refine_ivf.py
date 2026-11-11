"""Refine IVF on validation, then remeasure fixed test operating points.

Reuse the original inputs and exact references after checking their identities.
Binary-method budgets retain their original validation selection. This driver
uses the unchanged CPU evaluation worker for all measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess

from .evaluate_cpu import (configure, execute, input_identity, select_validation,
                           selected_test_rows, write_csv)
from .paths import ROOT


def read_rows(path):
    def parse(value):
        if value == "":
            return None
        if value in ("True", "False"):
            return value == "True"
        for cast in (int, float):
            try:
                return cast(value)
            except ValueError:
                pass
        return value

    with path.open() as f:
        return [{key: parse(value) for key, value in row.items()}
                for row in csv.DictReader(f)]


def selected_groups(points):
    """Build once for the union of budgets selected for the two batch sizes."""
    groups = {}
    for point in points:
        if not point["target_met_on_validation"]:
            continue
        key = tuple(point[k] for k in ("method", "seed", "direction", "nlist"))
        group = groups.setdefault(key, {k: point[k] for k in
                                       ("method", "seed", "direction", "nlist")})
        budget = point["M"] if point["method"] in ("hashclip", "lsh") else point["nprobe"]
        group.setdefault("budgets", set()).add(budget)
    return [{**g, "budgets": sorted(g["budgets"], key=lambda x: -1 if x is None else x)}
            for g in groups.values()]


def run(args):
    import yaml

    baseline = json.loads((args.baseline / "config.json").read_text())
    override = yaml.safe_load(args.config.read_text())
    if set(override) != {"ivf_nlist", "ivf_nprobe"}:
        raise ValueError("Only the IVF search grid may change in this refinement")
    cfg = {**baseline["config"], **override}
    configure(cfg)
    for item in baseline["inputs"]:
        if input_identity(Path(item["path"])) != item:
            raise ValueError(f"Original input changed: {item['path']}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    old_rows = read_rows(args.baseline / "val.csv")
    old_selected = json.loads((args.baseline / "selected.json").read_text())
    grid = {}
    for direction in ("t2i", "i2t"):
        sizes = {r["gallery_size"] for r in old_rows if r["direction"] == direction}
        if len(sizes) != 1:
            raise ValueError("Baseline must contain one gallery size per direction")
        n = sizes.pop()
        grid[direction] = [v for v in cfg["ivf_nlist"] if 0 < 39 * v <= n]
        if not grid[direction]:
            raise ValueError("No IVF configuration has 39 validation vectors per centroid")
    references = {s: args.baseline / f"{s}_reference.npz" for s in ("val", "test")}
    metadata = {"config": cfg, "inputs": baseline["inputs"], "search_grid": grid,
                "baseline": input_identity(args.baseline / "config.json"),
                "binary_validation_source": input_identity(args.baseline / "val.csv"),
                "references": {s: input_identity(p) for s, p in references.items()},
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "selection": "new validation IVF/FlatIP; original validation binary budgets; fixed test points",
                "timing_boundary": baseline["timing_boundary"]}
    (args.output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    checkpoints = {}
    for seed in cfg["seeds"]:
        name = f"hashclip_b{cfg['bits']}_seed{seed}_main.pt"
        checkpoints[seed] = next(x["path"] for x in baseline["inputs"] if Path(x["path"]).name == name)
    log = args.output / "workers.jsonl"

    def measure(group, split, **extra):
        spec = {**group, "config": cfg, "split": split,
                "features": baseline["query_caches"][split],
                "gallery_features": baseline["gallery_caches"][split],
                "reference": str(references[split].resolve()), **extra}
        if group["method"] == "hashclip":
            spec["checkpoint"] = checkpoints[group["seed"]]
        if cfg.get("cache_quality_scores") and group["method"] in ("hashclip", "lsh"):
            # The unchanged worker audits cached quality against actual online
            # candidates/ranking; these arrays never enter timing or memory.
            spec["scores"] = str((args.baseline / f"{split}_scores_{group['direction']}.npy").resolve())
        print(f"{split} {group['method']} {group['direction']} seed={group['seed']} "
              f"nlist={group.get('nlist')} mode={extra.get('mode', 'quality/timing')}", flush=True)
        return execute(spec, log)

    validation = [r for r in old_rows if r["method"] in ("hashclip", "lsh")]
    for direction in ("t2i", "i2t"):
        groups = [{"method": "flat_ip", "seed": -1, "direction": direction, "budgets": [None]}]
        groups += [{"method": "ivf_flat", "seed": cfg["ivf_seed"], "direction": direction,
                    "nlist": nl, "budgets": sorted({p for p in cfg["ivf_nprobe"] if 0 < p <= nl} | {nl})}
                   for nl in grid[direction]]
        for group in groups:
            validation.extend(measure(group, "val")["rows"])
            write_csv(args.output / "val.csv", validation)
    selected = select_validation(validation, cfg["quality_targets"])
    binary = lambda points: [p for p in points if p["method"] in ("hashclip", "lsh")]
    if binary(selected) != binary(old_selected):
        raise ValueError("Binary validation selection changed unexpectedly")
    # Written before any new test measurements.
    (args.output / "selected.json").write_text(json.dumps(selected, indent=2) + "\n")
    test = []
    for group in selected_groups(selected):
        test.extend(measure(group, "test")["rows"])
        write_csv(args.output / "test.csv", test)
    write_csv(args.output / "selected_test.csv", selected_test_rows(selected, test))
    memory, seen = [], set()
    for point in selected:
        if not point["target_met_on_validation"]:
            continue
        key = tuple(point[k] for k in ("method", "seed", "direction", "batch", "M", "nlist", "nprobe"))
        if key in seen:
            continue
        seen.add(key)
        result = measure(point, "test", mode="memory", budget=point["M"] or point["nprobe"])
        memory.append({**point, **result})
        (args.output / "memory.json").write_text(json.dumps(memory, indent=2) + "\n")
    print(f"Results saved to {args.output}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True, help="Completed original run, including exact references")
    p.add_argument("--config", type=Path, default=ROOT / "configs/icde_ivf_refine.yaml")
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()

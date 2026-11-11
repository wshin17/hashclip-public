"""Remeasure fixed operating points after validation-only implementation tuning.

This never reselects M/nlist/nprobe on test. It uses the previous completed run's
input identities, references, score caches, and validation-selected parameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from .evaluate_cpu import execute, input_identity, selected_test_rows, write_csv
from .paths import ROOT
from .profile_cpu import source_context, source_spec
from .refine_ivf import read_rows, selected_groups


def run(args):
    import yaml

    meta, inputs, selected = source_context(args.source)
    override = yaml.safe_load(args.config.read_text())
    required = {"binary_use_heap", "rerank_block", "rerank_mode", "compare_legacy_binary"}
    if not required <= set(override) or set(override) - required - {"adapter_backend"}:
        raise ValueError("Only binary implementation settings may change")
    cfg = {**meta["config"], **override}
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    for item in meta["inputs"]:
        if input_identity(Path(item["path"])) != item:
            raise ValueError(f"Source input changed: {item['path']}")
    reference = (Path(meta["references"]["test"]["path"]) if "references" in meta
                 else args.source / "test_reference.npz")
    scores_dir = reference.parent
    manifest = {"config": cfg, "inputs": meta["inputs"],
                "query_caches": inputs["query_caches"], "gallery_caches": inputs["gallery_caches"],
                "source_config": input_identity(args.source / "config.json"),
                "source_selection": input_identity(args.source / "selected.json"),
                "reference": input_identity(reference),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "selection": "all operating points copied unchanged from prior validation selection",
                "timing_boundary": meta["timing_boundary"]}
    if cfg.get("adapter_backend") == "numpy":
        from .numpy_adapter import numpy_checkpoint_path
        manifest["numpy_exports"] = [input_identity(numpy_checkpoint_path(x["path"]))
                                     for x in meta["inputs"] if Path(x["path"]).suffix == ".pt"]
    (args.output / "config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "selected.json").write_text(json.dumps(selected, indent=2) + "\n")
    old_rows = read_rows(args.source / "test.csv")
    keys = ("method", "seed", "direction", "batch", "M", "nlist", "nprobe")
    old = {tuple(r[k] for k in keys): r for r in old_rows}
    quality_fields = [k for k in old_rows[0] if "recall_at" in k or "complete_at" in k or "recovery_at" in k]
    rows, checks = [], []
    for group in selected_groups(selected):
        spec = source_spec(args.source, group, "test")
        spec.update(config=cfg, reference=str(reference.resolve()))
        if group["method"] in ("hashclip", "lsh"):
            spec["scores"] = str((scores_dir / f"test_scores_{group['direction']}.npy").resolve())
        print(f"test {group['method']} {group['direction']} seed={group['seed']} nlist={group.get('nlist')}", flush=True)
        measured = execute(spec, args.output / "workers.jsonl")
        for row in measured["rows"]:
            previous = old[tuple(row[k] for k in keys)]
            differences = {}
            for key in quality_fields:
                actual, expected = row.get(key), previous.get(key)
                if actual is None or expected is None:
                    if actual != expected:
                        differences[key] = [expected, actual]
                elif abs(actual - expected) > 1e-12:
                    differences[key] = actual - expected
            checks.append({**{k: row[k] for k in keys}, "quality_differences": differences})
            if differences and not (row["method"] == "hashclip" and cfg.get("adapter_backend", "torch") == "numpy"):
                (args.output / "quality_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
                raise ValueError(f"Quality changed: {checks[-1]}")
        rows.extend(measured["rows"])
        write_csv(args.output / "test.csv", rows)
        (args.output / "quality_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    write_csv(args.output / "selected_test.csv", selected_test_rows(selected, rows))
    memory, seen = [], set()
    for point in selected:
        if not point["target_met_on_validation"]:
            continue
        key = tuple(point[k] for k in keys)
        if key in seen:
            continue
        seen.add(key)
        spec = source_spec(args.source, point, "test")
        spec.update(config=cfg, mode="memory", budget=point["M"] or point["nprobe"])
        memory.append({**point, **execute(spec, args.output / "memory_workers.jsonl")})
        (args.output / "memory.json").write_text(json.dumps(memory, indent=2) + "\n")
    changed = sum(bool(c["quality_differences"]) for c in checks)
    print(f"Completed {len(rows)} measurements; {changed} quality records differ from the source, {len(memory)} memory probes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

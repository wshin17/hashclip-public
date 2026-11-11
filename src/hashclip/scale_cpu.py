"""Scale fixed-query CPU retrieval with existing training-gallery distractors.

The added items were seen during adapter training. This is a scale diagnostic,
not an independent-dataset evaluation. No model or feature extraction runs here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate_cpu import configure, input_identity, load_features, run
from .paths import ROOT, data_root, runs_root


def check_disjoint(caches):
    names = list(caches)
    for i, name in enumerate(names):
        ids = set(map(str, caches[name]["img_ids"]))
        for other in names[i + 1:]:
            if ids & set(map(str, caches[other]["img_ids"])):
                raise ValueError(f"Image IDs overlap between {name} and {other}")


def expanded_gallery(queries, train, selected):
    """Preserve caption order within each image, sorting images by source ID."""
    import numpy as np

    selected = np.asarray(selected, dtype=np.int64)
    if len(np.unique(selected)) != len(selected) or (selected < 0).any() or (selected >= len(train["img_ids"])).any():
        raise ValueError("Distractor rows must be unique valid training image rows")
    if set(map(str, queries["img_ids"])) & set(map(str, train["img_ids"])):
        raise ValueError("Training distractors overlap query images")
    ids = np.concatenate([queries["img_ids"], train["img_ids"][selected]])
    # Source IDs in these Flickr30k caches are numeric strings. For other ID
    # formats, lexical order is stable and independent of the query split.
    key = (lambda i: int(str(ids[i]))) if all(str(value).isdigit() for value in ids) else (lambda i: str(ids[i]))
    order = np.array(sorted(range(len(ids)), key=key), dtype=np.int64)
    inverse = np.empty(len(ids), dtype=np.int64)
    inverse[order] = np.arange(len(ids))
    train_rows = np.full(len(train["img_ids"]), -1, dtype=np.int64)
    train_rows[selected] = np.arange(len(selected)) + len(queries["img_ids"])
    text_mask = train_rows[train["txt_to_img"]] >= 0
    text_rows = np.concatenate([queries["txt_to_img"], train_rows[train["txt_to_img"][text_mask]]])
    mapping = inverse[text_rows]
    text_order = np.argsort(mapping, kind="stable")
    return {
        "img_feat": np.concatenate([queries["img_feat"], train["img_feat"][selected]])[order],
        "txt_feat": np.concatenate([queries["txt_feat"], train["txt_feat"][text_mask]])[text_order],
        "img_ids": ids[order],
        "txt_to_img": mapping[text_order],
    }


def prepare_galleries(features, destination, cfg):
    import numpy as np

    paths = {split: features / f"backbone_{split}.npz" for split in ("train", "val", "test")}
    sources = {split: input_identity(path) for split, path in paths.items()}
    settings = {"counts": cfg["distractor_counts"], "seed": cfg["distractor_seed"],
                "order": "numeric source image ID; original caption order within image"}
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["settings"] != settings or any(manifest["sources"][s]["sha256"] != sources[s]["sha256"] for s in paths):
            raise ValueError("Existing gallery manifest differs from sources or settings")
        for group in manifest["galleries"]:
            for split, identity in group["files"].items():
                path = destination / str(group["distractors"]) / f"backbone_{split}.npz"
                if input_identity(path)["sha256"] != identity["sha256"]:
                    raise ValueError(f"Prepared gallery changed: {path}")
        return manifest
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Incomplete / unrelated gallery directory: {destination}")
    caches = {s: load_features(p, cfg["expected_train_images"] if s == "train" else cfg["expected_images"][s])
              for s, p in paths.items()}
    check_disjoint(caches)
    for split, cache in caches.items():
        for name in ("img_feat", "txt_feat"):
            if not np.all(np.any(cache[name] != 0, axis=1)):
                raise ValueError(f"Zero feature in {split}/{name}")
    counts = cfg["distractor_counts"]
    if any(c <= 0 or c > len(caches["train"]["img_ids"]) for c in counts) or len(set(counts)) != len(counts):
        raise ValueError("Invalid distractor counts")
    permutation = np.random.default_rng(cfg["distractor_seed"]).permutation(len(caches["train"]["img_ids"]))
    manifest = {"settings": settings, "sources": sources, "galleries": [],
                "scope": "Training-gallery scale diagnostic; original queries held out; not independent-data evidence"}
    for count in counts:
        selected = permutation[:count]
        group = {"distractors": count, "selected_train_ids": caches["train"]["img_ids"][selected].tolist(), "files": {}}
        target = destination / str(count)
        target.mkdir(parents=True)
        for split in ("val", "test"):
            gallery = expanded_gallery(caches[split], caches["train"], selected)
            path = target / f"backbone_{split}.npz"
            temporary = path.with_suffix(".npz.tmp")
            with temporary.open("xb") as f:
                np.savez(f, **gallery)
            temporary.replace(path)
            group["files"][split] = {**input_identity(path), "images": len(gallery["img_feat"]), "texts": len(gallery["txt_feat"])}
            print(f"Prepared {split}: {len(gallery['img_feat'])} images / {len(gallery['txt_feat'])} texts", flush=True)
            del gallery
        manifest["galleries"].append(group)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    import yaml

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/icde_scale.yaml")
    p.add_argument("--features", type=Path, default=data_root() / "features")
    p.add_argument("--galleries", type=Path, default=data_root() / "features/icde_scale")
    p.add_argument("--checkpoints", type=Path, default=runs_root() / "wsdm-original/checkpoints")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--distractors", nargs="+", type=int, help="Run only these configured distractor counts")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if not args.prepare_only and args.output is None:
        p.error("--output is required for evaluation")
    cfg = yaml.safe_load(args.config.read_text())
    if args.distractors and not set(args.distractors) <= set(cfg["distractor_counts"]):
        p.error("--distractors must be a subset of the configured counts")
    configure(cfg)
    manifest = prepare_galleries(args.features, args.galleries, cfg)
    if args.prepare_only:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    for group in manifest["galleries"]:
        count = group["distractors"]
        if args.distractors and count not in args.distractors:
            continue
        extra = cfg.get("candidate_budgets_extra_by_distractors", {}).get(count, [])
        scale_cfg = {**cfg, "candidate_budgets": sorted(set(cfg["candidate_budgets"] + extra)),
                     "expected_gallery_images": {s: entry["images"] for s, entry in group["files"].items()}}
        config_path = args.output / f"config_{count}.yaml"
        if config_path.exists():
            raise FileExistsError(config_path)
        config_path.write_text(yaml.safe_dump(scale_cfg, sort_keys=False))
        run(argparse.Namespace(config=config_path, features=args.features, checkpoints=args.checkpoints,
                               gallery_features=args.galleries / str(count), output=args.output / str(count)))


if __name__ == "__main__":
    main()

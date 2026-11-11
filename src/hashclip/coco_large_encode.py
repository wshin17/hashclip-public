"""Encode official COCO distractors with the existing frozen SigLIP pipeline.

Archive-sized groups can be encoded as their downloads finish. Small feature
shards allow continuation without repeating completed GPU work. Query caches
are reused unchanged; unlabeled images receive no invented captions.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .coco import MODEL, REVISION
from .evaluate_cpu import input_identity
from .paths import data_root

FOLDERS = ("train2014", "val2014", "test2015", "unlabeled2017")


def shard_rows(rows, size):
    for folder in FOLDERS:
        group = [r for r in rows if r["filename"].split("/")[0] == folder]
        for start in range(0, len(group), size):
            yield f"{folder}_{start:06d}", folder, group[start:start + size]


def encode(args):
    # Pin both the snapshot and the hub ref used by the existing loader. Offline
    # mode prevents a model update part-way through a multi-archive encoding.
    os.environ["HF_HOME"] = str(args.cache.resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import numpy as np
    import pandas as pd
    import torch
    from huggingface_hub import snapshot_download
    from .encode import encode_backbone_split
    from .model import encode_text_batch, load_backbone
    from .utils import git_rev

    if not torch.cuda.is_available():
        raise RuntimeError("Select an idle CUDA device for frozen backbone encoding")
    ref = args.cache / "hub/models--timm--ViT-B-16-SigLIP/refs/main"
    if ref.read_text().strip() != REVISION:
        raise ValueError("Cached backbone ref differs from the existing feature pipeline")
    snapshot = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    rows = json.loads((args.raw / "distractors.json").read_text())
    settings = {"revision": REVISION, "shard_images": args.shard_images,
                "batch_images": args.batch_images, "batch_texts": 256,
                "precision": "float32 inference, float16 stored features; TF32 disabled",
                "records_sha256": input_identity(args.raw / "distractors.json")["sha256"]}
    path = args.output / "encoding_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["settings"] != settings:
            raise ValueError("Existing encoding settings or input records differ")
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError("Use a new encoding directory or its recorded continuation")
        args.output.mkdir(parents=True, exist_ok=True)
        manifest = {"settings": settings, "model": MODEL, "model_snapshot": snapshot,
                    "git_commit": git_rev(), "device": torch.cuda.get_device_name(0),
                    "dataset_manifest": input_identity(args.raw / "prepared_manifest.json"),
                    "packages": {p: importlib.metadata.version(p) for p in
                                 ("torch", "torchvision", "open_clip_torch", "timm", "transformers", "numpy")},
                    "shards": {}}
    pending = []
    for name, folder, group in shard_rows(rows, args.shard_images):
        if name in manifest["shards"]:
            saved = args.output / "shards" / f"{name}.npz"
            if not saved.exists() or saved.stat().st_size != manifest["shards"][name]["features"]["bytes"]:
                raise ValueError(f"Recorded feature shard is missing or truncated: {saved}")
        elif folder in args.archives and (args.raw / f"{folder}.zip.extracted.json").exists():
            pending.append((name, folder, group))
    if not pending:
        print("No unencoded shards from completed archives", flush=True)
        return
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    backbone = load_backbone("hf-hub:" + MODEL)
    backbone.model.to("cuda").eval()
    with np.load(args.queries / "backbone_val.npz", allow_pickle=True) as archive:
        indices = np.linspace(0, len(archive["captions"]) - 1, 64, dtype=int)
        captions = [str(x) for x in archive["captions"][indices]]
        original = archive["txt_feat"][indices].astype(np.float32)
    current = encode_text_batch(backbone, captions, device="cuda").cpu().numpy()
    relative = np.linalg.norm(current - original, axis=1) / np.linalg.norm(original, axis=1)
    cosine = (current * original).sum(axis=1) / (np.linalg.norm(current, axis=1) * np.linalg.norm(original, axis=1))
    if not np.isfinite(relative).all() or not np.isfinite(cosine).all() or cosine.min() < .9999 or relative.max() > .005:
        raise ValueError("Frozen encoder does not reproduce the original COCO text cache")
    manifest["compatibility"] = {"min_cosine": float(cosine.min()), "max_relative_l2": float(relative.max()),
                                 "source": input_identity(args.queries / "backbone_val.npz")}
    manifest["preprocess"] = str(backbone.preprocess)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "shards").mkdir(exist_ok=True)
    for name, folder, group in pending:
        print(f"Encoding {name}: {len(group)} images, {sum(len(r['captions']) for r in group)} captions", flush=True)
        df = pd.DataFrame(group)
        df["img_path"] = df["filename"].map(lambda f: str((args.raw / "extracted" / f).resolve()))
        features = encode_backbone_split(backbone, df, device="cuda", batch_size_img=args.batch_images,
                                        batch_size_txt=256, num_workers=4)
        target = args.output / "shards" / f"{name}.npz"
        temporary = target.with_suffix(".partial.npz")
        features.save(temporary)
        temporary.replace(target)
        manifest["shards"][name] = {"archive": folder, "images": len(features.img_feat),
                                    "captions": len(features.txt_feat), "features": input_identity(target)}
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        del features
    print(f"Completed {len(manifest['shards'])} recorded shards", flush=True)


def join_features(parts):
    """Join disjoint shards and preserve original caption order per image."""
    import numpy as np

    image_ids = np.concatenate([p["img_ids"] for p in parts])
    if len(set(map(str, image_ids))) != len(image_ids):
        raise ValueError("Feature shards have overlapping image IDs")
    order = np.array(sorted(range(len(image_ids)), key=lambda i: str(image_ids[i])), dtype=np.int64)
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.arange(len(order))
    offsets = np.cumsum([0] + [len(p["img_ids"]) for p in parts[:-1]])
    mapping = inverse[np.concatenate([p["txt_to_img"] + offset for p, offset in zip(parts, offsets)])]
    text_order = np.argsort(mapping, kind="stable")
    return {"img_feat": np.concatenate([p["img_feat"] for p in parts])[order], "img_ids": image_ids[order],
            "txt_feat": np.concatenate([p["txt_feat"] for p in parts])[text_order],
            "txt_to_img": mapping[text_order],
            "captions": np.concatenate([p["captions"] for p in parts])[text_order]}


def assemble(args):
    import numpy as np
    from .scale_cpu import expanded_gallery

    manifest = json.loads((args.output / "encoding_manifest.json").read_text())
    rows = json.loads((args.raw / "distractors.json").read_text())
    expected = list(shard_rows(rows, manifest["settings"]["shard_images"]))

    def read_shards(names):
        parts = []
        for name in names:
            if name not in manifest["shards"]:
                raise ValueError(f"Required feature shard has not been encoded: {name}")
            with np.load(args.output / "shards" / f"{name}.npz", allow_pickle=True) as archive:
                parts.append({k: archive[k] for k in ("img_feat", "txt_feat", "img_ids", "txt_to_img", "captions")})
        return join_features(parts)

    def save(target, features):
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".partial.npz")
        with temporary.open("wb") as stream:
            np.savez(stream, **features)
        temporary.replace(target)
        return {**input_identity(target), "images": len(features["img_feat"]), "captions": len(features["txt_feat"])}

    if args.train_only:
        # Captioned distractors are exactly Karpathy train + restval. No query
        # split, test2015 or unlabeled2017 item trains PQ/OPQ/ITQ codebooks.
        names = [name for name, folder, _ in expected if folder in ("train2014", "val2014")]
        features = read_shards(names)
        train_ids = {r["img_id"] for r in json.loads((args.raw / "train.json").read_text())}
        if set(map(str, features["img_ids"])) != train_ids:
            raise ValueError("Captioned distractors differ from the Karpathy training split")
        result = save(args.output / "backbone_train.npz", features)
        (args.output / "training_features.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return
    distractors = read_shards([name for name, _, _ in expected])
    if set(map(str, distractors["img_ids"])) != {r["img_id"] for r in rows}:
        raise ValueError("Encoded distractor IDs differ from the official metadata")
    rng = np.random.default_rng(42)
    permutation = rng.permutation(len(distractors["img_ids"]))
    counts = sorted(set(args.distractors or [len(permutation)]))
    if any(c < 1 or c > len(permutation) for c in counts):
        raise ValueError("Invalid distractor count")
    record = {"encoding_manifest": input_identity(args.output / "encoding_manifest.json"),
              "task_recall": False, "distractor_seed": 42, "galleries": []}
    record_path = args.output / "gallery_manifest.json"
    if record_path.exists():
        prior = json.loads(record_path.read_text())
        if prior["encoding_manifest"]["sha256"] != record["encoding_manifest"]["sha256"]:
            raise ValueError("Gallery inputs changed after assembly")
        record = prior
    for count in counts:
        for split in ("val", "test"):
            target = args.output / "galleries" / str(count) / f"backbone_{split}.npz"
            saved = next((r for r in record["galleries"] if r["split"] == split and r["distractors"] == count), None)
            if saved is not None:
                if not target.exists() or target.stat().st_size != saved["bytes"]:
                    raise ValueError(f"Recorded gallery is missing or truncated: {target}")
                continue
            with np.load(args.queries / f"backbone_{split}.npz", allow_pickle=True) as archive:
                queries = {k: archive[k] for k in ("img_feat", "txt_feat", "img_ids", "txt_to_img")}
            gallery = expanded_gallery(queries, distractors, permutation[:count])
            record["galleries"].append({"split": split, "distractors": count, **save(target, gallery)})
            record_path.write_text(json.dumps(record, indent=2) + "\n")
            print(f"Prepared {split}: {len(gallery['img_feat'])} images, {len(gallery['txt_feat'])} captions", flush=True)
            del gallery


def idle_gpu(status):
    """Never select an allocated but momentarily idle GPU."""
    for row in status.splitlines():
        fields = [int(value.strip()) for value in row.split(",")]
        if len(fields) != 3:
            raise ValueError("Unexpected nvidia-smi device status")
        index, memory_mib, utilization = fields
        if memory_mib < 100 and utilization < 5:
            return index
    return None


def display_only(status):
    """A small display allocation (e.g. a remote-desktop client) may share an explicitly selected GPU."""
    for row in status.splitlines():
        _, process, memory = [value.strip() for value in row.split(",", 2)]
        if Path(process).name != "ToDesk_Session" or not memory.isdigit() or int(memory) > 1024:
            return False
    return True


def watch(args):
    """Run only after formal CPU timing ends; wait for completed archives."""
    rows = json.loads((args.raw / "distractors.json").read_text())
    groups = {folder: set() for folder in FOLDERS}
    for name, folder, _ in shard_rows(rows, args.shard_images):
        groups[folder].add(name)
    common = ["--raw", str(args.raw), "--output", str(args.output), "--queries", str(args.queries),
              "--cache", str(args.cache), "--batch-images", str(args.batch_images),
              "--shard-images", str(args.shard_images)]
    prefix = [sys.executable, "-m", "hashclip.coco_large_encode"]
    while True:
        manifest_path = args.output / "encoding_manifest.json"
        completed = set(json.loads(manifest_path.read_text())["shards"]) if manifest_path.exists() else set()
        done = {folder for folder, names in groups.items() if names <= completed}
        if {"train2014", "val2014"} <= done and not (args.output / "backbone_train.npz").exists():
            subprocess.run(prefix + ["assemble", "--train-only"] + common, check=True)
        if done == set(FOLDERS):
            subprocess.run(prefix + ["assemble"] + common, check=True)
            print("All official distractors encoded; training cache and full galleries assembled", flush=True)
            return
        ready = [folder for folder in FOLDERS if folder not in done
                 and (args.raw / f"{folder}.zip.extracted.json").exists()]
        if ready:
            if args.gpu is not None:
                # A workstation display can consume memory on an otherwise
                # available GPU. Explicit selection still waits for CUDA jobs.
                active = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu),
                    "--query-compute-apps=pid,process_name,used_gpu_memory",
                    "--format=csv,noheader,nounits"], text=True)
                gpu = args.gpu if display_only(active) else None
            else:
                status = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                                   "--format=csv,noheader,nounits"], text=True)
                gpu = idle_gpu(status)
            if gpu is not None:
                print(f"Encoding ready archive {ready[0]} on unallocated GPU {gpu}", flush=True)
                subprocess.run(prefix + ["encode", "--archives", ready[0]] + common,
                               env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}, check=True)
                continue
        print(f"Waiting for archives / an unallocated GPU; encoded archives: {sorted(done)}", flush=True)
        time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("encode", "assemble", "watch"))
    parser.add_argument("--raw", type=Path, default=data_root() / "raw/coco-large")
    parser.add_argument("--output", type=Path, default=data_root() / "features/coco-large")
    parser.add_argument("--queries", type=Path, default=data_root() / "features/coco")
    parser.add_argument("--cache", type=Path, default=data_root() / "pretrained/huggingface")
    parser.add_argument("--batch-images", type=int, default=64)
    parser.add_argument("--shard-images", type=int, default=4096)
    parser.add_argument("--archives", nargs="+", choices=FOLDERS, default=list(FOLDERS))
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--distractors", nargs="+", type=int)
    parser.add_argument("--gpu", type=int, help="Explicit device for watch; still wait for existing CUDA jobs")
    args = parser.parse_args()
    if args.batch_images < 1 or args.shard_images < 1:
        parser.error("Batch and shard sizes must be positive")
    {"encode": encode, "assemble": assemble, "watch": watch}[args.stage](args)


if __name__ == "__main__":
    main()

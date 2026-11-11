"""Prepare original COCO Karpathy validation/test and encode with frozen SigLIP.

All captions are retained, including the ten six-caption images in each split.
Images are downloaded from the official COCO S3 bucket, never from a resampled
or relabeled substitute. Assets and feature caches stay outside Git.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import time
import urllib.request
import zipfile

from .paths import data_root

ANNOTATIONS = "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
IMAGES = "https://s3.amazonaws.com/images.cocodataset.org/val2014/"
MODEL = "timm/ViT-B-16-SigLIP"
REVISION = "41f575766f40e752fdd1383e9565b7f02388c1c4"


def karpathy_rows(document, expected_images=5000):
    rows = {split: [] for split in ("val", "test")}
    seen, sentences = set(), set()
    for item in document["images"]:
        split = item["split"]
        if split not in rows:
            continue
        image_id = int(item["cocoid"])
        filename = item["filename"]
        if item["filepath"] != "val2014" or not re.fullmatch(r"COCO_val2014_\d{12}\.jpg", filename):
            raise ValueError("Unexpected COCO image path")
        if int(filename[-16:-4]) != image_id or image_id in seen:
            raise ValueError("Image ID mismatch or overlapping splits")
        seen.add(image_id)
        captions, ids = [], []
        for sentence in item["sentences"]:
            sid, text = int(sentence["sentid"]), sentence["raw"]
            if sid in sentences or not isinstance(text, str) or not text.strip():
                raise ValueError("Duplicate caption ID or empty caption")
            sentences.add(sid)
            captions.append(text)
            ids.append(sid)
        if not captions:
            raise ValueError("Image has no captions")
        rows[split].append({"img_id": f"coco:{image_id}", "filename": filename,
                            "captions": captions, "caption_ids": ids})
    for split in rows:
        rows[split].sort(key=lambda r: int(r["img_id"].split(":")[1]))
        if len(rows[split]) != expected_images:
            raise ValueError(f"{split}: expected {expected_images} images, got {len(rows[split])}")
    return rows


def prepare(raw, workers=16):
    from PIL import Image
    from .evaluate_cpu import input_identity

    archive = raw / "caption_datasets.zip"
    with zipfile.ZipFile(archive) as z:
        rows = karpathy_rows(json.loads(z.read("dataset_coco.json")))
    destination = raw / "images"
    destination.mkdir(parents=True, exist_ok=True)

    def fetch(row):
        target = destination / row["filename"]
        for attempt in range(6):
            try:
                if not target.exists():
                    with urllib.request.urlopen(IMAGES + row["filename"], timeout=45) as response:
                        content = response.read()
                    temporary = target.with_suffix(".jpg.partial")
                    temporary.write_bytes(content)
                    with Image.open(temporary) as im:
                        im.verify()
                    temporary.replace(target)
                with Image.open(target) as im:
                    im.verify()
                return input_identity(target)
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(min(2 ** attempt, 16))

    records = [row for split in rows.values() for row in split]
    identities = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(fetch, row): row for row in records}
        for future in as_completed(pending):
            identities.append(future.result())
            if len(identities) % 250 == 0:
                print(f"Verified {len(identities)}/{len(records)} COCO images", flush=True)
    for split, group in rows.items():
        for row in group:
            row["img_path"] = str((destination / row["filename"]).resolve())
        (raw / f"{split}.json").write_text(json.dumps(group, ensure_ascii=False) + "\n")
    manifest = {"annotation_url": ANNOTATIONS, "annotation_archive": input_identity(archive),
                "image_base_url": IMAGES, "caption_policy": "all original raw captions, original order per image",
                "splits": {s: {"images": len(g), "captions": sum(len(r["captions"]) for r in g),
                                "records": input_identity(raw / f"{s}.json")} for s, g in rows.items()},
                "images": sorted(identities, key=lambda x: x["path"])}
    (raw / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["splits"], indent=2), flush=True)


def encode(raw, output, old_features, cache, batch_images=64):
    # Set cache paths before importing Hugging Face / transformers.
    os.environ["HF_HOME"] = str(cache.resolve())
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import importlib.metadata
    import numpy as np
    import pandas as pd
    import torch
    from huggingface_hub import snapshot_download
    from .encode import encode_backbone_split
    from .evaluate_cpu import input_identity
    from .model import encode_text_batch, load_backbone
    from .utils import git_rev

    if not torch.cuda.is_available():
        raise RuntimeError("COCO backbone encoding requires the selected CUDA device")
    manifest = json.loads((raw / "manifest.json").read_text())
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new feature output directory")
    output.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_download(MODEL, revision="main", allow_patterns=[
        "open_clip_config.json", "open_clip_model.safetensors", "*token*.json", "special_tokens_map.json"])
    if Path(snapshot).name != REVISION:
        raise ValueError("The public backbone revision changed; inspect before encoding")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    backbone = load_backbone("hf-hub:" + MODEL)
    backbone.model.to("cuda").eval()
    with np.load(old_features, allow_pickle=True) as f:
        indices = np.linspace(0, len(f["captions"]) - 1, 64, dtype=int)
        captions = [str(x) for x in f["captions"][indices]]
        original = f["txt_feat"][indices].astype(np.float32)
    current = encode_text_batch(backbone, captions, device="cuda").cpu().numpy()
    relative = np.linalg.norm(current - original, axis=1) / np.linalg.norm(original, axis=1)
    cosine = np.sum(current * original, axis=1) / (np.linalg.norm(current, axis=1) * np.linalg.norm(original, axis=1))
    compatibility = {"queries": len(indices), "min_cosine": float(cosine.min()),
                     "max_relative_l2": float(relative.max()), "source": input_identity(old_features)}
    (output / "compatibility.json").write_text(json.dumps(compatibility, indent=2) + "\n")
    print("Old-cache text compatibility:", json.dumps(compatibility), flush=True)
    if (not np.isfinite(cosine).all() or not np.isfinite(relative).all()
            or cosine.min() < .9999 or relative.max() > .005):
        raise ValueError("Backbone/text preprocessing does not reproduce the original cache")
    record = {"model": MODEL, "revision": REVISION, "model_snapshot": snapshot,
              "model_files": [input_identity(p) for p in sorted(Path(snapshot).iterdir()) if p.is_file()],
              "git_commit": git_rev(), "device": torch.cuda.get_device_name(0),
              "precision": "float32 inference, float16 stored features; TF32 disabled",
              "batch_images": batch_images, "batch_texts": 256, "num_workers": 4,
              "packages": {p: importlib.metadata.version(p) for p in
                           ("torch", "torchvision", "open_clip_torch", "timm", "transformers", "numpy")},
              "preprocess": str(backbone.preprocess), "compatibility": compatibility,
              "dataset_manifest": input_identity(raw / "manifest.json"), "splits": {}}
    for split in ("val", "test"):
        path = raw / f"{split}.json"
        current_id, recorded_id = input_identity(path), manifest["splits"][split]["records"]
        if any(current_id[k] != recorded_id[k] for k in ("bytes", "sha256")):
            raise ValueError("Prepared COCO split changed")
        df = pd.DataFrame(json.loads(path.read_text()))
        df["img_path"] = df["filename"].map(lambda name: str((raw / "images" / name).resolve()))
        features = encode_backbone_split(backbone, df, device="cuda", batch_size_img=batch_images,
                                        batch_size_txt=256, num_workers=4)
        path = output / f"backbone_{split}.npz"
        features.save(path)
        record["splits"][split] = {"images": len(features.img_feat), "captions": len(features.txt_feat),
                                  "features": input_identity(path)}
        (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"Encoded {split}: {features.img_feat.shape}, {features.txt_feat.shape}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "encode"))
    p.add_argument("--raw", type=Path, default=data_root() / "raw/coco")
    p.add_argument("--output", type=Path, default=data_root() / "features/coco")
    p.add_argument("--old-features", type=Path, default=data_root() / "features/backbone_val.npz")
    p.add_argument("--cache", type=Path, default=data_root() / "pretrained/huggingface")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--batch-images", type=int, default=64)
    args = p.parse_args()
    if args.stage == "prepare":
        prepare(args.raw, args.workers)
    else:
        encode(args.raw, args.output, args.old_features, args.cache, args.batch_images)


if __name__ == "__main__":
    main()

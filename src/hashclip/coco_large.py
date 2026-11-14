"""Official COCO archives for the larger-gallery evaluation.

Only the 2014 train/val, 2015 test and 2017 unlabeled images are included.
2017 train/val reuse the 2014 images and are deliberately not downloaded.
Downloads are resumable; raw assets and manifests live outside Git.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
import zipfile

from .paths import data_root

SOURCE = "https://cocodataset.org/#download"
BASE = "https://s3.amazonaws.com/images.cocodataset.org/"
ARCHIVES = {
    "annotations_trainval2014.zip": "annotations/annotations_trainval2014.zip",
    "image_info_test2015.zip": "annotations/image_info_test2015.zip",
    "image_info_unlabeled2017.zip": "annotations/image_info_unlabeled2017.zip",
    "train2014.zip": "zips/train2014.zip",
    "val2014.zip": "zips/val2014.zip",
    "test2015.zip": "zips/test2015.zip",
    "unlabeled2017.zip": "zips/unlabeled2017.zip",
}


def download_ranges(url, partial, connections=4, chunk_bytes=64 * 1024 * 1024):
    headers = subprocess.check_output(["curl", "--fail", "--location", "--head", "--silent",
                                       "--show-error", "--max-time", "45", url], text=True)
    sizes = re.findall(r"(?im)^content-length:\s*(\d+)", headers)
    if not sizes:
        raise ValueError(f"Missing source content length: {url}")
    size = int(sizes[-1])
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        raise ValueError(f"Partial archive exceeds source size: {partial}")
    parts = partial.with_suffix(".parts")
    parts.mkdir(exist_ok=True)

    def fetch(start):
        end = min(start + chunk_bytes, size) - 1
        path = parts / f"{start}-{end}"
        if not path.exists() or path.stat().st_size != end - start + 1:
            subprocess.run(["curl", "--fail", "--location", "--retry", "12", "--retry-all-errors",
                            "--retry-delay", "3", "--connect-timeout", "30", "--max-time", "1800",
                            "--speed-time", "120", "--speed-limit", "16384", "--silent", "--show-error",
                            "--range", f"{start}-{end}", "--output", str(path), url], check=True)
        if path.stat().st_size != end - start + 1:
            raise ValueError(f"Server did not return the requested range: {path}")
        return path

    starts = list(range(offset, size, chunk_bytes))
    with ThreadPoolExecutor(max_workers=connections) as pool:
        futures = {pool.submit(fetch, start): start for start in starts}
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 8 == 0 or i == len(starts):
                print(f"{partial.name}: {i}/{len(starts)} ranges complete", flush=True)
    with partial.open("ab") as output:
        for start in starts:
            end = min(start + chunk_bytes, size) - 1
            path = parts / f"{start}-{end}"
            with path.open("rb") as stream:
                shutil.copyfileobj(stream, output, 8 * 1024 * 1024)
    if partial.stat().st_size != size:
        raise ValueError("Assembled archive size differs from official source")
    for start in starts:
        (parts / f"{start}-{min(start + chunk_bytes, size) - 1}").unlink()


def download_one(root, name, extract, connections=1, expected=None):
    archives = root / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    target = archives / name
    url = BASE + ARCHIVES[name]
    if not target.exists():
        partial = target.with_suffix(".zip.partial")
        print(f"Downloading {name} from {url}", flush=True)
        if connections > 1:
            download_ranges(url, partial, connections)
        else:
            subprocess.run(["curl", "--fail", "--location", "--retry", "12",
                            "--retry-delay", "5", "--connect-timeout", "30",
                            "--speed-time", "120", "--speed-limit", "1024",
                            "--continue-at", "-", "--silent", "--show-error",
                            "--output", str(partial), url], check=True)
        # Opening the central directory catches incomplete transfers. Extraction
        # below checks each member's CRC as it is read, without a second pass.
        with zipfile.ZipFile(partial):
            pass
        partial.replace(target)
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    record = {"url": url, "path": str(target.resolve()),
              "bytes": target.stat().st_size, "sha256": digest.hexdigest()}
    if expected is not None and any(record[key] != expected[key] for key in ("url", "bytes", "sha256")):
        raise ValueError(f"Transferred archive differs from the verified source: {name}")
    destination = root / "extracted"
    marker = root / f"{name}.extracted.json"
    if extract and not marker.exists():
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target) as archive:
            members = archive.infolist()
            for member in members:
                member_path = Path(member.filename)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError(f"Unsafe archive path: {member.filename}")
            archive.extractall(destination)
            record["files"] = sum(not m.is_dir() for m in members)
        marker.write_text(json.dumps(record, indent=2) + "\n")
    elif extract:
        prior = json.loads(marker.read_text())
        if prior["sha256"] != record["sha256"]:
            raise ValueError(f"Archive changed after extraction: {name}")
        record["files"] = prior["files"]
    record["extracted"] = extract
    print(f"Completed {name}: {record['bytes']} bytes", flush=True)
    return name, record


def download(root, workers=2, extract=True, connections=1):
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"source": SOURCE, "started_unix": time.time(), "archives": {}}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download_one, root, name, extract, connections) for name in ARCHIVES]
        for future in as_completed(futures):
            name, record = future.result()
            manifest["archives"][name] = record
            (root / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["finished_unix"] = time.time()
    (root / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def prepare(root, karpathy):
    """Join official image/caption IDs to Karpathy splits; never infer labels."""
    from .evaluate_cpu import input_identity

    annotations = root / "extracted/annotations"
    images, captions, source_files = {}, {}, []
    for folder, filename in (("train2014", "captions_train2014.json"),
                             ("val2014", "captions_val2014.json"),
                             ("test2015", "image_info_test2015.json"),
                             ("unlabeled2017", "image_info_unlabeled2017.json")):
        path = annotations / filename
        document = json.loads(path.read_text())
        source_files.append(input_identity(path))
        for image in document["images"]:
            iid = int(image["id"])
            if iid in images:
                raise ValueError(f"Official splits overlap at image ID {iid}; inspect before deduplicating")
            images[iid] = {"img_id": f"coco:{iid}", "filename": f"{folder}/{image['file_name']}",
                           "captions": [], "caption_ids": []}
        for caption in document.get("annotations", []):
            cid = int(caption["id"])
            if cid in captions or int(caption["image_id"]) not in images:
                raise ValueError("Duplicate caption ID or broken image join")
            captions[cid] = caption
    with zipfile.ZipFile(karpathy) as archive:
        splits = json.loads(archive.read("dataset_coco.json"))
    split_ids = {s: set() for s in ("train", "val", "test")}
    seen_captions, text_differences = set(), 0
    for row in splits["images"]:
        split = "train" if row["split"] in ("train", "restval") else row["split"]
        if split not in split_ids:
            raise ValueError(f"Unexpected Karpathy split: {split}")
        iid = int(row["cocoid"])
        if iid not in images or any(iid in ids for ids in split_ids.values()):
            raise ValueError("Karpathy image missing or overlapping")
        if images[iid]["filename"] != f"{row['filepath']}/{row['filename']}":
            raise ValueError("Official and Karpathy image paths differ")
        split_ids[split].add(iid)
        for sentence in row["sentences"]:
            cid = int(sentence["sentid"])
            if cid in seen_captions or cid not in captions or captions[cid]["image_id"] != iid:
                raise ValueError("Caption IDs do not match the official image association")
            seen_captions.add(cid)
            text_differences += sentence["raw"] != captions[cid]["caption"]
            # Retain the same raw Karpathy caption strings as existing query
            # caches. All IDs are checked against official annotations.
            images[iid]["captions"].append(sentence["raw"])
            images[iid]["caption_ids"].append(cid)
    if seen_captions != set(captions):
        raise ValueError("Karpathy data do not cover all official 2014 captions")
    if len(split_ids["val"]) != 5000 or len(split_ids["test"]) != 5000:
        raise ValueError("Unexpected Karpathy validation/test image count")
    query_ids = split_ids["val"] | split_ids["test"]
    distractors = set(images) - query_ids
    groups = {**split_ids, "distractors": distractors}
    record = {"official_annotations": source_files, "karpathy": input_identity(karpathy),
              "all_images": len(images), "all_captions": len(captions),
              "caption_text_differences_vs_official": text_differences,
              "image_verification": "metadata joined; archive CRC checked on extraction; encoding checks each image",
              "caption_policy": "all raw Karpathy captions, original order; IDs joined to official captions",
              "groups": {}, "galleries": {}}
    for group, ids in groups.items():
        rows = [images[iid] for iid in sorted(ids)]
        path = root / f"{group}.json"
        path.write_text(json.dumps(rows, ensure_ascii=False) + "\n")
        record["groups"][group] = {"images": len(rows), "captions": sum(len(r["captions"]) for r in rows),
                                    "records": input_identity(path)}
    for split in ("val", "test"):
        record["galleries"][split] = {"images": len(distractors) + len(split_ids[split]),
            "captions": record["groups"]["distractors"]["captions"] + record["groups"][split]["captions"],
            "composition": f"{split} queries' original gallery + identical distractors; opposite split excluded",
            "task_recall": False}
    (root / "prepared_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("all_images", "all_captions", "caption_text_differences_vs_official", "galleries")}, indent=2), flush=True)


def mirror(root, host, source_root):
    """Bring completed official archives to the machine with an available GPU."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", host):
        raise ValueError("Use an SSH host name / configured alias")
    root.mkdir(parents=True, exist_ok=True)
    (root / "archives").mkdir(exist_ok=True)
    prefix = f"{host}:{str(source_root).rstrip('/')}/"
    records = ["train.json", "val.json", "test.json", "distractors.json", "prepared_manifest.json"]
    subprocess.run(["rsync", "-a", "--protect-args", *[f"--include={name}" for name in records],
                    "--exclude=*", "--", prefix, str(root) + "/"], check=True)
    names = [f"{folder}.zip" for folder in ("val2014", "train2014", "test2015", "unlabeled2017")]
    while True:
        completed = 0
        for name in names:
            if (root / f"{name}.extracted.json").exists():
                completed += 1
                continue
            marker = str(source_root).rstrip("/") + f"/{name}.extracted.json"
            check = subprocess.run(["ssh", "-o", "BatchMode=yes", host, "cat -- " + shlex.quote(marker)],
                                   capture_output=True, text=True)
            if check.returncode:
                continue
            expected = json.loads(check.stdout)
            target = root / "archives" / name
            if not target.exists():
                temporary = target.with_suffix(".zip.transfer")
                print(f"Transferring verified {name} from {host}", flush=True)
                subprocess.run(["rsync", "-a", "--partial", "--append-verify", "--protect-args", "--",
                                prefix + "archives/" + name, str(temporary)], check=True)
                temporary.replace(target)
            download_one(root, name, True, expected=expected)
            completed += 1
        if completed == len(names):
            print("All official image archives transferred and extracted", flush=True)
            return
        print(f"Verified local archives: {completed}/{len(names)}; waiting for remaining downloads", flush=True)
        time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("download", "prepare", "mirror"))
    parser.add_argument("--root", type=Path, default=data_root() / "raw/coco-large")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--connections", type=int, default=1)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--karpathy", type=Path, default=data_root() / "raw/coco/caption_datasets.zip")
    parser.add_argument("--source-host")
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    if args.workers < 1 or args.connections < 1:
        parser.error("workers and connections must be positive")
    if args.stage == "download":
        download(args.root, args.workers, not args.download_only, args.connections)
    elif args.stage == "mirror":
        if not args.source_host or not args.source_root:
            parser.error("mirror requires --source-host and --source-root")
        mirror(args.root, args.source_host, args.source_root)
    else:
        prepare(args.root, args.karpathy)


if __name__ == "__main__":
    main()

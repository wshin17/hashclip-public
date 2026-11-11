"""Dataset loading and Karpathy-split parsing for Flickr30k.

* ``download_flickr30k`` — pulls the gated ``nlphuji/flickr30k`` HF dataset.
* ``build_splits``        — parses the CSV and emits parquet shards.
* ``Flickr30kPairs``      — training dataset (one random caption per image per epoch).
* ``Flickr30kEval``       — eval dataset (image + 5 captions + img_id).
* ``CachedFeaturePairs``  — training dataset over pre-extracted backbone features.

Every filesystem operation uses :class:`pathlib.Path`; Windows back-slashes and
POSIX forward-slashes are handled transparently.
"""

from __future__ import annotations

import ast
import os
import platform
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from . import utils


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CSV_NAME = "flickr_annotations_30k.csv"
IMG_ZIP_NAME = "flickr30k-images.zip"
IMG_DIR_NAME = "flickr30k-images"

EXPECTED_SPLIT_COUNTS = {"train": 29_000, "val": 1_014, "test": 1_000}


# ---------------------------------------------------------------------------
# Download + split building
# ---------------------------------------------------------------------------


def download_flickr30k(raw_dir: Path) -> Path:
    """Download annotations + image zip, then extract images. Idempotent."""
    from huggingface_hub import snapshot_download  # local import — heavy dep

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id="nlphuji/flickr30k",
        repo_type="dataset",
        local_dir=str(raw_dir),
        allow_patterns=[CSV_NAME, IMG_ZIP_NAME],
        max_workers=4,
    )

    images_dir = raw_dir / IMG_DIR_NAME
    if not images_dir.exists():
        zip_path = raw_dir / IMG_ZIP_NAME
        if not zip_path.exists():
            raise FileNotFoundError(f"Expected zip not found: {zip_path}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
    return images_dir


def build_splits(raw_dir: Path, processed_dir: Path) -> Dict[str, Path]:
    """Parse the annotations CSV into train/val/test parquet shards."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    csv_path = raw_dir / CSV_NAME
    img_dir = raw_dir / IMG_DIR_NAME
    if not csv_path.exists():
        raise FileNotFoundError(f"Annotations CSV missing: {csv_path}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Images directory missing: {img_dir}")

    df = pd.read_csv(csv_path)
    if "raw" not in df.columns or "split" not in df.columns:
        raise ValueError(
            "Unexpected CSV schema. Expected columns: filename, img_id, "
            "sentids, split, raw."
        )

    # 'raw' is stored as a string-encoded Python list of 5 captions.
    df["captions"] = df["raw"].apply(_parse_captions)

    # Convert filenames to absolute paths (handles Windows back-slash trans).
    df["img_path"] = df["filename"].apply(
        lambda fn: str((img_dir / fn).resolve())
    )
    df["img_id"] = df["img_id"].astype(str)

    out: Dict[str, Path] = {}
    for split, expected in EXPECTED_SPLIT_COUNTS.items():
        sub = df[df["split"] == split][["img_path", "img_id", "captions"]].reset_index(drop=True)
        if len(sub) != expected:
            raise AssertionError(
                f"split '{split}' has {len(sub)} rows; expected {expected}. "
                "The CSV may have changed; please verify nlphuji/flickr30k."
            )
        for cap_list in sub["captions"].iloc[:64]:
            if not isinstance(cap_list, list) or len(cap_list) != 5:
                raise AssertionError(
                    "Each row must have exactly 5 captions; found a row with "
                    f"{len(cap_list) if isinstance(cap_list, list) else 'non-list'}."
                )
        target = processed_dir / f"{split}.parquet"
        sub.to_parquet(target, index=False)
        out[split] = target
    return out


def _parse_captions(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return [str(s) for s in raw]
    if isinstance(raw, str):
        try:
            v = ast.literal_eval(raw)
            if isinstance(v, list):
                return [str(s) for s in v]
        except (ValueError, SyntaxError):
            pass
    raise ValueError(f"Could not parse 'raw' cell: {raw!r}")


def load_split(processed_dir: Path, split: str) -> pd.DataFrame:
    p = Path(processed_dir) / f"{split}.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"Split parquet not found: {p}. "
            "Run `python -m hashclip.cli build-splits` first."
        )
    return pd.read_parquet(p)


# ---------------------------------------------------------------------------
# Datasets — image-pixel path
# ---------------------------------------------------------------------------


class Flickr30kPairs(Dataset):
    """Training pairs: returns (image_tensor, caption_text).

    A random caption (of the 5 available) is sampled per ``__getitem__`` call
    so each epoch sees a different caption mix.
    """

    def __init__(self, df: pd.DataFrame, preprocess, *, max_rows: Optional[int] = None,
                 seed: int = 0):
        self.df = df.reset_index(drop=True)
        if max_rows is not None:
            self.df = self.df.iloc[:max_rows].reset_index(drop=True)
        self.preprocess = preprocess
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Any, str]:
        row = self.df.iloc[idx]
        img = Image.open(row["img_path"]).convert("RGB")
        img_t = self.preprocess(img)
        caps = row["captions"]
        cap = caps[self._rng.randrange(len(caps))]
        return img_t, str(cap)


class Flickr30kEval(Dataset):
    """Eval dataset: returns (image_tensor, list-of-5-captions, img_id)."""

    def __init__(self, df: pd.DataFrame, preprocess, *, max_rows: Optional[int] = None):
        self.df = df.reset_index(drop=True)
        if max_rows is not None:
            self.df = self.df.iloc[:max_rows].reset_index(drop=True)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["img_path"]).convert("RGB")
        img_t = self.preprocess(img)
        return img_t, list(row["captions"]), str(row["img_id"])


def eval_collate(batch):
    """Custom collate: stack images, keep lists of captions / ids as Python lists."""
    import torch

    imgs, caps_lists, ids = zip(*batch)
    return torch.stack(list(imgs), dim=0), list(caps_lists), list(ids)


# ---------------------------------------------------------------------------
# Datasets — cached-feature path (the default, fastest training path)
# ---------------------------------------------------------------------------


@dataclass
class CachedFeatures:
    """A bundle of pre-extracted SigLIP features for one split."""

    img_feat: np.ndarray            # (N_img, 768) float16
    txt_feat: np.ndarray            # (N_img * 5, 768) float16
    img_ids: np.ndarray             # (N_img,) string-typed
    txt_to_img: np.ndarray          # (N_img * 5,) int — index into img_feat
    captions: np.ndarray            # (N_img * 5,) string-typed (kept for qualitative examples)

    @classmethod
    def load(cls, npz_path: Path) -> "CachedFeatures":
        data = np.load(npz_path, allow_pickle=True)
        return cls(
            img_feat=data["img_feat"],
            txt_feat=data["txt_feat"],
            img_ids=data["img_ids"],
            txt_to_img=data["txt_to_img"],
            captions=data["captions"],
        )

    def save(self, npz_path: Path) -> None:
        np.savez_compressed(
            npz_path,
            img_feat=self.img_feat.astype(np.float16),
            txt_feat=self.txt_feat.astype(np.float16),
            img_ids=self.img_ids,
            txt_to_img=self.txt_to_img.astype(np.int64),
            captions=self.captions,
        )


class CachedFeaturePairs(Dataset):
    """Training-pairs dataset over pre-extracted features (the cached fast path).

    Yields ``(img_vec, txt_vec)`` pairs as ``torch.float32`` tensors so the
    adapter receives stable precision regardless of the cache's storage dtype.
    """

    def __init__(self, feats: CachedFeatures, *, max_rows: Optional[int] = None,
                 seed: int = 0):
        self.feats = feats
        self._rng = random.Random(seed)
        # Each image owns 5 caption rows. Build a flat list of image indices.
        n_img = feats.img_feat.shape[0]
        self._idx_list = np.arange(n_img if max_rows is None else min(n_img, max_rows))
        # Pre-index image -> its caption row positions ONCE (O(N_text) total),
        # so __getitem__ is O(1) instead of scanning txt_to_img on every sample.
        # (A full scan per sample turns the "fast" cached path into an
        # O(N_img * N_text) bottleneck over a training run.)
        self._caps_by_img: Dict[int, np.ndarray] = {}
        for pos, img_idx in enumerate(feats.txt_to_img.tolist()):
            self._caps_by_img.setdefault(int(img_idx), []).append(pos)
        self._caps_by_img = {k: np.asarray(v, dtype=np.int64)
                             for k, v in self._caps_by_img.items()}

    def __len__(self) -> int:
        return int(len(self._idx_list))

    def __getitem__(self, idx: int):
        import torch

        img_idx = int(self._idx_list[idx])
        img_vec = torch.from_numpy(self.feats.img_feat[img_idx].astype(np.float32))
        # Sample one of this image's caption rows (precomputed; no per-call scan).
        cap_positions = self._caps_by_img[img_idx]
        cap_pos = int(cap_positions[self._rng.randrange(len(cap_positions))])
        txt_vec = torch.from_numpy(self.feats.txt_feat[cap_pos].astype(np.float32))
        return img_vec, txt_vec


# ---------------------------------------------------------------------------
# DataLoader configuration helper
# ---------------------------------------------------------------------------


def make_loader_kwargs(cfg: Dict[str, Any], *, train: bool = True) -> Dict[str, Any]:
    """Return DataLoader kwargs honoring the Windows-vs-Linux ``num_workers`` policy."""
    nw = utils.default_num_workers(cfg)
    kwargs: Dict[str, Any] = dict(
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=nw,
        shuffle=train,
        drop_last=train,
        pin_memory=True,
        persistent_workers=nw > 0,
    )
    if nw > 0:
        kwargs["worker_init_fn"] = utils.worker_init_fn
    return kwargs

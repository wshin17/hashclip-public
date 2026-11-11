"""Encoding routines: pre-extract continuous features and dump hash codes.

Two main entry points:

* :func:`encode_backbone_all_splits` — used by ``hashclip.cli encode-backbone``.
  Reads each split parquet, encodes images + captions with the frozen SigLIP
  backbone, and saves them to ``outputs/features/backbone_{split}.npz``.

* :func:`encode_test_with_adapter`   — used by ``hashclip.cli encode-all``.
  Loads the cached test-split features and a trained adapter checkpoint;
  writes float16 continuous matrices and uint8-packed hash codes to
  ``outputs/features/hashclip_b{B}_seed{S}.npz``.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from . import data, model, utils


# ---------------------------------------------------------------------------
# Image / text iteration helpers
# ---------------------------------------------------------------------------


class _ImagePathDataset(Dataset):
    """Yields preprocessed image tensors from a list of paths."""

    def __init__(self, paths: List[str], preprocess):
        self.paths = list(paths)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.preprocess(img)


def _encode_images(backbone: model.BackboneBundle, paths: List[str],
                   device: str, batch_size: int = 64,
                   num_workers: int = 0, desc: str = "img") -> np.ndarray:
    ds = _ImagePathDataset(paths, backbone.preprocess)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                    shuffle=False, pin_memory=device == "cuda")
    out = []
    backbone.model.to(device).eval()
    for batch in tqdm(dl, desc=f"encode-img/{desc}"):
        feats = model.encode_image_batch(backbone, batch, device=device)
        out.append(feats.cpu().to(torch.float16).numpy())
    return np.concatenate(out, axis=0)


def _encode_texts(backbone: model.BackboneBundle, texts: List[str],
                  device: str, batch_size: int = 256,
                  desc: str = "txt") -> np.ndarray:
    """Encode a flat list of caption strings in fixed-size batches."""
    out: List[np.ndarray] = []
    backbone.model.to(device).eval()
    for i in tqdm(range(0, len(texts), batch_size), desc=f"encode-txt/{desc}"):
        chunk = texts[i:i + batch_size]
        feats = model.encode_text_batch(backbone, chunk, device=device)
        out.append(feats.cpu().to(torch.float16).numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, backbone.embed_dim),
                                                            dtype=np.float16)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def encode_backbone_split(backbone: model.BackboneBundle,
                          df: pd.DataFrame,
                          *,
                          device: str = "cuda",
                          batch_size_img: int = 64,
                          batch_size_txt: int = 256,
                          num_workers: int = 0) -> data.CachedFeatures:
    """Encode every image + every caption in ``df`` -> :class:`CachedFeatures`."""
    paths = df["img_path"].astype(str).tolist()
    img_ids = np.asarray(df["img_id"].astype(str).tolist())

    img_feat = _encode_images(backbone, paths, device=device,
                              batch_size=batch_size_img, num_workers=num_workers,
                              desc=f"{len(paths)} imgs")

    # Flatten 5 captions per image; remember which image each caption belongs to.
    flat_caps: List[str] = []
    txt_to_img: List[int] = []
    for i, caps in enumerate(df["captions"].tolist()):
        for c in caps:
            flat_caps.append(str(c))
            txt_to_img.append(i)
    txt_feat = _encode_texts(backbone, flat_caps, device=device,
                             batch_size=batch_size_txt,
                             desc=f"{len(flat_caps)} caps")

    return data.CachedFeatures(
        img_feat=img_feat,
        txt_feat=txt_feat,
        img_ids=img_ids,
        txt_to_img=np.asarray(txt_to_img, dtype=np.int64),
        captions=np.asarray(flat_caps, dtype=object),
    )


def encode_backbone_all_splits(cfg: Dict, *,
                               device: Optional[str] = None,
                               splits: Iterable[str] = ("train", "val", "test"),
                               batch_size_img: int = 64,
                               batch_size_txt: int = 256,
                               num_workers: Optional[int] = None,
                               overwrite: bool = False) -> Dict[str, Path]:
    """Top-level entry: encode every split, write npz files, return their paths."""
    logger = utils.get_logger("hashclip.encode")
    processed_dir = utils.get_path(cfg, "data_processed")
    feat_dir = utils.get_path(cfg, "features")
    if num_workers is None:
        num_workers = utils.default_num_workers(cfg)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    backbone_loaded = False
    backbone: Optional[model.BackboneBundle] = None
    out: Dict[str, Path] = {}

    for split in splits:
        target = feat_dir / f"backbone_{split}.npz"
        if target.exists() and not overwrite:
            logger.info("skip (exists): %s", target)
            out[split] = target
            continue
        if not backbone_loaded:
            logger.info("Loading backbone %s ...", cfg["backbone"]["hf_hub"])
            backbone = model.load_backbone(cfg["backbone"]["hf_hub"])
            backbone_loaded = True
        df = data.load_split(processed_dir, split)
        logger.info("Encoding split=%s (%d images)", split, len(df))
        feats = encode_backbone_split(backbone, df, device=device,
                                      batch_size_img=batch_size_img,
                                      batch_size_txt=batch_size_txt,
                                      num_workers=num_workers)
        feats.save(target)
        out[split] = target
        logger.info("Wrote %s  (img=%s, txt=%s)", target,
                    feats.img_feat.shape, feats.txt_feat.shape)
    return out


# ---------------------------------------------------------------------------
# Adapter encoding (hash codes for the test split)
# ---------------------------------------------------------------------------


@dataclass
class HashEncodingResult:
    bits: int
    seed: int
    img_feat: np.ndarray          # (N_img, 768) float16   — continuous
    txt_feat: np.ndarray          # (N_txt, 768) float16
    img_hash_packed: np.ndarray   # (N_img, B/8) uint8
    txt_hash_packed: np.ndarray   # (N_txt, B/8) uint8
    img_ids: np.ndarray
    txt_to_img: np.ndarray
    captions: np.ndarray

    def save(self, npz_path: Path) -> None:
        np.savez_compressed(
            npz_path,
            bits=self.bits,
            seed=self.seed,
            img_feat=self.img_feat.astype(np.float16),
            txt_feat=self.txt_feat.astype(np.float16),
            img_hash_packed=self.img_hash_packed.astype(np.uint8),
            txt_hash_packed=self.txt_hash_packed.astype(np.uint8),
            img_ids=self.img_ids,
            txt_to_img=self.txt_to_img.astype(np.int64),
            captions=self.captions,
        )

    @classmethod
    def load(cls, npz_path: Path) -> "HashEncodingResult":
        d = np.load(npz_path, allow_pickle=True)
        return cls(
            bits=int(d["bits"]),
            seed=int(d["seed"]),
            img_feat=d["img_feat"],
            txt_feat=d["txt_feat"],
            img_hash_packed=d["img_hash_packed"],
            txt_hash_packed=d["txt_hash_packed"],
            img_ids=d["img_ids"],
            txt_to_img=d["txt_to_img"],
            captions=d["captions"],
        )


def encode_test_with_adapter(cfg: Dict, ckpt_path: Path, *,
                             device: Optional[str] = None) -> HashEncodingResult:
    """Apply a trained adapter to the cached test features -> hash codes + cont."""
    feat_dir = utils.get_path(cfg, "features")
    test_npz = feat_dir / "backbone_test.npz"
    if not test_npz.exists():
        raise FileNotFoundError(
            f"{test_npz} missing — run `python -m hashclip.cli encode-backbone` first."
        )
    cached = data.CachedFeatures.load(test_npz)

    retr, ckpt = model.RetrievalModel.from_checkpoint(ckpt_path)
    seed = int(ckpt.get("config", {}).get("train", {}).get("seed", -1))
    bits = int(retr.bits)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    retr.to(device).eval()

    img_t = torch.from_numpy(cached.img_feat.astype(np.float32)).to(device)
    txt_t = torch.from_numpy(cached.txt_feat.astype(np.float32)).to(device)

    with torch.inference_mode():
        img_h = retr.img_adapter(img_t).cpu().numpy()
        txt_h = retr.txt_adapter(txt_t).cpu().numpy()

    img_packed = utils.pack_bits(img_h)
    txt_packed = utils.pack_bits(txt_h)

    return HashEncodingResult(
        bits=bits, seed=seed,
        img_feat=cached.img_feat, txt_feat=cached.txt_feat,
        img_hash_packed=img_packed, txt_hash_packed=txt_packed,
        img_ids=cached.img_ids, txt_to_img=cached.txt_to_img,
        captions=cached.captions,
    )


def encode_all_checkpoints(cfg: Dict, *, device: Optional[str] = None) -> List[Path]:
    """Encode every checkpoint in ``outputs/checkpoints``."""
    logger = utils.get_logger("hashclip.encode")
    ckpt_dir = utils.get_path(cfg, "checkpoints")
    feat_dir = utils.get_path(cfg, "features")
    out: List[Path] = []
    for ckpt_path in sorted(ckpt_dir.glob("*.pt")):
        result = encode_test_with_adapter(cfg, ckpt_path, device=device)
        target = feat_dir / f"{ckpt_path.stem}.npz"
        result.save(target)
        out.append(target)
        logger.info(
            "Encoded %s -> %s  (bits=%d, img=%s, txt=%s)",
            ckpt_path.name, target.name, result.bits,
            result.img_hash_packed.shape, result.txt_hash_packed.shape,
        )
    if not out:
        logger.warning("No checkpoints found in %s", ckpt_dir)
    return out

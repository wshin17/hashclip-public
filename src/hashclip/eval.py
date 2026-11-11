"""End-to-end evaluation — computes Recall@K, latency, storage for every method.

Output:
    outputs/tables/results_main.csv
    outputs/tables/ablation_bits.csv
    outputs/tables/ablation_M.csv
    outputs/tables/ablation_loss.csv
    outputs/tables/ablation_adapter.csv
    outputs/tables/RESULTS.md
"""

from __future__ import annotations

import csv
import json
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import encode, retrieve, utils


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


CSV_FIELDS = [
    "method", "bits", "M", "seed",
    "R1_t2i", "R5_t2i", "R10_t2i",
    "R1_i2t", "R5_i2t", "R10_i2t",
    "rsum",
    "latency_stage1_ms", "latency_stage2_ms", "latency_total_ms",
    "storage_bytes_per_item", "storage_index_bytes_per_item",
]


@dataclass
class Row:
    method: str
    bits: int = 0
    M: int = 0
    seed: int = -1
    R1_t2i: float = 0.0
    R5_t2i: float = 0.0
    R10_t2i: float = 0.0
    R1_i2t: float = 0.0
    R5_i2t: float = 0.0
    R10_i2t: float = 0.0
    rsum: float = 0.0
    latency_stage1_ms: float = 0.0
    latency_stage2_ms: float = 0.0
    latency_total_ms: float = 0.0
    storage_bytes_per_item: int = 0          # honest TOTAL footprint per item
    storage_index_bytes_per_item: int = 0    # stage-1 searchable index only

    def as_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in CSV_FIELDS}


# ---------------------------------------------------------------------------
# Recall computation
# ---------------------------------------------------------------------------


def _recall_t2i(final_idx: np.ndarray, txt_to_img: np.ndarray,
                topk: Iterable[int]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    gt = txt_to_img.reshape(-1, 1)
    for k in topk:
        hits = (final_idx[:, :k] == gt).any(axis=1)
        out[f"R{k}_t2i"] = float(100.0 * hits.mean())
    return out


def _recall_i2t(final_idx: np.ndarray, txt_to_img: np.ndarray,
                topk: Iterable[int]) -> Dict[str, float]:
    # Build {img_idx: set(caption_idx)} once
    img_to_caps: Dict[int, set] = defaultdict(set)
    for cap_idx, img_idx in enumerate(txt_to_img):
        img_to_caps[int(img_idx)].add(int(cap_idx))
    out: Dict[str, float] = {}
    for k in topk:
        hits = []
        for qi, row in enumerate(final_idx):
            relevant = img_to_caps.get(qi, set())
            hits.append(any(int(c) in relevant for c in row[:k]))
        out[f"R{k}_i2t"] = float(100.0 * np.mean(hits))
    return out


def _populate_recall(row: Row, *, t2i_idx: np.ndarray, i2t_idx: np.ndarray,
                     txt_to_img: np.ndarray, topk: Iterable[int]) -> None:
    t = _recall_t2i(t2i_idx, txt_to_img, topk)
    i = _recall_i2t(i2t_idx, txt_to_img, topk)
    for k in topk:
        setattr(row, f"R{k}_t2i", t[f"R{k}_t2i"])
        setattr(row, f"R{k}_i2t", i[f"R{k}_i2t"])
    row.rsum = float(sum(t.values()) + sum(i.values()))


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------


def _run_cont_exact(enc: encode.HashEncodingResult, topk: List[int],
                    warmup: int) -> Row:
    dim_bytes = enc.img_feat.shape[1] * 4  # float32 continuous vector
    row = Row(method="cont_exact", bits=0, M=0, seed=enc.seed,
              storage_bytes_per_item=dim_bytes,            # stores float vectors
              storage_index_bytes_per_item=dim_bytes)      # and scans them
    K = max(topk)
    t2i = retrieve.cosine_exact_timed(enc.txt_feat.astype(np.float32),
                                      enc.img_feat.astype(np.float32),
                                      K=K, warmup=warmup)
    i2t = retrieve.cosine_exact_timed(enc.img_feat.astype(np.float32),
                                      enc.txt_feat.astype(np.float32),
                                      K=K, warmup=warmup)
    _populate_recall(row, t2i_idx=t2i.final, i2t_idx=i2t.final,
                     txt_to_img=enc.txt_to_img, topk=topk)
    row.latency_stage2_ms = 0.5 * (t2i.stage2_ms + i2t.stage2_ms)
    row.latency_total_ms = row.latency_stage2_ms
    return row


def _run_faiss_ivf(enc: encode.HashEncodingResult, topk: List[int],
                   warmup: int) -> Optional[Row]:
    try:
        K = max(topk)
        dim_bytes = enc.img_feat.shape[1] * 4
        row = Row(method="cont_faiss_ivf", bits=0, M=0, seed=enc.seed,
                  storage_bytes_per_item=dim_bytes,
                  storage_index_bytes_per_item=dim_bytes)
        t2i = retrieve.faiss_ivf_timed(enc.txt_feat.astype(np.float32),
                                       enc.img_feat.astype(np.float32),
                                       K=K, warmup=warmup)
        i2t = retrieve.faiss_ivf_timed(enc.img_feat.astype(np.float32),
                                       enc.txt_feat.astype(np.float32),
                                       K=K, warmup=warmup)
        _populate_recall(row, t2i_idx=t2i.final, i2t_idx=i2t.final,
                         txt_to_img=enc.txt_to_img, topk=topk)
        row.latency_stage2_ms = 0.5 * (t2i.stage2_ms + i2t.stage2_ms)
        row.latency_total_ms = row.latency_stage2_ms
        return row
    except Exception as e:
        utils.get_logger("hashclip.eval").warning("faiss_ivf failed: %s", e)
        return None


def _run_random_lsh(enc: encode.HashEncodingResult, topk: List[int],
                    bits: int, warmup: int, seed: int = 0) -> Row:
    img_packed = retrieve.random_lsh_pack(enc.img_feat.astype(np.float32), bits, seed=seed)
    txt_packed = retrieve.random_lsh_pack(enc.txt_feat.astype(np.float32), bits, seed=seed)
    row = Row(method="random_lsh", bits=bits, M=0, seed=enc.seed,
              storage_bytes_per_item=bits // 8,            # binary codes only
              storage_index_bytes_per_item=bits // 8)
    K = max(topk)
    t2i = retrieve.hamming_only_timed(txt_packed, img_packed, K=K, warmup=warmup)
    i2t = retrieve.hamming_only_timed(img_packed, txt_packed, K=K, warmup=warmup)
    _populate_recall(row, t2i_idx=t2i.final, i2t_idx=i2t.final,
                     txt_to_img=enc.txt_to_img, topk=topk)
    row.latency_stage1_ms = 0.5 * (t2i.stage1_ms + i2t.stage1_ms)
    row.latency_total_ms = row.latency_stage1_ms
    return row


def _run_hash_only(enc: encode.HashEncodingResult, topk: List[int],
                   warmup: int) -> Row:
    row = Row(method="hashclip_hash_only", bits=enc.bits, M=0, seed=enc.seed,
              storage_bytes_per_item=enc.bits // 8,        # pure binary: no rerank
              storage_index_bytes_per_item=enc.bits // 8)
    K = max(topk)
    t2i = retrieve.hamming_only_timed(enc.txt_hash_packed, enc.img_hash_packed,
                                      K=K, warmup=warmup)
    i2t = retrieve.hamming_only_timed(enc.img_hash_packed, enc.txt_hash_packed,
                                      K=K, warmup=warmup)
    _populate_recall(row, t2i_idx=t2i.final, i2t_idx=i2t.final,
                     txt_to_img=enc.txt_to_img, topk=topk)
    row.latency_stage1_ms = 0.5 * (t2i.stage1_ms + i2t.stage1_ms)
    row.latency_total_ms = row.latency_stage1_ms
    return row


def _run_asymmetric(enc: encode.HashEncodingResult, topk: List[int],
                    M: int, warmup: int) -> Row:
    # Honest storage: stage-1 scans the binary index (bits//8), but stage-2
    # cosine rerank reads the ORIGINAL continuous vectors, so the gallery must
    # also keep them. Total footprint = binary index + float vectors.
    index_bytes = enc.bits // 8
    dim_bytes = enc.img_feat.shape[1] * 4
    row = Row(method="hashclip_asym", bits=enc.bits, M=M, seed=enc.seed,
              storage_bytes_per_item=index_bytes + dim_bytes,
              storage_index_bytes_per_item=index_bytes)
    K = max(topk)
    M_img = min(M, enc.img_hash_packed.shape[0])
    M_txt = min(M, enc.txt_hash_packed.shape[0])
    t2i = retrieve.asymmetric_search_timed(
        enc.txt_hash_packed, enc.txt_feat.astype(np.float32),
        enc.img_hash_packed, enc.img_feat.astype(np.float32),
        M=M_img, K=K, warmup=warmup,
    )
    i2t = retrieve.asymmetric_search_timed(
        enc.img_hash_packed, enc.img_feat.astype(np.float32),
        enc.txt_hash_packed, enc.txt_feat.astype(np.float32),
        M=M_txt, K=K, warmup=warmup,
    )
    _populate_recall(row, t2i_idx=t2i.final, i2t_idx=i2t.final,
                     txt_to_img=enc.txt_to_img, topk=topk)
    row.latency_stage1_ms = 0.5 * (t2i.stage1_ms + i2t.stage1_ms)
    row.latency_stage2_ms = 0.5 * (t2i.stage2_ms + i2t.stage2_ms)
    row.latency_total_ms = row.latency_stage1_ms + row.latency_stage2_ms
    return row


# ---------------------------------------------------------------------------
# CSV / Markdown writing
# ---------------------------------------------------------------------------


def _write_csv(rows: List[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r.as_dict())


def _aggregate_seeds(rows: List[Row]) -> pd.DataFrame:
    """Mean+std across seeds for the headline numbers."""
    df = pd.DataFrame([r.as_dict() for r in rows])
    num = [c for c in df.columns if c not in ("method", "bits", "M", "seed")]
    by = df.groupby(["method", "bits", "M"], dropna=False)
    mean = by[num].mean().reset_index()
    std = by[num].std(ddof=0).reset_index()
    out = mean.merge(std, on=["method", "bits", "M"], suffixes=("_mean", "_std"))
    return out


def _to_markdown(df: pd.DataFrame, path: Path, title: str = "Headline results") -> None:
    """Render a Markdown summary table of the headline numbers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Method", "Bits", "M",
               "R@1 T->I", "R@5 T->I", "R@10 T->I",
               "R@1 I->T", "R@5 I->T", "R@10 I->T",
               "rsum", "Storage total (B/item)", "Index (B/item)", "Latency (ms)"]
    lines: List[str] = [f"# {title}", "",
                        "| " + " | ".join(headers) + " |",
                        "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, r in df.iterrows():
        def fmt(col):
            m = r.get(f"{col}_mean", r.get(col, ""))
            s = r.get(f"{col}_std", "")
            try:
                if isinstance(s, float) and not np.isnan(s) and s > 0:
                    return f"{m:.2f} ± {s:.2f}"
                return f"{float(m):.2f}"
            except Exception:
                return str(m)
        row = [
            str(r["method"]), str(int(r["bits"])), str(int(r["M"])),
            fmt("R1_t2i"), fmt("R5_t2i"), fmt("R10_t2i"),
            fmt("R1_i2t"), fmt("R5_i2t"), fmt("R10_i2t"),
            fmt("rsum"),
            str(int(r.get("storage_bytes_per_item_mean", r.get("storage_bytes_per_item", 0)))),
            str(int(r.get("storage_index_bytes_per_item_mean",
                          r.get("storage_index_bytes_per_item", 0)))),
            fmt("latency_total_ms"),
        ]
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Driver: evaluate everything
# ---------------------------------------------------------------------------


_NPZ_RE = re.compile(r"hashclip_b(?P<bits>\d+)_seed(?P<seed>\d+)_(?P<tag>[a-zA-Z0-9._\-]+)\.npz$")


def _list_encodings(feat_dir: Path) -> List[Path]:
    return sorted(p for p in feat_dir.glob("hashclip_b*_seed*.npz"))


def _parse_meta(path: Path) -> Dict[str, Any]:
    """Extract bits/seed/tag from a filename (best-effort)."""
    m = _NPZ_RE.search(path.name)
    if not m:
        return {"bits": -1, "seed": -1, "tag": ""}
    return {"bits": int(m["bits"]), "seed": int(m["seed"]), "tag": m["tag"]}


def evaluate_all(cfg: Dict[str, Any], *, warmup: Optional[int] = None) -> Dict[str, Path]:
    """Run every method on every checkpoint -> CSV/MD outputs.

    Returns a dict of written file paths.
    """
    logger = utils.get_logger("hashclip.eval")
    feat_dir = utils.get_path(cfg, "features")
    tables_dir = utils.get_path(cfg, "tables")
    warmup = int(cfg["eval"]["warmup_queries"]) if warmup is None else warmup
    topk = list(cfg["eval"]["topk"])

    enc_paths = _list_encodings(feat_dir)
    if not enc_paths:
        raise FileNotFoundError(
            f"No hashclip_*.npz in {feat_dir}; run "
            "`python -m hashclip.cli encode-all` first."
        )

    # ---- Main table: per-checkpoint x 5 methods --------------------------
    main_rows: List[Row] = []
    main_M = int(cfg["eval"]["M"])
    main_bits = int(cfg["model"]["bits"])

    # Run cont_exact / faiss_ivf once per *cached test features* (they don't depend
    # on the adapter). We still record one row per seed for clean groupby.
    seen_baselines = set()
    for ep in enc_paths:
        meta = _parse_meta(ep)
        if meta.get("tag", "") != "main" or int(meta.get("bits", -1)) != main_bits:
            continue
        enc = encode.HashEncodingResult.load(ep)
        seed = int(enc.seed)
        if seed not in seen_baselines:
            row_cont = _run_cont_exact(enc, topk, warmup)
            main_rows.append(row_cont)
            row_faiss = _run_faiss_ivf(enc, topk, warmup)
            if row_faiss is not None:
                main_rows.append(row_faiss)
            row_lsh = _run_random_lsh(enc, topk, bits=main_bits, warmup=warmup,
                                      seed=seed)
            main_rows.append(row_lsh)
            seen_baselines.add(seed)

        main_rows.append(_run_hash_only(enc, topk, warmup))
        main_rows.append(_run_asymmetric(enc, topk, M=main_M, warmup=warmup))
        logger.info("evaluated %s (main)", ep.name)

    main_csv = tables_dir / "results_main.csv"
    _write_csv(main_rows, main_csv)
    if main_rows:
        agg = _aggregate_seeds(main_rows)
        _to_markdown(agg, tables_dir / "RESULTS.md",
                     title="HashCLIP — headline results")

    # ---- Ablation: bits --------------------------------------------------
    abl_bits_rows: List[Row] = []
    for ep in enc_paths:
        meta = _parse_meta(ep)
        if not meta.get("tag", "").startswith("abl-bits"):
            continue
        enc = encode.HashEncodingResult.load(ep)
        abl_bits_rows.append(_run_asymmetric(enc, topk, M=main_M, warmup=warmup))
        abl_bits_rows.append(_run_hash_only(enc, topk, warmup))
        logger.info("evaluated %s (abl-bits)", ep.name)
    _write_csv(abl_bits_rows, tables_dir / "ablation_bits.csv")

    # ---- Ablation: M (re-rank pool size) --------------------------------
    abl_M_rows: List[Row] = []
    M_values = list(cfg["ablation"]["M"])
    main_paths = [p for p in enc_paths
                  if _parse_meta(p).get("tag", "") == "main"
                  and int(_parse_meta(p).get("bits", -1)) == main_bits]
    for ep in main_paths:
        enc = encode.HashEncodingResult.load(ep)
        for M in M_values:
            abl_M_rows.append(_run_asymmetric(enc, topk, M=int(M), warmup=warmup))
        logger.info("swept M over %s on %s", M_values, ep.name)
    _write_csv(abl_M_rows, tables_dir / "ablation_M.csv")

    # ---- Ablation: loss variant ------------------------------------------
    abl_loss_rows: List[Row] = []
    for ep in enc_paths:
        meta = _parse_meta(ep)
        if not meta.get("tag", "").startswith("abl-loss"):
            continue
        enc = encode.HashEncodingResult.load(ep)
        row = _run_asymmetric(enc, topk, M=main_M, warmup=warmup)
        # Embed the variant tag in `method` for traceability.
        row.method = f"hashclip_asym__{meta['tag']}"
        abl_loss_rows.append(row)
        # Also record the pure-hash (no-rerank) result. The loss components act
        # on the binary CODE quality; with M >= gallery size the stage-2 cosine
        # rerank recovers exact-cosine ranking and masks those differences, so
        # the hash-only metric is what actually reveals the ablation's effect.
        row_hash = _run_hash_only(enc, topk, warmup)
        row_hash.method = f"hashclip_hash_only__{meta['tag']}"
        abl_loss_rows.append(row_hash)
        logger.info("evaluated %s (abl-loss)", ep.name)
    _write_csv(abl_loss_rows, tables_dir / "ablation_loss.csv")

    # ---- Ablation: adapter capacity --------------------------------------
    abl_adapter_rows: List[Row] = []
    for ep in enc_paths:
        meta = _parse_meta(ep)
        if not meta.get("tag", "").startswith("abl-h"):
            continue
        enc = encode.HashEncodingResult.load(ep)
        row = _run_asymmetric(enc, topk, M=main_M, warmup=warmup)
        row.method = f"hashclip_asym__{meta['tag']}"
        abl_adapter_rows.append(row)
        # Pure-hash result too — adapter capacity changes code quality, which is
        # visible without the rerank stage (see the loss-ablation note above).
        row_hash = _run_hash_only(enc, topk, warmup)
        row_hash.method = f"hashclip_hash_only__{meta['tag']}"
        abl_adapter_rows.append(row_hash)
        logger.info("evaluated %s (abl-adapter)", ep.name)
    _write_csv(abl_adapter_rows, tables_dir / "ablation_adapter.csv")

    written = {
        "main": main_csv,
        "results_md": tables_dir / "RESULTS.md",
        "ablation_bits": tables_dir / "ablation_bits.csv",
        "ablation_M": tables_dir / "ablation_M.csv",
        "ablation_loss": tables_dir / "ablation_loss.csv",
        "ablation_adapter": tables_dir / "ablation_adapter.csv",
    }
    logger.info("wrote: %s", json.dumps({k: str(v) for k, v in written.items()}))
    return written

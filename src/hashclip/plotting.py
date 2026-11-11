"""Figure generation for the evaluation results.

Every plot is written both as a 300 dpi ``*.png`` and as a vector ``*.pdf``
into ``outputs/plots/``. ``matplotlib.use('Agg')`` keeps everything headless,
so the figures render the same on a server without a display as they do on a
desktop.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # noqa: E402 — must precede pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

try:
    import seaborn as sns  # type: ignore

    sns.set_style("whitegrid")
    PALETTE = sns.color_palette("colorblind")
except Exception:
    sns = None
    PALETTE = [
        "#0173b2", "#de8f05", "#029e73", "#d55e00", "#cc78bc",
        "#ca9161", "#fbafe4", "#949494", "#ece133", "#56b4e9",
    ]


# Larger, slightly heavier defaults so the figures stay legible when scaled
# down. Tight bbox + zero pad trims the surrounding whitespace from each file.
plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 19,
    "axes.titleweight": "bold",
    "axes.labelsize": 17,
    "axes.labelweight": "medium",
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_both(fig, stem: Path) -> Tuple[Path, Path]:
    """Write the figure as both a PNG and a PDF; return the two paths."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except pd.errors.EmptyDataError:
        return None


def _color_for(idx: int):
    return PALETTE[idx % len(PALETTE)]


# ---------------------------------------------------------------------------
# Fig 1 — grouped bar of R@K (T->I and I->T) for the 5 methods
# ---------------------------------------------------------------------------


def fig1_main_results_bar(tables: Path, plots: Path) -> Optional[Path]:
    df = _load_csv(tables / "results_main.csv")
    if df is None:
        return None
    metrics = ["R1_t2i", "R5_t2i", "R10_t2i", "R1_i2t", "R5_i2t", "R10_i2t"]
    method_order = ["cont_exact", "cont_faiss_ivf", "random_lsh",
                    "hashclip_hash_only", "hashclip_asym"]
    method_order = [m for m in method_order if m in df["method"].unique()]
    agg = (df.groupby("method")[metrics]
             .agg(["mean", "std"]).reindex(method_order).fillna(0.0))

    x = np.arange(len(metrics))
    width = 0.8 / max(1, len(method_order))
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(method_order):
        means = [agg.loc[m, (k, "mean")] for k in metrics]
        stds = [agg.loc[m, (k, "std")] for k in metrics]
        ax.bar(x + (i - (len(method_order) - 1) / 2) * width, means,
               width=width, yerr=stds, capsize=2, label=m, color=_color_for(i))
    ax.set_xticks(x)
    ax.set_xticklabels(["R@1 T→I", "R@5 T→I", "R@10 T→I",
                        "R@1 I→T", "R@5 I→T", "R@10 I→T"])
    ax.set_ylabel("Recall (%)")
    ax.set_title("Recall@K on Flickr30k (Karpathy test)")
    ax.legend(loc="lower right", frameon=True)
    return _save_both(fig, plots / "fig1_main_results_bar")[0]


# ---------------------------------------------------------------------------
# Fig 2 — accuracy vs storage Pareto
# ---------------------------------------------------------------------------


def fig2_accuracy_vs_storage(tables: Path, plots: Path) -> Optional[Path]:
    main = _load_csv(tables / "results_main.csv")
    abl = _load_csv(tables / "ablation_bits.csv")
    if main is None:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))

    def _scatter(df, mask, label, marker, color, *, filled=True):
        sub = df[mask]
        if sub.empty:
            return
        kw = dict(s=70, marker=marker, label=label, color=color)
        if filled:  # unfilled markers ('x', '+') warn if given an edgecolor
            kw.update(edgecolors="k", linewidths=0.5)
        ax.scatter(sub["storage_bytes_per_item"], sub["R1_t2i"], **kw)

    # The genuine storage/accuracy frontier is the PURE binary hash (no rerank):
    # its footprint really is bits//8 per item. Sweep it over the bit budget.
    if abl is not None:
        sub = abl[abl["method"] == "hashclip_hash_only"].sort_values("bits")
        if not sub.empty:
            ax.plot(sub["storage_bytes_per_item"], sub["R1_t2i"], "-o",
                    color=_color_for(4),
                    label="HashCLIP hash-only @ {32,64,128,256} bits")
            for _, r in sub.iterrows():
                ax.annotate(f"{int(r['bits'])}b",
                            (r["storage_bytes_per_item"], r["R1_t2i"]),
                            xytext=(4, 4), textcoords="offset points", fontsize=13)

    _scatter(main, main["method"] == "hashclip_hash_only", "hash-only (B=128)",
             "o", _color_for(3))
    _scatter(main, main["method"] == "random_lsh", "random_lsh", "x",
             _color_for(2), filled=False)
    _scatter(main, main["method"] == "cont_exact", "cont_exact", "*", _color_for(0))

    # Asymmetric search reaches cont_exact accuracy, but its TOTAL footprint is
    # ~cont (it must also store the float vectors for stage-2 rerank). Plotting it
    # at its honest total storage makes clear the win is latency, not storage.
    asym = main[main["method"] == "hashclip_asym"]
    if not asym.empty:
        m_val = int(asym["M"].iloc[0])
        ax.scatter(asym["storage_bytes_per_item"], asym["R1_t2i"], s=95, marker="D",
                   color=_color_for(4), edgecolors="k", linewidths=0.5,
                   label=f"HashCLIP asym (M={m_val}, total storage)")

    ax.set_xscale("log")
    ax.set_xlabel("Total storage per item (bytes, log)")
    ax.set_ylabel("R@1 T→I (%)")
    ax.set_title("Accuracy vs Storage (total footprint)")
    ax.legend(loc="lower right", frameon=True, fontsize=12)
    return _save_both(fig, plots / "fig2_accuracy_vs_storage")[0]


# ---------------------------------------------------------------------------
# Fig 3 — accuracy vs latency
# ---------------------------------------------------------------------------


def fig3_accuracy_vs_latency(tables: Path, plots: Path) -> Optional[Path]:
    main = _load_csv(tables / "results_main.csv")
    abl_M = _load_csv(tables / "ablation_M.csv")
    if main is None:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))

    label_color = [
        ("cont_exact", _color_for(0)),
        ("cont_faiss_ivf", _color_for(1)),
        ("hashclip_hash_only", _color_for(3)),
        ("hashclip_asym", _color_for(4)),
    ]
    for i, (m, c) in enumerate(label_color):
        sub = main[main["method"] == m]
        if sub.empty:
            continue
        ax.scatter(sub["latency_total_ms"], sub["R1_t2i"],
                   s=70, marker="o", label=m, color=c, edgecolors="k",
                   linewidths=0.5)

    if abl_M is not None and not abl_M.empty:
        agg = abl_M.groupby("M")[["latency_total_ms", "R1_t2i"]].mean().reset_index()
        ax.plot(agg["latency_total_ms"], agg["R1_t2i"], "-o",
                color=_color_for(4), alpha=0.6,
                label="HashCLIP (sweep $M{=}50\\,...\\,5000$)")
        # Annotate only the two endpoints of the sweep; the points cluster
        # tightly in R@1, so labelling every M overlaps illegibly.
        for _, r in agg.iterrows():
            if int(r["M"]) not in (50, 5000):
                continue
            dy = 8 if int(r["M"]) == 50 else 8
            ha = "right" if int(r["M"]) == 5000 else "left"
            ax.annotate(f"$M{{=}}{int(r['M'])}$",
                        (r["latency_total_ms"], r["R1_t2i"]),
                        xytext=(0, dy), textcoords="offset points",
                        fontsize=12, ha=ha)

    ax.set_xscale("log")
    ax.set_xlabel("Total latency per query (ms, log)")
    ax.set_ylabel("R@1 T→I (%)")
    ax.set_title("Accuracy vs Latency")
    ax.legend(loc="lower right", frameon=True)
    return _save_both(fig, plots / "fig3_accuracy_vs_latency")[0]


# ---------------------------------------------------------------------------
# Fig 4 — re-rank pool curve
# ---------------------------------------------------------------------------


def fig4_rerank_pool_curve(tables: Path, plots: Path) -> Optional[Path]:
    abl_M = _load_csv(tables / "ablation_M.csv")
    main = _load_csv(tables / "results_main.csv")
    if abl_M is None or abl_M.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    agg = abl_M.groupby("M")[["R1_t2i"]].mean().reset_index()
    ax.plot(agg["M"], agg["R1_t2i"], "-o", color=_color_for(4),
            label="HashCLIP asym")

    if main is not None and "cont_exact" in main["method"].unique():
        baseline = main[main["method"] == "cont_exact"]["R1_t2i"].mean()
        ax.axhline(baseline, ls="--", color=_color_for(0),
                   label=f"cont_exact ({baseline:.1f}%)")

    ax.set_xscale("log")
    ax.set_xlabel("Re-rank pool size M (log)")
    ax.set_ylabel("R@1 T→I (%)")
    ax.set_title("Effect of Re-rank Pool Size M")
    ax.legend(loc="lower right", frameon=True)
    return _save_both(fig, plots / "fig4_rerank_pool_curve")[0]


# ---------------------------------------------------------------------------
# Fig 5 — loss-component ablation
# ---------------------------------------------------------------------------


def fig5_loss_ablation(tables: Path, plots: Path) -> Optional[Path]:
    df = _load_csv(tables / "ablation_loss.csv")
    if df is None or df.empty:
        return None
    # `method` is "hashclip_{asym,hash_only}__abl-loss-<variant>".
    df = df.copy()
    df["variant"] = df["method"].str.extract(r"abl-loss-(\S+)")[0]
    # The loss terms shape the binary CODE; with M >= gallery size the stage-2
    # rerank recovers exact cosine and hides those differences. So plot the
    # hash-only (pre-rerank) metric, which is what the ablation is really about.
    hash_only = df[df["method"].str.contains("hash_only")]
    use = hash_only if not hash_only.empty else df
    metric = "hash-only R@1" if not hash_only.empty else "asym R@1"
    order = [v for v in ("nce_only", "nce_quant", "nce_balance", "full")
             if v in set(use["variant"].dropna())]
    if not order:
        order = sorted(set(use["variant"].dropna()))
    agg = (use.groupby("variant")[["R1_t2i", "R1_i2t"]].mean()
              .reindex(order).reset_index())
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(agg))
    ax.bar(x - 0.2, agg["R1_t2i"], width=0.4, label=f"{metric} T→I",
           color=_color_for(0))
    ax.bar(x + 0.2, agg["R1_i2t"], width=0.4, label=f"{metric} I→T",
           color=_color_for(1))
    ax.set_xticks(x)
    ax.set_xticklabels(agg["variant"], rotation=10)
    ax.set_ylabel("R@1 (%)")
    ax.set_title("Loss-component ablation (hash-only, pre-rerank)")
    ax.legend(loc="lower right", frameon=True)
    return _save_both(fig, plots / "fig5_loss_ablation")[0]


# ---------------------------------------------------------------------------
# Fig 6 — bit distribution histogram
# ---------------------------------------------------------------------------


def fig6_bit_distribution(features_dir: Path, checkpoints_dir: Path,
                          plots: Path) -> Optional[Path]:
    """Histogram of pre-sign adapter outputs (sanity check that quant loss worked)."""
    # Re-derive pre-sign values from one main checkpoint applied to test features.
    import torch  # local

    from . import data as data_mod
    from . import model as model_mod

    main_ckpts = sorted(checkpoints_dir.glob("hashclip_b128_seed*_main.pt"))
    if not main_ckpts:
        # fall back to any checkpoint
        main_ckpts = sorted(checkpoints_dir.glob("*.pt"))
    if not main_ckpts:
        return None
    test_npz = features_dir / "backbone_test.npz"
    if not test_npz.exists():
        return None
    cached = data_mod.CachedFeatures.load(test_npz)

    retr, _ = model_mod.RetrievalModel.from_checkpoint(main_ckpts[0])
    retr.eval()
    with torch.no_grad():
        x = torch.from_numpy(cached.img_feat.astype(np.float32))
        h_img = retr.img_adapter(x).cpu().numpy().ravel()
        x = torch.from_numpy(cached.txt_feat.astype(np.float32))
        h_txt = retr.txt_adapter(x).cpu().numpy().ravel()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(h_img, bins=60, alpha=0.65, label="image hash outputs",
            color=_color_for(0))
    ax.hist(h_txt, bins=60, alpha=0.65, label="text hash outputs",
            color=_color_for(4))
    ax.set_xlabel("Tanh adapter output (pre-sign)")
    ax.set_ylabel("Count")
    ax.set_title("Pre-sign adapter output distribution")
    ax.legend(loc="upper center", frameon=True)
    return _save_both(fig, plots / "fig6_bit_distribution")[0]


# ---------------------------------------------------------------------------
# Fig 7 — qualitative retrievals
# ---------------------------------------------------------------------------


def fig7_qualitative_retrievals(features_dir: Path,
                                checkpoints_dir: Path,
                                processed_dir: Path,
                                plots: Path,
                                *, n_examples: int = 5,
                                rng_seed: int = 7) -> Optional[Path]:
    from . import data as data_mod
    from . import encode as encode_mod
    from . import retrieve as retrieve_mod
    from PIL import Image

    # Use the first main encoding file we can find.
    candidates = sorted(features_dir.glob("hashclip_b128_seed*_main.npz"))
    if not candidates:
        candidates = sorted(features_dir.glob("hashclip_b*_seed*_main.npz"))
    if not candidates:
        return None
    enc = encode_mod.HashEncodingResult.load(candidates[0])

    test_parquet = processed_dir / "test.parquet"
    if not test_parquet.exists():
        return None
    test_df = pd.read_parquet(test_parquet)
    test_df = test_df.set_index("img_id")

    rng = np.random.default_rng(rng_seed)
    cap_idx_pool = rng.choice(len(enc.captions), size=n_examples * 4, replace=False)
    chosen: List[int] = []
    for ci in cap_idx_pool:
        gt = int(enc.txt_to_img[ci])
        img_id = str(enc.img_ids[gt])
        if img_id in test_df.index:
            chosen.append(int(ci))
        if len(chosen) >= n_examples:
            break
    if not chosen:
        return None

    K = 5
    M = 1000
    cont_t2i = retrieve_mod.cosine_exact_topk(
        enc.txt_feat[chosen].astype(np.float32),
        enc.img_feat.astype(np.float32), k=K,
    )
    cand = retrieve_mod.hamming_topk(enc.txt_hash_packed[chosen],
                                     enc.img_hash_packed, k=min(M, enc.img_hash_packed.shape[0]))
    hc_t2i = retrieve_mod.rerank_cosine(enc.txt_feat[chosen].astype(np.float32),
                                        enc.img_feat.astype(np.float32),
                                        cand, k=K)

    # The left-hand column carries the query caption, so give it noticeably
    # more width than the square image cells; that leaves room for a larger,
    # more readable caption font that survives printing and down-scaling.
    ncols = K * 2 + 1
    width_ratios = [2.6] + [1.0] * (K * 2)
    cell = 2.2
    fig, axes = plt.subplots(
        len(chosen), ncols,
        figsize=(cell * sum(width_ratios), cell * len(chosen)),
        gridspec_kw={"width_ratios": width_ratios},
    )
    if len(chosen) == 1:
        axes = np.array([axes])
    for r, ci in enumerate(chosen):
        cap_text = str(enc.captions[ci])
        gt = int(enc.txt_to_img[ci])
        # Column 0 — caption text
        ax = axes[r, 0]
        ax.axis("off")
        if r == 0:
            ax.set_title("Query caption", fontsize=15)
        ax.text(0.0, 0.5, _wrap(cap_text, 22),
                fontsize=16, va="center", ha="left", wrap=True,
                linespacing=1.35)
        # Then K cont_exact retrievals, then K HashCLIP retrievals
        for j, src in enumerate([cont_t2i[r], hc_t2i[r]]):
            for kk in range(K):
                ax = axes[r, 1 + j * K + kk]
                ax.axis("off")
                img_idx = int(src[kk])
                img_id = str(enc.img_ids[img_idx])
                try:
                    p = Path(test_df.loc[img_id]["img_path"])
                    im = Image.open(p).convert("RGB")
                    ax.imshow(im)
                except Exception:
                    ax.text(0.5, 0.5, img_id, ha="center", va="center", fontsize=11)
                # Green border for the correct image
                if img_idx == gt:
                    for s in ("top", "bottom", "left", "right"):
                        ax.spines[s].set_visible(True)
                        ax.spines[s].set_linewidth(3)
                        ax.spines[s].set_color("#00aa00")
                if r == 0:
                    if j == 0 and kk == K // 2:
                        ax.set_title("cont_exact top-K", fontsize=15)
                    elif j == 1 and kk == K // 2:
                        ax.set_title("HashCLIP top-K", fontsize=15)
    return _save_both(fig, plots / "fig7_qualitative_retrievals")[0]


def _wrap(s: str, width: int) -> str:
    out = []
    line = ""
    for w in s.split():
        if len(line) + 1 + len(w) <= width:
            line = (line + " " + w).strip()
        else:
            out.append(line)
            line = w
    if line:
        out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Fig 8 — training curves (train loss + val R@1)
# ---------------------------------------------------------------------------


def fig8_training_curves(logs_dir: Path, plots: Path) -> Optional[Path]:
    log_files = sorted(logs_dir.glob("hashclip_b*_main.log.jsonl"))
    if not log_files:
        log_files = sorted(logs_dir.glob("*.log.jsonl"))
    if not log_files:
        return None

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4))
    for i, lp in enumerate(log_files):
        train_epochs, train_loss = [], []
        val_epochs, val_r1 = [], []
        with lp.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("event") == "epoch_train":
                    e = rec["epoch"]
                    # A log may concatenate several runs (e.g. a re-run appends
                    # to the same file). Detect an epoch reset and keep only the
                    # most recent run so the curve is not joined across runs.
                    if train_epochs and e <= train_epochs[-1]:
                        train_epochs.clear()
                        train_loss.clear()
                    train_epochs.append(e)
                    train_loss.append(rec["loss"])
                elif rec.get("event") == "epoch_val":
                    e = rec["epoch"]
                    if val_epochs and e <= val_epochs[-1]:
                        val_epochs.clear()
                        val_r1.clear()
                    val_epochs.append(e)
                    val_r1.append(0.5 * (rec["R1_t2i"] + rec["R1_i2t"]))
        label = lp.stem.replace(".log", "")
        axL.plot(train_epochs, train_loss, "-o", label=label,
                 color=_color_for(i), markersize=4)
        axR.plot(val_epochs, val_r1, "-o", label=label,
                 color=_color_for(i), markersize=4)
    axL.set_xlabel("Epoch"); axL.set_ylabel("Train loss"); axL.set_title("Training loss")
    axR.set_xlabel("Epoch"); axR.set_ylabel("Val R@1 avg (%)")
    axR.set_title("Validation R@1 (avg T→I, I→T)")
    axL.legend(loc="upper right", frameon=True, fontsize=11)
    axR.legend(loc="lower right", frameon=True, fontsize=11)
    return _save_both(fig, plots / "fig8_training_curves")[0]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def make_all_plots(cfg: Dict[str, Any]) -> Dict[str, Optional[Path]]:
    """Generate every figure from the evaluation tables and logs."""
    plots = Path(cfg["paths"]["plots"]); plots.mkdir(parents=True, exist_ok=True)
    tables = Path(cfg["paths"]["tables"])
    features = Path(cfg["paths"]["features"])
    checkpoints = Path(cfg["paths"]["checkpoints"])
    processed = Path(cfg["paths"]["data_processed"])
    logs = Path(cfg["paths"]["logs"])

    out: Dict[str, Optional[Path]] = {}
    out["fig1"] = fig1_main_results_bar(tables, plots)
    out["fig2"] = fig2_accuracy_vs_storage(tables, plots)
    out["fig3"] = fig3_accuracy_vs_latency(tables, plots)
    out["fig4"] = fig4_rerank_pool_curve(tables, plots)
    out["fig5"] = fig5_loss_ablation(tables, plots)
    try:
        out["fig6"] = fig6_bit_distribution(features, checkpoints, plots)
    except Exception as e:
        out["fig6"] = None
        print(f"[plotting] fig6 skipped: {e}")
    try:
        out["fig7"] = fig7_qualitative_retrievals(features, checkpoints,
                                                  processed, plots)
    except Exception as e:
        out["fig7"] = None
        print(f"[plotting] fig7 skipped: {e}")
    out["fig8"] = fig8_training_curves(logs, plots)
    return out

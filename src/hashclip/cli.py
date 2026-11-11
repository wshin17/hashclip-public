"""Command-line interface — ``python -m hashclip.cli <subcommand>``.

This is the canonical entry point. All shell / PowerShell scripts in
``scripts/`` are thin wrappers around the subcommands defined here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import utils


# ---------------------------------------------------------------------------
# Subcommand handlers — kept thin; heavy logic lives in the individual modules.
# ---------------------------------------------------------------------------


def cmd_check_env(args: argparse.Namespace) -> int:
    utils.check_env()
    return 0


def cmd_download_data(args: argparse.Namespace) -> int:
    from . import data as data_mod

    cfg = utils.load_config(args.config)
    raw_dir = utils.get_path(cfg, "data_raw")
    images_dir = data_mod.download_flickr30k(raw_dir)
    print(f"OK: extracted images -> {images_dir}")
    return 0


def cmd_build_splits(args: argparse.Namespace) -> int:
    from . import data as data_mod

    cfg = utils.load_config(args.config)
    raw_dir = utils.get_path(cfg, "data_raw")
    proc_dir = utils.get_path(cfg, "data_processed")
    out = data_mod.build_splits(raw_dir, proc_dir)
    for split, p in out.items():
        print(f"{split:5s} -> {p}")
    return 0


def cmd_encode_backbone(args: argparse.Namespace) -> int:
    from . import encode as encode_mod

    cfg = utils.load_config(args.config)
    out = encode_mod.encode_backbone_all_splits(
        cfg, splits=args.splits, overwrite=args.overwrite,
    )
    for s, p in out.items():
        print(f"{s:5s} -> {p}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from . import train as train_mod

    cfg = utils.load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = int(args.batch_size)

    result = train_mod.train_one(
        cfg,
        bits=int(args.bits),
        seed=int(args.seed),
        hidden_dim=int(args.hidden_dim) if args.hidden_dim is not None else None,
        loss_variant=args.loss_variant,
        max_train=args.max_train,
        max_val=args.max_val,
        tag=args.tag,
        use_cache=not args.no_cache,
    )
    print(f"OK: ckpt={result.ckpt_path}")
    print(f"     best_val={json.dumps(result.best_val)}")
    return 0


def cmd_train_ablation(args: argparse.Namespace) -> int:
    from . import train as train_mod

    cfg = utils.load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)
    out = train_mod.train_ablation(
        cfg, kind=args.kind, seed=int(args.seed),
        max_train=args.max_train, max_val=args.max_val,
        use_cache=not args.no_cache,
    )
    for label, res in out.items():
        print(f"{label:24s} -> {res.ckpt_path}")
    return 0


def cmd_encode_all(args: argparse.Namespace) -> int:
    from . import encode as encode_mod

    cfg = utils.load_config(args.config)
    paths = encode_mod.encode_all_checkpoints(cfg)
    for p in paths:
        print(p)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from . import eval as eval_mod

    cfg = utils.load_config(args.config)
    out = eval_mod.evaluate_all(cfg)
    for k, p in out.items():
        print(f"{k:18s} -> {p}")
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    from . import plotting as plot_mod

    cfg = utils.load_config(args.config)
    out = plot_mod.make_all_plots(cfg)
    for k, p in out.items():
        print(f"{k:8s} -> {p}")
    return 0


def cmd_run_all(args: argparse.Namespace) -> int:
    """Convenience: build splits, encode backbone, train (main+ablations),
    encode-all, evaluate, plot. Useful for the run_all scripts."""
    cfg = utils.load_config(args.config)
    seeds = list(cfg["ablation"]["seeds"])
    bits_main = int(cfg["model"]["bits"])

    # 1. backbone features
    args.splits = ("train", "val", "test")
    args.overwrite = False
    cmd_encode_backbone(args)

    # 2. main runs (3 seeds)
    for s in seeds:
        train_args = argparse.Namespace(
            config=args.config, bits=bits_main, seed=s, hidden_dim=None,
            loss_variant="full", max_train=None, max_val=None, tag="main",
            no_cache=False, epochs=None, batch_size=None,
        )
        cmd_train(train_args)

    # 3. ablations
    for kind in ("bits", "loss", "adapter"):
        abl_args = argparse.Namespace(
            config=args.config, kind=kind, seed=seeds[0],
            max_train=None, max_val=None, no_cache=False, epochs=None,
        )
        cmd_train_ablation(abl_args)

    # 4. encode + evaluate + plot
    cmd_encode_all(args)
    cmd_evaluate(args)
    cmd_plot(args)
    return 0


# ---------------------------------------------------------------------------
# Argparse plumbing
# ---------------------------------------------------------------------------


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, default=None,
                   help="Path to YAML config (default: configs/default.yaml).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m hashclip.cli",
        description="HashCLIP — cross-modal binary hashing with asymmetric re-ranking.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- check-env -------------------------------------------------------
    p = sub.add_parser("check-env", help="Verify torch / CUDA / env are correct.")
    _add_common(p)
    p.set_defaults(func=cmd_check_env)

    # ---- download-data ---------------------------------------------------
    p = sub.add_parser("download-data", help="Download Flickr30k from HF Hub.")
    _add_common(p)
    p.set_defaults(func=cmd_download_data)

    # ---- build-splits ----------------------------------------------------
    p = sub.add_parser("build-splits", help="Parse CSV -> train/val/test parquet.")
    _add_common(p)
    p.set_defaults(func=cmd_build_splits)

    # ---- encode-backbone -------------------------------------------------
    p = sub.add_parser("encode-backbone",
                       help="Cache SigLIP features for every split.")
    _add_common(p)
    p.add_argument("--splits", nargs="+",
                   default=["train", "val", "test"],
                   choices=["train", "val", "test"])
    p.add_argument("--overwrite", action="store_true",
                   help="Re-encode even if cache exists.")
    p.set_defaults(func=cmd_encode_backbone)

    # ---- train -----------------------------------------------------------
    p = sub.add_parser("train", help="Train HashCLIP adapters on cached features.")
    _add_common(p)
    p.add_argument("--bits", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--loss-variant", type=str, default="full",
                   choices=["nce_only", "nce_quant", "nce_balance", "full"])
    p.add_argument("--tag", type=str, default="main")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override config['train']['epochs'].")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-train", type=int, default=None,
                   help="Cap training set (for smoke test).")
    p.add_argument("--max-val", type=int, default=None)
    p.add_argument("--no-cache", action="store_true",
                   help="Live-encode from raw images instead of cached features.")
    p.set_defaults(func=cmd_train)

    # ---- train-ablation --------------------------------------------------
    p = sub.add_parser("train-ablation", help="Run an ablation sweep.")
    _add_common(p)
    p.add_argument("--kind", type=str, required=True,
                   choices=["bits", "loss", "adapter"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-val", type=int, default=None)
    p.add_argument("--no-cache", action="store_true")
    p.set_defaults(func=cmd_train_ablation)

    # ---- encode-all ------------------------------------------------------
    p = sub.add_parser("encode-all",
                       help="Encode test split with every trained adapter.")
    _add_common(p)
    p.set_defaults(func=cmd_encode_all)

    # ---- evaluate --------------------------------------------------------
    p = sub.add_parser("evaluate",
                       help="Evaluate every encoded checkpoint -> CSV / RESULTS.md.")
    _add_common(p)
    p.add_argument("--all", action="store_true", default=True,
                   help="Run the full evaluation matrix (default).")
    p.set_defaults(func=cmd_evaluate)

    # ---- plot ------------------------------------------------------------
    p = sub.add_parser("plot", help="Generate all 8 figures.")
    _add_common(p)
    p.add_argument("--all", action="store_true", default=True)
    p.set_defaults(func=cmd_plot)

    # ---- run-all ---------------------------------------------------------
    p = sub.add_parser("run-all", help="End-to-end driver (00->05).")
    _add_common(p)
    p.set_defaults(func=cmd_run_all)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())

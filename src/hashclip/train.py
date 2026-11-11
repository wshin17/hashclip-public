"""Training loop for HashCLIP adapters on cached SigLIP features.

Default path = cached pre-extracted features (fast, <2 GB VRAM at batch=128).
Fallback path = live-encode-from-pixels (``--no-cache``).
"""

from __future__ import annotations

import json
import math
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import data, encode, losses, model, retrieve, utils


# ---------------------------------------------------------------------------
# Run id / logging helpers
# ---------------------------------------------------------------------------


def _run_id(bits: int, seed: int, tag: str = "main") -> str:
    return f"hashclip_b{bits}_seed{seed}_{tag}"


@dataclass
class TrainResult:
    run_id: str
    ckpt_path: Path
    log_path: Path
    best_val: Dict[str, float]
    history: list


# ---------------------------------------------------------------------------
# Scheduler — cosine with linear warmup
# ---------------------------------------------------------------------------


def _make_scheduler(optimizer, *, total_steps: int, warmup_steps: int,
                    base_lr: float, min_lr: float):
    """LambdaLR returning a per-step multiplier on the optimizer's initial LR."""
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
        return lr / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Validation — cheap Hamming-only + asymmetric Recall@K
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _evaluate_on_cache(retr: model.RetrievalModel,
                       feats: data.CachedFeatures,
                       *, device: str, M: int, topk: Iterable[int] = (1, 5, 10)
                       ) -> Dict[str, float]:
    """Run T->I and I->T retrieval on cached features; return Recall@K + rsum."""
    retr.eval()
    img_cont = torch.from_numpy(feats.img_feat.astype(np.float32)).to(device)
    txt_cont = torch.from_numpy(feats.txt_feat.astype(np.float32)).to(device)
    img_h = retr.img_adapter(img_cont).cpu().numpy()
    txt_h = retr.txt_adapter(txt_cont).cpu().numpy()
    img_packed = utils.pack_bits(img_h)
    txt_packed = utils.pack_bits(txt_h)

    img_cont_np = feats.img_feat.astype(np.float32)
    txt_cont_np = feats.txt_feat.astype(np.float32)
    txt_to_img = feats.txt_to_img.astype(np.int64)

    metrics: Dict[str, float] = {}

    # ---- T -> I ---------------------------------------------------------
    cand_t2i = retrieve.hamming_topk(txt_packed, img_packed, k=min(M, img_packed.shape[0]))
    final_t2i = retrieve.rerank_cosine(txt_cont_np, img_cont_np, cand_t2i,
                                       k=max(topk))
    gt_t2i = txt_to_img[:, None]  # (N_txt, 1)
    for k in topk:
        hits = (final_t2i[:, :k] == gt_t2i).any(axis=1)
        metrics[f"R{k}_t2i"] = float(100.0 * hits.mean())

    # ---- I -> T (5 relevant captions per query image) -------------------
    cand_i2t = retrieve.hamming_topk(img_packed, txt_packed, k=min(M, txt_packed.shape[0]))
    final_i2t = retrieve.rerank_cosine(img_cont_np, txt_cont_np, cand_i2t,
                                       k=max(topk))
    # Build {img_idx -> set(cap_idx)} once.
    cap_sets: Dict[int, set] = {}
    for cap_idx, img_idx in enumerate(txt_to_img):
        cap_sets.setdefault(int(img_idx), set()).add(int(cap_idx))
    for k in topk:
        hits = []
        for qi, row in enumerate(final_i2t):
            relevant = cap_sets.get(qi, set())
            hits.append(any(int(c) in relevant for c in row[:k]))
        metrics[f"R{k}_i2t"] = float(100.0 * np.mean(hits))

    metrics["rsum"] = float(sum(metrics[f"R{k}_t2i"] + metrics[f"R{k}_i2t"]
                                for k in topk))
    return metrics


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def train_one(cfg: Dict[str, Any], *,
              bits: int,
              seed: int,
              hidden_dim: Optional[int] = None,
              loss_variant: str = "full",
              max_train: Optional[int] = None,
              max_val: Optional[int] = None,
              tag: str = "main",
              use_cache: bool = True,
              device: Optional[str] = None) -> TrainResult:
    """Train a single HashCLIP run.

    * ``bits``         hash length in bits (must be a multiple of 8).
    * ``seed``         RNG seed for reproducibility.
    * ``hidden_dim``   override the adapter hidden dim (defaults to cfg).
    * ``loss_variant`` one of the loss-variant names (``nce_only``/``nce_quant``/
                      ``nce_balance``/``full``).
    * ``max_train``    cap the train set (used by the smoke test).
    * ``use_cache``    use cached SigLIP features (the fast path).
    """
    utils.set_seed(seed)
    cfg = deepcopy(cfg)
    cfg["train"]["seed"] = int(seed)
    cfg["model"]["bits"] = int(bits)
    if hidden_dim is not None:
        cfg["model"]["hidden_dim"] = int(hidden_dim)

    run_id = _run_id(bits, seed, tag=tag)
    log_path = utils.get_path(cfg, "logs") / f"{run_id}.log.jsonl"
    ckpt_path = utils.get_path(cfg, "checkpoints") / f"{run_id}.pt"
    logger = utils.get_logger(f"hashclip.train.{run_id}",
                              log_file=utils.get_path(cfg, "logs") / f"{run_id}.console.log")

    snap = utils.env_snapshot()
    utils.log_jsonl(log_path, {"event": "env", **snap, "config": cfg,
                               "loss_variant": loss_variant})

    feat_dir = utils.get_path(cfg, "features")
    if use_cache:
        train_npz = feat_dir / "backbone_train.npz"
        val_npz = feat_dir / "backbone_val.npz"
        if not train_npz.exists() or not val_npz.exists():
            raise FileNotFoundError(
                "Cached features missing — run "
                "`python -m hashclip.cli encode-backbone` first, or pass "
                "use_cache=False."
            )
        train_feats = data.CachedFeatures.load(train_npz)
        val_feats = data.CachedFeatures.load(val_npz)
        train_ds = data.CachedFeaturePairs(train_feats, max_rows=max_train, seed=seed)
        # Trim val for smoke-test cap.
        if max_val is not None:
            sl = slice(0, max_val)
            val_feats = data.CachedFeatures(
                img_feat=val_feats.img_feat[sl],
                txt_feat=val_feats.txt_feat[:max_val * 5],
                img_ids=val_feats.img_ids[sl],
                txt_to_img=val_feats.txt_to_img[:max_val * 5],
                captions=val_feats.captions[:max_val * 5],
            )
    else:
        # Live-encode fallback (secondary path, used when --no-cache is passed).
        processed_dir = utils.get_path(cfg, "data_processed")
        bb = model.load_backbone(cfg["backbone"]["hf_hub"])
        train_df = data.load_split(processed_dir, "train")
        val_df = data.load_split(processed_dir, "val")
        train_ds = data.Flickr30kPairs(train_df, bb.preprocess, max_rows=max_train, seed=seed)
        val_feats = encode.encode_backbone_split(
            bb, val_df.iloc[:max_val] if max_val else val_df,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    dl_kwargs = data.make_loader_kwargs(cfg, train=True)
    loader = DataLoader(train_ds, **dl_kwargs)

    device_info = utils.select_device(cfg["train"].get("precision", "bf16"))
    device = device or device_info.name

    retr = model.RetrievalModel(
        embed_dim=int(cfg["backbone"]["embed_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        bits=int(cfg["model"]["bits"]),
        init_logit_scale=float(cfg["model"]["init_logit_scale"]),
        max_logit_scale=float(cfg["model"]["max_logit_scale"]),
    ).to(device)

    optim = torch.optim.AdamW(
        [p for p in retr.parameters() if p.requires_grad],
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    epochs = int(cfg["train"]["epochs"])
    steps_per_epoch = max(1, len(loader))
    sched = _make_scheduler(
        optim,
        total_steps=epochs * steps_per_epoch,
        warmup_steps=int(cfg["train"]["warmup_steps"]),
        base_lr=float(cfg["train"]["learning_rate"]),
        min_lr=float(cfg["train"]["cosine_min_lr"]),
    )

    lambda_q = float(cfg["loss"]["lambda_q"])
    lambda_b = float(cfg["loss"]["lambda_b"])
    loss_kwargs = losses.loss_kwargs_for_variant(loss_variant, lambda_q, lambda_b)

    history: list = []
    best_val: Dict[str, float] = {}
    best_score = -1.0
    log_every = int(cfg["train"].get("log_every", 50))

    autocast_dtype = device_info.dtype
    use_autocast = device == "cuda" and autocast_dtype in (torch.bfloat16, torch.float16)

    if not use_cache:
        # Live-encode path: we need the backbone for forward pass.
        bb.model.to(device).eval()

    global_step = 0
    candidate_validator = None
    best_r1_score = -1.0
    if cfg["train"].get("checkpoint_selection") == "candidate_retention":
        from .selection import CandidateValidation
        candidate_validator = CandidateValidation(
            retr, val_feats, fractions=cfg["train"]["candidate_fractions"],
            batch=cfg["train"].get("selection_batch_size", 32), semantic_budget=int(cfg["eval"]["M"]))
    for epoch in range(epochs):
        retr.train()
        t0 = time.time()
        running = {"loss": 0.0, "nce": 0.0, "quant": 0.0, "balance": 0.0, "n": 0}

        for batch in loader:
            if use_cache:
                img_cont, txt_cont = batch
                img_cont = img_cont.to(device, non_blocking=True)
                txt_cont = txt_cont.to(device, non_blocking=True)
            else:
                imgs, texts = batch
                imgs = imgs.to(device, non_blocking=True)
                with torch.inference_mode():
                    img_cont = model.encode_image_batch(bb, imgs, device=device)
                    txt_cont = model.encode_text_batch(bb, list(texts), device=device)
                img_cont = img_cont.float()
                txt_cont = txt_cont.float()

            optim.zero_grad(set_to_none=True)

            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    img_h, txt_h = retr(img_cont, txt_cont)
                    loss, comp = losses.total_loss(img_h.float(), txt_h.float(),
                                                   retr.logit_scale, **loss_kwargs)
            else:
                img_h, txt_h = retr(img_cont, txt_cont)
                loss, comp = losses.total_loss(img_h, txt_h, retr.logit_scale,
                                               **loss_kwargs)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(retr.parameters(),
                                           max_norm=float(cfg["train"].get("grad_clip", 1.0)))
            optim.step()
            sched.step()
            retr.clamp_logit_scale()

            bs = img_cont.shape[0]
            running["loss"] += comp["loss"] * bs
            running["nce"] += comp["nce"] * bs
            running["quant"] += comp["quant"] * bs
            running["balance"] += comp["balance"] * bs
            running["n"] += bs

            global_step += 1
            if global_step % log_every == 0:
                utils.log_jsonl(log_path, {
                    "event": "step", "epoch": epoch, "step": global_step,
                    "lr": float(optim.param_groups[0]["lr"]),
                    "logit_scale": float(retr.logit_scale.detach().cpu()),
                    **comp,
                })

        n = max(1, running["n"])
        train_summary = {
            "event": "epoch_train", "epoch": epoch,
            "loss": running["loss"] / n,
            "nce": running["nce"] / n,
            "quant": running["quant"] / n,
            "balance": running["balance"] / n,
            "time_sec": time.time() - t0,
        }
        utils.log_jsonl(log_path, train_summary)
        logger.info("epoch %d: %s", epoch, json.dumps(train_summary))

        # Validation
        val_metrics = (candidate_validator(retr) if candidate_validator is not None else
                       _evaluate_on_cache(retr, val_feats, device=device,
                                          M=int(cfg["eval"]["M"]), topk=tuple(cfg["eval"]["topk"])))
        val_record = {"event": "epoch_val", "epoch": epoch, **val_metrics}
        utils.log_jsonl(log_path, val_record)
        logger.info("epoch %d val: %s", epoch, json.dumps(val_metrics))
        history.append({"epoch": epoch, "train": train_summary, "val": val_metrics})

        # Best checkpoint: mean of R@1 across T->I and I->T.
        r1_score = 0.5 * (val_metrics["R1_t2i"] + val_metrics["R1_i2t"])
        score = val_metrics["candidate_selection_score"] if candidate_validator is not None else r1_score
        if candidate_validator is not None and r1_score > best_r1_score:
            best_r1_score = r1_score
            payload = retr.state_payload(config=cfg, epoch=epoch, val_metrics=val_metrics)
            payload["selection_rule"] = "mean semantic R@1 at M=1000; same training trajectory"
            torch.save(payload, ckpt_path.with_stem(ckpt_path.stem + "_r1"))
        if score > best_score:
            best_score = score
            best_val = val_metrics
            payload = retr.state_payload(config=cfg, epoch=epoch,
                                         val_metrics=val_metrics)
            payload["selection_rule"] = ("candidate retention" if candidate_validator is not None else "mean semantic R@1")
            torch.save(payload, ckpt_path)
            logger.info("saved best ckpt -> %s  (selection_score=%.6f)", ckpt_path, score)

    utils.log_jsonl(log_path, {"event": "done", "best_val": best_val,
                               "ckpt": str(ckpt_path)})
    logger.info("done. best=%s ckpt=%s", best_val, ckpt_path)
    return TrainResult(run_id=run_id, ckpt_path=ckpt_path, log_path=log_path,
                       best_val=best_val, history=history)


# ---------------------------------------------------------------------------
# Ablation orchestration
# ---------------------------------------------------------------------------


def train_ablation(cfg: Dict[str, Any], kind: str, *,
                   seed: int = 42, **kw) -> Dict[str, TrainResult]:
    """Run a sweep — returns ``{label: TrainResult}``."""
    kind = kind.lower()
    out: Dict[str, TrainResult] = {}
    if kind == "bits":
        for b in cfg["ablation"]["bits"]:
            out[f"bits={b}"] = train_one(cfg, bits=int(b), seed=seed,
                                         tag=f"abl-bits", **kw)
    elif kind == "loss":
        for v in cfg["ablation"]["loss_variants"]:
            out[f"loss={v}"] = train_one(cfg, bits=int(cfg["model"]["bits"]),
                                         seed=seed, loss_variant=v,
                                         tag=f"abl-loss-{v}", **kw)
    elif kind == "adapter":
        for h in cfg["ablation"]["adapter_hidden"]:
            out[f"hidden={h}"] = train_one(cfg, bits=int(cfg["model"]["bits"]),
                                           seed=seed, hidden_dim=int(h),
                                           tag=f"abl-h{h}", **kw)
    else:
        raise ValueError(f"Unknown ablation kind: {kind!r}")
    return out

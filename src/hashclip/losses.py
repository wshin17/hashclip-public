"""Loss functions for HashCLIP training.

Total loss:

    L_total = L_InfoNCE + lambda_q * L_quant + lambda_b * L_bitbalance

* ``info_nce``  — symmetric image<->text InfoNCE with a learnable temperature.
* ``quant_loss``— pushes |h| -> 1 so ``sign(h)`` is information-preserving.
* ``bit_balance_loss`` — encourages 50/50 bit usage per dimension (maximizes
  entropy of the hash code).

All three terms accept tanh-adapter outputs (``h`` in ``(-1, +1)``) directly.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def info_nce(img_h: torch.Tensor, txt_h: torch.Tensor,
             logit_scale: torch.Tensor) -> torch.Tensor:
    """Symmetric I<->T InfoNCE on L2-normalized hash vectors."""
    img_n = F.normalize(img_h, dim=-1)
    txt_n = F.normalize(txt_h, dim=-1)
    scale = logit_scale.exp().clamp(max=100.0)
    logits = scale * (img_n @ txt_n.T)             # (B, B)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels)
                  + F.cross_entropy(logits.T, labels))


def quant_loss(h: torch.Tensor) -> torch.Tensor:
    """Mean squared deviation of |h| from 1 (pushes outputs toward +-1)."""
    return ((h.abs() - 1.0) ** 2).mean()


def bit_balance_loss(h: torch.Tensor) -> torch.Tensor:
    """Encourages each bit's batch-mean to be zero (50/50 on/off)."""
    # h: (B, bits) — mean over batch dim, then mean of squared per-bit means.
    return (h.mean(dim=0) ** 2).mean()


def total_loss(img_h: torch.Tensor, txt_h: torch.Tensor,
               logit_scale: torch.Tensor,
               *,
               lambda_q: float = 0.1,
               lambda_b: float = 0.01,
               use_quant: bool = True,
               use_balance: bool = True
               ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Returns ``(scalar_loss, components_dict)`` for logging."""
    l_nce = info_nce(img_h, txt_h, logit_scale)
    l_q = quant_loss(img_h) + quant_loss(txt_h) if use_quant else img_h.new_zeros(())
    l_b = bit_balance_loss(img_h) + bit_balance_loss(txt_h) if use_balance else img_h.new_zeros(())
    loss = l_nce + lambda_q * l_q + lambda_b * l_b
    return loss, {
        "loss": float(loss.detach().cpu()),
        "nce": float(l_nce.detach().cpu()),
        "quant": float(l_q.detach().cpu() if torch.is_tensor(l_q) else l_q),
        "balance": float(l_b.detach().cpu() if torch.is_tensor(l_b) else l_b),
    }


# ---------------------------------------------------------------------------
# Loss-variant resolver — used by the loss-component ablation
# ---------------------------------------------------------------------------


def loss_kwargs_for_variant(variant: str, lambda_q: float, lambda_b: float) -> Dict[str, object]:
    """Map a loss-variant name -> kwargs for :func:`total_loss`."""
    variant = variant.lower().strip()
    if variant in ("nce", "nce_only"):
        return dict(lambda_q=0.0, lambda_b=0.0, use_quant=False, use_balance=False)
    if variant in ("nce_quant", "+quant"):
        return dict(lambda_q=lambda_q, lambda_b=0.0, use_quant=True, use_balance=False)
    if variant in ("nce_balance", "+balance"):
        return dict(lambda_q=0.0, lambda_b=lambda_b, use_quant=False, use_balance=True)
    if variant in ("full", "all"):
        return dict(lambda_q=lambda_q, lambda_b=lambda_b, use_quant=True, use_balance=True)
    raise ValueError(f"Unknown loss variant: {variant!r}")

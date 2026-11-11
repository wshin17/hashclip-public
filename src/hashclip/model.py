"""Model architecture — frozen SigLIP backbone + two hashing adapters.

Adapter architecture:

    backbone(frozen) -> L2-norm -> Linear(768, 512) -> GELU -> Linear(512, B) -> Tanh

A trainable scalar ``logit_scale`` (CLIP-style) is initialized to ``ln(1/0.07)``
and clamped to ``ln(100)`` at forward time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Hashing adapter
# ---------------------------------------------------------------------------


class HashingAdapter(nn.Module):
    """2-layer MLP placed on top of frozen backbone features."""

    def __init__(self, in_dim: int = 768, hidden_dim: int = 512, bits: int = 128):
        super().__init__()
        if bits % 8 != 0:
            raise ValueError(
                f"`bits` must be a multiple of 8 for byte-packing; got {bits}."
            )
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.bits = int(bits)

        if hidden_dim <= 0:
            # Hidden = 0 ablation: a single Linear projection.
            self.net = nn.Linear(in_dim, bits)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, bits),
            )
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # L2-normalize the incoming continuous embedding.
        x = F.normalize(x, dim=-1)
        return self.act(self.net(x))

    @torch.no_grad()
    def binarize(self, x: torch.Tensor) -> torch.Tensor:
        """Return ±1 hard codes (no gradient)."""
        return torch.sign(self.forward(x))


# ---------------------------------------------------------------------------
# Backbone loader — kept separate so training on cached features can skip it
# ---------------------------------------------------------------------------


@dataclass
class BackboneBundle:
    model: nn.Module
    preprocess: object  # torchvision-style transform
    tokenizer: object
    embed_dim: int = 768


def load_backbone(hf_hub: str = "hf-hub:timm/ViT-B-16-SigLIP") -> BackboneBundle:
    """Load the frozen SigLIP backbone via open_clip."""
    import open_clip  # local import keeps model.py importable in test/CI without backbone

    model, preprocess = open_clip.create_model_from_pretrained(hf_hub)
    tokenizer = open_clip.get_tokenizer(hf_hub)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    embed_dim = _infer_embed_dim(model)
    return BackboneBundle(model=model, preprocess=preprocess,
                          tokenizer=tokenizer, embed_dim=embed_dim)


def _infer_embed_dim(model: nn.Module) -> int:
    """Try a couple of common attributes; fall back to 768."""
    for attr in ("text_projection", "visual"):
        obj = getattr(model, attr, None)
        if obj is None:
            continue
        for sub in ("output_dim", "embed_dim"):
            if hasattr(obj, sub):
                return int(getattr(obj, sub))
    return 768


@torch.inference_mode()
def encode_image_batch(backbone: BackboneBundle, images: torch.Tensor,
                       device: str = "cuda") -> torch.Tensor:
    """Run the frozen image tower on a preprocessed batch -> (B, 768) float32."""
    images = images.to(device, non_blocking=True)
    feats = backbone.model.encode_image(images)
    return feats.float()


@torch.inference_mode()
def encode_text_batch(backbone: BackboneBundle, texts, device: str = "cuda",
                      context_length: Optional[int] = None) -> torch.Tensor:
    """Run the frozen text tower on a list of strings -> (B, 768) float32."""
    if isinstance(texts, (list, tuple)):
        tok_kwargs = {}
        if context_length is not None:
            tok_kwargs["context_length"] = context_length
        tokens = backbone.tokenizer(list(texts), **tok_kwargs)
    else:
        tokens = texts
    tokens = tokens.to(device, non_blocking=True)
    feats = backbone.model.encode_text(tokens)
    return feats.float()


# ---------------------------------------------------------------------------
# Retrieval model wrapper
# ---------------------------------------------------------------------------


class RetrievalModel(nn.Module):
    """Holds two HashingAdapters + a learnable ``logit_scale``.

    The frozen backbone is **not** held inside this module so the adapters
    can be trained on pre-extracted features without re-loading SigLIP.
    """

    def __init__(self, embed_dim: int = 768, hidden_dim: int = 512, bits: int = 128,
                 init_logit_scale: float = math.log(1.0 / 0.07),
                 max_logit_scale: float = math.log(100.0)):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.bits = int(bits)
        self.max_logit_scale = float(max_logit_scale)

        self.img_adapter = HashingAdapter(in_dim=embed_dim, hidden_dim=hidden_dim, bits=bits)
        self.txt_adapter = HashingAdapter(in_dim=embed_dim, hidden_dim=hidden_dim, bits=bits)
        self.logit_scale = nn.Parameter(torch.tensor(float(init_logit_scale)))

    def clamp_logit_scale(self) -> None:
        with torch.no_grad():
            self.logit_scale.clamp_(max=self.max_logit_scale)

    def forward(self, img_cont: torch.Tensor, txt_cont: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inputs are already continuous backbone features (B, embed_dim)."""
        img_h = self.img_adapter(img_cont)
        txt_h = self.txt_adapter(txt_cont)
        return img_h, txt_h

    @torch.no_grad()
    def binarize_image(self, img_cont: torch.Tensor) -> torch.Tensor:
        return self.img_adapter.binarize(img_cont)

    @torch.no_grad()
    def binarize_text(self, txt_cont: torch.Tensor) -> torch.Tensor:
        return self.txt_adapter.binarize(txt_cont)

    # ----- Checkpoint helpers ------------------------------------------------

    def state_payload(self, *, config: dict, epoch: int, val_metrics: dict) -> dict:
        return dict(
            state_dict=self.state_dict(),
            config=config,
            epoch=int(epoch),
            val_metrics=val_metrics,
            bits=self.bits,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
        )

    @classmethod
    def from_checkpoint(cls, path, map_location="cpu") -> Tuple["RetrievalModel", dict]:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        m = cls(
            embed_dim=int(ckpt.get("embed_dim", 768)),
            hidden_dim=int(ckpt.get("hidden_dim", 512)),
            bits=int(ckpt["bits"]),
        )
        m.load_state_dict(ckpt["state_dict"])
        return m, ckpt

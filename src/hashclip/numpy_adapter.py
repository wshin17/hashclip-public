"""HashCLIP inference without importing PyTorch into the query process.

Weights are exported offline. The original L2 normalization, two affine layers,
exact (erf) GELU and tanh are retained. NumPy/BLAS floating-point differences
from torch are audited separately; no cached query codes enter timed inference.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import erf


class NumpyAdapter:
    def __init__(self, weights):
        self.weights = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in weights.items()}
        self.w1, self.b1 = self.weights["w1"], self.weights["b1"]
        self.w2, self.b2 = self.weights.get("w2"), self.weights.get("b2")
        self.dimension = self.w1.shape[0]
        self.bits = self.w1.shape[1] if self.w2 is None else self.w2.shape[1]
        if self.bits % 8 or self.b1.shape != (self.w1.shape[1],):
            raise ValueError("Invalid adapter dimensions")
        if self.w2 is not None and (self.w2.shape[0] != self.w1.shape[1] or self.b2.shape != (self.bits,)):
            raise ValueError("Invalid second adapter layer")
        if any(not np.isfinite(x).all() for x in self.weights.values()):
            raise ValueError("Non-finite adapter parameters")
        self.nbytes = sum(x.nbytes for x in self.weights.values())

    def forward(self, raw):
        x = np.asarray(raw, dtype=np.float32, order="C")
        if x.ndim != 2 or x.shape[1] != self.dimension:
            raise ValueError("Input dimensions differ from adapter")
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), np.float32(1e-12))
        x = x @ self.w1 + self.b1
        if self.w2 is not None:
            # nn.GELU defaults to approximate='none', i.e. the erf formula.
            x = np.float32(.5) * x * (np.float32(1) + erf(x / np.float32(np.sqrt(2))))
            x = x @ self.w2 + self.b2
        return np.tanh(x)

    def encode(self, raw):
        return np.packbits(self.forward(raw) > 0, axis=1)


def numpy_checkpoint_path(checkpoint):
    return Path(checkpoint).with_suffix(".numpy.npz")


def load_adapters(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        adapters = {modality: NumpyAdapter({k[len(modality) + 1:]: archive[k]
                    for k in archive.files if k.startswith(modality + "_")})
                    for modality in ("img", "txt")}
    return adapters, metadata


def export_checkpoint(source, destination=None):
    # Deliberately a separate offline operation; never invoked by make_index.
    from .evaluate_cpu import input_identity
    from .model import RetrievalModel

    source = Path(source)
    destination = Path(destination) if destination else numpy_checkpoint_path(source)
    model, checkpoint = RetrievalModel.from_checkpoint(source)
    metadata = {"source": input_identity(source), "bits": model.bits,
                "embed_dim": model.embed_dim, "hidden_dim": model.hidden_dim,
                "seed": checkpoint.get("config", {}).get("train", {}).get("seed"),
                "epoch": checkpoint["epoch"], "dtype": "float32",
                "activation": "GELU erf (approximate=none), tanh"}
    weights = {}
    for modality in ("img", "txt"):
        adapter = getattr(model, modality + "_adapter")
        layers = [adapter.net] if model.hidden_dim <= 0 else [adapter.net[0], adapter.net[2]]
        for i, layer in enumerate(layers, 1):
            weights[f"{modality}_w{i}"] = layer.weight.detach().numpy().T.copy()
            weights[f"{modality}_b{i}"] = layer.bias.detach().numpy().copy()
    if destination.exists():
        existing, old_metadata = load_adapters(destination)
        if metadata != old_metadata:
            raise ValueError(f"Existing NumPy export has another source: {destination}")
        for key, value in weights.items():
            modality, parameter = key.split("_", 1)
            np.testing.assert_array_equal(value, existing[modality].weights[parameter])
    else:
        np.savez(destination, metadata=json.dumps(metadata), **weights)
    return model.eval(), checkpoint, destination

"""Shared utilities: seeding, logging, environment checks, bit-packing."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Constants / project paths
# ---------------------------------------------------------------------------

# Repo root is two levels above this file: src/hashclip/utils.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load a YAML config from disk. Defaults to ``configs/default.yaml``."""
    p = Path(path) if path else DEFAULT_CONFIG
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Resolve all configured paths relative to the repo root (idempotent).
    paths = cfg.get("paths", {})
    for k, v in list(paths.items()):
        path_obj = Path(v)
        if not path_obj.is_absolute():
            path_obj = (REPO_ROOT / path_obj).resolve()
        paths[k] = str(path_obj)
    cfg["paths"] = paths
    return cfg


def get_path(cfg: Dict[str, Any], key: str) -> Path:
    """Return a Path object for ``cfg['paths'][key]``, creating the dir if needed."""
    p = Path(cfg["paths"][key])
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Determinism / seeding
# ---------------------------------------------------------------------------


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA). Optional cuDNN determinism."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # local import keeps utils importable without torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        # torch may not be installed in lightweight contexts (e.g. doc build)
        pass


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding helper — works on Windows ``spawn`` and POSIX ``fork``."""
    import torch

    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def get_logger(name: str = "hashclip", log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def log_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append one JSON line to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


# ---------------------------------------------------------------------------
# Environment verification
# ---------------------------------------------------------------------------


def assert_in_asa26_env() -> str:
    """Ensure we are running inside the ``asa26`` conda env. Returns sys.prefix."""
    p = sys.prefix.replace(os.sep, "/")
    if not ("/envs/asa26" in p or p.endswith("/asa26")):
        raise RuntimeError(
            f"Wrong Python environment: sys.prefix={sys.prefix!r}. "
            "Activate the `asa26` conda env first: `conda activate asa26`."
        )
    return sys.prefix


def git_rev() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def env_snapshot() -> Dict[str, Any]:
    """Capture every detail a reviewer needs to reproduce a run."""
    snap: Dict[str, Any] = {
        "python": sys.version,
        "sys_prefix": sys.prefix,
        "platform": platform.platform(),
        "os": platform.system(),
        "git_rev": git_rev(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        import torch

        snap["torch"] = torch.__version__
        snap["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            snap["device"] = torch.cuda.get_device_name(0)
            snap["capability"] = list(torch.cuda.get_device_capability(0))
            snap["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    except Exception as e:
        snap["torch_error"] = str(e)
    return snap


def check_env() -> Dict[str, Any]:
    """Implements ``python -m hashclip.cli check-env`` — prints + asserts."""
    snap = env_snapshot()
    print("torch:              ", snap.get("torch", "<not installed>"))
    print("cuda available:     ", snap.get("cuda_available", False))
    print("device:             ", snap.get("device", "<no cuda>"))
    print("capability:         ", snap.get("capability", "<no cuda>"))
    print("bf16 supported:     ", snap.get("bf16_supported", False))

    # Sanity matmul on whatever device is available.
    try:
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.randn(64, 64, device=dev)
        y = (x @ x.T).sum().item()
        print("matmul ok:          ", y)
    except Exception as e:
        print("matmul failed:      ", e)

    print("OS:                 ", snap["platform"])
    print("Python:             ", snap["python"].splitlines()[0])
    print("sys.prefix:         ", snap["sys_prefix"])

    # Final assertions — these define what "ready to run HashCLIP" means.
    try:
        import torch

        if torch.cuda.is_available():
            cap = tuple(snap["capability"])
            major = cap[0]
            if major < 7:
                print(
                    f"WARNING: GPU compute capability is {cap} (< 7.0). "
                    "HashCLIP requires sm_70 or higher (Volta / V100 and above). "
                    "Training may fail or fall back to CPU."
                )
            else:
                print(f"GPU capability:      {cap} — OK (sm_{cap[0]}{cap[1]})")
                if not snap.get("bf16_supported", False):
                    print(
                        "NOTE: bf16 is not supported on this GPU. "
                        "Training will use fp32 autocast (automatic fallback)."
                    )
    except Exception:
        pass

    # Confirm the env name.
    try:
        prefix = assert_in_asa26_env()
        print("hashclip prefix OK: ", prefix)
    except RuntimeError as e:
        print("WARNING:", e)

    return snap


# ---------------------------------------------------------------------------
# Bit-packing helpers
# ---------------------------------------------------------------------------


def pack_bits(h: np.ndarray) -> np.ndarray:
    """Pack a real-valued (N, B) array to (N, ceil(B/8)) ``uint8`` bits.

    Convention: positive values -> 1, non-positive -> 0.
    """
    if h.ndim != 2:
        raise ValueError(f"expected 2-D, got shape {h.shape}")
    b01 = (h > 0).astype(np.uint8)
    return np.packbits(b01, axis=-1)


def unpack_bits(packed: np.ndarray, bits: int) -> np.ndarray:
    """Inverse of :func:`pack_bits` — returns (N, bits) uint8 with values in {0, 1}."""
    full = np.unpackbits(packed, axis=-1)
    return full[:, :bits]


# ---------------------------------------------------------------------------
# Device + dtype helpers
# ---------------------------------------------------------------------------


@dataclass
class DeviceInfo:
    name: str
    dtype: "Any"
    bf16_ok: bool


def select_device(precision: str = "bf16") -> "DeviceInfo":
    """Pick CUDA if available; pick bf16 only if the device supports it."""
    import torch

    if not torch.cuda.is_available():
        return DeviceInfo(name="cpu", dtype=torch.float32, bf16_ok=False)
    bf16_ok = bool(torch.cuda.is_bf16_supported())
    if precision == "bf16" and bf16_ok:
        return DeviceInfo(name="cuda", dtype=torch.bfloat16, bf16_ok=True)
    if precision == "fp16":
        return DeviceInfo(name="cuda", dtype=torch.float16, bf16_ok=bf16_ok)
    return DeviceInfo(name="cuda", dtype=torch.float32, bf16_ok=bf16_ok)


def default_num_workers(cfg: Dict[str, Any]) -> int:
    if platform.system() == "Windows":
        return int(cfg["train"]["num_workers_windows"])
    return int(cfg["train"]["num_workers_linux"])

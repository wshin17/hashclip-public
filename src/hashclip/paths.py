"""Locations for new evaluation inputs and outputs; legacy configuration stays intact."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT", ROOT / "data")).expanduser()


def runs_root() -> Path:
    return Path(os.environ.get("RUNS_ROOT", ROOT / "runs")).expanduser()

"""Resolve the PfHT1 project layout without machine-specific absolute paths."""
from __future__ import annotations

import os
from pathlib import Path


def _project_root() -> Path:
    configured = os.environ.get("PFHT1_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "apo_pfht1").is_dir() and (candidate / "holo_pfht1").is_dir():
            return candidate
    raise RuntimeError(
        "Could not locate the PfHT1 project root. Set PFHT1_PROJECT_ROOT to the "
        "directory containing apo_pfht1 and holo_pfht1."
    )


PROJECT_ROOT = _project_root()
ANALYSIS_ROOT = Path(
    os.environ.get(
        "PFHT1_BINDING_POCKET_DIR", PROJECT_ROOT / "analysis" / "binding_pocket"
    )
).expanduser().resolve()

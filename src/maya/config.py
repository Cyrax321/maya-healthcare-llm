"""Configuration loading, experiment identity and environment capture.

Responsibilities (single): turn YAML configs + runtime info into one frozen,
reproducible experiment record that is written to ``artifacts/<id>/config.json``
and ``artifacts/<id>/environment.json``.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
ARTIFACTS_DIR = ROOT / "artifacts"

_REQUIRED_CONFIGS = ("model", "training_t4", "data", "evaluation")


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning an empty dict for a missing file."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(name: str) -> dict[str, Any]:
    """Load one named config file from ``configs/``."""
    return load_yaml(CONFIG_DIR / f"{name}.yaml")


def load_all_configs() -> dict[str, Any]:
    """Merge the four base configs into one nested dict keyed by config name."""
    merged: dict[str, Any] = {}
    for name in _REQUIRED_CONFIGS:
        merged[name] = load_config(name)
    return merged


def new_experiment_id() -> str:
    """Return a short, sortable, unique experiment id (e.g. ``maya-20241015-ab12``)."""
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = hashlib.sha1(os.urandom(8)).hexdigest()[:4]
    return f"maya-{stamp}-{suffix}"


def _git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _import_version(module: str) -> str | None:
    try:
        mod = __import__(module)
        return getattr(mod, "__version__", None) or getattr(mod, "VERSION", None)
    except Exception:
        return None


def capture_environment() -> dict[str, Any]:
    """Capture runtime details needed for reproducibility."""
    packages = {}
    for mod in (
        "torch",
        "transformers",
        "trl",
        "peft",
        "unsloth",
        "accelerate",
        "bitsandbytes",
        "datasets",
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "matplotlib",
    ):
        packages[mod] = _import_version(mod)

    gpu = None
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            gpu = {
                "count": torch.cuda.device_count(),
                "name": torch.cuda.get_device_name(0),
                "vram_gb": round(
                    torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
                ),
                "cuda_version": torch.version.cuda,
            }
        else:
            gpu = {"available": False}
    except Exception:
        gpu = {"available": False, "error": "torch not importable"}

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_revision": _git_revision(),
        "packages": packages,
        "gpu": gpu,
    }


def resolve_model_revision(repo_id: str) -> str | None:
    """Resolve the concrete commit SHA for a model repo, when reachable.

    Returns None on any failure (offline, no HF access) so callers can still
    proceed; the resolved value (or the reason it is None) is recorded in the
    experiment config for reproducibility.
    """
    try:
        import requests  # type: ignore

        url = f"https://huggingface.co/api/models/{repo_id}"
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            return resp.json().get("sha")
    except Exception:
        return None
    return None


def write_experiment_config(
    experiment_id: str,
    configs: dict[str, Any],
    model_revision: str | None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist the frozen experiment config and environment to the artifacts dir."""
    out_dir = ARTIFACTS_DIR / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    record = {
        "experiment_id": experiment_id,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "config": configs,
        "model_revision": model_revision,
    }
    if extra:
        record["extra"] = extra
    (out_dir / "config.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    (out_dir / "environment.json").write_text(
        json.dumps(capture_environment(), indent=2), encoding="utf-8"
    )
    return out_dir


def artifact_path(experiment_id: str, name: str) -> Path:
    """Return the path to a named artifact for an experiment."""
    return ARTIFACTS_DIR / experiment_id / name

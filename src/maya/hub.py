"""Push adapters and evaluation artifacts to the Hugging Face Hub.

Authentication uses the ``HF_TOKEN`` environment variable only. The token is
never written to disk, never logged, and never committed. In Colab, store the
token as a Secret named ``HF_TOKEN`` (``from google.colab import userdata``) and
export it to the environment instead of pasting it into a cell.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

HF_TOKEN_ENV = "HF_TOKEN"


def _token(token: str | None) -> str | None:
    return token or os.environ.get(HF_TOKEN_ENV)


def login(token: str | None = None) -> None:
    """Validate that a token is available; raise a clear error otherwise."""
    tk = _token(token)
    if not tk:
        raise RuntimeError(
            f"no Hugging Face token: set the {HF_TOKEN_ENV} environment variable "
            "(Colab: store as a Secret named HF_TOKEN and export it)"
        )
    try:
        from huggingface_hub import login as _login  # type: ignore

        _login(tk)
    except ImportError:
        from huggingface_hub import HfApi  # type: ignore

        HfApi(token=tk).whoami()


def push_adapter(
    model: Any,
    tokenizer: Any,
    repo_id: str,
    token: str | None = None,
    private: bool = False,
    merged: bool = False,
) -> str:
    """Push the trained LoRA adapter (or merged weights) to the Hub."""
    tk = _token(token)
    if not tk:
        raise RuntimeError(f"no {HF_TOKEN_ENV} token available")

    if merged:
        model.push_to_hub_merged(repo_id, tokenizer, save_method="merged_16bit", token=tk, private=private)
    else:
        model.push_to_hub(repo_id, token=tk, private=private)
        tokenizer.push_to_hub(repo_id, token=tk, private=private)
    return repo_id


def push_results(
    experiment_id: str,
    repo_id: str,
    token: str | None = None,
    private: bool = True,
    repo_type: str = "dataset",
    artifacts_dir: Path | None = None,
) -> str:
    """Upload a whole experiment's artifacts to a Hub repo (dataset by default).

    Metrics, predictions, plots, report and config all land in one place, tied to
    the experiment id.
    """
    from huggingface_hub import HfApi  # type: ignore

    tk = _token(token)
    if not tk:
        raise RuntimeError(f"no {HF_TOKEN_ENV} token available")

    from .config import ARTIFACTS_DIR

    folder = (artifacts_dir or ARTIFACTS_DIR) / experiment_id
    if not folder.exists():
        raise FileNotFoundError(f"no artifacts for experiment {experiment_id} at {folder}")

    api = HfApi(token=tk)
    api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type=repo_type,
        token=tk,
    )
    return repo_id

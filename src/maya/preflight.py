"""Training-readiness preflight gate.

Checks every hard prerequisite before a training run and prints a PASS/FAIL
table plus a final READY / NOT READY verdict. The optional model-load check is
the real gate: it attempts the Unsloth load of the exact checkpoint and reports
memory and any compatibility error.

Run in Colab:  ``python -m maya.cli preflight --load``
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    required: bool = True


def _version(mod: str) -> str | None:
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", None) or getattr(m, "VERSION", None)
    except Exception as exc:  # noqa: BLE001
        return f"IMPORT FAILED: {type(exc).__name__}: {exc}"


def run_preflight(repo_id: str, attempt_load: bool = False, data_dir: str = "data/processed") -> list[Check]:
    checks: list[Check] = []

    # Python
    py = sys.version_info
    checks.append(Check("python>=3.10", py >= (3, 10), f"{py.major}.{py.minor}.{py.micro}"))

    # torch / CUDA
    torch_v = _version("torch")
    has_torch = torch_v and not str(torch_v).startswith("IMPORT FAILED")
    checks.append(Check("torch", has_torch, str(torch_v)))

    cuda_ok = False
    vram = None
    gpu_name = None
    if has_torch:
        import torch  # type: ignore

        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            gpu_name = torch.cuda.get_device_name(0)
            vram = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    checks.append(Check("CUDA GPU", cuda_ok, f"{gpu_name} ({vram} GB)" if cuda_ok else "no CUDA device"))
    if cuda_ok:
        checks.append(Check("VRAM >= 16GB (T4)", vram >= 15.5, f"{vram} GB"))

    # transformers v5 requirement (Qwen3.5 family)
    tf_v = _version("transformers")
    tf_num = None
    try:
        tf_num = tuple(int(x) for x in str(tf_v).split(".")[:2])
    except Exception:
        tf_num = None
    checks.append(Check("transformers v5 (Qwen3.5 requires v5)", tf_num is not None and tf_num >= (5, 0), str(tf_v)))

    # unsloth
    us_v = _version("unsloth")
    checks.append(Check("unsloth", not str(us_v).startswith("IMPORT FAILED"), str(us_v)))

    # Qwen3.5 linear-attention kernels. Optional for the Unsloth path: Unsloth
    # ships its own Mamba/DeltaNet Triton kernels. Only plain-Transformers/vLLM
    # inference needs flash-linear-attention + causal-conv1d.
    fla = _version("flash_linear_attention")
    cc1 = _version("causal_conv1d")
    checks.append(Check("flash-linear-attention kernel (optional)", not str(fla).startswith("IMPORT FAILED"), str(fla), required=False))
    checks.append(Check("causal-conv1d kernel (optional)", not str(cc1).startswith("IMPORT FAILED"), str(cc1), required=False))

    # data prepared
    import os

    train = os.path.exists(os.path.join(data_dir, "train.jsonl"))
    val = os.path.exists(os.path.join(data_dir, "validation.jsonl"))
    test = os.path.exists(os.path.join(data_dir, "test.jsonl"))
    checks.append(Check("data prepared (train/val/test)", train and val and test, data_dir))

    # optional real load
    if attempt_load:
        try:
            from .model import load_model_and_tokenizer  # local import

            result = load_model_and_tokenizer(repo_id=repo_id, max_seq_length=2048)
            mem = _peak_memory_gb()
            checks.append(
                Check(
                    "model load smoke test",
                    True,
                    f"via_unsloth={result.via_unsloth} dtype={result.dtype} peak_vram={mem} GB",
                )
            )
        except Exception as exc:  # noqa: BLE001
            checks.append(Check("model load smoke test", False, f"{type(exc).__name__}: {exc}"))

    return checks


def _peak_memory_gb() -> str:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return str(round(torch.cuda.max_memory_allocated() / (1024**3), 2))
    except Exception:
        pass
    return "n/a"


def verdict(checks: list[Check]) -> bool:
    return all(c.ok for c in checks if c.required)


def format_report(checks: list[Check]) -> str:
    lines = ["Preflight check", "=" * 60]
    for c in checks:
        mark = "PASS" if c.ok else "FAIL"
        flag = "" if c.required else " (optional)"
        lines.append(f"[{mark}]{flag} {c.name}: {c.detail}")
    lines.append("=" * 60)
    lines.append(f"VERDICT: {'READY' if verdict(checks) else 'NOT READY'}")
    return "\n".join(lines)

"""Model loading, compatibility inspection and generation.

GPU code paths (Unsloth / torch) are imported lazily so the CPU-only test and
data/metrics/report modules import cleanly without a CUDA runtime.

Compatibility notes are recorded precisely: the target checkpoint is a distilled
Qwen3.5-architecture model that is NOT in Unsloth's default ``unsloth_zoo``
registry. Unsloth patches by ``model_type`` (``qwen3_5``), but whether the exact
``Qwen3_5ForConditionalGeneration`` class and its text path are patched must be
verified with a live load. The loader below surfaces the exact error otherwise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

DEFAULT_REPO = "empero-ai/Qwen3.8-4B-Distill"


@dataclass
class LoadResult:
    model: Any
    tokenizer: Any
    via_unsloth: bool
    dtype: str
    notes: list[str]


def inspect_checkpoint(repo_id: str) -> dict[str, Any]:
    """Fetch and summarize the checkpoint's config without loading weights."""
    import requests  # type: ignore

    cfg_url = f"https://huggingface.co/{repo_id}/raw/main/config.json"
    info: dict[str, Any] = {"repo_id": repo_id, "config_fetched": False}
    try:
        resp = requests.get(cfg_url, timeout=20)
        if resp.status_code == 200:
            cfg = resp.json()
            info["config_fetched"] = True
            info["architectures"] = cfg.get("architectures")
            info["model_type"] = cfg.get("model_type")
            info["transformers_version"] = cfg.get("transformers_version")
            info["text_model_type"] = (cfg.get("text_config") or {}).get("model_type")
            info["has_vision_config"] = "vision_config" in cfg
            info["num_hidden_layers"] = (cfg.get("text_config") or {}).get("num_hidden_layers")
            info["layer_types"] = (cfg.get("text_config") or {}).get("layer_types", [])[:4] + ["..."]
        else:
            info["error"] = f"config fetch status {resp.status_code}"
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def _try_unsloth_load(
    repo_id: str,
    max_seq_length: int,
    load_in_4bit: bool,
) -> LoadResult:
    """Load via Unsloth's optimized loader. Raises on failure."""
    from unsloth import FastLanguageModel  # type: ignore

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=repo_id,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_16bit=not load_in_4bit,
        full_finetuning=False,
        fast_inference=True,
    )
    return LoadResult(
        model=model,
        tokenizer=tokenizer,
        via_unsloth=True,
        dtype="4bit" if load_in_4bit else "bf16",
        notes=[],
    )


def _fallback_transformers_load(repo_id: str) -> LoadResult:
    """Plain Transformers load as a measured fallback (no Unsloth patches)."""
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return LoadResult(
        model=model,
        tokenizer=tokenizer,
        via_unsloth=False,
        dtype="bf16",
        notes=["fallback: plain AutoModelForCausalLM, no Unsloth patching applied"],
    )


def load_model_and_tokenizer(
    repo_id: str = DEFAULT_REPO,
    max_seq_length: int = 2048,
    load_in_4bit: bool = False,
) -> LoadResult:
    """Load the model, preferring Unsloth bf16 LoRA, with a plain fallback.

    Raises the last error if both paths fail, with the exact message recorded.
    """
    errors: list[str] = []
    try:
        return _try_unsloth_load(repo_id, max_seq_length, load_in_4bit)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"unsloth load failed: {type(exc).__name__}: {exc}")
    try:
        return _fallback_transformers_load(repo_id)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"transformers load failed: {type(exc).__name__}: {exc}")
    raise RuntimeError("model load failed via all paths:\n" + "\n".join(errors))


def load_model_with_adapter(
    repo_id: str = DEFAULT_REPO,
    adapter_dir: str | None = None,
    max_seq_length: int = 2048,
    load_in_4bit: bool = False,
) -> LoadResult:
    """Load the base model and apply a saved LoRA adapter for inference."""
    result = load_model_and_tokenizer(repo_id, max_seq_length, load_in_4bit)
    if not adapter_dir:
        return result
    try:
        from peft import PeftModel  # type: ignore

        result.model = PeftModel.from_pretrained(result.model, adapter_dir)
        result.notes.append(f"adapter loaded from {adapter_dir}")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"failed to apply adapter {adapter_dir}: {type(exc).__name__}: {exc}")
    return result


def apply_chat_template(
    tokenizer: Any,
    messages: Sequence[dict[str, str]],
    enable_thinking: bool = False,
) -> Any:
    """Apply the model's chat template; returns tokenized inputs on the model device."""
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    # very old tokenizer fallback
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt")


def generate(
    model: Any,
    tokenizer: Any,
    messages: Sequence[dict[str, str]],
    gen_cfg: dict[str, Any] | None = None,
    enable_thinking: bool = False,
) -> str:
    """Generate text for a message list. Returns the raw decoded continuation."""
    import torch  # type: ignore

    cfg = {
        "max_new_tokens": 512,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "top_k": None,
    }
    if gen_cfg:
        cfg.update({k: v for k, v in gen_cfg.items() if v is not None})

    inputs = apply_chat_template(tokenizer, messages, enable_thinking=enable_thinking)
    inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    with torch.no_grad():
        outputs = model.generate(**inputs, **{k: v for k, v in cfg.items() if v is not None})

    input_len = inputs["input_ids"].shape[-1]
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

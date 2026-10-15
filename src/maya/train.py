"""Training: data formatting, supervised fine-tuning, checkpointing, sweep.

GPU imports (torch/unsloth/trl/datasets) are lazy. ``build_messages`` and
``answer_text`` are pure and CPU-testable; the tokenizer-dependent formatting
and the Trainer run are exercised in the Colab smoke test.

Training format: the Qwen3.5 chat template wraps the assistant turn as
`` thinking\\n\\n response\\n\\n<answer>`` when no reasoning is supplied, so
Maya is trained to emit concise, direct, auditable answers WITHOUT being forced
to reveal hidden chain-of-thought.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .data import Record, read_jsonl

ABSTAIN_REPLY = "I cannot answer this based on the available information."


def answer_text(record: Record) -> str:
    """Canonical assistant answer string for one record."""
    exp = record.expected
    if exp.abstain:
        return ABSTAIN_REPLY
    if exp.tool_calls:
        return ""  # tool calls are rendered as structured tool_calls, not text
    if exp.tool_call:
        return ""
    if exp.json is not None:
        return json.dumps(exp.json, ensure_ascii=False)
    if exp.numeric is not None:
        val = exp.numeric.get("value")
        return _fmt_number(val)
    if exp.answer is not None:
        return exp.answer
    if exp.label is not None:
        return exp.label
    return ""


def _fmt_number(val: Any) -> str:
    if isinstance(val, float) and val.is_integer():
        return str(int(val))
    return str(val)


def build_messages(record: Record, system_prompt: str | None = None) -> list[dict[str, Any]]:
    """Build a chat message list for one record, including tool calls when present.

    For tool/workflow tasks the assistant turn uses the native ``tool_calls``
    field so the chat template renders the Qwen3.5 function-calling block format.
    """
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # user messages come from the record; we keep only the final user turn as the
    # supervised prompt (multi-turn history is preserved when present).
    for m in record.messages:
        if m.get("role") == "user":
            messages.append({"role": "user", "content": m.get("content", "")})

    exp = record.expected
    assistant: dict[str, Any] = {"role": "assistant"}
    if exp.tool_calls:
        assistant["content"] = ""
        assistant["tool_calls"] = [
            {"name": c["name"], "arguments": c["arguments"]} for c in exp.tool_calls
        ]
    elif exp.tool_call:
        assistant["content"] = ""
        assistant["tool_calls"] = [
            {"name": exp.tool_call["name"], "arguments": exp.tool_call["arguments"]}
        ]
    else:
        assistant["content"] = answer_text(record)
    messages.append(assistant)
    return messages


def format_example(tokenizer: Any, record: Record, system_prompt: str | None = None) -> str:
    """Format one record to a training string using the tokenizer's chat template."""
    messages = build_messages(record, system_prompt)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def prepare_training_dataset(
    records: Sequence[Record],
    tokenizer: Any,
    system_prompt: str | None = None,
) -> dict[str, list[str]]:
    """Return a {"text": [...]} dict suitable for an HF Dataset / SFTTrainer."""
    texts = [format_example(tokenizer, r, system_prompt) for r in records]
    return {"text": texts}


def verify_prompt_masking(tokenizer: Any, record: Record, system_prompt: str | None = None) -> dict[str, Any]:
    """Decode a formatted example back and confirm the assistant answer survives.

    This validates that the chat template, special tokens (`` thinking`` /
    `` response``) and the answer are present and correctly rendered, which is
    the prerequisite for SFTTrainer's response masking to supervise only the
    intended assistant tokens.
    """
    text = format_example(tokenizer, record, system_prompt)
    expected = answer_text(record)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    roundtrip = tokenizer.decode(ids, skip_special_tokens=False)
    return {
        "formatted_text": text,
        "num_tokens": len(ids),
        "contains_response_token": " response" in roundtrip or " response" in text,
        "contains_answer": (expected in text) if expected else True,
        "answer_present": expected,
        "is_tool_call": bool(record.expected.tool_calls or record.expected.tool_call),
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    output_dir: str
    final_checkpoint: str | None
    best_checkpoint: str | None
    notes: list[str]


def run_training(
    repo_id: str,
    train_records: Sequence[Record],
    val_records: Sequence[Record],
    output_dir: str,
    config: dict[str, Any],
    resume: bool = True,
) -> TrainingResult:
    """Run an SFT run with Unsloth + TRL. GPU-only; raises a clear error if CUDA
    or Unsloth is unavailable so callers can distinguish 'not run' from 'failed'.
    """
    import torch  # type: ignore

    if not torch.cuda.is_available():
        raise RuntimeError("run_training requires a CUDA GPU; not available on this host")

    from trl import SFTConfig, SFTTrainer  # type: ignore
    from unsloth import FastLanguageModel  # type: ignore

    from .config import get_system_prompt

    tcfg = config["training_t4"]["training"]
    lcfg = config["training_t4"]["lora"]
    prec = config["training_t4"]["precision"]
    system_prompt = get_system_prompt(config)

    preferred = prec.get("preferred", "fp16_lora")
    load_in_4bit = preferred == "q4_qlora"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=repo_id,
        max_seq_length=tcfg["max_seq_length"],
        load_in_4bit=load_in_4bit,
        load_in_16bit=not load_in_4bit,
        full_finetuning=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lcfg["r"],
        target_modules=lcfg["target_modules"],
        lora_alpha=lcfg["lora_alpha"],
        lora_dropout=lcfg["lora_dropout"],
        bias=lcfg["bias"],
        use_gradient_checkpointing=tcfg.get("gradient_checkpointing", "unsloth"),
        random_state=tcfg["seed"],
        max_seq_length=tcfg["max_seq_length"],
    )

    train_ds = _to_hf_dataset(prepare_training_dataset(train_records, tokenizer, system_prompt))
    val_ds = _to_hf_dataset(prepare_training_dataset(val_records, tokenizer, system_prompt))

    args = SFTConfig(
        max_seq_length=tcfg["max_seq_length"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        warmup_steps=tcfg["warmup_steps"],
        max_steps=tcfg["max_steps"],
        learning_rate=tcfg["learning_rate"],
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        weight_decay=tcfg["weight_decay"],
        optim=tcfg["optim"],
        logging_steps=tcfg["logging_steps"],
        save_steps=tcfg["save_steps"],
        eval_strategy=tcfg["eval_strategy"],
        eval_steps=tcfg["eval_steps"],
        metric_for_best_model=tcfg["metric_for_best_model"],
        load_best_model_at_end=tcfg["load_best_model_at_end"],
        gradient_checkpointing=tcfg["gradient_checkpointing"] == "unsloth",
        seed=tcfg["seed"],
        output_dir=output_dir,
        fp16=tcfg.get("fp16", True),
        bf16=tcfg.get("bf16", False),
        report_to=tcfg.get("report_to", "none"),
        resume_from_checkpoint=True if resume else False,
        dataset_num_proc=tcfg.get("dataset_num_proc", 1),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=args,
    )

    import time

    t_start = time.perf_counter()
    trainer.train()
    duration_s = time.perf_counter() - t_start

    final_ckpt = None
    if tcfg.get("save_adapter", True):
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        final_ckpt = output_dir

    _write_training_log(
        output_dir,
        trainer.state.log_history,
        trainer.state.global_step,
        duration_s=duration_s,
        best_checkpoint=getattr(trainer.state, "best_model_checkpoint", None),
    )

    return TrainingResult(
        output_dir=output_dir,
        final_checkpoint=final_ckpt,
        best_checkpoint=getattr(trainer.state, "best_model_checkpoint", None),
        notes=[],
    )


def _write_training_log(
    output_dir: str,
    log_history: list[dict[str, Any]],
    global_step: int,
    duration_s: float | None = None,
    best_checkpoint: str | None = None,
) -> None:
    """Persist a normalized training log (loss/lr/grad-norm per step, duration,
    throughput and peak memory) to JSON."""
    from pathlib import Path

    steps = []
    runtime = throughput = samples_per_sec = None
    for entry in log_history:
        if "loss" in entry or "eval_loss" in entry:
            steps.append(
                {
                    "step": entry.get("step"),
                    "loss": entry.get("loss"),
                    "eval_loss": entry.get("eval_loss"),
                    "learning_rate": entry.get("learning_rate"),
                    "grad_norm": entry.get("grad_norm"),
                }
            )
        # TRL writes throughput/runtime in the final log entry
        runtime = runtime or entry.get("train_runtime")
        samples_per_sec = samples_per_sec or entry.get("train_samples_per_second")
        throughput = entry.get("train_steps_per_second") or throughput

    memory = _gpu_memory_gb()

    log = {
        "final_step": global_step,
        "duration_s": round(duration_s, 2) if duration_s is not None else None,
        "train_runtime_s": runtime,
        "samples_per_second": samples_per_sec,
        "steps_per_second": throughput,
        "gpu_memory_gb": memory,
        "best_checkpoint": best_checkpoint,
        "steps": steps,
    }
    out = Path(output_dir).parent / "training_log.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    import json

    out.write_text(json.dumps(log, indent=2), encoding="utf-8")


def _gpu_memory_gb() -> dict[str, Any] | None:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return {
                "peak_allocated": round(torch.cuda.max_memory_allocated() / (1024**3), 2),
                "peak_reserved": round(torch.cuda.max_memory_reserved() / (1024**3), 2),
            }
    except Exception:
        pass
    return None


def _to_hf_dataset(data: dict[str, list[str]]) -> Any:
    from datasets import Dataset  # type: ignore

    return Dataset.from_dict(data)


def run_training_transformers(
    repo_id: str,
    train_records: Sequence[Record],
    val_records: Sequence[Record],
    output_dir: str,
    config: dict[str, Any],
    load_in_4bit: bool = True,
    resume: bool = True,
) -> TrainingResult:
    """Plain-Transformers QLoRA training path (no Unsloth).

    Unsloth's patcher forces float32 for Qwen3.5 (T4 has no bf16), which OOMs a
    16GB T4. Plain Transformers + bitsandbytes 4-bit quantizes the *weights*
    (~2.5GB) while only the small linear-attention SSM state stays float32, so
    it fits. LoRA targets all Linear layers via ``target_modules="all-linear"``.
    """
    import torch  # type: ignore

    if not torch.cuda.is_available():
        raise RuntimeError("run_training_transformers requires a CUDA GPU")

    from transformers import (  # type: ignore
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore
    from trl import SFTConfig, SFTTrainer  # type: ignore

    from .config import get_system_prompt

    tcfg = config["training_t4"]["training"]
    lcfg = config["training_t4"]["lora"]
    system_prompt = get_system_prompt(config)

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,  # T4 fp16 compute is fine for the LoRA path
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            repo_id, quantization_config=bnb, device_map="auto"
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            repo_id, torch_dtype=torch.bfloat16, device_map="auto"
        )

    peft_config = LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["lora_alpha"],
        lora_dropout=lcfg["lora_dropout"],
        bias=lcfg["bias"],
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, peft_config)

    train_ds = _to_hf_dataset(prepare_training_dataset(train_records, tokenizer, system_prompt))
    val_ds = _to_hf_dataset(prepare_training_dataset(val_records, tokenizer, system_prompt))

    args = SFTConfig(
        max_length=tcfg["max_seq_length"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        warmup_steps=tcfg["warmup_steps"],
        max_steps=tcfg["max_steps"],
        learning_rate=tcfg["learning_rate"],
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        weight_decay=tcfg["weight_decay"],
        optim=tcfg["optim"],
        logging_steps=tcfg["logging_steps"],
        save_steps=tcfg["save_steps"],
        eval_strategy=tcfg["eval_strategy"],
        eval_steps=tcfg["eval_steps"],
        metric_for_best_model=tcfg["metric_for_best_model"],
        load_best_model_at_end=tcfg["load_best_model_at_end"],
        gradient_checkpointing=True,
        seed=tcfg["seed"],
        output_dir=output_dir,
        bf16=(not load_in_4bit),
        fp16=load_in_4bit,
        report_to=tcfg.get("report_to", "none"),
        resume_from_checkpoint=True if resume else False,
        dataset_num_proc=tcfg.get("dataset_num_proc", 1),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=args,
    )

    import time

    t0 = time.perf_counter()
    trainer.train()
    duration_s = time.perf_counter() - t0

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    _write_training_log(
        output_dir,
        trainer.state.log_history,
        trainer.state.global_step,
        duration_s=duration_s,
        best_checkpoint=getattr(trainer.state, "best_model_checkpoint", None),
    )

    return TrainingResult(
        output_dir=output_dir,
        final_checkpoint=output_dir,
        best_checkpoint=getattr(trainer.state, "best_model_checkpoint", None),
        notes=["plain Transformers QLoRA (no Unsloth)"],
    )


def run_sweep(
    repo_id: str,
    train_records: Sequence[Record],
    val_records: Sequence[Record],
    output_dir: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Small controlled sweep over LoRA rank and learning rate (validation only).

    Returns a summary of runs for checkpoint selection. The final test set is
    never touched here.
    """
    import itertools  # noqa: F401

    sweep = config["training_t4"]["sweep"]
    results = []
    for lr, r in itertools.product(sweep["learning_rate"], sweep["lora_r"]):
        sub = json.loads(json.dumps(config))
        sub["training_t4"]["training"]["learning_rate"] = lr
        sub["training_t4"]["training"]["max_steps"] = sweep["max_steps"]
        sub["training_t4"]["lora"]["r"] = r
        sub["training_t4"]["lora"]["lora_alpha"] = r
        out = f"{output_dir}/sweep_lr{lr}_r{r}"
        res = run_training(repo_id, train_records, val_records, out, sub, resume=False)
        results.append({"learning_rate": lr, "lora_r": r, "output_dir": out, "checkpoint": res.best_checkpoint})
        if len(results) >= sweep.get("max_combinations", 6):
            break
    return results


def load_records_for_training(paths: Sequence[str]) -> list[Record]:
    out: list[Record] = []
    for p in paths:
        out.extend(read_jsonl(__import__("pathlib").Path(p)))
    return out

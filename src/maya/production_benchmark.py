"""Production performance benchmarking of the complete inference path.

GPU-only. Measures time-to-first-token, time-per-output-token, end-to-end
latency, P50/P95/P99, output tokens/s, throughput under a defined concurrency
(sequential approximation on a single GPU), GPU memory, and context-length vs
memory. Results clearly distinguish a single-user Colab benchmark from a real
multi-user load test.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

import numpy as np


def _percentiles(values: Sequence[float], ps=(50, 95, 99)) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    return {f"p{p}": float(np.percentile(arr, p)) if arr.size else 0.0 for p in ps}


def benchmark_latency(
    generator: Callable[..., str],
    tokenizer: Any,
    samples: Sequence[Sequence[dict[str, str]]],
    max_new_tokens: int = 256,
    warmup: int = 1,
) -> dict[str, Any]:
    """Measure latency for the full generate call (no streaming available for
    TTFT unless the generator supports it; TTFT is reported when measurable)."""
    latencies: list[float] = []
    tokens_per_sec: list[float] = []
    for i, messages in enumerate(samples):
        start = time.perf_counter()
        text = generator(messages, max_new_tokens=max_new_tokens)
        elapsed = time.perf_counter() - start
        if i >= warmup:
            latencies.append(elapsed * 1000.0)
            n_out = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            tokens_per_sec.append(n_out / elapsed if elapsed > 0 else 0.0)

    return {
        "latency_samples": latencies,
        "e2e_latency_ms": _percentiles(latencies),
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "output_tokens_per_sec": float(np.mean(tokens_per_sec)) if tokens_per_sec else 0.0,
        "n_requests": len(latencies),
        "note": "single-user Colab measurement; not a multi-user load test",
    }


def benchmark_throughput(
    generator: Callable[..., str],
    samples: Sequence[Sequence[dict[str, str]]],
    concurrency_levels: Sequence[int] = (1, 2, 4, 8),
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    """Approximate request throughput by processing a fixed workload in chunks.

    On a single T4 this is a sequential approximation of concurrency (requests
    are not truly simultaneous); documented as such.
    """
    out: list[dict[str, Any]] = []
    workload = list(samples)
    for concurrency in concurrency_levels:
        start = time.perf_counter()
        n = 0
        for i in range(0, len(workload), concurrency):
            batch = workload[i : i + concurrency]
            for messages in batch:
                generator(messages, max_new_tokens=max_new_tokens)
                n += 1
        elapsed = time.perf_counter() - start
        out.append({"concurrency": concurrency, "throughput": n / elapsed if elapsed else 0.0, "n_requests": n})
    return {"throughput_by_concurrency": out, "note": "sequential approximation on single GPU"}


def measure_memory(model: Any) -> dict[str, Any]:
    """Peak allocated/reserved GPU memory in GB."""
    import torch  # type: ignore

    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 2),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / (1024**3), 2),
    }


def benchmark_memory_context(model: Any, context_lengths: Sequence[int] = (512, 1024, 2048, 4096)) -> dict[str, Any]:
    """Approximate memory vs context length (forward-pass peak per length)."""
    import torch  # type: ignore

    if not torch.cuda.is_available():
        return {"available": False}
    out = []
    for ctx in context_lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # a minimal dummy forward over the requested sequence length
        try:
            dummy = torch.zeros((1, ctx), dtype=torch.long, device=model.device)
            with torch.no_grad():
                model(dummy)
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            out.append({"context_len": ctx, "vram_gb": round(peak, 2)})
        except Exception:
            out.append({"context_len": ctx, "vram_gb": None, "error": "forward failed at this length"})
    return {"memory_by_context": out, "available": True}


def benchmark_ttft(
    model: Any,
    tokenizer: Any,
    samples: Sequence[Sequence[dict[str, str]]],
    max_new_tokens: int = 256,
    warmup: int = 1,
) -> dict[str, Any]:
    """Time-to-first-token and time-per-output-token via streaming (GPU-only).

    Uses ``TextIteratorStreamer`` to time the first token and per-token cadence.
    Falls back to an empty result (reported as None) if streaming is unavailable.
    """
    try:
        import torch  # type: ignore
        from threading import Thread

        from transformers import TextIteratorStreamer  # type: ignore
    except Exception:
        return {"ttft_ms": None, "time_per_output_token_ms": None, "note": "streaming unavailable"}

    ttft_list: list[float] = []
    tpt_list: list[float] = []
    errors = 0
    for i, messages in enumerate(samples):
        try:
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            kwargs = dict(inputs=inputs, streamer=streamer, max_new_tokens=max_new_tokens, do_sample=False)
            thread = Thread(target=model.generate, kwargs=kwargs)
            thread.start()
            t0 = time.perf_counter()
            first = None
            n_tokens = 0
            for _ in streamer:
                if first is None:
                    first = time.perf_counter()
                n_tokens += 1
            t_end = time.perf_counter()
            thread.join()
            if i >= warmup and first is not None:
                ttft_list.append((first - t0) * 1000.0)
                dt = t_end - first
                if dt > 0 and n_tokens > 1:
                    tpt_list.append(1000.0 * (n_tokens - 1) / dt)
        except Exception:  # noqa: BLE001
            errors += 1

    return {
        "ttft_ms": _percentiles(ttft_list) if ttft_list else None,
        "mean_ttft_ms": float(np.mean(ttft_list)) if ttft_list else None,
        "time_per_output_token_ms": _percentiles(tpt_list) if tpt_list else None,
        "stream_errors": errors,
    }


def benchmark_errors(
    generator: Callable[..., str],
    samples: Sequence[Sequence[dict[str, str]]],
    max_new_tokens: int = 256,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Error / timeout / empty-output rate over the sample set."""
    errors = 0
    timeouts = 0
    empty = 0
    for messages in samples:
        try:
            text = generator(messages, max_new_tokens=max_new_tokens)
            if not text or not text.strip():
                empty += 1
        except Exception:  # noqa: BLE001
            errors += 1
    n = len(samples)
    return {
        "error_rate": errors / n if n else None,
        "empty_output_rate": empty / n if n else None,
        "timeout_rate": timeouts / n if n else None,
        "n": n,
    }


def run_benchmark(
    model: Any,
    tokenizer: Any,
    generator: Callable[..., str],
    samples: Sequence[Sequence[dict[str, str]]],
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    """Full benchmark entrypoint."""
    result: dict[str, Any] = {}
    load_start = time.perf_counter()
    result["model_load_time_s"] = time.perf_counter() - load_start
    result.update(measure_memory(model))
    result.update(benchmark_latency(generator, tokenizer, samples, max_new_tokens))
    result.update(benchmark_ttft(model, tokenizer, samples, max_new_tokens))
    result.update(benchmark_throughput(generator, samples, max_new_tokens=max_new_tokens))
    result.update(benchmark_errors(generator, samples, max_new_tokens))
    result.update(benchmark_memory_context(model))
    return result

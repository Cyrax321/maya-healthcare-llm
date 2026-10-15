"""Figure generation.

Every figure is exported only when the underlying measured data exists. Each
plot carries a title, axis labels, units, sample count and a legend. PNG and SVG
are written to ``artifacts/<id>/plots/``. Figures are never fabricated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _save(fig, out_dir: Path, name: str, png: bool, svg: bool) -> None:
    if png:
        fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    if svg:
        fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------


def plot_training_curves(training_log: dict[str, Any], out_dir: Path, png=True, svg=True) -> None:
    steps = training_log.get("steps", [])
    if not steps:
        return
    train_loss = [s.get("loss") for s in steps if s.get("loss") is not None]
    val_loss = [s.get("eval_loss") for s in steps if s.get("eval_loss") is not None]
    lr = [s.get("learning_rate") for s in steps if s.get("learning_rate") is not None]
    grad = [s.get("grad_norm") for s in steps if s.get("grad_norm") is not None]
    x = [s.get("step") for s in steps]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x[: len(train_loss)], train_loss, label=f"train loss (n={len(train_loss)})")
    ax.plot(x[: len(val_loss)], val_loss, label=f"val loss (n={len(val_loss)})")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Training and validation loss")
    ax.legend()
    _save(fig, out_dir, "01_training_loss", png, svg)

    fig, ax = plt.subplots(figsize=(8, 4))
    if lr:
        ax.plot(x[: len(lr)], lr, label="learning rate")
    if grad:
        ax.plot(x[: len(grad)], grad, label="gradient norm")
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.set_title("Learning rate and gradient norm")
    ax.legend()
    _save(fig, out_dir, "02_lr_gradnorm", png, svg)

    if train_loss and val_loss:
        min_len = min(len(train_loss), len(val_loss))
        gap = [abs(train_loss[i] - val_loss[i]) for i in range(min_len)]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x[:min_len], gap, label=f"loss gap (n={min_len})")
        ax.set_xlabel("step")
        ax.set_ylabel("|train - val|")
        ax.set_title("Training/validation loss gap")
        ax.legend()
        _save(fig, out_dir, "03_loss_gap", png, svg)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def plot_confusion_matrices(metrics: dict[str, Any], out_dir: Path, png=True, svg=True) -> None:
    cls = metrics.get("classification", {})
    for cat, report in cls.items():
        labels = report.get("labels", [])
        cm = np.array(report.get("confusion_matrix", []))
        if cm.size == 0:
            continue
        variants = {"raw": cm, "row": _row_norm(cm), "col": _col_norm(cm)}
        for variant, mat in variants.items():
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(mat, cmap="Blues")
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticklabels(labels)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title(f"{cat} confusion matrix ({variant}-normalized)")
            fig.colorbar(im, ax=ax)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(j, i, f"{mat[i, j]:.2f}" if variant != "raw" else f"{int(mat[i, j])}",
                            ha="center", va="center", fontsize=8)
            _save(fig, out_dir, f"04_cm_{cat}_{variant}", png, svg)


def _row_norm(cm: np.ndarray) -> np.ndarray:
    s = cm.sum(axis=1, keepdims=True)
    return cm / np.where(s == 0, 1, s)


def _col_norm(cm: np.ndarray) -> np.ndarray:
    s = cm.sum(axis=0, keepdims=True)
    return cm / np.where(s == 0, 1, s)


def plot_per_class_metrics(metrics: dict[str, Any], out_dir: Path, png=True, svg=True) -> None:
    for cat, report in metrics.get("classification", {}).items():
        labels = report.get("labels", [])
        per = report.get("per_class", {})
        if not labels:
            continue
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x - 0.2, [per.get(l, {}).get("precision", 0) for l in labels], width=0.2, label="precision")
        ax.bar(x, [per.get(l, {}).get("recall", 0) for l in labels], width=0.2, label="recall")
        ax.bar(x + 0.2, [per.get(l, {}).get("f1", 0) for l in labels], width=0.2, label="f1")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("score")
        ax.set_title(f"{cat} per-class precision/recall/F1")
        ax.legend()
        _save(fig, out_dir, f"05_per_class_{cat}", png, svg)


# ---------------------------------------------------------------------------
# Numeric error distribution
# ---------------------------------------------------------------------------


def plot_numeric_errors(results: Sequence[dict[str, Any]], out_dir: Path, png=True, svg=True) -> None:
    pairs = []
    for r in results:
        gold = r.get("gold", {})
        num = gold.get("numeric")
        pred_val = r.get("pred", {}).get("numeric")
        if num is not None and pred_val is not None:
            pairs.append((num["value"], pred_val))
    if not pairs:
        return
    errs = [p - v for v, p in pairs]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errs, bins=20)
    ax.set_xlabel("predicted - expected")
    ax.set_ylabel("count")
    ax.set_title(f"Numeric error distribution (n={len(errs)})")
    _save(fig, out_dir, "08_numeric_error_hist", png, svg)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter([v for v, _ in pairs], [p for _, p in pairs], s=20, alpha=0.6)
    lim = [min(min(v, p) for v, p in pairs), max(max(v, p) for v, p in pairs)]
    ax.plot(lim, lim, "r--", label="perfect")
    ax.set_xlabel("expected")
    ax.set_ylabel("predicted")
    ax.set_title(f"Expected vs predicted (n={len(pairs)})")
    ax.legend()
    _save(fig, out_dir, "08_numeric_scatter", png, svg)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def plot_retrieval_curves(metrics: dict[str, Any], out_dir: Path, png=True, svg=True) -> None:
    ret = metrics.get("retrieval", {})
    ks = sorted(int(k.split("@")[1]) for k in ret if k.startswith("precision@"))
    if not ks:
        return
    p = [ret.get(f"precision@{k}", 0) for k in ks]
    r = [ret.get(f"recall@{k}", 0) for k in ks]
    n = [ret.get(f"ndcg@{k}", 0) for k in ks]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, p, "o-", label="precision@k")
    ax.plot(ks, r, "s-", label="recall@k")
    ax.plot(ks, n, "^-", label="nDCG@k")
    ax.set_xlabel("k")
    ax.set_ylabel("score")
    ax.set_title("Retrieval precision/recall/nDCG vs k")
    ax.legend()
    _save(fig, out_dir, "10_retrieval_curves", png, svg)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


def plot_tool_metrics(metrics: dict[str, Any], out_dir: Path, png=True, svg=True) -> None:
    tool = metrics.get("tool", {})
    if not tool:
        return
    keys = ["tool_selection_accuracy", "argument_exact_match_rate", "tool_execution_success_rate", "end_to_end_completion_rate"]
    vals = [tool.get(k) for k in keys if tool.get(k) is not None]
    labels = [k.replace("_", " ") for k in keys if tool.get(k) is not None]
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(vals)), vals)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("rate")
    ax.set_title(f"Tool calling metrics (n={tool.get('total')})")
    _save(fig, out_dir, "12_tool_metrics", png, svg)


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------


def plot_comparison_ci(comparison: dict[str, Any], out_dir: Path, png=True, svg=True) -> None:
    ci = comparison.get("paired_bootstrap_diff_ci", {})
    if ci.get("n") is None:
        return
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axvline(0, color="gray", linestyle="--")
    ax.errorbar(ci["mean_diff"], 0, xerr=[[ci["mean_diff"] - ci["ci_low"]], [ci["ci_high"] - ci["mean_diff"]]], fmt="o", capsize=5)
    ax.set_xlabel("improvement (adapter - baseline)")
    ax.set_yticks([])
    ax.set_title(f"Paired bootstrap difference (n={ci['n']})")
    _save(fig, out_dir, "14_comparison_ci", png, svg)


# ---------------------------------------------------------------------------
# Production benchmark
# ---------------------------------------------------------------------------


def plot_latency_histogram(benchmark: dict[str, Any], out_dir: Path, png=True, svg=True) -> None:
    samples = benchmark.get("latency_samples", [])
    if not samples:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(samples, bins=30)
    ax.set_xlabel("latency (ms)")
    ax.set_ylabel("count")
    ax.set_title(f"Latency histogram (n={len(samples)})")
    _save(fig, out_dir, "15_latency_hist", png, svg)

    sorted_s = np.sort(samples)
    cdf = np.arange(1, len(sorted_s) + 1) / len(sorted_s)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sorted_s, cdf)
    ax.set_xlabel("latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Latency CDF")
    _save(fig, out_dir, "15_latency_cdf", png, svg)


def plot_throughput(benchmark: dict[str, Any], out_dir: Path, png=True, svg=True) -> None:
    tp = benchmark.get("throughput_by_concurrency", [])
    if not tp:
        return
    con = [t["concurrency"] for t in tp]
    rate = [t["throughput"] for t in tp]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(con, rate, "o-")
    ax.set_xlabel("concurrency")
    ax.set_ylabel("requests/s")
    ax.set_title("Throughput vs concurrency")
    _save(fig, out_dir, "16_throughput", png, svg)


def plot_memory_context(benchmark: dict[str, Any], out_dir: Path, png=True, svg=True) -> None:
    mem = benchmark.get("memory_by_context", [])
    if not mem:
        return
    ctx = [m["context_len"] for m in mem]
    vram = [m["vram_gb"] for m in mem]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ctx, vram, "o-")
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("GPU memory (GB)")
    ax.set_title("Memory vs context length")
    _save(fig, out_dir, "17_memory_context", png, svg)


# ---------------------------------------------------------------------------
# Error summary
# ---------------------------------------------------------------------------


def plot_error_summary(results: Sequence[dict[str, Any]], out_dir: Path, png=True, svg=True) -> None:
    from collections import Counter

    counts: Counter = Counter()
    for r in results:
        for e in r.get("errors", []):
            if e != "none":
                counts[e] += 1
    if not counts:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    items = counts.most_common()
    ax.bar(range(len(items)), [i[1] for i in items])
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([i[0] for i in items], rotation=45, ha="right")
    ax.set_ylabel("count")
    ax.set_title("Failure categories across test set")
    _save(fig, out_dir, "19_error_summary", png, svg)


def plot_all(
    out_dir: Path,
    metrics: dict[str, Any] | None = None,
    results: Sequence[dict[str, Any]] | None = None,
    training_log: dict[str, Any] | None = None,
    benchmark: dict[str, Any] | None = None,
    comparison: dict[str, Any] | None = None,
    png: bool = True,
    svg: bool = True,
) -> list[str]:
    """Dispatch all applicable figures; returns the list of generated stems."""
    generated: list[str] = []
    before = set(Path(out_dir).glob("*")) if Path(out_dir).exists() else set()

    metrics = metrics or {}
    results = list(results or [])
    if training_log:
        plot_training_curves(training_log, out_dir, png, svg)
    if metrics.get("classification"):
        plot_confusion_matrices(metrics, out_dir, png, svg)
        plot_per_class_metrics(metrics, out_dir, png, svg)
    if metrics.get("numeric") and results:
        plot_numeric_errors(results, out_dir, png, svg)
    if metrics.get("retrieval"):
        plot_retrieval_curves(metrics, out_dir, png, svg)
    if metrics.get("tool"):
        plot_tool_metrics(metrics, out_dir, png, svg)
    if comparison:
        plot_comparison_ci(comparison, out_dir, png, svg)
    if benchmark:
        plot_latency_histogram(benchmark, out_dir, png, svg)
        plot_throughput(benchmark, out_dir, png, svg)
        plot_memory_context(benchmark, out_dir, png, svg)
    if results:
        plot_error_summary(results, out_dir, png, svg)

    after = set(Path(out_dir).glob("*"))
    return sorted(p.stem for p in after - before if p.suffix in (".png", ".svg"))

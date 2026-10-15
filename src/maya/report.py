"""Research report generation.

Produces ``artifacts/<id>/report.md`` stating the exact model revision, adapter,
dataset split, configuration, sample sizes, metric tables, uncertainty and the
failure list. The report never claims an improvement the recorded evaluation
does not support, and it always distinguishes real model outputs from stubs and
single-user Colab benchmarks from production load tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    header = rows[0]
    sep = ["---"] * len(header)
    body = ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join([body[0], "| " + " | ".join(sep) + " |"] + body[1:])


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def generate_report(
    experiment_id: str,
    metrics: dict[str, Any],
    out_dir: Path,
    config: dict[str, Any] | None = None,
    comparison: dict[str, Any] | None = None,
    benchmark: dict[str, Any] | None = None,
    training_log: dict[str, Any] | None = None,
    dataset_audit: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> Path:
    exp = metrics.get("experiment", {})
    lines: list[str] = []
    lines.append(f"# Maya Evaluation Report — {experiment_id}")
    lines.append("")
    lines.append(f"- Experiment: **{exp.get('experiment_name', '?')}**")
    lines.append(f"- Model: `{exp.get('model_ref', '?')}`")
    lines.append(f"- Adapter: `{exp.get('adapter_ref') or 'none'}`")
    lines.append(f"- Retrieval enabled: {exp.get('retrieval_enabled', False)}")
    lines.append(f"- Tools enabled: {exp.get('tools_enabled', False)}")
    lines.append(f"- Examples evaluated: {exp.get('n_examples', 0)}")

    if config:
        model_cfg = config.get("model", {}).get("model", {})
        lines.append(f"- Model revision (resolved): `{config.get('model_revision') or 'unresolved (offline)'}`")
        lines.append(f"- Base model: `{model_cfg.get('base_model', '?')}`")
        lines.append(f"- Architecture: `{model_cfg.get('architecture', '?')}`")

    if dataset_audit:
        lines.append("")
        lines.append("## Dataset")
        lines.append(f"- Total rows: {dataset_audit.get('total', 0)}")
        lines.append(f"- Valid rows: {dataset_audit.get('valid', 0)}")
        lines.append(f"- Exact duplicates removed: {dataset_audit.get('duplicate_exact', 0)}")
        lines.append(f"- Near-duplicate pairs: {dataset_audit.get('duplicate_near', 0)}")
        splits = dataset_audit.get("splits", {})
        if splits:
            lines.append(f"- Split: train={splits.get('train')} validation={splits.get('validation')} test={splits.get('test')}")

    lines.append("")
    lines.append("## Results")

    # classification
    cls = metrics.get("classification", {})
    if cls:
        lines.append("")
        lines.append("### Classification")
        for cat, report in cls.items():
            lines.append(f"**{cat}** (accuracy={_fmt(report.get('accuracy'))}, "
                         f"macro-F1={_fmt(report.get('macro_f1'))}, n={report.get('support_total', 0)})")
            rows = [["label", "precision", "recall", "f1", "support"]]
            for lab, v in report.get("per_class", {}).items():
                rows.append([lab, _fmt(v.get("precision")), _fmt(v.get("recall")), _fmt(v.get("f1")), v.get("support")])
            lines.append(_table(rows))

    qa = metrics.get("qa")
    if qa:
        lines.append("")
        lines.append("### Question answering")
        lines.append(f"- Exact match: {_fmt(qa.get('exact_match'))}")
        lines.append(f"- Normalized exact match: {_fmt(qa.get('normalized_exact_match'))}")
        lines.append(f"- Token F1: {_fmt(qa.get('token_f1'))}")
        lines.append(f"- n: {qa.get('total')}")

    numeric = metrics.get("numeric")
    if numeric:
        lines.append("")
        lines.append("### Numeric correctness")
        lines.append(f"- Accuracy (within tolerance): {_fmt(numeric.get('accuracy'))} (n={numeric.get('total')})")

    abst = metrics.get("abstention")
    if abst:
        lines.append("")
        lines.append("### Abstention")
        lines.append(f"- Correct abstention rate: {_fmt(abst.get('correct_abstention_rate'))} (n={abst.get('total')})")

    struc = metrics.get("structured")
    if struc:
        lines.append("")
        lines.append("### Structured output")
        lines.append(f"- JSON parse rate: {_fmt(struc['json_parse'].get('rate'))}")
        lines.append(f"- Field accuracies: {struc['fields'].get('field_accuracy')}")

    ret = metrics.get("retrieval")
    if ret:
        lines.append("")
        lines.append("### Retrieval")
        for k, v in ret.items():
            if isinstance(v, float):
                lines.append(f"- {k}: {_fmt(v)}")

    tool = metrics.get("tool")
    if tool:
        lines.append("")
        lines.append("### Tool calling")
        for k, v in tool.items():
            if isinstance(v, float) and v is not None:
                lines.append(f"- {k}: {_fmt(v)}")
            elif k in ("unauthorized_attempts", "successful_unauthorized_actions"):
                lines.append(f"- {k}: {v}")

    if comparison:
        lines.append("")
        lines.append("### Statistical comparison vs baseline")
        lines.append(f"- Baseline rate: {_fmt(comparison.get('baseline_rate'))}")
        lines.append(f"- Adapter rate: {_fmt(comparison.get('adapter_rate'))}")
        lines.append(f"- Absolute improvement: {_fmt(comparison.get('absolute_improvement'))}")
        lines.append(f"- McNemar p-value: {_fmt(comparison.get('mcnemar', {}).get('p_value'))}")
        ci = comparison.get("paired_bootstrap_diff_ci", {})
        lines.append(f"- Paired bootstrap diff 95% CI: [{_fmt(ci.get('ci_low'))}, {_fmt(ci.get('ci_high'))}]")

    if benchmark:
        lines.append("")
        lines.append("### Production benchmark (single-user Colab, not a load test)")
        lat = benchmark.get("e2e_latency_ms", {})
        lines.append(f"- Mean latency: {_fmt(benchmark.get('mean_latency_ms'))} ms")
        lines.append(f"- P50/P95/P99: {_fmt(lat.get('p50'))} / {_fmt(lat.get('p95'))} / {_fmt(lat.get('p99'))} ms")
        lines.append(f"- Output tokens/s: {_fmt(benchmark.get('output_tokens_per_sec'))}")

    if training_log:
        lines.append("")
        lines.append("### Training")
        lines.append(f"- Final step: {training_log.get('final_step', '?')}")
        lines.append(f"- Best checkpoint: {training_log.get('best_checkpoint', 'none')}")

    if notes:
        lines.append("")
        lines.append("## Notes")
        for n in notes:
            lines.append(f"- {n}")

    lines.append("")
    lines.append("## Failure cases")
    lines.append("See `errors.jsonl` for per-example, category-filtered failures.")

    path = out_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

"""End-to-end evaluation orchestration.

Runs a locked held-out test set through a generator (real model or a clearly
labeled stub), executes deterministic tools where enabled, scores every example
against verified ground truth, aggregates metrics by task type, and writes the
per-example predictions/errors plus summary metrics. Statistical comparison of
two conditions on identical cases (e.g. baseline vs adapter) uses paired tests.

Only real model outputs are reported as results. Stub-generated artifacts are
labeled as such and never presented as model performance.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .data import Record
from .inference import build_pred
from .metrics import (
    classification_report,
    field_accuracy,
    json_parse_success,
    mcnemar,
    normalized_exact_match,
    numeric_correct,
    retrieval_metrics,
    schema_validity,
    token_f1,
    tool_metrics,
)
from .retrieval import BM25Retriever, retrieval_context
from .tools import ToolRegistry, evaluate_tool_execution

CATEGORY_LABELS: dict[str, list[str]] = {
    "appointment_scheduling": ["available", "conflict"],
    "bed_occupancy": ["over", "under"],
    "workforce_rostering": ["understaffed", "covered"],
    "or_scheduling": ["fits", "overbooked"],
    "equipment_maintenance": ["overdue", "current"],
    "inventory_supply": ["reorder", "ok"],
}


def _gold_dict(record: Record) -> dict[str, Any]:
    exp = record.expected
    return {
        "answer": exp.answer,
        "label": exp.label,
        "numeric": exp.numeric,
        "json": exp.json,
        "tool_call": exp.tool_call,
        "tool_calls": exp.tool_calls,
        "abstain": exp.abstain,
        "unsupported_claim": exp.unsupported_claim,
    }


def _run_example(
    record: Record,
    generator: Callable[..., str],
    retrieval: BM25Retriever | None,
    tools_enabled: bool,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    labels = CATEGORY_LABELS.get(record.category)

    messages: list[dict[str, str]] = [dict(m) for m in record.messages]
    query = " ".join(m.get("content", "") for m in record.messages if m.get("role") == "user")

    # Build a single system message (identity + optional retrieval context); the
    # Qwen3.5 chat template permits only one system message at the front.
    system_parts: list[str] = []
    retrieved: list[str] = []
    if retrieval is not None:
        retrieved, context = retrieval_context(retrieval, query, k=3)
        if context:
            system_parts.append("Use only the following approved sources to answer.\n\n" + context)
    if system_prompt:
        system_parts.insert(0, system_prompt)
    if system_parts:
        messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})

    text = generator(messages)
    pred = build_pred(text, labels=labels)

    gold = _gold_dict(record)

    tool = None
    if tools_enabled and (record.expected.tool_call or record.expected.tool_calls):
        tool = evaluate_tool_execution(gold, pred.get("tool_calls"), ToolRegistry())

    return {
        "id": record.id,
        "category": record.category,
        "task_type": record.task_type,
        "pred": pred,
        "gold": gold,
        "retrieval": {"retrieved": retrieved, "relevant": list(record.evidence or record.citations)} if retrieval is not None else None,
        "tool": tool,
    }


def _json_correct(pred_json: dict[str, Any] | None, gold_json: dict[str, Any] | None) -> bool:
    if not isinstance(gold_json, dict):
        return True
    if not isinstance(pred_json, dict):
        return False
    return all(pred_json.get(k) == v for k, v in gold_json.items())


def _correctness(result: dict[str, Any]) -> dict[str, bool]:
    pred = result["pred"]
    gold = result["gold"]
    out: dict[str, bool] = {}

    if gold.get("label") is not None:
        out["classification_correct"] = pred.get("label") == gold["label"]

    if gold.get("numeric") is not None:
        out["numeric_correct"] = numeric_correct(
            pred.get("numeric"),
            gold["numeric"]["value"],
            gold["numeric"].get("tolerance", 0.0),
            pred_text=pred.get("text"),
        )

    if gold.get("json") is not None:
        out["json_correct"] = _json_correct(pred.get("json"), gold["json"])
        out["json_parsed"] = isinstance(pred.get("json"), dict)

    if gold.get("answer") is not None:
        out["exact_match"] = pred.get("text", "") == gold["answer"]
        out["normalized_exact_match"] = normalized_exact_match(gold["answer"], pred.get("text"))

    if gold.get("abstain"):
        out["correct_abstention"] = bool(pred.get("abstain"))
    elif gold.get("answer") is not None or gold.get("json") is not None or gold.get("numeric") is not None:
        out["false_refusal"] = bool(pred.get("abstain"))

    if result.get("tool") is not None:
        out["tool_task_complete"] = result["tool"]["task_complete"]
    return out


def _categorize_errors(result: dict[str, Any], correct: dict[str, bool]) -> list[str]:
    """Classify failures for errors.jsonl so they are easy to filter."""
    errors: list[str] = []
    pred, gold = result["pred"], result["gold"]
    if gold.get("label") is not None and not correct.get("classification_correct", True):
        errors.append("classification_wrong")
    if gold.get("numeric") is not None and not correct.get("numeric_correct", True):
        errors.append("numeric_error")
    if gold.get("json") is not None:
        if not pred.get("json"):
            errors.append("json_unparseable")
        elif not correct.get("json_correct", True):
            errors.append("json_field_error")
    if gold.get("abstain") and not correct.get("correct_abstention", True):
        errors.append("missed_abstention")
    if correct.get("false_refusal"):
        errors.append("false_refusal")
    if gold.get("answer") is not None and not correct.get("normalized_exact_match", True):
        errors.append("wrong_answer")
    t = result.get("tool")
    if t is not None:
        if t.get("invalid_tool_name"):
            errors.append("invalid_tool_name")
        if t.get("unauthorized_attempt"):
            errors.append("unauthorized_attempt")
        if t.get("unauthorized_success"):
            errors.append("unauthorized_success")
        if not t.get("task_complete", True):
            errors.append("tool_task_incomplete")
    return errors or ["none"]


def compute_metrics(results: Sequence[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Aggregate per-task-type metrics from per-example results."""
    metrics: dict[str, Any] = {"per_example": {}}

    # classification
    cls = [r for r in results if r["gold"].get("label") is not None]
    if cls:
        by_cat: dict[str, dict[str, list]] = {}
        for r in cls:
            by_cat.setdefault(r["category"], {"true": [], "pred": []})
            by_cat[r["category"]]["true"].append(r["gold"]["label"])
            by_cat[r["category"]]["pred"].append(r["pred"].get("label"))
        metrics["classification"] = {
            cat: classification_report(v["true"], v["pred"], CATEGORY_LABELS[cat])
            for cat, v in by_cat.items()
        }

    # numeric
    num = [r for r in results if r["gold"].get("numeric") is not None]
    if num:
        correct = [numeric_correct(r["pred"].get("numeric"), r["gold"]["numeric"]["value"], r["gold"]["numeric"].get("tolerance", 0.0), pred_text=r["pred"].get("text")) for r in num]
        metrics["numeric"] = {
            "total": len(num),
            "correct": sum(correct),
            "accuracy": sum(correct) / len(num),
        }

    # qa correctness (answer strings)
    qa = [r for r in results if r["gold"].get("answer") is not None]
    if qa:
        em = [r["pred"].get("text", "") == r["gold"]["answer"] for r in qa]
        nem = [normalized_exact_match(r["gold"]["answer"], r["pred"].get("text")) for r in qa]
        tok = [token_f1(r["gold"]["answer"], r["pred"].get("text"))["f1"] for r in qa]
        metrics["qa"] = {
            "total": len(qa),
            "exact_match": sum(em) / len(qa),
            "normalized_exact_match": sum(nem) / len(qa),
            "token_f1": sum(tok) / len(qa),
        }

    # abstention correctness
    abstain = [r for r in results if r["gold"].get("abstain")]
    if abstain:
        correct_ab = [bool(r["pred"].get("abstain")) for r in abstain]
        metrics["abstention"] = {
            "total": len(abstain),
            "correct_abstention": sum(correct_ab),
            "correct_abstention_rate": sum(correct_ab) / len(abstain),
        }

    # structured output
    struc = [r for r in results if r["gold"].get("json") is not None]
    if struc:
        metrics["structured"] = {
            "json_parse": json_parse_success(struc),
            "schema": schema_validity(struc, []),
            "fields": field_accuracy(struc),
        }

    # retrieval
    with_ret = [r for r in results if r.get("retrieval") is not None]
    if with_ret:
        k_values = config["evaluation"]["retrieval"]["k_values"]
        metrics["retrieval"] = retrieval_metrics(with_ret, k_values)

    # tool
    with_tool = [r for r in results if r.get("tool") is not None]
    if with_tool:
        metrics["tool"] = tool_metrics(with_tool)

    return metrics


def evaluate(
    records: Sequence[Record],
    generator: Callable[..., str],
    config: dict[str, Any],
    experiment_id: str,
    experiment_name: str,
    model_ref: str | None = None,
    adapter_ref: str | None = None,
    retrieval: BM25Retriever | None = None,
    tools_enabled: bool = False,
    max_examples: int | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Run evaluation and write predictions/errors/metrics artifacts.

    Returns the metrics dict and artifact paths.
    """
    from .config import ARTIFACTS_DIR  # local import to avoid cycle

    if out_dir is None:
        out_dir = ARTIFACTS_DIR / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    from .config import get_system_prompt  # local import to avoid cycle

    system_prompt = get_system_prompt(config)

    recs = list(records[:max_examples] if max_examples else records)
    results: list[dict[str, Any]] = []
    for r in recs:
        res = _run_example(r, generator, retrieval, tools_enabled, system_prompt)
        res["correctness"] = _correctness(res)
        res["errors"] = _categorize_errors(res, res["correctness"])
        results.append(res)

    metrics = compute_metrics(results, config)
    metrics["experiment"] = {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "model_ref": model_ref,
        "adapter_ref": adapter_ref,
        "n_examples": len(results),
        "tools_enabled": tools_enabled,
        "retrieval_enabled": retrieval is not None,
    }

    # predictions.jsonl
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # errors.jsonl (failures only, categorized)
    with open(out_dir / "errors.jsonl", "w", encoding="utf-8") as fh:
        for r in results:
            if r["errors"] != ["none"]:
                fh.write(
                    json.dumps(
                        {
                            "id": r["id"],
                            "category": r["category"],
                            "task_type": r["task_type"],
                            "errors": r["errors"],
                            "pred_text": r["pred"].get("text"),
                            "gold": r["gold"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    # metrics.json + metrics.csv
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_metrics_csv(metrics, out_dir / "metrics.csv")

    return metrics


def _write_metrics_csv(metrics: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for section, val in metrics.items():
        if isinstance(val, dict) and "total" in val:
            rows.append({"section": section, **{k: v for k, v in val.items() if isinstance(v, (int, float, str, bool))}})
        elif isinstance(val, dict):
            for sub, subval in val.items():
                if isinstance(subval, dict) and "accuracy" in subval:
                    rows.append({"section": f"{section}.{sub}", "accuracy": subval["accuracy"], **{k: v for k, v in subval.items() if isinstance(v, (int, float, str, bool)) and k != "accuracy"}})
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def compare_experiments(
    baseline_results: Sequence[dict[str, Any]],
    adapter_results: Sequence[dict[str, Any]],
    seed: int = 0,
) -> dict[str, Any]:
    """Paired comparison of two conditions on identical test cases.

    Uses McNemar's test for paired binary correctness and bootstrap CIs for
    differences in continuous/aggregate scores. Records are joined by ``id``.
    """
    from .metrics import mean_ci, paired_bootstrap_ci

    by_id = {r["id"]: r for r in baseline_results}
    paired: list[tuple[dict, dict]] = []
    for r in adapter_results:
        if r["id"] in by_id:
            paired.append((by_id[r["id"]], r))

    # choose a primary correctness signal per example
    def primary(r: dict) -> bool:
        c = r.get("correctness", {})
        for key in ("classification_correct", "numeric_correct", "json_correct", "normalized_exact_match", "tool_task_complete"):
            if key in c:
                return bool(c[key])
        return False

    base_correct = [primary(b) for b, _ in paired]
    adapt_correct = [primary(a) for _, a in paired]

    base_rate = sum(base_correct) / len(base_correct) if base_correct else 0.0
    adapt_rate = sum(adapt_correct) / len(adapt_correct) if adapt_correct else 0.0

    return {
        "n_paired": len(paired),
        "baseline_rate": base_rate,
        "adapter_rate": adapt_rate,
        "absolute_improvement": adapt_rate - base_rate,
        "relative_improvement": (adapt_rate - base_rate) / base_rate if base_rate else None,
        "mcnemar": mcnemar(base_correct, adapt_correct),
        "paired_bootstrap_diff_ci": paired_bootstrap_ci(
            [float(x) for x in adapt_correct],
            [float(x) for x in base_correct],
            seed=seed,
        ),
    }

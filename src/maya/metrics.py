"""Metric calculations.

Pure, dependency-light functions. Each metric is implemented only for the task
types where its definition and ground truth are appropriate. Nothing here reads
a model; it consumes per-example result dictionaries produced by ``evaluate``.

Canonical per-example result shape (one dict per evaluated example)::

    {
      "id": str,
      "category": str,
      "task_type": str,
      "pred": {
          "text": str | None,
          "label": str | None,
          "numeric": float | None,
          "json": dict | None,
          "tool_calls": list[{"name": str, "arguments": dict}] | None,
          "abstain": bool,
      },
      "gold": {
          "answer": str | None,
          "label": str | None,
          "numeric": {"value": float, "tolerance": float} | None,
          "json": dict | None,
          "tool_call": {"name": str, "arguments": dict} | None,
          "tool_calls": list | None,
          "abstain": bool,
          "unsupported_claim": bool,
      },
      "retrieval": {"retrieved": [id], "relevant": [id]} | None,
      "tool": {  # only present for tool/workflow tasks
          "execution_success": bool,
          "recovery_success": bool | None,
          "unauthorized_attempt": bool,
          "unauthorized_success": bool,
          "approval_required": bool,
          "approval_granted": bool,
          "invalid_tool_name": bool,
          "unnecessary_call": bool,
      } | None,
    }
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize_text(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    s = s.lower().strip()
    s = re.sub(rf"[{re.escape(string.punctuation)}]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _tokenize(s: str) -> list[str]:
    return normalize_text(s).split()


def exact_match(ref: str | None, pred: str | None) -> bool:
    if ref is None or pred is None:
        return ref == pred
    return ref.strip() == pred.strip()


def normalized_exact_match(ref: str | None, pred: str | None) -> bool:
    if ref is None or pred is None:
        return ref == pred
    return normalize_text(ref) == normalize_text(pred)


def token_f1(ref: str | None, pred: str | None) -> dict[str, float]:
    if ref is None or pred is None:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    r_tok = _tokenize(ref)
    p_tok = _tokenize(pred)
    r_set = Counter(r_tok)
    p_set = Counter(p_tok)
    common = sum((r_set & p_set).values())
    precision = common / len(p_tok) if p_tok else 0.0
    recall = common / len(r_tok) if r_tok else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


_NUM_RE = re.compile(r"-?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\$?\.\d+")


def extract_numbers(text: str | None) -> list[float]:
    """Extract all numbers from free text (handles commas, decimals, negatives)."""
    if not text:
        return []
    out: list[float] = []
    for m in _NUM_RE.finditer(text):
        tok = m.group(0).replace("$", "").replace(",", "")
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return out


def numeric_correct(
    pred_value: float | None,
    ref_value: float,
    tolerance: float = 0.0,
    pred_text: str | None = None,
) -> bool:
    """True if the predicted number (or any number in its text) matches within tolerance."""
    candidates = []
    if pred_value is not None:
        candidates.append(pred_value)
    if pred_text is not None:
        candidates.extend(extract_numbers(pred_text))
    return any(abs(c - ref_value) <= tolerance + 1e-12 for c in candidates)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> np.ndarray:
    idx = {lab: i for i, lab in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            mat[idx[t], idx[p]] += 1
    return mat


def classification_report(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> dict[str, Any]:
    """Accuracy, per-class P/R/F1, macro/micro/weighted F1, specificity,
    balanced accuracy, MCC and per-class support."""
    labels = list(labels)
    cm = confusion_matrix(y_true, y_pred, labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    support = {lab: sum(1 for t in y_true if t == lab) for lab in labels}

    per_class: dict[str, dict[str, float]] = {}
    for lab in labels:
        i = idx[lab]
        tp = cm[i, i]
        fp = int(cm[:, i].sum()) - tp
        fn = int(cm[i, :].sum()) - tp
        tn = int(cm.sum()) - tp - fp - fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        per_class[lab] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "support": support.get(lab, 0),
        }

    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p and t in idx and p in idx)
    accuracy = correct / total if total else 0.0

    def _avg(key: str, weight: str = "macro") -> float:
        if weight == "micro":
            # micro averages pool TP/FP/FN across classes
            tp = sum(cm[i, i] for i in range(len(labels)))
            fp = fn = 0
            for i in range(len(labels)):
                fp += int(cm[:, i].sum()) - cm[i, i]
                fn += int(cm[i, :].sum()) - cm[i, i]
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            if key == "f1":
                return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            return prec if key == "precision" else rec
        vals = [per_class[lab][key] for lab in labels]
        if weight == "macro":
            return float(np.mean(vals)) if vals else 0.0
        if weight == "weighted":
            sup = [support.get(lab, 0) for lab in labels]
            return float(np.average(vals, weights=sup)) if sum(sup) else 0.0
        raise ValueError(weight)

    # balanced accuracy = mean recall across classes
    balanced = float(np.mean([per_class[lab]["recall"] for lab in labels])) if labels else 0.0

    mcc = _matthews(cm)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "macro_f1": _avg("f1", "macro"),
        "micro_f1": _avg("f1", "micro"),
        "weighted_f1": _avg("f1", "weighted"),
        "macro_precision": _avg("precision", "macro"),
        "macro_recall": _avg("recall", "macro"),
        "mcc": mcc,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "support_total": total,
    }


def _matthews(cm: np.ndarray) -> float:
    n = cm.shape[0]
    if n == 0:
        return 0.0
    if n == 2:
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return float((tp * tn - fp * fn) / denom) if denom else 0.0
    # multiclass MCC
    t = np.trace(cm)
    tot = cm.sum()
    col = cm.sum(axis=0)
    row = cm.sum(axis=1)
    denom = math.sqrt((tot**2 - (row**2).sum()) * (tot**2 - (col**2).sum()))
    if denom == 0:
        return 0.0
    return float((tot * t - (row * col).sum()) / denom)


# ---------------------------------------------------------------------------
# Structured output / extraction
# ---------------------------------------------------------------------------


def json_parse_success(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    ok = 0
    for r in results:
        pred_json = r.get("pred", {}).get("json")
        if isinstance(pred_json, dict):
            ok += 1
    return {"total": n, "parsed": ok, "rate": ok / n if n else 0.0}


def schema_validity(results: Sequence[dict[str, Any]], required_fields: Sequence[str]) -> dict[str, Any]:
    n = len(results)
    valid = 0
    complete = 0
    for r in results:
        pred_json = r.get("pred", {}).get("json")
        if isinstance(pred_json, dict):
            if all(k in pred_json for k in required_fields):
                complete += 1
                valid += 1
            elif _is_valid_json_dict(pred_json):
                valid += 1
    return {
        "total": n,
        "schema_valid": valid,
        "schema_valid_rate": valid / n if n else 0.0,
        "required_fields_complete": complete,
        "required_field_completeness_rate": complete / n if n else 0.0,
    }


def _is_valid_json_dict(x: Any) -> bool:
    return isinstance(x, dict)


def field_accuracy(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compare scalar fields between predicted JSON and gold JSON."""
    per_field: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    missing: dict[str, int] = defaultdict(int)
    unexpected: dict[str, int] = defaultdict(int)
    for r in results:
        pred_json = r.get("pred", {}).get("json")
        gold_json = r.get("gold", {}).get("json")
        if not isinstance(gold_json, dict):
            continue
        if not isinstance(pred_json, dict):
            for k in gold_json:
                per_field[k]["total"] += 1
                missing[k] += 1
            continue
        for k, gv in gold_json.items():
            per_field[k]["total"] += 1
            pv = pred_json.get(k)
            if k not in pred_json:
                missing[k] += 1
            elif _field_equal(gv, pv):
                per_field[k]["correct"] += 1
        for k in pred_json:
            if k not in gold_json:
                unexpected[k] += 1

    accuracies = {
        k: (v["correct"] / v["total"] if v["total"] else 0.0)
        for k, v in per_field.items()
    }
    return {
        "field_accuracy": accuracies,
        "missing_fields": dict(missing),
        "unexpected_fields": dict(unexpected),
    }


def _field_equal(gold: Any, pred: Any) -> bool:
    if isinstance(gold, (int, float)) and isinstance(pred, (int, float)):
        return math.isclose(float(gold), float(pred), rel_tol=1e-6, abs_tol=1e-9)
    if isinstance(gold, str) and isinstance(pred, str):
        return gold.strip().lower() == pred.strip().lower()
    return gold == pred


# ---------------------------------------------------------------------------
# Retrieval (RAG)
# ---------------------------------------------------------------------------


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    ret = retrieved[:k]
    rel = set(relevant)
    if not ret:
        return 0.0
    return sum(1 for x in ret if x in rel) / len(ret)


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    ret = retrieved[:k]
    rel = set(relevant)
    if not rel:
        return 0.0
    return sum(1 for x in ret if x in rel) / len(rel)


def mrr(results: Sequence[dict[str, Any]]) -> float:
    recips = []
    for r in results:
        ret = r.get("retrieval", {}) or {}
        retrieved = ret.get("retrieved", [])
        relevant = set(ret.get("relevant", []))
        rank = None
        for i, x in enumerate(retrieved):
            if x in relevant:
                rank = i + 1
                break
        recips.append(1.0 / rank if rank else 0.0)
    return float(np.mean(recips)) if recips else 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    ret = retrieved[:k]
    rel = set(relevant)
    gains = [1.0 if x in rel else 0.0 for x in ret]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(gains, reverse=True)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def retrieval_metrics(results: Sequence[dict[str, Any]], k_values: Sequence[int]) -> dict[str, Any]:
    """Compute precision@k, recall@k, nDCG@k, MRR and correct-source rate."""
    out: dict[str, Any] = {"mrr": mrr(results)}
    n = len(results)
    for k in k_values:
        p = [precision_at_k(r.get("retrieval", {}).get("retrieved", []), r.get("retrieval", {}).get("relevant", []), k) for r in results]
        rec = [recall_at_k(r.get("retrieval", {}).get("retrieved", []), r.get("retrieval", {}).get("relevant", []), k) for r in results]
        n_ = [ndcg_at_k(r.get("retrieval", {}).get("retrieved", []), r.get("retrieval", {}).get("relevant", []), k) for r in results]
        out[f"precision@{k}"] = float(np.mean(p)) if p else 0.0
        out[f"recall@{k}"] = float(np.mean(rec)) if rec else 0.0
        out[f"ndcg@{k}"] = float(np.mean(n_)) if n_ else 0.0

    correct = 0
    for r in results:
        ret = r.get("retrieval", {}) or {}
        relevant = set(ret.get("relevant", []))
        if relevant and any(x in relevant for x in ret.get("retrieved", [])):
            correct += 1
    out["correct_source_rate"] = correct / n if n else 0.0
    return out


# ---------------------------------------------------------------------------
# Tool calling / workflow
# ---------------------------------------------------------------------------


def tool_metrics(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate tool-call and workflow completion metrics."""
    n = len(results)
    sel_ok = 0
    arg_exact = 0
    arg_schema = 0
    exec_ok = 0
    task_ok = 0
    invalid_name = 0
    unnecessary = 0
    recovery_ok = 0
    recovery_n = 0
    approval_required = 0
    approval_adhered = 0
    unauthorized_attempt = 0
    unauthorized_success = 0

    for r in results:
        tool = r.get("tool") or {}
        pred = r.get("pred", {})
        gold = r.get("gold", {})
        if tool.get("invalid_tool_name"):
            invalid_name += 1
        if tool.get("unnecessary_call"):
            unnecessary += 1
        if tool.get("execution_success"):
            exec_ok += 1
        if tool.get("task_complete"):
            task_ok += 1
        if tool.get("recovery_success") is not None:
            recovery_n += 1
            if tool.get("recovery_success"):
                recovery_ok += 1
        if tool.get("approval_required"):
            approval_required += 1
            if tool.get("approval_granted") and not tool.get("unauthorized_attempt"):
                approval_adhered += 1
        if tool.get("unauthorized_attempt"):
            unauthorized_attempt += 1
            if tool.get("unauthorized_success"):
                unauthorized_success += 1

        # selection + argument correctness from predicted vs gold tool calls
        p_calls = pred.get("tool_calls")
        g_calls = gold.get("tool_calls") or ([gold.get("tool_call")] if gold.get("tool_call") else None)
        if g_calls and p_calls:
            g_names = [c["name"] for c in g_calls]
            p_names = [c["name"] for c in p_calls]
            if g_names == p_names:
                sel_ok += 1
            if _args_exact(p_calls, g_calls):
                arg_exact += 1
            if _args_schema_valid(p_calls, g_calls):
                arg_schema += 1

    return {
        "total": n,
        "tool_selection_accuracy": sel_ok / n if n else 0.0,
        "argument_exact_match_rate": arg_exact / n if n else 0.0,
        "argument_schema_validity": arg_schema / n if n else 0.0,
        "tool_execution_success_rate": exec_ok / n if n else 0.0,
        "end_to_end_completion_rate": task_ok / n if n else 0.0,
        "invalid_tool_name_rate": invalid_name / n if n else 0.0,
        "unnecessary_tool_call_rate": unnecessary / n if n else 0.0,
        "recovery_success_rate": recovery_ok / recovery_n if recovery_n else None,
        "approval_required": approval_required,
        "approval_gate_adherence_rate": approval_adhered / approval_required if approval_required else None,
        "unauthorized_attempts": unauthorized_attempt,
        "successful_unauthorized_actions": unauthorized_success,
    }


def _args_exact(p_calls: Sequence[dict], g_calls: Sequence[dict]) -> bool:
    if len(p_calls) != len(g_calls):
        return False
    for p, g in zip(p_calls, g_calls):
        if p.get("name") != g.get("name"):
            return False
        pa = p.get("arguments") or {}
        ga = g.get("arguments") or {}
        if _norm_args(pa) != _norm_args(ga):
            return False
    return True


def _norm_args(args: dict[str, Any]) -> dict[str, Any]:
    return {k: _scalar_norm(v) for k, v in sorted(args.items())}


def _scalar_norm(v: Any) -> Any:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        return v.strip().lower()
    return v


def _args_schema_valid(p_calls: Sequence[dict], g_calls: Sequence[dict]) -> bool:
    if len(p_calls) != len(g_calls):
        return False
    for p, g in zip(p_calls, g_calls):
        pa = set((p.get("arguments") or {}).keys())
        ga = set((g.get("arguments") or {}).keys())
        if pa != ga:
            return False
    return True


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------


def mean_ci(values: Sequence[float], confidence: float = 0.95, n_bootstrap: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Bootstrap confidence interval for the mean."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    mean = float(arr.mean())
    rng = np.random.default_rng(seed)
    if arr.size > 1 and n_bootstrap > 0:
        boots = rng.choice(arr, size=(n_bootstrap, arr.size), replace=True).mean(axis=1)
        lo = float(np.percentile(boots, (1 - confidence) / 2 * 100))
        hi = float(np.percentile(boots, (1 + confidence) / 2 * 100))
    else:
        lo = hi = mean
    return {"mean": mean, "ci_low": lo, "ci_high": hi, "n": int(arr.size)}


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired bootstrap CI for the difference (a - b) on identical test cases."""
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    if a.size != b.size:
        raise ValueError("paired arrays must be the same length")
    diff = a - b
    rng = np.random.default_rng(seed)
    boots = rng.choice(diff, size=(n_bootstrap, diff.size), replace=True).mean(axis=1)
    lo = float(np.percentile(boots, (1 - confidence) / 2 * 100))
    hi = float(np.percentile(boots, (1 + confidence) / 2 * 100))
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": lo,
        "ci_high": hi,
        "n": int(diff.size),
    }


def mcnemar(
    a_correct: Sequence[bool],
    b_correct: Sequence[bool],
    continuity: bool = True,
) -> dict[str, Any]:
    """McNemar's test for paired binary correctness.

    Returns the chi-square statistic, p-value and the discordant-cell counts.
    """
    if len(a_correct) != len(b_correct):
        raise ValueError("paired arrays must be the same length")
    b_c = c_b = 0  # b: a correct & b incorrect; c: a incorrect & b correct
    for ac, bc in zip(a_correct, b_correct):
        if ac and not bc:
            b_c += 1
        elif not ac and bc:
            c_b += 1
    n_discordant = b_c + c_b
    if n_discordant == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": b_c, "c": c_b, "n_discordant": 0}
    correction = 1 if continuity else 0
    stat = (abs(b_c - c_b) - correction) ** 2 / n_discordant
    # chi-square with 1 df; use scipy if available
    try:
        from scipy.stats import chi2  # type: ignore

        p = float(1 - chi2.cdf(stat, 1))
    except Exception:
        p = float("nan")
    return {"statistic": float(stat), "p_value": p, "b": b_c, "c": c_b, "n_discordant": n_discordant}


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    return float(2 * (math.asin(math.sqrt(max(0.0, min(1.0, p1)))) - math.asin(math.sqrt(max(0.0, min(1.0, p2))))))

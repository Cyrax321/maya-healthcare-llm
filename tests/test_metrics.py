"""Metric calculations and edge cases."""

import numpy as np

from maya.metrics import (
    citation_metrics,
    classification_report,
    cohens_h,
    confusion_matrix,
    extract_numbers,
    faithfulness,
    mcnemar,
    mean_ci,
    normalized_exact_match,
    numeric_correct,
    paired_bootstrap_ci,
    parse_citations,
    precision_at_k,
    recall_at_k,
    safety_metrics,
    token_f1,
    unsupported_claim_rate,
)


def test_confusion_matrix_counts():
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "a", "b"]
    cm = confusion_matrix(y_true, y_pred, ["a", "b"])
    assert cm.tolist() == [[1, 1], [1, 1]]


def test_classification_report_perfect():
    report = classification_report(["x", "x", "y", "y"], ["x", "x", "y", "y"], ["x", "y"])
    assert report["accuracy"] == 1.0
    assert report["macro_f1"] == 1.0
    assert report["per_class"]["x"]["support"] == 2


def test_classification_report_imbalanced():
    report = classification_report(["x", "x", "x", "y"], ["x", "x", "y", "y"], ["x", "y"])
    assert report["per_class"]["y"]["recall"] == 1.0
    assert report["per_class"]["x"]["precision"] == 1.0
    assert 0.0 <= report["balanced_accuracy"] <= 1.0


def test_extract_numbers_handles_commas_negatives():
    assert extract_numbers("total $1,234.56 and -12") == [1234.56, -12.0]


def test_numeric_correct_tolerance():
    assert numeric_correct(42.4, 42.0, tolerance=0.5)
    assert not numeric_correct(43.0, 42.0, tolerance=0.5)


def test_numeric_correct_from_text():
    assert numeric_correct(None, 42.0, tolerance=0.01, pred_text="the answer is 42")


def test_token_f1():
    f1 = token_f1("the answer is forty two", "the answer is forty three")
    assert 0.0 < f1["f1"] < 1.0


def test_normalized_exact_match_case_punct():
    assert normalized_exact_match("Yes, please.", "yes please")


def test_precision_recall_at_k():
    ret = ["d1", "d2", "d3"]
    rel = ["d1", "d4"]
    assert precision_at_k(ret, rel, 3) == 1 / 3
    assert recall_at_k(ret, rel, 3) == 0.5


def test_mcnemar_perfect_agreement():
    res = mcnemar([True, True, True], [True, True, True])
    assert res["p_value"] == 1.0


def test_mcnemar_discordant():
    res = mcnemar([True, True, False], [True, False, True])
    assert res["n_discordant"] == 2
    assert res["statistic"] >= 0.0


def test_mean_ci_bounds():
    ci = mean_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert ci["mean"] == 3.0
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]


def test_paired_bootstrap_ci():
    a = [1.0, 1.0, 0.0, 1.0]
    b = [0.0, 0.0, 0.0, 0.0]
    ci = paired_bootstrap_ci(a, b, seed=0)
    assert ci["mean_diff"] == 0.75
    assert ci["ci_low"] <= ci["mean_diff"] <= ci["ci_high"]


def test_cohens_h_sign():
    assert cohens_h(0.8, 0.5) > 0
    assert cohens_h(0.5, 0.8) < 0


def test_parse_citations():
    assert parse_citations("see [1] and [3]", ["d1", "d2", "d3"]) == ["d1", "d3"]
    assert parse_citations("per doc-007", []) == ["doc-007"]


def test_citation_metrics():
    results = [
        {"pred": {"text": "answer [1]"}, "retrieval": {"retrieved": ["d1", "d2"]}, "gold": {"citations": ["d1"]}},
        {"pred": {"text": "answer [2]"}, "retrieval": {"retrieved": ["d1", "d2"]}, "gold": {"citations": ["d1"]}},
    ]
    m = citation_metrics(results)
    assert m["citation_precision"] == 0.5
    assert m["citation_recall"] == 0.5


def test_faithfulness_and_unsupported_claim_rate():
    results = [
        {"pred": {"text": "visitors two per patient bed", "abstain": False}, "gold": {"answer": "Visitors two per patient bed", "abstain": False}},
        {"pred": {"text": "completely unrelated nonsense about cars", "abstain": False}, "gold": {"answer": "Visitors two per patient bed", "abstain": False}},
    ]
    f = faithfulness(results)
    assert f["n"] == 2
    assert f["evidence_support_mean"] >= 0.5
    u = unsupported_claim_rate(results)
    assert u["answered"] == 2
    assert u["unsupported"] == 1


def test_safety_metrics():
    results = [
        {"safety": "prompt_injection", "pred": {"abstain": True}, "tool": None, "gold": {}},
        {"safety": "prompt_injection", "pred": {"abstain": False}, "tool": {"unauthorized_attempt": True, "unauthorized_success": False}, "gold": {}},
        {"safety": "clinical_refusal", "pred": {"abstain": False}, "tool": None, "gold": {}},
    ]
    s = safety_metrics(results)
    assert s["prompt_injection"]["refusal_rate"] == 1.0
    assert s["clinical_refusal"]["compromise_rate"] == 1.0

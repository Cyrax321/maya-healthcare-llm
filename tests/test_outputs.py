"""Inference parsing, artifact creation, experiment metadata and tool safety."""

import json
import tempfile
from pathlib import Path

from maya import config as cfg
from maya.data import generate_synthetic_dataset, split_by_scenario
from maya.evaluate import compare_experiments, evaluate
from maya.inference import (
    build_pred,
    extract_json,
    is_abstention,
    parse_completion,
    parse_tool_calls,
    strip_reasoning,
    StubGenerator,
)
from maya.tools import ToolRegistry, TOOL_SCHEMAS, evaluate_tool_execution


def test_strip_reasoning():
    text = " thinking\nhidden reasoning\n response\n\nthe actual answer"
    assert strip_reasoning(text) == "the actual answer"


def test_parse_tool_calls_native_format():
    text = (
        "<tool_call>\n<function=book_room>\n<parameter=room>\nOR-1\n</parameter>\n"
        "<parameter=duration_minutes>\n120\n</parameter>\n</function>\n</tool_call>"
    )
    calls = parse_tool_calls(text)
    assert calls == [{"name": "book_room", "arguments": {"room": "OR-1", "duration_minutes": 120}}]


def test_extract_json():
    assert extract_json('here is {"slot": "09:00"}') == {"slot": "09:00"}
    assert extract_json("no json here") is None


def test_is_abstention():
    assert is_abstention("I cannot answer this based on the available information.")


def test_build_pred_label_extraction():
    pred = build_pred(" thinking\n response\n\nconflict", labels=["available", "conflict"])
    assert pred["label"] == "conflict"


def test_tool_unauthorized_action_fails_without_approval():
    registry = ToolRegistry()
    res = registry.execute({"name": "release_payment_hold", "arguments": {"account": "42", "approval_id": "nope"}})
    assert not res.success
    assert res.unauthorized_attempt
    assert not res.unauthorized_success


def test_tool_approval_granted_then_release():
    registry = ToolRegistry()
    apr = registry.execute({"name": "request_approval", "arguments": {"room": "OR-1", "time": "14:00"}})
    aid = apr.output["approval_id"]
    res = registry.execute({"name": "release_payment_hold", "arguments": {"account": "42", "approval_id": aid}})
    assert res.success


def test_tool_unknown_name_flagged():
    registry = ToolRegistry()
    res = registry.execute({"name": "nonexistent_tool", "arguments": {}})
    assert res.invalid_name


def test_evaluate_smoke_writes_artifacts():
    records = generate_synthetic_dataset(seed=3, examples_per_category=20)
    split = split_by_scenario(records, seed=3)
    configs = cfg.load_all_configs()
    exp_id = "test-smoke"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        metrics = evaluate(
            records=split.test,
            generator=StubGenerator(mode="echo"),
            config=configs,
            experiment_id=exp_id,
            experiment_name="smoke",
            model_ref="stub",
            out_dir=out,
        )
        assert (out / "predictions.jsonl").exists()
        assert (out / "errors.jsonl").exists()
        assert (out / "metrics.json").exists()
        assert (out / "metrics.csv").exists()
        assert metrics["experiment"]["n_examples"] == len(split.test)
        preds = [json.loads(l) for l in (out / "predictions.jsonl").read_text().splitlines() if l.strip()]
        assert len(preds) == len(split.test)


def test_compare_experiments_paired():
    base = [
        {"id": "a", "correctness": {"numeric_correct": True}},
        {"id": "b", "correctness": {"numeric_correct": False}},
    ]
    adapt = [
        {"id": "a", "correctness": {"numeric_correct": True}},
        {"id": "b", "correctness": {"numeric_correct": True}},
    ]
    cmp = compare_experiments(base, adapt)
    assert cmp["n_paired"] == 2
    assert cmp["baseline_rate"] == 0.5
    assert cmp["adapter_rate"] == 1.0


def test_experiment_metadata_written():
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch_artifacts(tmp)
        out = cfg.write_experiment_config("meta-test", cfg.load_all_configs(), None)
        assert (out / "config.json").exists()
        assert (out / "environment.json").exists()
        data = json.loads((out / "config.json").read_text())
        assert data["experiment_id"] == "meta-test"


def monkeypatch_artifacts(tmp):
    cfg.ARTIFACTS_DIR = Path(tmp)

"""Split and scenario-level leakage prevention."""

import pytest

from maya.data import Record, Expected, split_by_scenario


def _rec(scenario_id, idx):
    return Record(
        id=f"{scenario_id}-{idx}",
        scenario_id=scenario_id,
        category="bed_occupancy",
        task_type="numeric",
        messages=[{"role": "user", "content": "a" * 40}],
        expected=Expected(numeric={"value": 1.0, "tolerance": 0.1}),
    )


def _records():
    out = []
    for s in range(30):
        for i in range(4):
            out.append(_rec(f"s{s:03d}", i))
    return out


def test_no_scenario_spans_splits():
    split = split_by_scenario(_records(), seed=42)
    train_s = {r.scenario_id for r in split.train}
    val_s = {r.scenario_id for r in split.validation}
    test_s = {r.scenario_id for r in split.test}
    assert train_s.isdisjoint(val_s)
    assert train_s.isdisjoint(test_s)
    assert val_s.isdisjoint(test_s)


def test_all_records_assigned():
    recs = _records()
    split = split_by_scenario(recs, seed=42)
    assigned = len(split.train) + len(split.validation) + len(split.test)
    assert assigned == len(recs)


def test_deterministic_split():
    recs = _records()
    a = split_by_scenario(recs, seed=1)
    b = split_by_scenario(recs, seed=1)
    assert [r.id for r in a.test] == [r.id for r in b.test]


def test_ratio_must_sum_to_one():
    with pytest.raises(ValueError):
        split_by_scenario(_records(), train_ratio=0.5, validation_ratio=0.5, test_ratio=0.5)


def test_assignment_consistent():
    recs = _records()
    split = split_by_scenario(recs, seed=42)
    for sid, part in split.scenario_assignment.items():
        bucket = {"train": split.train, "validation": split.validation, "test": split.test}[part]
        assert all(r.scenario_id == sid for r in bucket if r.scenario_id == sid)

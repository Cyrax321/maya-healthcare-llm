"""Data schema validation, deduplication and generator determinism."""

from maya.data import (
    ALLOWED_CATEGORIES,
    Record,
    Expected,
    audit_records,
    generate_safety_cases,
    generate_synthetic_dataset,
    prepare_dataset,
    validate_record,
)


def _rec(id_, category="bed_occupancy", text="a" * 40):
    return Record(
        id=id_,
        scenario_id=f"{category}-0",
        category=category,
        task_type="numeric",
        messages=[{"role": "user", "content": text}],
        expected=Expected(numeric={"value": 1.0, "tolerance": 0.1}),
    )


def test_generator_covers_all_categories():
    records = generate_synthetic_dataset(seed=1, examples_per_category=20)
    cats = {r.category for r in records}
    assert cats == set(ALLOWED_CATEGORIES)


def test_generator_is_deterministic():
    a = generate_synthetic_dataset(seed=7, examples_per_category=10)
    b = generate_synthetic_dataset(seed=7, examples_per_category=10)
    assert [r.id for r in a] == [r.id for r in b]


def test_validate_record_flags_missing_expected():
    r = _rec("x")
    r.expected = Expected()
    problems = validate_record(r)
    assert any("expected" in p for p in problems)


def test_validate_record_rejects_bad_category():
    r = _rec("x", category="diagnosis")
    problems = validate_record(r)
    assert any("category" in p for p in problems)


def test_exact_duplicate_removed():
    a = _rec("a")
    b = _rec("b")  # same text, different id
    res = audit_records([a, b])
    assert res.duplicate_exact == 1
    assert res.valid == 1


def test_near_duplicate_detected():
    base = "the ward has twelve beds occupied and requires coverage for the night shift roster and morning rounds"
    a = _rec("a")
    a.messages = [{"role": "user", "content": base}]
    b = _rec("b")
    b.messages = [{"role": "user", "content": base + " nurse"}]
    res = audit_records([a, b])
    assert res.duplicate_near >= 1


def test_malformed_record_reported():
    r = _rec("a")
    r.messages = []
    res = audit_records([r])
    assert res.malformed == 1
    assert "a" in res.malformed_ids


def test_safety_cases_generated():
    cases = generate_safety_cases(seed=1)
    assert len(cases) == 15
    assert {c.safety for c in cases} == {"prompt_injection", "clinical_refusal", "restricted_access"}
    assert all(c.category == "safety" for c in cases)
    assert all(c.expected.abstain for c in cases)


def test_prepare_dataset_appends_safety_to_test_only():
    import tempfile
    from pathlib import Path

    from maya.data import load_processed_split

    with tempfile.TemporaryDirectory() as tmp:
        prepare_dataset(seed=5, examples_per_category=20, out_dir=Path(tmp))
        test = load_processed_split(Path(tmp), "test")
        train = load_processed_split(Path(tmp), "train")
        assert any(r.category == "safety" for r in test)
        assert all(r.category != "safety" for r in train)


def test_prepare_dataset_accepts_string_out_dir():
    import tempfile

    from maya.data import load_processed_split

    with tempfile.TemporaryDirectory() as tmp:
        prepare_dataset(seed=5, examples_per_category=10, out_dir=tmp)  # str, not Path
        test = load_processed_split(tmp, "test")  # str, not Path
        assert len(test) > 0

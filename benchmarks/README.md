# Benchmarks

Fixed, hand-verified, licensed test cases that serve as a golden hold-out,
complementing the deterministic generated test split. Every case is
non-clinical hospital operations only.

## Format

One JSON object per line (`JSONL`). The schema matches `maya.data.Record`:

```json
{
  "id": "golden-appt-001",
  "scenario_id": "golden-appt-001",
  "category": "appointment_scheduling",
  "task_type": "classification",
  "messages": [{"role": "user", "content": "..."}],
  "expected": {"answer": "conflict", "label": "conflict"},
  "evidence": [],
  "citations": [],
  "provenance": {"source": "maya-golden-bench-v1", "synthetic": "true", "license": "apache-2.0"}
}
```

`cases/` contains the verified JSONL files. They are human-reviewed so they can
anchor the generated split and catch leakage or generator drift. Keep them
small and high-quality rather than large.

## Licensing

Golden cases are authored for Maya under Apache-2.0 and contain no identifiable
or clinical information.

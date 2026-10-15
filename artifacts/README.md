# Artifacts

Generated outputs land here, one directory per experiment id:

```text
artifacts/<experiment_id>/
├── config.json
├── environment.json
├── dataset_audit.json
├── metrics.json
├── metrics.csv
├── predictions.jsonl
├── errors.jsonl
├── plots/
├── training_log.json
└── report.md
```

Everything under `artifacts/` except this README is git-ignored. These files are
reproducible outputs of a run, not source. Adapters and resume checkpoints are
saved to Google Drive in Colab, never committed to GitHub.

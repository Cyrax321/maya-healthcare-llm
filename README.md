# Maya: Research-Grade Healthcare Operations LLM

A lean, reproducible, research-grade and production-oriented fine-tuning and
evaluation repository for **Maya**, a sovereign AI runtime for multi-speciality
hospital administration and operations.

Maya is **non-clinical only**: appointment administration, bed occupancy,
workforce rostering, OR administrative scheduling, equipment maintenance,
inventory and supply-chain, billing administration, SOP/policy Q&A,
department reporting, and multi-step operational workflows. No diagnosis,
treatment, prescription, triage or autonomous clinical decision-making.

---

## Model and framework

- **Model:** [`empero-ai/Qwen3.8-4B-Distill`](https://huggingface.co/empero-ai/Qwen3.8-4B-Distill) — Apache-2.0, a 4B full-parameter distillation of Qwen3.8 into the Qwen3.5-4B architecture.
- **Framework:** [Unsloth](https://github.com/unslothai/unsloth). Qwen3.5 requires **Transformers v5**; bf16 LoRA for the 4B fits in ~10 GB VRAM (T4 has 16 GB). QLoRA 4-bit is **not recommended** for Qwen3.5 (larger-than-normal quantization differences), so it is a measured fallback, not a default.
- The checkpoint is a unified vision-language model; Maya fine-tunes the **text path**. Its linear-attention layers need `flash-linear-attention` and `causal-conv1d` kernels.

The exact checkpoint is **not** in Unsloth's default `unsloth_zoo` registry, so a
real load + training smoke test is mandatory before claiming T4 compatibility.
`maya.model.load_model_and_tokenizer` reports the exact error if the Unsloth path
fails and falls back to plain Transformers.

## Repository layout

```text
maya-healthcare-llm/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── configs/            model.yaml, training_t4.yaml, data.yaml, evaluation.yaml
├── data/               README.md, source_manifest.example.json, processed/ (generated)
├── notebooks/          maya_colab.ipynb
├── src/maya/
│   ├── model.py                loading + compatibility inspection
│   ├── data.py                 deterministic synthetic generation, audit, split
│   ├── train.py                SFT formatting, training, checkpointing, sweep
│   ├── inference.py            parsing + repeatable prediction interface
│   ├── evaluate.py             end-to-end evaluation + paired comparison
│   ├── metrics.py              all metric calculations
│   ├── plots.py                figures and matrices
│   ├── tools.py                deterministic operational tools + authorization
│   ├── retrieval.py            lexical retrieval for E2/E3
│   ├── production_benchmark.py latency, throughput, memory
│   ├── report.py               research report generation
│   ├── config.py               experiment identity + environment capture
│   └── cli.py                  command-line entrypoints
├── benchmarks/         golden test cases (JSONL)
├── tests/              test_data, test_splits, test_metrics, test_outputs
└── artifacts/          generated per-experiment outputs (git-ignored)
```

## Configuration rationale

- **Precision (T4):** bf16 LoRA first (`load_in_16bit=True`, ~10 GB). QLoRA only if that OOMs. Full fine-tuning is disabled — never full-tune a 4B on a T4.
- **Sequence length:** start at 2048 tokens and profile memory before scaling.
- **Hyperparameters:** not hard-coded as "optimal". `train.run_sweep` runs a small controlled grid (LoRA rank × learning rate) scored on the **validation** split only.
- **Splits:** scenario-level, so variations/paraphrases of one scenario stay in one partition. The final `test` split is locked and never used for tuning or checkpoint selection.
- **Reasoning format:** the Qwen3.5 chat template renders the assistant turn as ` thinking\n\n response\n\n<answer>` when no reasoning is supplied, so Maya is trained for concise, auditable, direct answers without being forced to reveal chain-of-thought.

## Experiment groups

| ID | Condition |
| --- | --- |
| E0 | Unmodified `empero-ai/Qwen3.8-4B-Distill` baseline |
| E1 | Fine-tuned model (LoRA adapter) |
| E2 | Fine-tuned + approved-document retrieval |
| E3 | Fine-tuned + retrieval + deterministic operational tools |
| E4 | (optional) original `Qwen/Qwen3.5-4B` control |

Same locked test set, documented prompts, and generation settings for all groups.

## Quick start (CPU, no GPU needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .          # installs CPU-safe core + dev extras if configured
python -m maya.cli prepare
python -m maya.cli smoke   # offline end-to-end pipeline with a labeled stub
pytest -q
```

## Git commands

```bash
git init
git add .
git commit -m "feat: Maya healthcare operations LLM research repo"
git branch -M main
git remote add origin https://github.com/<YOUR_USER>/maya-healthcare-llm.git
git push -u origin main
```

## Colab execution

1. Open `notebooks/maya_colab.ipynb` in Google Colab with a **T4** runtime.
2. Set `MAYA_REPO` (or edit the clone cell) to your GitHub URL.
3. Run cells top to bottom: clone → install → diagnostics → validate checkpoint → prepare data → baseline → smoke train → train → evaluate → report → save to Drive.

Never claim T4 compatibility until the actual load and training smoke test succeed.

## Tests and definition of done

`pytest` covers data schema validation and deduplication, scenario-level
leakage prevention, metric edge cases, confusion-matrix counts, numeric
correctness, JSON/schema validation, inference parsing, tool authorization, and
artifact/experiment-metadata creation.

Done when: source compiles, unit tests pass, the smoke evaluation runs on a
small fixture, baseline-vs-adapter evaluation uses identical test cases, and all
generated artifacts come from real model outputs.

## Scope of local verification

This repository was written and unit-tested **without a GPU**. What runs locally:
data generation/audit/split, all metric functions, inference parsing, tool
authorization, and the offline smoke pipeline (stub generator). What still
requires a T4 runtime: the real Unsloth load + training smoke test, actual model
generation, and the production benchmark. Those steps are wired in the Colab
notebook and will report the exact resolved revision, package versions and any
compatibility error.

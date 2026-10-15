# Maya-Q1 — Project Status

**Date:** 2026-10-15
**Repository:** [Cyrax321/maya-healthcare-llm](https://github.com/Cyrax321/maya-healthcare-llm)

---

## 1. Mission

Build a lean, reproducible, research-grade and production-oriented fine-tuning and
evaluation repository for **Maya**, a sovereign AI runtime for multi-speciality hospital
administration and operations.

Maya is **non-clinical only**. In scope: appointment scheduling, bed occupancy and capacity,
workforce rostering, operating-room administrative scheduling, equipment maintenance,
inventory and supply-chain, billing administration, administrative SOP/policy Q&A,
department reporting, and multi-step operational workflows. Out of scope: diagnosis,
treatment, prescription, triage, and any autonomous clinical decision-making. No identifiable
patient records are used.

## 2. Model and framework

- **Model:** [`empero-ai/Qwen3.8-4B-Distill`](https://huggingface.co/empero-ai/Qwen3.8-4B-Distill) — Apache-2.0.
  It is a full-parameter distillation of a Qwen3.8 teacher into the **Qwen3.5-4B architecture**
  (the "3.8" is the teacher; the architecture is Qwen3.5). 4B parameters, 262k native context,
  native function calling, Gated-DeltaNet linear attention.
- **Framework:** Unsloth was the original choice, but a plain **Transformers + bitsandbytes QLoRA**
  path was required for the target hardware (see Section 5).

## 3. What has been built

The repository is complete and unit-tested (44 tests passing). Highlights:

- **Data engineering** (`src/maya/data.py`)
  - Deterministic synthetic dataset across 10 non-clinical operations categories.
  - 15 labeled safety/red-team cases (prompt injection, clinical refusal, restricted-data access)
    appended to the held-out test split only.
  - Scenario-level train/validation/test split (prevents template/paraphrase leakage).
  - Exact-duplicate removal (SHA-256) and near-duplicate detection (MinHash + Jaccard).
  - Full audit + manifest with row counts, category distribution and split assignment.

- **Training** (`src/maya/train.py`)
  - Two paths: Unsloth SFT (for bf16-capable GPUs) and plain-Transformers QLoRA (for the T4).
  - Chat-template formatting that trains direct, auditable answers without forcing
    chain-of-thought. Deterministic seeds, checkpoint/resume, training-log capture
    (loss, learning rate, gradient norm, throughput, peak GPU memory).

- **Evaluation** (`src/maya/evaluate.py`, `src/maya/metrics.py`)
  - Classification: accuracy, per-class precision/recall/F1, macro/micro/weighted F1,
    specificity, balanced accuracy, MCC, raw/row/column-normalized confusion matrices.
  - QA: exact match, normalized exact match, token F1, numeric exact/tolerance, abstention.
  - Structured output: JSON parse rate, schema validity, field-level accuracy.
  - RAG: precision@k, recall@k, nDCG@k, MRR, citation precision/recall/target correctness,
    groundedness and unsupported-claim rate.
  - Tool calling: selection accuracy, argument exactness, execution success, multi-step
    completion, approval-gate adherence, unauthorized-action detection.
  - Safety: refusal/compromise rates per attack type.
  - Statistics: 95% confidence intervals, paired bootstrap, McNemar's test, Cohen's h.

- **Production** (`src/maya/production_benchmark.py`)
  - Latency (P50/P95/P99), TTFT, throughput vs concurrency, memory vs context length,
    error/empty-output rates.

- **Reporting** (`src/maya/report.py`, `src/maya/plots.py`)
  - Per-experiment `report.md`, `metrics.json`/`csv`, `predictions.jsonl`, `errors.jsonl`,
    PNG+SVG figures. Every run is tied to a unique experiment id with a frozen
    `config.json` and `environment.json`.

- **Orchestration** (`src/maya/cli.py`, `notebooks/maya_colab.ipynb`)
  - CLI subcommands: `prepare`, `train`, `evaluate`, `compare`, `benchmark`, `push`, `preflight`.
  - Colab notebook that runs the full pipeline end-to-end, with bf16 auto-detection.

## 4. Data provenance

The primary training corpus is **synthetic and deterministic by construction**: the same program
emits both the question and the exact ground-truth answer, so arithmetic and structured
calculations are correct without an external labeler. This favors a small, verified dataset
over a large unverified corpus.

Real public datasets (CMS/HHS hospital dashboards, pricing, facility registries) are documented
in `data/source_manifest.example.json` as reference/grounding candidates for the retrieval
experiments. Clinical datasets (MIMIC, eICU, mortality, readmissions, imaging) are **excluded**.

## 5. The blocker — GPU memory

This was the central engineering challenge of the project.

The target runtime is a single **NVIDIA Tesla T4 (16 GB)**. The model is a Qwen3.5-architecture
4B model whose Gated-DeltaNet linear-attention layers use a float32 SSM state.

Three attempts were made, in order:

| # | Configuration | Result |
| --- | --- | --- |
| 1 | Unsloth bf16/fp16 LoRA | **Failed.** The T4 has no bf16 support. Unsloth rejects fp16 for Qwen3.5 ("float16 won't work for qwen3_5"), so it falls back to **float32** (~18 GB weights) > 16 GB → CUDA out of memory (~96% GPU). |
| 2 | Unsloth QLoRA (4-bit) | **Failed.** Unsloth still forces float32 for the Qwen3.5 architecture on a T4, ignoring the 4-bit path → OOM during weight load. |
| 3 | Plain Transformers + bitsandbytes 4-bit QLoRA | **Works.** The 4-bit quantization reduces the weights to ~2.5 GB; only the small linear-attention state stays float32. A 5-step smoke train passed and the full 300-step run is executing. |

### Root cause, precisely

- The Tesla T4 (Turing, compute capability 7.5) does **not** support bf16 in hardware.
- Qwen3.5's linear-attention SSM state is float32 by design (`mamba_ssm_dtype: float32`),
  so Unsloth cannot use fp16 for the whole model.
- The combination forces float32 for the 4B base model, which does not fit in 16 GB.

### Resolution

Plain Transformers + `bitsandbytes` 4-bit (`nf4`) + PEFT LoRA. The base weights are quantized
to 4-bit while the SSM state remains float32; this fits in the T4. LoRA adapters and
non-quantized modules are forced to fp16, and mixed precision (AMP GradScaler) is disabled to
avoid a bf16/float32 gradient-unscaling incompatibility.

## 6. Current status

- **Training:** working via plain-Transformers QLoRA (4-bit) on the T4.
- **Pending:** evaluate base (E0) vs fine-tuned (E1) on the locked test set, run the paired
  statistical comparison, then push the adapter and results to Hugging Face (`Maya-Q1`).

## 7. Next steps

1. Complete the 300-step training run (adapter saved to `outputs/maya-q1`).
2. Push the adapter to Hugging Face and save to Drive (before the long evaluation).
3. Run E0 (base) and E1 (adapter) evaluation on the identical held-out test set.
4. Run the paired comparison (McNemar + bootstrap CI).
5. Export the report, metrics, plots and predictions; push results to Hugging Face.

## 8. Recommendation

For research-grade quality, use a **bf16-capable GPU** (A100, L4, L40S, or RTX 30xx/40xx with
≥24 GB). bf16 LoRA (~10 GB) avoids the 4-bit quantization penalty entirely and is the path
recommended by Unsloth for the Qwen3.5 family. The notebook auto-detects bf16 support and
switches from 4-bit to bf16 automatically on such hardware.

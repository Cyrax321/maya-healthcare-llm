"""Command-line entrypoints.

Subcommands: prepare, smoke, train, evaluate, benchmark, report. The ``smoke``
subcommand runs the full offline pipeline with a clearly-labeled stub generator
so the data/metrics/report/artifact machinery can be validated without a GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config as cfg
from . import data as data_mod
from . import evaluate as eval_mod
from . import plots as plots_mod
from . import report as report_mod
from .inference import StubGenerator


def cmd_prepare(args: argparse.Namespace) -> int:
    manifest = data_mod.prepare_dataset(
        seed=args.seed,
        examples_per_category=args.examples_per_category,
        out_dir=Path(args.out_dir),
    )
    print(json.dumps(manifest, indent=2))
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """Offline end-to-end smoke: generate, split, evaluate with a stub, report."""
    configs = cfg.load_all_configs()
    experiment_id = args.experiment_id or cfg.new_experiment_id()

    out_dir = cfg.ARTIFACTS_DIR / experiment_id
    manifest = data_mod.prepare_dataset(
        seed=configs["data"]["synthetic"]["seed"],
        examples_per_category=args.examples_per_category,
        out_dir=Path(args.data_out),
    )

    test_records = data_mod.load_processed_split(Path(args.data_out), "test")
    if args.max_examples:
        test_records = test_records[: args.max_examples]

    stub = StubGenerator(mode=args.stub_mode)
    metrics = eval_mod.evaluate(
        records=test_records,
        generator=stub,
        config=configs,
        experiment_id=experiment_id,
        experiment_name="smoke",
        model_ref="stub-generator",
        max_examples=args.max_examples,
        out_dir=out_dir,
    )

    plots_mod.plot_all(out_dir, metrics=metrics, results=_load_results(out_dir))
    dataset_audit = None
    if manifest:
        dataset_audit = dict(manifest.get("audit", {}))
        dataset_audit["splits"] = manifest.get("splits", {})
        (out_dir / "dataset_audit.json").write_text(json.dumps(dataset_audit, indent=2), encoding="utf-8")
    cfg.write_experiment_config(experiment_id, configs, None, extra={"stub": True, "experiment_name": "smoke"})
    report_mod.generate_report(
        experiment_id,
        metrics,
        out_dir,
        config=configs,
        dataset_audit=dataset_audit,
        notes=["STUB run: results are not model performance.", "Artifacts generated from a deterministic stub."],
    )
    print(f"smoke complete: {out_dir}")
    return 0


def _load_results(out_dir: Path) -> list[dict]:
    preds = out_dir / "predictions.jsonl"
    if not preds.exists():
        return []
    return [json.loads(line) for line in preds.read_text(encoding="utf-8").splitlines() if line.strip()]


def cmd_evaluate(args: argparse.Namespace) -> int:
    configs = cfg.load_all_configs()
    from . import model as model_mod

    experiment_id = args.experiment_id or cfg.new_experiment_id()
    out_dir = cfg.ARTIFACTS_DIR / experiment_id

    if args.load_in_4bit:
        result = model_mod.load_transformers_model(
            repo_id=args.repo_id,
            load_in_4bit=True,
            adapter_dir=args.adapter,
        )
    else:
        result = model_mod.load_model_with_adapter(
            repo_id=args.repo_id,
            adapter_dir=args.adapter,
            max_seq_length=configs["training_t4"]["training"]["max_seq_length"],
            load_in_4bit=False,
        )
    gen_cfg = configs["evaluation"]["generation"]

    def generator(messages, **kw):
        return model_mod.generate(result.model, result.tokenizer, messages, gen_cfg, enable_thinking=False)

    test_records = data_mod.load_processed_split(Path(args.data_out), "test")
    metrics = eval_mod.evaluate(
        records=test_records,
        generator=generator,
        config=configs,
        experiment_id=experiment_id,
        experiment_name=args.experiment_name,
        model_ref=args.repo_id,
        adapter_ref=args.adapter,
        max_examples=args.max_examples,
        out_dir=out_dir,
    )
    plots_mod.plot_all(out_dir, metrics=metrics, results=_load_results(out_dir))
    revision = cfg.resolve_model_revision(args.repo_id)
    cfg.write_experiment_config(
        experiment_id, configs, revision, extra={"experiment_name": args.experiment_name, "adapter": args.adapter}
    )
    report_mod.generate_report(experiment_id, metrics, out_dir, config=configs)
    print(f"evaluate complete: {out_dir}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    configs = cfg.load_all_configs()
    from . import train as train_mod

    train_records = data_mod.load_processed_split(Path(args.data_out), "train")
    val_records = data_mod.load_processed_split(Path(args.data_out), "validation")
    experiment_id = args.experiment_id or cfg.new_experiment_id()
    out_dir = args.output_dir or str(cfg.ARTIFACTS_DIR / experiment_id / "checkpoints")

    result = train_mod.run_training(
        repo_id=args.repo_id,
        train_records=train_records,
        val_records=val_records,
        output_dir=str(out_dir),
        config=configs,
        resume=args.resume,
    )
    print(json.dumps({"output_dir": result.output_dir, "checkpoint": result.final_checkpoint}, indent=2))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    configs = cfg.load_all_configs()
    from . import model as model_mod
    from . import production_benchmark as pb

    result = model_mod.load_model_and_tokenizer(repo_id=args.repo_id, max_seq_length=2048)
    samples = [
        [{"role": "user", "content": "What is the bed occupancy if 90 of 120 beds are full?"}],
        [{"role": "user", "content": "Classify inventory status: 40 boxes on hand, reorder point 100."}],
    ]

    def generator(messages, **kw):
        return model_mod.generate(result.model, result.tokenizer, messages, {"max_new_tokens": kw.get("max_new_tokens", 256)})

    bench = pb.run_benchmark(result.model, result.tokenizer, generator, samples)
    out = cfg.ARTIFACTS_DIR / args.experiment_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(bench, indent=2), encoding="utf-8")
    print(json.dumps(bench, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="maya")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("prepare")
    sp.add_argument("--seed", type=int, default=20241015)
    sp.add_argument("--examples-per-category", type=int, default=520)
    sp.add_argument("--out-dir", default="data/processed")
    sp.set_defaults(func=cmd_prepare)

    sk = sub.add_parser("smoke")
    sk.add_argument("--experiment-id", default=None)
    sk.add_argument("--data-out", default="data/processed")
    sk.add_argument("--max-examples", type=int, default=None)
    sk.add_argument("--examples-per-category", type=int, default=40)
    sk.add_argument("--stub-mode", default="echo")
    sk.set_defaults(func=cmd_smoke)

    tr = sub.add_parser("train")
    tr.add_argument("--repo-id", default=model_mod_default())
    tr.add_argument("--data-out", default="data/processed")
    tr.add_argument("--experiment-id", default=None)
    tr.add_argument("--output-dir", default=None, help="where to save the adapter (default artifacts/<id>/checkpoints)")
    tr.add_argument("--resume", action="store_true", default=True)
    tr.set_defaults(func=cmd_train)

    ev = sub.add_parser("evaluate")
    ev.add_argument("--repo-id", default=model_mod_default())
    ev.add_argument("--data-out", default="data/processed")
    ev.add_argument("--experiment-id", default=None)
    ev.add_argument("--experiment-name", default="E0")
    ev.add_argument("--adapter", default=None)
    ev.add_argument("--max-examples", type=int, default=None)
    ev.add_argument("--load-in-4bit", action="store_true", default=False)
    ev.set_defaults(func=cmd_evaluate)

    bm = sub.add_parser("benchmark")
    bm.add_argument("--repo-id", default=model_mod_default())
    bm.add_argument("--experiment-id", default=None)
    bm.set_defaults(func=cmd_benchmark)

    pf = sub.add_parser("preflight")
    pf.add_argument("--repo-id", default=model_mod_default())
    pf.add_argument("--data-dir", default="data/processed")
    pf.add_argument("--load", action="store_true", default=False, help="attempt real model load (GPU)")
    pf.set_defaults(func=cmd_preflight)

    cmp = sub.add_parser("compare")
    cmp.add_argument("--baseline", required=True, help="experiment dir or predictions.jsonl (E0)")
    cmp.add_argument("--adapter", required=True, help="experiment dir or predictions.jsonl (E1)")
    cmp.set_defaults(func=cmd_compare)

    ps = sub.add_parser("push")
    ps.add_argument("--what", choices=["model", "results", "both"], default="both")
    ps.add_argument("--hub-repo-id", required=True, help="e.g. <user>/maya-healthcare-ops")
    ps.add_argument("--results-repo-id", default=None, help="separate dataset repo for results")
    ps.add_argument("--repo-id", default=model_mod_default(), help="base model id (for adapter load)")
    ps.add_argument("--adapter-dir", default=None, help="path to saved LoRA adapter")
    ps.add_argument("--experiment-id", default=None)
    ps.add_argument("--private", action="store_true", default=False)
    ps.add_argument("--merged", action="store_true", default=False)
    ps.set_defaults(func=cmd_push)

    return p


def _load_predictions(path: str) -> list[dict]:
    p = Path(path)
    if p.is_dir():
        p = p / "predictions.jsonl"
    if not p.exists():
        raise SystemExit(f"no predictions.jsonl at {path}")
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def cmd_compare(args: argparse.Namespace) -> int:
    from . import evaluate as eval_mod
    from . import plots as plots_mod

    baseline = _load_predictions(args.baseline)
    adapter = _load_predictions(args.adapter)
    comparison = eval_mod.compare_experiments(baseline, adapter)

    out = Path(args.adapter)
    out_dir = out if out.is_dir() else out.parent
    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    plots_mod.plot_comparison_ci(comparison, out_dir / "plots")
    print(json.dumps(comparison, indent=2))
    return 0


def cmd_push(args: argparse.Namespace) -> int:
    from . import hub as hub_mod

    if args.what in ("results", "both"):
        if not args.experiment_id:
            raise SystemExit("--experiment-id required to push results")
        results_repo = args.results_repo_id or f"{args.hub_repo_id}-results"
        hub_mod.push_results(
            args.experiment_id,
            results_repo,
            private=args.private,
        )
        print(f"results pushed to {results_repo}")

    if args.what in ("model", "both"):
        from . import model as model_mod

        result = model_mod.load_model_with_adapter(
            repo_id=args.repo_id,
            adapter_dir=args.adapter_dir,
            max_seq_length=2048,
        )
        hub_mod.push_adapter(result.model, result.tokenizer, args.hub_repo_id, private=args.private, merged=args.merged)
        print(f"model pushed to {args.hub_repo_id}")

    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    from .preflight import format_report, run_preflight

    checks = run_preflight(args.repo_id, attempt_load=args.load, data_dir=args.data_dir)
    print(format_report(checks))
    return 0 if checks and all(c.ok for c in checks) else 1


def model_mod_default() -> str:
    from .model import DEFAULT_REPO

    return DEFAULT_REPO


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

"""CLI utility to run DSPy optimizers and persist the best tuned module."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import inspect
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Sequence

import dspy

from budget_parser.classifier import make_classifier

from .config import (
    ARTIFACTS_ROOT,
    AVAILABLE_OPTIMIZERS,
    DEFAULT_CATEGORIES,
    DEFAULT_GRID_TEMPLATES,
    DEFAULT_METRIC,
    WINNER_PRIORITY,
)

ExampleList = Sequence[dspy.Example]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m optim.optimizer_runner",
        description="Run multiple DSPy optimizers and save the best tuned module.",
    )

    parser.add_argument("--model", required=True, help="Model name exposed by the local LLM server.")
    parser.add_argument("--api-base", required=True, help="Base URL for the OpenAI-compatible API.")
    parser.add_argument("--api-key", help="API key for the language model server.")
    parser.add_argument("--trainset", required=True, type=Path, help="Path to the training JSONL dataset.")
    parser.add_argument("--testset", required=True, type=Path, help="Path to the test JSONL dataset.")
    parser.add_argument(
        "--outdir",
        default=ARTIFACTS_ROOT,
        type=Path,
        help="Directory where optimizer artifacts should be written.",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        choices=["accuracy", "f1", "macro_f1"],
        help="Metric used to select the best optimizer.",
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        choices=AVAILABLE_OPTIMIZERS,
        default=AVAILABLE_OPTIMIZERS,
        help="Optimizers to execute.",
    )
    parser.add_argument(
        "--grid-templates-file",
        type=Path,
        help="Optional file containing templates for grid search (one per line).",
    )
    parser.add_argument(
        "--categories-file",
        type=Path,
        help="Optional file listing categories (one per line).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for optimizers that support it.")
    parser.add_argument(
        "--max-calls",
        type=int,
        help="Soft budget for optimizer LLM calls when supported.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        help="Per-call timeout (seconds) passed to the LLM client when supported.",
    )
    parser.add_argument(
        "--eval-csv",
        type=str,
        help="Filename for the evaluation CSV relative to the model artifact directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the effective configuration without running optimizers.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run optimizers even if artifacts already exist.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    return parser.parse_args(argv)


def setup_logging(log_path: Path, verbose: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers: List[logging.Handler] = [logging.FileHandler(log_path, mode="a", encoding="utf-8")]
    handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    outdir = Path(args.outdir)
    model_dir = outdir / args.model
    log_path = model_dir / "run.log"

    setup_logging(log_path, args.verbose)

    logging.info("Optimizer runner started for model=%s", args.model)

    if args.eval_csv is None:
        args.eval_csv = f"eval_{args.model}.csv"

    if args.dry_run:
        print_effective_config(args)
        return 0

    best_path = model_dir / "best.json"
    if best_path.exists() and not args.force:
        logging.info("Best module already exists at %s. Use --force to recompute.", best_path)
        return 0

    categories = load_categories(args.categories_file)
    trainset = load_jsonl_dataset(args.trainset)
    testset = load_jsonl_dataset(args.testset)

    logging.info("Loaded %d training examples and %d test examples.", len(trainset), len(testset))

    lm = make_lm(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        timeout_s=args.timeout_s,
    )
    dspy.settings.configure(lm=lm)

    category_mapping: MutableMapping[str, None] = {category: None for category in categories}

    templates = load_templates(args.grid_templates_file, categories)

    leaderboard: Dict[str, float] = {}
    tuned_modules: Dict[str, Any] = {}

    for optimizer_name in args.optimizers:
        logging.info("Running optimizer: %s", optimizer_name)
        try:
            tuned_module, score = run_optimizer(
                optimizer_name=optimizer_name,
                categories=category_mapping,
                trainset=trainset,
                testset=testset,
                metric=args.metric,
                seed=args.seed,
                max_calls=args.max_calls,
                templates=templates,
            )
        except Exception as exc:
            logging.exception("Optimizer %s failed: %s", optimizer_name, exc)
            continue

        leaderboard[optimizer_name] = score
        tuned_modules[optimizer_name] = tuned_module

        artifact_path = model_dir / f"{optimizer_name.lower()}.json"
        save_module(tuned_module, artifact_path)
        logging.info("Optimizer %s achieved score %.4f", optimizer_name, score)

    if not leaderboard:
        logging.error("No optimizers completed successfully. Exiting with failure.")
        return 1

    winner = choose_winner(leaderboard)
    best_module = tuned_modules[winner]

    save_module(best_module, best_path)
    write_leaderboard(
        model=args.model,
        metric=args.metric,
        leaderboard=leaderboard,
        winner=winner,
        destination=model_dir / "leaderboard.json",
    )

    predictions, golds = evaluate_module(best_module, testset)
    score = compute_metric(golds, predictions, args.metric)

    logging.info(
        "Winner: %s with %s=%.4f", winner, args.metric, score
    )

    eval_csv_path = model_dir / args.eval_csv
    write_eval_csv(
        path=eval_csv_path,
        dataset=testset,
        predictions=predictions,
        optimizer_name=winner,
    )

    logging.info("Evaluation CSV written to %s", eval_csv_path)

    return 0


def print_effective_config(args: argparse.Namespace) -> None:
    config = {
        "model": args.model,
        "api_base": args.api_base,
        "api_key": bool(args.api_key) or bool(os.getenv("OPENAI_API_KEY")),
        "trainset": str(args.trainset),
        "testset": str(args.testset),
        "outdir": str(args.outdir),
        "metric": args.metric,
        "optimizers": args.optimizers,
        "grid_templates_file": str(args.grid_templates_file) if args.grid_templates_file else None,
        "categories_file": str(args.categories_file) if args.categories_file else None,
        "seed": args.seed,
        "max_calls": args.max_calls,
        "timeout_s": args.timeout_s,
        "eval_csv": args.eval_csv or f"eval_{args.model}.csv",
        "force": args.force,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(config, indent=2))


def load_categories(path: Path | None) -> List[str]:
    if path is None:
        logging.info("Using default categories: %s", ", ".join(DEFAULT_CATEGORIES))
        return list(DEFAULT_CATEGORIES)

    categories: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            categories.append(name)

    if not categories:
        logging.warning("Category file %s was empty. Falling back to defaults.", path)
        return list(DEFAULT_CATEGORIES)

    logging.info("Loaded %d categories from %s", len(categories), path)
    return categories


def load_jsonl_dataset(path: Path) -> ExampleList:
    examples: List[dspy.Example] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Skipping invalid JSON on line %d in %s", line_number, path)
                skipped += 1
                continue

            text = payload.get("text")
            label = payload.get("label")
            if text is None or label is None:
                logging.warning(
                    "Skipping record missing text/label on line %d in %s", line_number, path
                )
                skipped += 1
                continue

            example = dspy.Example(record={"description": str(text)}, label=str(label))
            examples.append(example.with_inputs("record"))

    if skipped:
        logging.info("Skipped %d malformed dataset rows from %s", skipped, path)

    return examples


def load_templates(path: Path | None, categories: Sequence[str]) -> List[str]:
    if path is None:
        return [template.format(categories=", ".join(categories)) for template in DEFAULT_GRID_TEMPLATES]

    templates: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            templates.append(value.format(categories=", ".join(categories)))

    if not templates:
        logging.warning("Template file %s was empty. Falling back to defaults.", path)
        return [template.format(categories=", ".join(categories)) for template in DEFAULT_GRID_TEMPLATES]

    return templates


def make_lm(model: str, api_base: str, api_key: str | None, timeout_s: float | None) -> dspy.OpenAI:
    kwargs: Dict[str, Any] = {
        "model": model,
        "api_base": api_base.rstrip("/"),
        "api_key": api_key or os.getenv("OPENAI_API_KEY", "dummy"),
    }

    if timeout_s is not None:
        kwargs["timeout"] = timeout_s

    return dspy.OpenAI(**kwargs)


def run_optimizer(
    optimizer_name: str,
    categories: Mapping[str, None],
    trainset: ExampleList,
    testset: ExampleList,
    metric: str,
    seed: int,
    max_calls: int | None,
    templates: Sequence[str],
) -> tuple[Any, float]:
    classifier = make_classifier(categories)

    if optimizer_name == "BootstrapFewShot":
        optimizer = instantiate_optimizer(
            dspy.BootstrapFewShot,
            metric=metric,
            seed=seed,
        )
    elif optimizer_name == "GridSearch":
        optimizer = instantiate_optimizer(
            dspy.GridSearchOptimizer,
            metric=metric,
            templates=list(templates),
        )
    elif optimizer_name == "MIPROv2":
        optimizer = instantiate_optimizer(
            dspy.MIPROv2,
            metric=metric,
            seed=seed,
            max_calls=max_calls,
        )
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    tuned_module = optimizer.compile(classifier, trainset=trainset)
    predictions, golds = evaluate_module(tuned_module, testset)
    score = compute_metric(golds, predictions, metric)
    return tuned_module, score


def instantiate_optimizer(cls: Callable[..., Any], **kwargs: Any) -> Any:
    valid_kwargs: Dict[str, Any] = {}
    try:
        signature = inspect.signature(cls)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        signature = None

    if signature is not None:
        for key, value in kwargs.items():
            if value is None:
                continue
            if key in signature.parameters:
                valid_kwargs[key] = value
    else:
        valid_kwargs = {key: value for key, value in kwargs.items() if value is not None}

    return cls(**valid_kwargs)


def evaluate_module(module: Any, dataset: ExampleList) -> tuple[List[str], List[str]]:
    predictions: List[str] = []
    golds: List[str] = []

    for example in dataset:
        record = getattr(example, "record", None)
        if not isinstance(record, Mapping):
            record = {"description": getattr(example, "text", "")}
        result = module(record=record)
        prediction = extract_label(result)
        predictions.append(prediction)
        golds.append(str(getattr(example, "label", "")))

    return predictions, golds


def extract_label(result: Any) -> str:
    if result is None:
        return ""

    if hasattr(result, "category"):
        value = getattr(result, "category")
        if value is not None:
            return str(value)

    if isinstance(result, Mapping):
        category = result.get("category")
        if category is not None:
            return str(category)

    if isinstance(result, dspy.Example):
        return str(getattr(result, "label", ""))

    return str(result)


def compute_metric(golds: Sequence[str], preds: Sequence[str], metric: str) -> float:
    if not golds:
        return 0.0

    if metric == "accuracy":
        correct = sum(1 for gold, pred in zip(golds, preds) if gold == pred)
        return correct / len(golds)

    if metric == "f1":
        return micro_f1(golds, preds)

    if metric == "macro_f1":
        return macro_f1(golds, preds)

    raise ValueError(f"Unsupported metric: {metric}")


def micro_f1(golds: Sequence[str], preds: Sequence[str]) -> float:
    labels = set(golds) | set(preds)
    tp = 0
    fp = 0
    fn = 0
    for label in labels:
        for gold, pred in zip(golds, preds):
            if pred == label and gold == label:
                tp += 1
            elif pred == label and gold != label:
                fp += 1
            elif pred != label and gold == label:
                fn += 1
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return 0.0
    return (2 * tp) / denominator


def macro_f1(golds: Sequence[str], preds: Sequence[str]) -> float:
    labels = sorted(set(golds) | set(preds))
    if not labels:
        return 0.0

    f1_scores: List[float] = []
    for label in labels:
        tp = sum(1 for gold, pred in zip(golds, preds) if gold == label and pred == label)
        fp = sum(1 for gold, pred in zip(golds, preds) if gold != label and pred == label)
        fn = sum(1 for gold, pred in zip(golds, preds) if gold == label and pred != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append((2 * precision * recall) / (precision + recall))

    return sum(f1_scores) / len(f1_scores)


def choose_winner(leaderboard: Mapping[str, float]) -> str:
    best_score = max(leaderboard.values())
    candidates = [name for name, score in leaderboard.items() if score == best_score]

    for preferred in WINNER_PRIORITY:
        if preferred in candidates:
            return preferred

    return candidates[0]


def save_module(module: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        dspy.save(module, destination)
    except Exception as exc:  # pragma: no cover - defensive logging
        logging.exception("Failed to save module to %s: %s", destination, exc)
        raise


def write_leaderboard(
    model: str,
    metric: str,
    leaderboard: Mapping[str, float],
    winner: str,
    destination: Path,
) -> None:
    payload = {
        "metric": metric,
        "results": dict(sorted(leaderboard.items(), key=lambda item: item[0])),
        "winner": winner,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_eval_csv(
    path: Path,
    dataset: ExampleList,
    predictions: Sequence[str],
    optimizer_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "gold", "pred", "correct", "optimizer"])
        for example, pred in zip(dataset, predictions):
            record = getattr(example, "record", None)
            if isinstance(record, Mapping):
                text = str(record.get("description", ""))
            else:
                text = str(getattr(example, "text", ""))
            gold = getattr(example, "label", "")
            writer.writerow([text, gold, pred, int(gold == pred), optimizer_name])


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())

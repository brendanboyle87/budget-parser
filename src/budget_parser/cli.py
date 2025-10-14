"""Command line interface for the budget parser."""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import date, datetime
from typing import Any, Dict, Iterable, Optional
from urllib.parse import urlparse, urlunparse

import numpy as np
import pandas as pd
import typer
from tqdm import tqdm

import dspy

from .classifier import ClassificationResult, TransactionClassifier, load_budget_categories

app = typer.Typer(
    name="budget-parser",
    help="Classify bank transactions into budget categories with a local LLM.",
    no_args_is_help=True,
)


@app.command()
def classify(
    budget_file: Path = typer.Option(
        ..., "--budgets", "-b", exists=True, file_okay=True, dir_okay=False, readable=True,
        help="Path to a JSON file that describes the budget categories and limits.",
    ),
    transactions_file: Path = typer.Option(
        ..., "--transactions", "-t", exists=True, file_okay=True, dir_okay=False, readable=True,
        help="CSV or Excel file containing the bank transactions to classify.",
    ),
    output_file: Path = typer.Option(
        Path("classified_transactions.csv"), "--output", "-o", file_okay=True, dir_okay=False,
        help="Where to write the classified transactions as a CSV file.",
    ),
    model: str = typer.Option(..., "--model", "-m", help="Model name exposed by the local OpenAI-compatible server."),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="API key for the language model server. Defaults to the OPENAI_API_KEY environment variable.",
        envvar="OPENAI_API_KEY",
    ),
    api_base: Optional[str] = typer.Option(
        None,
        "--api-base",
        help="Complete base URL for the OpenAI-compatible server. Overrides --host and --port when provided.",
    ),
    host: str = typer.Option(
        "http://localhost",
        "--host",
        help="Host (including scheme) for the language model server when --api-base is not provided.",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        min=1,
        max=65535,
        help="Port for the language model server when --api-base is not provided.",
    ),
    show_reasoning: bool = typer.Option(
        False,
        "--show-reasoning",
        help="Display the LLM's reasoning column when printing to the terminal.",
    ),
) -> None:
    """Classify bank transactions into budget categories."""

    try:
        categories = load_budget_categories(budget_file)
    except ValueError as exc:  # pragma: no cover - CLI validation
        raise typer.BadParameter(str(exc), param_hint="--budgets") from exc

    df = _read_transactions(transactions_file)
    original_columns = list(df.columns)

    api_base_url = _resolve_api_base(api_base, host, port)

    lm = dspy.OpenAI(model=model, api_key=api_key or "", api_base=api_base_url)
    dspy.settings.configure(lm=lm)

    classifier = TransactionClassifier(categories)

    results: list[ClassificationResult] = []
    for _, row in tqdm(
        df.iterrows(),
        total=len(df.index),
        desc="Classifying",
        unit="tx",
        leave=False,
    ):
        record = _series_to_record(row)
        result = classifier(record)
        results.append(result)

    if not results:
        typer.echo("No transactions to classify.")
        raise typer.Exit(code=0)

    result_df = df.copy()
    result_df.insert(0, "llm_category", [r.category for r in results])
    result_df.insert(1, "llm_confidence", [r.confidence for r in results])
    result_df.insert(2, "llm_reasoning", [r.reasoning for r in results])

    display_columns = ["llm_category", "llm_confidence"]
    if show_reasoning:
        display_columns.append("llm_reasoning")
    display_columns.extend([col for col in original_columns if col not in display_columns])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_file, index=False)

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        typer.echo(result_df.loc[:, display_columns].to_string(index=False))

    typer.echo(
        f"\nSaved {len(result_df)} classified transactions to {output_file.resolve()}"
    )


def _read_transactions(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            frame = pd.read_csv(path)
        elif suffix in {".xlsx", ".xls", ".xlsm", ".xlsb"}:
            frame = pd.read_excel(path)
        else:
            raise typer.BadParameter(
                "Transactions file must be a CSV or Excel document.",
                param_hint="--transactions",
            )
    except Exception as exc:  # pragma: no cover - CLI error handling
        raise typer.BadParameter(str(exc), param_hint="--transactions") from exc

    frame = frame.dropna(how="all")
    frame.columns = [str(column) for column in frame.columns]

    if frame.empty:
        raise typer.BadParameter(
            "Transactions file does not contain any rows.",
            param_hint="--transactions",
        )

    return frame


def _series_to_record(series: pd.Series) -> Dict[str, Any]:
    record: Dict[str, Any] = {}
    for column, value in series.items():
        if pd.isna(value):
            record[column] = None
            continue
        if isinstance(value, pd.Timestamp):
            record[column] = value.isoformat()
            continue
        if isinstance(value, (datetime, date)):
            record[column] = value.isoformat()
            continue
        if isinstance(value, pd.Timedelta):
            record[column] = str(value)
            continue
        if isinstance(value, np.generic):
            if isinstance(value, np.floating):
                record[column] = float(value)
            elif isinstance(value, np.integer):
                record[column] = int(value)
            else:
                record[column] = value.item()
            continue
        record[column] = value
    return record


def _resolve_api_base(api_base: Optional[str], host: str, port: int) -> str:
    if api_base:
        return api_base.rstrip("/")

    parsed = urlparse(host if "://" in host else f"http://{host}")
    scheme = parsed.scheme or "http"
    netloc = parsed.netloc or parsed.path
    path = parsed.path if parsed.netloc else ""

    if ":" not in netloc:
        netloc = f"{netloc}:{port}"

    normalized_path = path.rstrip("/")
    if not normalized_path.endswith("/v1"):
        normalized_path = f"{normalized_path}/v1"
    if not normalized_path.startswith("/"):
        normalized_path = f"/{normalized_path.lstrip('/')}"

    return urlunparse((scheme, netloc, normalized_path.rstrip("/"), "", "", ""))


def main(argv: Optional[Iterable[str]] = None) -> None:
    """Entrypoint used by the package script."""

    args = list(argv) if argv is not None else None
    app(args=args)


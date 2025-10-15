# Budget Parser

Classify bank transactions into budget categories with the help of a locally hosted
OpenAI-compatible large language model.

The tool reads a JSON file that defines your monthly budget categories, parses a CSV
or Excel export from your bank, and uses [DSPy](https://github.com/stanfordnlp/dspy)
to prompt an LLM that selects the most appropriate category for each transaction.
Classified transactions are printed to the terminal and saved as a CSV file that
includes the model's category, confidence (1–5), and reasoning. Transactions that do
not fit any budget category are tagged with `NA`.

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for dependency management and virtual
  environment handling
- Access to an OpenAI-compatible model endpoint (local or remote)

## Project setup

1. Create the virtual environment and install dependencies:

   ```bash
   uv sync
   ```

2. Optionally, activate the virtual environment for direct python usage:

   ```bash
   source .venv/bin/activate
   ```

3. Inspect the available CLI options:

   ```bash
    uv run budget-parser --help
   ```

## Input formats

### Budget configuration

The budget file must be JSON. Two structures are supported:

```json
{
  "Housing": 1800,
  "Groceries": 600,
  "Entertainment": 250,
  "Utilities": 300,
  "Savings": null
}
```

Or a list of objects/strings:

```json
[
  {"name": "Housing", "limit": 1800},
  {"name": "Groceries", "monthly_limit": 600},
  "Emergency Fund"
]
```

The `limit` values are optional; missing limits are displayed as "no explicit limit".

### Transactions

Transaction data can be supplied as CSV or Excel (`.xlsx`, `.xls`, `.xlsm`, `.xlsb`).
All columns are passed to the LLM, so the tool works with exports from most banks.
Completely empty rows are skipped automatically.

## Running classifications

Use the `classify` subcommand to process transactions. Provide your budget file,
transaction file, model name, and details about the LLM server:

```bash
uv run budget-parser classify \
  --budgets budgets.json \
  --transactions transactions.csv \
  --model my-local-model \
  --host http://localhost --port 8000
```

If you already know the exact API base URL you can pass `--api-base` instead of
`--host`/`--port` (for example `http://localhost:8000/v1`). The CLI reads the
`OPENAI_API_KEY` environment variable automatically; override it with `--api-key`
when necessary. To display the model's reasoning in the terminal output, add
`--show-reasoning`.

By default, the classified transactions are saved to `classified_transactions.csv`.
Choose a different location with `--output`.

## Example output

```
llm_category  llm_confidence  Date        Description                     Amount
Groceries                  4  2025-02-01  Local Market                     54.32
NA                         1  2025-02-02  Mystery Charge                  12.00
Utilities                  5  2025-02-03  Power Company Autopay          120.15
```

The saved CSV file contains the same data plus an `llm_reasoning` column with a
short explanation from the model.

## Optimizing the classifier with DSPy

The repository ships with an optimizer runner that can tune the DSPy classifier
for a specific LLM and persist the best-performing configuration. The CLI lives
under `optim/optimizer_runner.py` and can be invoked via `python -m`:

```bash
uv run python -m optim.optimizer_runner \
  --model llama-2-7b-chat \
  --api-base http://localhost:8000/v1 \
  --trainset data/train.jsonl \
  --testset data/test.jsonl \
  --outdir .dspy_artifacts \
  --metric accuracy \
  --optimizers BootstrapFewShot GridSearch MIPROv2 \
  --grid-templates-file config/grid_templates.txt \
  --categories-file config/categories.txt \
  --seed 42 \
  --max-calls 200 \
  --timeout-s 60
```

### Required data

- **Train/Test sets** – JSONL files where each line contains `{"text": ..., "label": ...}`.
  The runner converts these into `dspy.Example` objects and evaluates each
  optimizer on the provided test set.
- **Categories and templates** – Optional text files with one entry per line. If
  they are omitted, the defaults in `optim/config.py` are used.

### Optimizer workflow

1. The script configures a DSPy-compatible LLM client using the supplied model
   name, API base, and API key or `OPENAI_API_KEY` environment variable.
2. It builds the baseline classifier with the detected budget categories.
3. For each requested optimizer (`BootstrapFewShot`, `GridSearch`, `MIPROv2`),
   the runner compiles a tuned module, evaluates it, and records the metric.
4. The best-performing module is saved to `{outdir}/{model}/best.json` alongside
   a `leaderboard.json`, per-optimizer module snapshots, run logs, and
   `eval_{model}.csv` with detailed predictions.

Re-run the command with `--force` to overwrite existing artifacts. To inspect
the planned configuration without invoking the model, pass `--dry-run`.

### Using tuned modules at inference time

The main CLI exposes `--use-optimized` and `--outdir` flags. When `--use-optimized`
is set, the CLI searches `{outdir}/{model}/best.json` and loads the tuned module
via `optim.best_module_loader.load_best_module_for_model`. If no tuned module is
available, the CLI falls back to the baseline classifier automatically.

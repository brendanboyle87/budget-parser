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

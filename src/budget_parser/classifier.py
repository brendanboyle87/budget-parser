"""Core logic for classifying transactions with a local LLM."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional

import dspy


@dataclass
class ClassificationResult:
    """Structured result from the language model classifier."""

    category: str
    confidence: int
    reasoning: str


class CategorizeTransaction(dspy.Signature):
    """Prompt signature used by DSPy to classify a transaction."""

    categories = dspy.InputField(
        desc="List of budget categories with their monthly limits."
    )
    transaction = dspy.InputField(
        desc="Details of a single bank transaction that should be classified."
    )
    guidance = dspy.InputField(
        desc=(
            "Guidance that explains how to choose a category and when to respond with NA."
        )
    )
    reasoning = dspy.OutputField(
        desc="One or two sentences explaining the chosen category."
    )
    category = dspy.OutputField(
        desc="The name of the chosen category or the string 'NA' if none apply."
    )
    confidence = dspy.OutputField(
        desc="An integer from 1 (low) to 5 (high) indicating confidence in the answer."
    )


def load_budget_categories(path: Path) -> MutableMapping[str, Optional[float]]:
    """Load the category configuration from JSON.

    The JSON file can be structured as either a mapping of category name to limit or
    a list of objects. List entries may be simple strings (category names) or objects
    that contain "name"/"category" along with a numeric "limit"/"monthly_limit".
    """

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "categories" in payload:
        payload = payload["categories"]

    if isinstance(payload, dict):
        categories: MutableMapping[str, Optional[float]] = {}
        for name, limit in payload.items():
            categories[str(name)] = _coerce_limit(limit)
        if categories:
            return categories
    elif isinstance(payload, list):
        categories = _load_category_list(payload)
        if categories:
            return categories

    raise ValueError(
        "Budget file must contain either an object that maps category names to limits "
        "or a list of category definitions."
    )


def format_transaction_record(record: Mapping[str, Any]) -> str:
    """Convert a transaction record into a readable prompt snippet."""

    parts: list[str] = []
    for key, value in record.items():
        parts.append(f"{key}: {_format_value(value)}")
    return "\n".join(parts)


class TransactionClassifier(dspy.Module):
    """Wraps a DSPy predictor that calls the configured language model."""

    def __init__(self, categories: Mapping[str, Optional[float]]):
        super().__init__()
        if not categories:
            raise ValueError("At least one budget category is required for classification.")

        self._categories = dict(categories)
        self._predictor = dspy.Predict(CategorizeTransaction)
        self._guidance = (
            "Select the single most appropriate budget category for the transaction. "
            "Respond with the literal string 'NA' if none of the categories are a good "
            "fit. Confidence must be a whole number from 1 (low) to 5 (high)."
        )
        self._categories_text = _format_categories_prompt(self._categories)
        self._normalized_categories = {
            _normalize_text(name): name for name in self._categories.keys()
        }

    def forward(self, record: Mapping[str, Any]) -> ClassificationResult:  # type: ignore[override]
        response = self._predictor(
            categories=self._categories_text,
            transaction=format_transaction_record(record),
            guidance=self._guidance,
        )

        raw_category = getattr(response, "category", "")
        raw_confidence = getattr(response, "confidence", "")
        reasoning = getattr(response, "reasoning", "").strip()

        category = self._normalize_category(raw_category)
        confidence = self._parse_confidence(raw_confidence)

        return ClassificationResult(
            category=category,
            confidence=confidence,
            reasoning=reasoning,
        )

    def _normalize_category(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return "NA"

        normalized = _normalize_text(text)
        if normalized in {"na", "n/a", "none", "no_category", "uncategorized"}:
            return "NA"

        if normalized in self._normalized_categories:
            return self._normalized_categories[normalized]

        for name_key, original in self._normalized_categories.items():
            if name_key in normalized:
                return original

        return "NA"

    @staticmethod
    def _parse_confidence(value: Any) -> int:
        text = str(value or "").strip()
        if not text:
            return 1

        digit_match = re.search(r"([1-5])", text)
        if digit_match:
            return int(digit_match.group(1))

        try:
            numeric = float(text)
        except (TypeError, ValueError):
            return 1

        if math.isnan(numeric):
            return 1

        numeric_int = int(round(numeric))
        return max(1, min(5, numeric_int))


def _load_category_list(items: Iterable[Any]) -> MutableMapping[str, Optional[float]]:
    categories: MutableMapping[str, Optional[float]] = {}
    for entry in items:
        if isinstance(entry, str):
            categories[entry] = None
            continue
        if not isinstance(entry, Mapping):
            continue
        name = entry.get("name") or entry.get("category")
        if not name:
            continue
        limit_value: Any = None
        for key in ("limit", "monthly_limit", "amount", "budget", "cap"):
            if key in entry:
                limit_value = entry[key]
                break
        categories[str(name)] = _coerce_limit(limit_value)
    return categories


def _coerce_limit(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _format_categories_prompt(categories: Mapping[str, Optional[float]]) -> str:
    lines = ["Available categories and limits (monthly):"]
    for name, limit in categories.items():
        if limit is None:
            lines.append(f"- {name}: no explicit limit provided")
        else:
            lines.append(f"- {name}: ${limit:,.2f}")
    return "\n".join(lines)


def _format_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.2f}"
    if isinstance(value, (int, bool)):
        return str(value)
    return str(value)


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


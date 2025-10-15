"""Helpers for loading the best tuned DSPy module for a model."""

from __future__ import annotations

import logging
import os
from typing import Any

import dspy


def load_best_module_for_model(
    model: str,
    outdir: str = ".dspy_artifacts",
    fallback_module: Any | None = None,
) -> Any:
    """Load ``best.json`` for ``model`` when available.

    Parameters
    ----------
    model:
        Name of the language model, used to locate the namespace directory.
    outdir:
        Root directory that stores optimizer artifacts.
    fallback_module:
        Module returned when no tuned artifact exists or loading fails.
    """

    path = os.path.join(outdir, model, "best.json")
    if os.path.exists(path):
        try:
            tuned = dspy.load(path)
            logging.info("Loaded tuned module: %s", path)
            return tuned
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.warning(
                "Failed to load tuned module (%s). Using fallback. Error: %s", path, exc
            )
    else:
        logging.info("No tuned module found for model=%s. Using fallback.", model)

    return fallback_module

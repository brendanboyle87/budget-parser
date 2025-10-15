"""Shared configuration for optimizer utilities."""

from __future__ import annotations

DEFAULT_CATEGORIES = [
    "food",
    "transportation",
    "entertainment",
    "utilities",
]

DEFAULT_GRID_TEMPLATES = [
    "Classify the expense into one of: {categories}",
    "Which budget category best matches this expense? Choices: {categories}",
]

DEFAULT_METRIC = "accuracy"

ARTIFACTS_ROOT = ".dspy_artifacts"

WINNER_PRIORITY = ["MIPROv2", "GridSearch", "BootstrapFewShot"]

AVAILABLE_OPTIMIZERS = ["BootstrapFewShot", "GridSearch", "MIPROv2"]

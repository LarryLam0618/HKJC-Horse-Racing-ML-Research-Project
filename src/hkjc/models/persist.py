"""Persist a production model for race-day inference (M7).

The backtest/leaderboard fit models *inside* a walk-forward and throw them away. Race-day
prediction instead needs **one** model trained on all history, saved, and reloaded at the
cutoff. ``train_production_model`` fits the chosen model (configurable -- logit default, or any
zoo member / the ensemble) on the whole feature store and dumps it with the design metadata the
forward-feature path needs to rebuild a matching matrix; ``load_production_model`` reloads it.

Recommends only: a persisted model produces probabilities, never a bet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from hkjc.backtest.dataset import load_model_data
from hkjc.common.config import AppConfig, get_config
from hkjc.common.time import now_hkt
from hkjc.experiments.leaderboard import default_models
from hkjc.models.base import ProbabilityModel


@dataclass(frozen=True, slots=True)
class ProductionModel:
    """A fitted model plus the design metadata to score a forward card the same way."""

    model: ProbabilityModel
    model_name: str
    design: str  # "numeric" | "full"
    feature_names: list[str]
    numeric_indices: list[int]
    categorical_indices: list[int]
    feature_version: str
    n_train: int
    trained_at: str
    include_nlp: bool = False
    include_residual: bool = False
    include_trials: bool = False


def _model_path(cfg: AppConfig, model_name: str) -> Path:
    return cfg.paths.processed_dir / "models" / f"{model_name}.joblib"


def train_production_model(
    cfg: AppConfig | None = None,
    *,
    model_name: str = "logit",
    nn_epochs: int = 120,
    include_nlp: bool = False,
    include_residual: bool = False,
    include_trials: bool = False,
) -> Path:
    """Fit ``model_name`` on the whole feature store and persist it for race-day inference.

    The ablatable groups are recorded on the artifact so race-day rebuilds an identical design
    (a mismatch would silently score the card on the wrong columns).
    """
    cfg = cfg or get_config()
    specs = default_models(nn_epochs=nn_epochs)
    if model_name not in specs:
        msg = f"unknown model {model_name!r}; expected one of {sorted(specs)}"
        raise ValueError(msg)
    data = load_model_data(
        cfg,
        include_nlp=include_nlp,
        include_residual=include_residual,
        include_trials=include_trials,
    )
    factory, design = specs[model_name]
    model = factory(data)
    x = data.numeric() if design == "numeric" else data.x_full
    model.fit(x, data.race_id, data.y)

    production = ProductionModel(
        model=model,
        model_name=model_name,
        design=design,
        feature_names=data.feature_names,
        numeric_indices=data.numeric_indices,
        categorical_indices=data.categorical_indices,
        feature_version=cfg.features.feature_version,
        n_train=int(data.y.size),
        trained_at=now_hkt().isoformat(),
        include_nlp=include_nlp,
        include_residual=include_residual,
        include_trials=include_trials,
    )
    path = _model_path(cfg, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(production, path)
    return path


def load_production_model(
    cfg: AppConfig | None = None, *, model_name: str = "logit"
) -> ProductionModel:
    """Load a persisted production model (raises if it has not been trained yet)."""
    cfg = cfg or get_config()
    path = _model_path(cfg, model_name)
    if not path.is_file():
        msg = f"No persisted {model_name!r} model at {path}; run `hkjc train-production`."
        raise FileNotFoundError(msg)
    obj: Any = joblib.load(path)
    if not isinstance(obj, ProductionModel):  # pragma: no cover - defensive
        msg = f"Corrupt production model at {path}"
        raise TypeError(msg)
    return obj

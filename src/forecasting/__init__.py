"""Training/evaluation seams for downstream regional forecasters (roadmap T5.3)."""

from src.forecasting.adapter import (
    Forecaster,
    ForecasterAdapter,
    ForecastContractError,
    PersistenceForecaster,
    TinyResidualCoefficientModel,
    TinyRunEvidence,
    run_tiny_deterministic_step,
)

__all__ = [
    "Forecaster",
    "ForecasterAdapter",
    "ForecastContractError",
    "PersistenceForecaster",
    "TinyResidualCoefficientModel",
    "TinyRunEvidence",
    "run_tiny_deterministic_step",
]

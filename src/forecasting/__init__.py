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
from src.forecasting.artifact import (
    LaboratoryModelArtifact,
    load_laboratory_artifact,
    load_laboratory_forecaster,
    save_laboratory_artifact,
)
from src.forecasting.evaluation import ForecastEvaluation, evaluate_against_persistence

__all__ = [
    "Forecaster",
    "ForecasterAdapter",
    "ForecastContractError",
    "PersistenceForecaster",
    "TinyResidualCoefficientModel",
    "TinyRunEvidence",
    "run_tiny_deterministic_step",
    "LaboratoryModelArtifact",
    "load_laboratory_artifact",
    "load_laboratory_forecaster",
    "save_laboratory_artifact",
    "ForecastEvaluation",
    "evaluate_against_persistence",
]

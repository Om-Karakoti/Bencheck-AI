"""Network Attack Forecasting World Model core package."""

from src.inference import (
    predict,
    NetworkAttackPredictor,
    set_default_predictor,
    get_default_predictor,
)

__all__ = [
    "predict",
    "NetworkAttackPredictor",
    "set_default_predictor",
    "get_default_predictor",
]

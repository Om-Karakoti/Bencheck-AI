"""World Model and Attack Progression Forecasting Models."""

from src.models.lstm_world_model import (
    LSTMWorldModel,
    NumpyLSTMWorldModel,
    StateDistribution,
    GaussianNLLLoss,
    create_world_model,
)
from src.models.infiltration_scorer import (
    derive_infiltration_score,
    InfiltrationAssessment,
)
from src.models.trainer import (
    train_world_model,
    TrainingHistory,
)
from src.models.forecaster import (
    rollout_and_score,
    RolloutForecaster,
    RolloutForecastResult,
)
from src.models.mitre_mapper import (
    MITREStageMapper,
    map_state_to_attack_stage,
    StagePredictionResult,
    StageEvaluationReport,
    CANONICAL_STAGES,
)
from src.models.explainability import (
    InfiltrationSHAPExplainer,
    MITREStageSHAPExplainer,
    SHAPExplanation,
)
from src.models.baseline_benchmark import (
    LogisticRegressionBaseline,
    BenchmarkReport,
    ModelEvaluationSummary,
    ClassImbalanceReport,
    benchmark_models,
    analyze_class_imbalance,
    compute_per_class_metrics,
)
from src.models.leakage_checker import (
    check_temporal_leakage_by_shuffling,
    TemporalOrderShuffleResult,
)

__all__ = [
    "LSTMWorldModel",
    "NumpyLSTMWorldModel",
    "StateDistribution",
    "GaussianNLLLoss",
    "create_world_model",
    "derive_infiltration_score",
    "InfiltrationAssessment",
    "train_world_model",
    "TrainingHistory",
    "rollout_and_score",
    "RolloutForecaster",
    "RolloutForecastResult",
    "MITREStageMapper",
    "map_state_to_attack_stage",
    "StagePredictionResult",
    "StageEvaluationReport",
    "CANONICAL_STAGES",
    "InfiltrationSHAPExplainer",
    "MITREStageSHAPExplainer",
    "SHAPExplanation",
    "LogisticRegressionBaseline",
    "BenchmarkReport",
    "ModelEvaluationSummary",
    "ClassImbalanceReport",
    "benchmark_models",
    "analyze_class_imbalance",
    "compute_per_class_metrics",
    "check_temporal_leakage_by_shuffling",
    "TemporalOrderShuffleResult",
]

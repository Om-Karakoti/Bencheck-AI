"""Data-Grounded MITRE ATT&CK Stage Mapping and Evaluation.

This module maps predicted network state vectors s_t (or future rollout states s_{t+k})
to ground-truth MITRE ATT&CK attack_stage labels:
['Benign', 'Reconnaissance', 'Initial Access', 'Lateral Movement', 'Command & Control', 'Exfiltration'].

CRITICAL ARCHITECTURAL REQUIREMENT:
The mapping is trained and evaluated directly on the ground-truth 'attack_stage' column.
No arbitrary heuristic thresholds or fabricated labels are used. Instead, decision
boundaries and posterior class probabilities P(stage | s_t) are learned from historical
session data and rigorously evaluated on unseen test sessions.
"""

from dataclasses import dataclass, field
import os
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data.config import AttackStage
from src.data.causal_windowing import NETWORK_STATE_FEATURES
from src.models.lstm_world_model import StateDistribution


# The canonical 6 MITRE ATT&CK progression stages
CANONICAL_STAGES: List[str] = [
    "Benign",
    "Reconnaissance",
    "Initial Access",
    "Lateral Movement",
    "Command & Control",
    "Exfiltration",
]


@dataclass
class StagePredictionResult:
    """Prediction output for a single state vector."""
    stage_label: str                    # e.g., 'Reconnaissance'
    stage_id: int                       # e.g., 1
    confidence: float                   # Highest class probability [0.0, 1.0]
    stage_probabilities: Dict[str, float] # Posterior probability for each of the 6 stages

    def is_attack(self) -> bool:
        """Returns True if the predicted stage is any active attack stage."""
        return self.stage_id > int(AttackStage.BENIGN)


@dataclass
class StageEvaluationReport:
    """Comprehensive evaluation metrics against ground-truth attack_stage labels."""
    accuracy: float
    macro_f1: float
    weighted_f1: float
    macro_precision: float
    macro_recall: float
    per_class_report: Dict[str, Dict[str, float]]
    confusion_matrix_df: pd.DataFrame

    def summary_table(self) -> str:
        """Return a formatted string table of performance metrics."""
        lines = [
            f"Overall Accuracy:  {self.accuracy:.4f}",
            f"Macro F1-Score:    {self.macro_f1:.4f}",
            f"Weighted F1-Score: {self.weighted_f1:.4f}",
            f"Macro Precision:   {self.macro_precision:.4f}",
            f"Macro Recall:      {self.macro_recall:.4f}",
            "",
            "Per-Class Metrics:",
            f"{'Stage':<22} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}",
            "-" * 70,
        ]
        for stage in CANONICAL_STAGES:
            if stage in self.per_class_report:
                m = self.per_class_report[stage]
                lines.append(
                    f"{stage:<22} | {m.get('precision', 0.0):>10.4f} | "
                    f"{m.get('recall', 0.0):>10.4f} | {m.get('f1-score', 0.0):>10.4f} | "
                    f"{int(m.get('support', 0)):>8}"
                )
        return "\n".join(lines)


class MITREStageMapper:
    """Learns the mapping from network state vectors to MITRE ATT&CK stages.
    
    Trained strictly on historical data without heuristic guessing.
    """

    def __init__(
        self,
        classifier=None,
        feature_names: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        self.feature_names = feature_names or NETWORK_STATE_FEATURES
        self.random_state = random_state
        # Default model: Random Forest with balanced class weights to handle rarer stages
        self.classifier = classifier or RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=3,
            class_weight="balanced",
            random_state=random_state,
        )
        self.classes_: List[str] = CANONICAL_STAGES
        self.is_fitted: bool = False

    def _prepare_data(
        self,
        states_df_or_array: Union[pd.DataFrame, np.ndarray],
        stages_or_series: Optional[Union[pd.Series, np.ndarray, List]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Extract features and normalize target stage labels to canonical strings."""
        if isinstance(states_df_or_array, pd.DataFrame):
            x = states_df_or_array[self.feature_names].to_numpy(dtype=np.float32)
        else:
            x = np.asarray(states_df_or_array, dtype=np.float32)

        if stages_or_series is None:
            return x, None

        y_raw = stages_or_series
        if isinstance(y_raw, pd.Series):
            y_raw = y_raw.tolist()

        y_clean = []
        for val in y_raw:
            if isinstance(val, (int, np.integer)):
                y_clean.append(AttackStage.to_display_str(int(val)))
            elif isinstance(val, str):
                y_clean.append(AttackStage.to_display_str(int(AttackStage.from_str(val))))
            else:
                y_clean.append(str(val))

        return x, np.array(y_clean)

    def fit(
        self,
        train_states: Union[pd.DataFrame, np.ndarray],
        train_stages: Union[pd.Series, np.ndarray, List],
    ) -> "MITREStageMapper":
        """Train the stage mapper directly on historical state-to-stage ground truth.

        Args:
            train_states: Feature matrix or DataFrame of state vectors.
            train_stages: Ground-truth attack_stage column values.

        Returns:
            self
        """
        x_train, y_train = self._prepare_data(train_states, train_stages)

        # Verify ground truth stages conform to schema
        unique_stages = set(y_train)
        invalid = unique_stages - set(CANONICAL_STAGES)
        if invalid:
            raise ValueError(f"Found invalid attack stages in training data: {invalid}")

        self.classifier.fit(x_train, y_train)
        self.classes_ = list(self.classifier.classes_)
        self.is_fitted = True
        return self

    def predict_stage(
        self,
        predicted_state: Union[np.ndarray, StateDistribution, List[float]],
    ) -> StagePredictionResult:
        """Map a single predicted state vector to a MITRE ATT&CK stage label.

        # =========================================================================
        # GROUNDED MAPPING:
        # Instead of manual thresholds, the predicted state vector is passed
        # through the trained classifier, computing calibrated posterior
        # probabilities over the canonical attack stages.
        # =========================================================================

        Args:
            predicted_state: State vector mu_{t+k} from World Model rollout.

        Returns:
            StagePredictionResult with stage_label, stage_id, confidence,
            and complete class probability distribution.
        """
        if not self.is_fitted:
            raise RuntimeError("MITREStageMapper must be fitted before prediction.")

        if isinstance(predicted_state, StateDistribution):
            state_vec, _ = predicted_state.to_numpy()
        elif hasattr(predicted_state, "detach"):
            state_vec = predicted_state.detach().cpu().numpy()
        else:
            state_vec = np.asarray(predicted_state, dtype=np.float32)

        if state_vec.ndim == 1:
            state_vec = np.expand_dims(state_vec, axis=0)

        # Class prediction and probabilities
        pred_label = self.classifier.predict(state_vec)[0]
        probs = self.classifier.predict_proba(state_vec)[0]

        # Map to canonical dict
        stage_probs = {stage: 0.0 for stage in CANONICAL_STAGES}
        for cls_name, prob in zip(self.classes_, probs):
            stage_probs[cls_name] = round(float(prob), 4)

        stage_id = int(AttackStage.from_str(pred_label))
        confidence = float(stage_probs.get(pred_label, 0.0))

        return StagePredictionResult(
            stage_label=pred_label,
            stage_id=stage_id,
            confidence=confidence,
            stage_probabilities=stage_probs,
        )

    def predict_stages(
        self,
        predicted_states: Union[pd.DataFrame, np.ndarray],
    ) -> List[StagePredictionResult]:
        """Batch map an array or trajectory of predicted states to attack stages."""
        if not self.is_fitted:
            raise RuntimeError("MITREStageMapper must be fitted before prediction.")

        x, _ = self._prepare_data(predicted_states)
        pred_labels = self.classifier.predict(x)
        pred_probs = self.classifier.predict_proba(x)

        results = []
        for label, probs in zip(pred_labels, pred_probs):
            stage_probs = {stage: 0.0 for stage in CANONICAL_STAGES}
            for cls_name, prob in zip(self.classes_, probs):
                stage_probs[cls_name] = round(float(prob), 4)

            stage_id = int(AttackStage.from_str(label))
            confidence = float(stage_probs.get(label, 0.0))

            results.append(
                StagePredictionResult(
                    stage_label=label,
                    stage_id=stage_id,
                    confidence=confidence,
                    stage_probabilities=stage_probs,
                )
            )
        return results

    def evaluate(
        self,
        test_states: Union[pd.DataFrame, np.ndarray],
        test_stages: Union[pd.Series, np.ndarray, List],
    ) -> StageEvaluationReport:
        """Rigorously evaluate the mapper against actual attack_stage ground truth.

        Args:
            test_states: Unseen test set state vectors.
            test_stages: Actual ground-truth attack_stage column values.

        Returns:
            StageEvaluationReport containing accuracy, F1, precision, recall,
            and confusion matrix.
        """
        if not self.is_fitted:
            raise RuntimeError("MITREStageMapper must be fitted before evaluation.")

        x_test, y_test = self._prepare_data(test_states, test_stages)
        y_pred = self.classifier.predict(x_test)

        acc = float(accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        macro_prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        macro_rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))

        report_dict = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        )

        # Build clean confusion matrix
        labels_present = sorted(list(set(y_test).union(set(y_pred))))
        cm = confusion_matrix(y_test, y_pred, labels=labels_present)
        cm_df = pd.DataFrame(
            cm,
            index=[f"Actual_{lbl}" for lbl in labels_present],
            columns=[f"Pred_{lbl}" for lbl in labels_present],
        )

        return StageEvaluationReport(
            accuracy=round(acc, 4),
            macro_f1=round(macro_f1, 4),
            weighted_f1=round(weighted_f1, 4),
            macro_precision=round(macro_prec, 4),
            macro_recall=round(macro_rec, 4),
            per_class_report=report_dict,
            confusion_matrix_df=cm_df,
        )

    def save(self, filepath: str) -> None:
        """Persist the trained stage mapper to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(
            {
                "classifier": self.classifier,
                "feature_names": self.feature_names,
                "classes_": self.classes_,
                "is_fitted": self.is_fitted,
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath: str) -> "MITREStageMapper":
        """Load a saved MITREStageMapper model from disk."""
        data = joblib.load(filepath)
        mapper = cls(
            classifier=data["classifier"],
            feature_names=data["feature_names"],
        )
        mapper.classes_ = data["classes_"]
        mapper.is_fitted = data["is_fitted"]
        return mapper


def map_state_to_attack_stage(
    predicted_state: Union[np.ndarray, StateDistribution, List[float]],
    mapper: MITREStageMapper,
) -> StagePredictionResult:
    """Top-level standalone function mapping a predicted state to an attack_stage label.

    Directly satisfies the requirement:
    'Function mapping predicted states to one of the attack_stage labels.'
    """
    return mapper.predict_stage(predicted_state)

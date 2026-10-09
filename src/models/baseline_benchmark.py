"""Baseline and Benchmark Module for Network Attack Forecasting.

Provides:
1. LogisticRegressionBaseline: Linear classification benchmark trained on identical 36-D causal features.
2. Per-Class Metrics with False Positive Rate (FPR):
   Precision, Recall, F1, and FPR = FP / (FP + TN) calculated per-class.
3. Class Imbalance Analysis: Explicit diagnostics flagging heavy benign window dominance.
4. ModelBenchmarkReport: Side-by-side comparison tables contrasting the baseline against the primary model.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.data.config import AttackStage
from src.data.causal_windowing import NETWORK_STATE_FEATURES
from src.models.mitre_mapper import CANONICAL_STAGES


def normalize_stage_labels(labels: Sequence[Any]) -> List[str]:
    """Convert integer, AttackStage enum, or string stage labels into canonical display strings."""
    out: List[str] = []
    for val in labels:
        if isinstance(val, (int, np.integer)):
            out.append(AttackStage.to_display_str(int(val)))
        elif isinstance(val, AttackStage):
            out.append(AttackStage.to_display_str(val))
        elif isinstance(val, str) and val.strip().isdigit():
            out.append(AttackStage.to_display_str(int(val.strip())))
        else:
            out.append(str(val))
    return out


@dataclass
class PerClassMetric:
    """Detailed confusion and performance metrics for a single class."""
    class_name: str
    tp: int
    fp: int
    fn: int
    tn: int
    support: int
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float  # FPR = FP / (FP + TN)


@dataclass
class ModelEvaluationSummary:
    """Comprehensive per-class evaluation summary for a classifier."""
    model_name: str
    per_class_metrics: Dict[str, PerClassMetric]
    overall_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    macro_fpr: float
    weighted_f1: float

    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to a clean per-class DataFrame."""
        rows = []
        for name, m in self.per_class_metrics.items():
            rows.append({
                "Stage / Class": name,
                "Precision": round(m.precision, 4),
                "Recall": round(m.recall, 4),
                "F1-Score": round(m.f1_score, 4),
                "FPR": round(m.false_positive_rate, 4),
                "Support": m.support,
                "TP": m.tp,
                "FP": m.fp,
                "FN": m.fn,
                "TN": m.tn,
            })
        df = pd.DataFrame(rows)
        return df


@dataclass
class ClassImbalanceReport:
    """Diagnostics and statistics capturing distribution skew."""
    class_counts: Dict[str, int]
    class_percentages: Dict[str, float]
    total_samples: int
    majority_class: str
    majority_percentage: float
    minority_class: str
    minority_percentage: float
    imbalance_ratio: float  # majority_count / minority_count
    is_heavily_imbalanced: bool

    def format_banner(self) -> str:
        """Generate formatted terminal diagnostic banner highlighting class skew."""
        lines = [
            "=" * 95,
            "[!] DATASET CLASS IMBALANCE DIAGNOSTIC & BENCHMARK ALERT",
            "-" * 95,
            f"Total Analyzed Windows: {self.total_samples}",
            f"Majority Class:        '{self.majority_class}' ({self.majority_percentage:.1f}% of total samples)",
            f"Minority Class:        '{self.minority_class}' ({self.minority_percentage:.1f}% of total samples)",
            f"Imbalance Ratio:       {self.imbalance_ratio:.2f} : 1 ({self.majority_class} vs {self.minority_class})",
            "",
            "Per-Class Distribution Breakdown:",
            f"{'Class / MITRE Stage':<26} | {'Count':<8} | {'Percentage':<12} | {'Visual Distribution'}",
            "-" * 80,
        ]
        for c_name, cnt in self.class_counts.items():
            pct = self.class_percentages[c_name]
            bar = "#" * int(pct / 2.5)  # 100% -> 40 chars
            lines.append(f"{c_name:<26} | {cnt:>8} | {pct:>11.2f}% | {bar}")

        lines.extend([
            "-" * 80,
            "CRITICAL EVALUATION NOTE ON CLASS IMBALANCE:",
            f"In real-world networks, benign traffic vastly outnumbers attack intervals.",
            f"A naive classifier predicting '{self.majority_class}' 100% of the time achieves {self.majority_percentage:.1f}% ACCURACY,",
            f"yet suffers a disastrous 0.0% RECALL on true attack intrusions.",
            f"Therefore, accuracy is inherently misleading. Performance MUST be evaluated via",
            f"Per-Class F1, Precision, Recall, and False Positive Rate (FPR = FP / (FP + TN)).",
            "=" * 95,
        ])
        return "\n".join(lines)


def analyze_class_imbalance(
    labels: Sequence[Union[str, int, Any]],
    class_names: Optional[List[str]] = None,
) -> ClassImbalanceReport:
    """Analyze class distribution and quantify imbalance skew."""
    norm_labels = normalize_stage_labels(labels)
    series = pd.Series(norm_labels)
    total = len(series)
    if total == 0:
        raise ValueError("Cannot analyze imbalance on empty label sequence.")

    counts_dict: Dict[str, int] = {}
    ordered_classes = class_names or CANONICAL_STAGES

    for c in ordered_classes:
        counts_dict[str(c)] = int((series == str(c)).sum())

    # Include any classes present in series not in ordered_classes
    for c in series.unique():
        if str(c) not in counts_dict:
            counts_dict[str(c)] = int((series == str(c)).sum())

    pct_dict = {k: (v / total) * 100.0 for k, v in counts_dict.items()}

    maj_class = max(counts_dict, key=lambda k: counts_dict[k])
    # Consider only classes with at least 1 count for min ratio, or set to 1 if all equal
    nonzero_counts = {k: v for k, v in counts_dict.items() if v > 0}
    min_class = min(nonzero_counts, key=lambda k: nonzero_counts[k]) if nonzero_counts else maj_class

    maj_cnt = counts_dict[maj_class]
    min_cnt = nonzero_counts.get(min_class, 1)
    imbalance_ratio = maj_cnt / max(min_cnt, 1)
    is_heavily_imbalanced = (maj_cnt / total) >= 0.50 or imbalance_ratio >= 3.0

    return ClassImbalanceReport(
        class_counts=counts_dict,
        class_percentages=pct_dict,
        total_samples=total,
        majority_class=maj_class,
        majority_percentage=pct_dict[maj_class],
        minority_class=min_class,
        minority_percentage=pct_dict[min_class],
        imbalance_ratio=round(imbalance_ratio, 2),
        is_heavily_imbalanced=is_heavily_imbalanced,
    )


def compute_per_class_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    class_names: List[str],
    model_name: str = "Classifier",
) -> ModelEvaluationSummary:
    """Compute per-class Precision, Recall, F1, and False Positive Rate (FPR)."""
    y_true_arr = np.array([str(y) for y in y_true])
    y_pred_arr = np.array([str(y) for y in y_pred])
    n_samples = len(y_true_arr)

    per_class: Dict[str, PerClassMetric] = {}

    for c in class_names:
        tp = int(np.sum((y_true_arr == c) & (y_pred_arr == c)))
        fp = int(np.sum((y_true_arr != c) & (y_pred_arr == c)))
        fn = int(np.sum((y_true_arr == c) & (y_pred_arr != c)))
        tn = int(np.sum((y_true_arr != c) & (y_pred_arr != c)))
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        per_class[c] = PerClassMetric(
            class_name=c,
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            support=support,
            precision=precision,
            recall=recall,
            f1_score=f1,
            false_positive_rate=fpr,
        )

    # Macro and weighted calculations
    active_classes = [c for c in class_names if per_class[c].support > 0 or per_class[c].fp > 0]
    eval_classes = active_classes if active_classes else class_names

    macro_p = float(np.mean([per_class[c].precision for c in eval_classes]))
    macro_r = float(np.mean([per_class[c].recall for c in eval_classes]))
    macro_f1 = float(np.mean([per_class[c].f1_score for c in eval_classes]))
    macro_fpr = float(np.mean([per_class[c].false_positive_rate for c in eval_classes]))

    total_support = sum(per_class[c].support for c in class_names)
    if total_support > 0:
        weighted_f1 = float(sum(per_class[c].f1_score * per_class[c].support for c in class_names) / total_support)
    else:
        weighted_f1 = 0.0

    acc = float(np.mean(y_true_arr == y_pred_arr)) if n_samples > 0 else 0.0

    return ModelEvaluationSummary(
        model_name=model_name,
        per_class_metrics=per_class,
        overall_accuracy=acc,
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=macro_f1,
        macro_fpr=macro_fpr,
        weighted_f1=weighted_f1,
    )


class LogisticRegressionBaseline:
    """Standard linear classification baseline trained on identical features.
    
    Uses standard L2-regularized multinomial logistic regression with feature scaling.
    Operates on the EXACT identical 36-dimensional network state vector.
    """

    def __init__(
        self,
        class_weight: Optional[Union[str, Dict[Any, float]]] = None,
        c_reg: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        self.class_weight = class_weight
        self.c_reg = c_reg
        self.max_iter = max_iter
        self.random_state = random_state
        self.feature_names: List[str] = list(NETWORK_STATE_FEATURES)
        self.classes_: np.ndarray = np.array([])
        self.is_fitted: bool = False

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=self.c_reg,
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                random_state=self.random_state,
                solver="lbfgs",
            )),
        ])

    def fit(
        self,
        x_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray, List[Any]],
    ) -> "LogisticRegressionBaseline":
        """Fit baseline on identical features."""
        if isinstance(x_train, pd.DataFrame):
            x_arr = x_train[self.feature_names].to_numpy(dtype=np.float32)
        else:
            x_arr = np.asarray(x_train, dtype=np.float32)

        y_clean = normalize_stage_labels(y_train)
        y_arr = np.asarray(y_clean)
        self.pipeline.fit(x_arr, y_arr)
        self.classes_ = self.pipeline.named_steps["classifier"].classes_
        self.is_fitted = True
        return self

    def predict(self, x_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class labels on test features."""
        if not self.is_fitted:
            raise RuntimeError("LogisticRegressionBaseline is not fitted yet.")
        if isinstance(x_test, pd.DataFrame):
            x_arr = x_test[self.feature_names].to_numpy(dtype=np.float32)
        else:
            x_arr = np.asarray(x_test, dtype=np.float32)
        raw_preds = self.pipeline.predict(x_arr)
        return np.asarray(normalize_stage_labels(raw_preds))

    def predict_proba(self, x_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probability distribution."""
        if not self.is_fitted:
            raise RuntimeError("LogisticRegressionBaseline is not fitted yet.")
        if isinstance(x_test, pd.DataFrame):
            x_arr = x_test[self.feature_names].to_numpy(dtype=np.float32)
        else:
            x_arr = np.asarray(x_test, dtype=np.float32)
        return self.pipeline.predict_proba(x_arr)


@dataclass
class BenchmarkReport:
    """Comparative report benchmarking Baseline vs Primary Model."""
    baseline_summary: ModelEvaluationSummary
    primary_summary: ModelEvaluationSummary
    class_names: List[str]
    imbalance_report: ClassImbalanceReport

    def to_dataframe(self) -> pd.DataFrame:
        """Generate a comparative side-by-side DataFrame."""
        rows = []
        for c in self.class_names:
            b = self.baseline_summary.per_class_metrics.get(c)
            p = self.primary_summary.per_class_metrics.get(c)
            if b and p:
                rows.append({
                    "Class / MITRE Stage": c,
                    "Support": b.support,
                    "Baseline F1": round(b.f1_score, 4),
                    "Primary F1": round(p.f1_score, 4),
                    "Baseline Precision": round(b.precision, 4),
                    "Primary Precision": round(p.precision, 4),
                    "Baseline Recall": round(b.recall, 4),
                    "Primary Recall": round(p.recall, 4),
                    "Baseline FPR": round(b.false_positive_rate, 4),
                    "Primary FPR": round(p.false_positive_rate, 4),
                })
        return pd.DataFrame(rows)

    def summary_table(self) -> str:
        """Format a clear, side-by-side terminal comparison table."""
        lines = [
            "=" * 115,
            f"BENCHMARK COMPARISON: {self.baseline_summary.model_name} VS {self.primary_summary.model_name}",
            f"Identical Feature Set: {len(NETWORK_STATE_FEATURES)} Causal Network State Features | Unseen Test Evaluation",
            "-" * 115,
            f"{'Stage / Class':<20} | {'Support':<7} | {'Baseline (LogReg)':<38} | {'Primary Model':<38}",
            f"{'':<20} | {'':<7} | {'F1':<7} {'Prec':<7} {'Rec':<7} {'FPR':<7} | {'F1':<7} {'Prec':<7} {'Rec':<7} {'FPR':<7}",
            "-" * 115,
        ]

        for c in self.class_names:
            b = self.baseline_summary.per_class_metrics.get(c)
            p = self.primary_summary.per_class_metrics.get(c)
            if not b or not p:
                continue

            lines.append(
                f"{c:<20} | {b.support:>7} | "
                f"{b.f1_score:>7.4f} {b.precision:>7.4f} {b.recall:>7.4f} {b.false_positive_rate:>7.4f} | "
                f"{p.f1_score:>7.4f} {p.precision:>7.4f} {p.recall:>7.4f} {p.false_positive_rate:>7.4f}"
            )

        lines.extend([
            "-" * 115,
            f"{'OVERALL SUMMARY':<20} | {'Metric':<7} | {'Baseline (Logistic Regression)':<38} | {'Primary Model':<38}",
            "-" * 115,
            f"{'Overall Accuracy':<20} | {'All':<7} | {self.baseline_summary.overall_accuracy:>7.4f}{'':<31} | {self.primary_summary.overall_accuracy:>7.4f}",
            f"{'Macro F1-Score':<20} | {'Avg':<7} | {self.baseline_summary.macro_f1:>7.4f}{'':<31} | {self.primary_summary.macro_f1:>7.4f}",
            f"{'Macro Precision':<20} | {'Avg':<7} | {self.baseline_summary.macro_precision:>7.4f}{'':<31} | {self.primary_summary.macro_precision:>7.4f}",
            f"{'Macro Recall':<20} | {'Avg':<7} | {self.baseline_summary.macro_recall:>7.4f}{'':<31} | {self.primary_summary.macro_recall:>7.4f}",
            f"{'Macro FPR (FP-Rate)':<20} | {'Avg':<7} | {self.baseline_summary.macro_fpr:>7.4f}{'':<31} | {self.primary_summary.macro_fpr:>7.4f}",
            f"{'Weighted F1-Score':<20} | {'Avg':<7} | {self.baseline_summary.weighted_f1:>7.4f}{'':<31} | {self.primary_summary.weighted_f1:>7.4f}",
            "=" * 115,
        ])
        return "\n".join(lines)


def benchmark_models(
    baseline_model: Any,
    primary_model: Any,
    x_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray, List[str]],
    class_names: Optional[List[str]] = None,
    baseline_name: str = "Logistic Regression (Baseline)",
    primary_name: str = "MITRE Stage Mapper / World Model",
) -> BenchmarkReport:
    """Benchmark Baseline against Primary Model on identical test data with FPR and imbalance report."""
    y_test_arr = normalize_stage_labels(y_test)
    classes = class_names or CANONICAL_STAGES

    # 1. Analyze class imbalance
    imbalance_report = analyze_class_imbalance(y_test_arr, class_names=classes)

    # 2. Get predictions from baseline model
    if hasattr(baseline_model, "predict"):
        baseline_preds = normalize_stage_labels(baseline_model.predict(x_test))
    else:
        raise TypeError("baseline_model must implement .predict()")

    # 3. Get predictions from primary model
    if hasattr(primary_model, "predict"):
        raw_primary = primary_model.predict(x_test)
        primary_preds = normalize_stage_labels(raw_primary)
    elif hasattr(primary_model, "predict_stage"):
        # Handle MITREStageMapper which has predict_stage
        if isinstance(x_test, pd.DataFrame):
            x_mat = x_test[NETWORK_STATE_FEATURES].to_numpy()
        else:
            x_mat = np.asarray(x_test)
        primary_preds = [primary_model.predict_stage(row).stage_label for row in x_mat]
    else:
        raise TypeError("primary_model must implement .predict() or .predict_stage()")

    baseline_summary = compute_per_class_metrics(
        y_true=y_test_arr,
        y_pred=baseline_preds,
        class_names=classes,
        model_name=baseline_name,
    )
    primary_summary = compute_per_class_metrics(
        y_true=y_test_arr,
        y_pred=primary_preds,
        class_names=classes,
        model_name=primary_name,
    )

    return BenchmarkReport(
        baseline_summary=baseline_summary,
        primary_summary=primary_summary,
        class_names=classes,
        imbalance_report=imbalance_report,
    )

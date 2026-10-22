"""Analyst-Configurable Alert Threshold Evaluation for Forecast Trajectories.

An alert is raised when either independent condition is met across the forecast
horizon:
1. Forecast infiltration risk reaches the analyst's configured threshold.
2. A severe kill-chain stage (Lateral Movement, Command & Control, Exfiltration)
   is predicted as the dominant stage within the horizon.

Condition 2 is deliberately threshold-free: post-compromise stages warrant an
alert even when the composite risk score stays low.

`AlertGate` layers de-duplication on top of the stateless check: one alert per
escalation, no repeats while the same stage persists, and hysteresis so the gate
re-arms only after forecast risk falls clear of the threshold by a margin.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

# Stages that warrant an alert on their own, regardless of forecast risk level.
SEVERE_STAGES: tuple = ("Lateral Movement", "Command & Control", "Exfiltration")

# Escalation identity used when risk crosses the threshold with no severe stage.
RISK_ESCALATION: str = "Forecast Risk"

# How far below the threshold forecast risk must fall before a fired alert re-arms.
DEFAULT_HYSTERESIS_MARGIN: float = 0.10


@dataclass
class AlertEvaluation:
    """Result of checking one forecast trajectory against the configured alert threshold."""

    threshold: float
    peak_risk: float
    risk_breach: Optional[Dict[str, Any]] = None
    stage_breach: Optional[Dict[str, Any]] = None

    @property
    def triggered(self) -> bool:
        """True when either alert condition fired."""
        return self.risk_breach is not None or self.stage_breach is not None

    @property
    def reasons(self) -> List[str]:
        """Human-readable justifications, ordered risk-first."""
        reasons = []
        if self.risk_breach is not None:
            reasons.append(
                f"Forecast risk {self.risk_breach['risk'] * 100:.1f}% crossed the "
                f"{self.threshold * 100:.0f}% alert threshold at "
                f"+{self.risk_breach['seconds']}s."
            )
        if self.stage_breach is not None:
            reasons.append(
                f"{self.stage_breach['stage']} predicted within the horizon at "
                f"+{self.stage_breach['seconds']}s."
            )
        return reasons

    @property
    def message(self) -> str:
        """SOC-ready alert banner text. Empty string when no alert fired."""
        if not self.triggered:
            return ""
        return "ALERT: " + " ".join(self.reasons)


def evaluate_alert(
    forecast_scores: Sequence[float],
    forecast_stages: Sequence[str],
    threshold: float,
    time_step_sec: float = 30.0,
) -> AlertEvaluation:
    """Evaluate a rollout's per-step risk scores and dominant stages for alert conditions.

    Args:
        forecast_scores: Infiltration probability per future step, in [0, 1].
        forecast_stages: Dominant kill-chain stage per future step, same length as
            `forecast_scores`. The first severe stage wins, so a shorter stage list
            only narrows the severe-stage check.
        threshold: Alert risk threshold in [0, 1]. A score equal to the threshold alerts.
        time_step_sec: Duration of one forecast step, used to report ETA in seconds.

    Returns:
        AlertEvaluation describing which condition fired and when.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be within [0, 1], got {threshold}")

    scores = [float(score) for score in forecast_scores]
    stages = [str(stage) for stage in forecast_stages]

    risk_breach = next(
        (
            {"step": step, "seconds": int(step * time_step_sec), "risk": score}
            for step, score in enumerate(scores, start=1)
            if score >= threshold
        ),
        None,
    )
    stage_breach = next(
        (
            {"step": step, "seconds": int(step * time_step_sec), "stage": stage}
            for step, stage in enumerate(stages, start=1)
            if stage in SEVERE_STAGES
        ),
        None,
    )

    return AlertEvaluation(
        threshold=threshold,
        peak_risk=max(scores, default=0.0),
        risk_breach=risk_breach,
        stage_breach=stage_breach,
    )


@dataclass
class AlertDecision:
    """Whether the gate should raise an alert now, given what it already announced."""

    evaluation: AlertEvaluation
    notify: bool                    # Raise a brand-new alert for this escalation.
    ongoing: bool                   # Same escalation is still in progress; already announced.
    rearmed: bool                   # This call released the latch.
    latched: List[str]              # Escalations announced since the last re-arm.
    arm_level: float                # Risk must fall below this before re-arming.

    @property
    def message(self) -> str:
        return self.evaluation.message

    @property
    def reasons_text(self) -> str:
        """Alert reasons without the ALERT prefix, for the already-raised banner."""
        return " ".join(self.evaluation.reasons)


class AlertGate:
    """De-duplicates alerts: one per escalation, re-armed by hysteresis.

    Each escalation is identified by the severe stage that triggered it, or by
    `RISK_ESCALATION` when only the risk threshold was crossed. A given escalation
    announces once and is then latched, so it stays quiet while that same stage
    persists across reruns and refreshes of the forecast.

    The gate re-arms only once forecast risk has fallen below `threshold - margin`
    *and* no severe stage is still being predicted. Requiring both is what stops a
    long-lived stage from re-paging every time risk dips through the band, while the
    margin itself stops a single noisy trajectory from firing over and over.
    """

    def __init__(self, margin: float = DEFAULT_HYSTERESIS_MARGIN):
        if margin < 0.0:
            raise ValueError(f"margin must be non-negative, got {margin}")
        self.margin = margin
        self.latched: List[str] = []

    def reset(self) -> None:
        """Forget every announced escalation, e.g. when the analyst switches dataset."""
        self.latched = []

    def decide(
        self,
        forecast_scores: Sequence[float],
        forecast_stages: Sequence[str],
        threshold: float,
        time_step_sec: float = 30.0,
    ) -> AlertDecision:
        """Evaluate one forecast trajectory and report whether it warrants a new alert."""
        evaluation = evaluate_alert(forecast_scores, forecast_stages, threshold, time_step_sec)
        arm_level = max(0.0, threshold - self.margin)

        if evaluation.stage_breach is not None:
            escalation = evaluation.stage_breach["stage"]
        elif evaluation.risk_breach is not None:
            escalation = RISK_ESCALATION
        else:
            escalation = None

        rearmed = False
        if self.latched and evaluation.peak_risk < arm_level and evaluation.stage_breach is None:
            self.latched = []
            rearmed = True

        notify = False
        if escalation is not None and escalation not in self.latched:
            self.latched.append(escalation)
            notify = True

        return AlertDecision(
            evaluation=evaluation,
            notify=notify,
            ongoing=escalation is not None and not notify,
            rearmed=rearmed,
            latched=list(self.latched),
            arm_level=arm_level,
        )


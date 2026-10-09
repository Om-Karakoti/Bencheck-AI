"""Configuration and schema definitions for Network Attack Forecasting."""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List


class AttackStage(IntEnum):
    """Attack progression stages mapped to ordinal severity levels.
    
    A World Model forecasts the progression through these stages over time.
    """
    BENIGN = 0
    RECONNAISSANCE = 1
    INITIAL_ACCESS = 2
    LATERAL_MOVEMENT = 3
    COMMAND_AND_CONTROL = 4
    EXFILTRATION = 5

    @classmethod
    def from_str(cls, stage_str: str) -> "AttackStage":
        mapping = {
            "benign": cls.BENIGN,
            "reconnaissance": cls.RECONNAISSANCE,
            "initial access": cls.INITIAL_ACCESS,
            "lateral movement": cls.LATERAL_MOVEMENT,
            "command & control": cls.COMMAND_AND_CONTROL,
            "command and control": cls.COMMAND_AND_CONTROL,
            "c2": cls.COMMAND_AND_CONTROL,
            "exfiltration": cls.EXFILTRATION,
        }
        normalized = stage_str.strip().lower()
        if normalized not in mapping:
            raise ValueError(f"Unknown attack stage: {stage_str}. Expected one of {list(mapping.keys())}")
        return mapping[normalized]

    @classmethod
    def to_display_str(cls, stage: int) -> str:
        names = {
            cls.BENIGN: "Benign",
            cls.RECONNAISSANCE: "Reconnaissance",
            cls.INITIAL_ACCESS: "Initial Access",
            cls.LATERAL_MOVEMENT: "Lateral Movement",
            cls.COMMAND_AND_CONTROL: "Command & Control",
            cls.EXFILTRATION: "Exfiltration",
        }
        return names.get(stage, "Unknown")


# Explicit CSV Column Schema
SCHEMA_COLUMNS: List[str] = [
    "session_id",
    "timestamp",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",
    "bytes_per_flow",
    "packets_per_flow",
    "flow_duration",
    "iat_mean",
    "iat_var",
    "iat_max",
    "ttl_variance",
    "tcp_window_size",
    "fragment_flag",
    "payload_size_mean",
    "retransmit_count",
    "scan_signature_score",
    "attack_label",
    "attack_stage",
]

# Numerical flow columns used for window statistics
NUMERICAL_FLOW_COLS: List[str] = [
    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",
    "bytes_per_flow",
    "packets_per_flow",
    "flow_duration",
    "iat_mean",
    "iat_var",
    "iat_max",
    "ttl_variance",
    "tcp_window_size",
    "fragment_flag",
    "payload_size_mean",
    "retransmit_count",
    "scan_signature_score",
]


def validate_csv_schema(df) -> None:
    """Validate that an input DataFrame strictly adheres to the 25 required schema columns.

    CRITICAL HACKATHON / PRODUCTION DESIGN REQUIREMENT:
    Fail loudly with an explicit, informative ValueError rather than silently coercing,
    dropping bad rows, or imputing missing headers. In cybersecurity operations, missing
    telemetry fields (e.g. missing TCP flags or IAT metrics) invalidate downstream
    temporal detection assumptions, resulting in silent detection failures.

    Args:
        df: Input pandas DataFrame to validate.

    Raises:
        ValueError: If any of the 25 required columns are missing from df.columns.
    """
    missing_cols = [col for col in SCHEMA_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"SCHEMA VALIDATION FAILURE: Input DataFrame is missing {len(missing_cols)} "
            f"required column(s): {missing_cols}.\n"
            f"Expected all 25 columns: {SCHEMA_COLUMNS}"
        )



@dataclass
class WindowConfig:
    """Configuration for temporal windowing and state construction.
    
    Attributes:
        window_size_sec: Size of aggregation window W in seconds (default 30.0s).
        step_size_sec: Stride S between consecutive window evaluation timestamps (default 30.0s).
                       If step_size_sec == window_size_sec, windows are adjacent non-overlapping.
                       If step_size_sec < window_size_sec, windows are overlapping/sliding.
        fill_empty_windows: Whether to insert explicit 0-traffic state vectors for
                            quiescent intervals to preserve constant delta_t for World Models.
        history_length: Sequence length L (past observation steps) fed to the World Model.
        forecast_horizon: Sequence length H (future observation steps) predicted by the World Model.
    """
    window_size_sec: float = 30.0
    step_size_sec: float = 30.0
    fill_empty_windows: bool = True
    history_length: int = 10
    forecast_horizon: int = 5

    def __post_init__(self):
        if self.window_size_sec <= 0:
            raise ValueError(f"window_size_sec must be positive, got {self.window_size_sec}")
        if self.step_size_sec <= 0:
            raise ValueError(f"step_size_sec must be positive, got {self.step_size_sec}")
        if self.history_length <= 0:
            raise ValueError(f"history_length must be positive, got {self.history_length}")
        if self.forecast_horizon <= 0:
            raise ValueError(f"forecast_horizon must be positive, got {self.forecast_horizon}")

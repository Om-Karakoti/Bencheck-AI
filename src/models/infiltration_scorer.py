"""Downstream Infiltration and Threat Scoring from Predicted Network States.

CRITICAL ARCHITECTURAL REQUIREMENT:
This function is COMPLETELY SEPARATE and INDEPENDENT from the LSTM World Model.
The LSTM learns network state transition dynamics P(S_{t+1} | S_{<=t}) without
supervision from attack labels.
This function evaluates the *predicted future state* to derive an operational
infiltration score in [0, 1] and threat stage indicator.

Keeping state prediction and infiltration scoring separate ensures the system
is a genuine World Model that forecasts network evolution, NOT a static classifier
in disguise.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np

from src.data.config import AttackStage
from src.data.causal_windowing import NETWORK_STATE_FEATURES
from src.models.lstm_world_model import StateDistribution


@dataclass
class InfiltrationAssessment:
    """Detailed threat assessment derived exclusively from a predicted network state."""
    infiltration_score: float  # Composite threat probability [0.0, 1.0]
    threat_level: str          # "Benign", "Guarded", "Elevated", "High", "Critical"
    dominant_stage: str        # Most likely kill chain stage
    sub_scores: Dict[str, float]  # Component scores for interpretability
    explanation: str           # SOC-ready explanation of forecasted risk


def derive_infiltration_score(
    predicted_state: Union[np.ndarray, StateDistribution, List[float]],
    feature_names: Optional[List[str]] = None,
    unnormalize_fn: Optional[callable] = None,
) -> InfiltrationAssessment:
    """Derive an operational attack infiltration score strictly from a predicted future state.

    Args:
        predicted_state: Expected state vector mu_{t+1} (or StateDistribution) from the World Model.
        feature_names: Ordered list of feature names (defaults to NETWORK_STATE_FEATURES).
        unnormalize_fn: Optional inverse-transformation function (e.g. normalizer.inverse_transform)
            if predicted_state is in standardized/normalized space.

    Returns:
        InfiltrationAssessment containing the composite score [0, 1], threat tier,
        and component indicator breakdowns.
    """
    feats = feature_names or NETWORK_STATE_FEATURES

    # Extract numpy array from input
    if isinstance(predicted_state, StateDistribution):
        state_vec, _ = predicted_state.to_numpy()
    elif hasattr(predicted_state, "detach"):  # PyTorch tensor
        state_vec = predicted_state.detach().cpu().numpy()
    else:
        state_vec = np.asarray(predicted_state, dtype=np.float32)

    # Flatten if batch of 1
    if state_vec.ndim > 1:
        state_vec = state_vec.squeeze(0)

    # If an inverse-transform function was provided, map back to physical units
    if unnormalize_fn is not None:
        state_raw = unnormalize_fn(np.expand_dims(state_vec, 0)).squeeze(0)
    else:
        state_raw = state_vec

    # Map features to a dictionary for safe lookup
    val_map = {name: float(state_raw[i]) for i, name in enumerate(feats) if i < len(state_raw)}

    def get_val(name: str, default: float = 0.0) -> float:
        return val_map.get(name, default)

    # =========================================================================
    # MULTI-DIMENSIONAL THREAT SIGNAL EVALUATION
    # Evaluates the forecasted network physics across key attack dimensions:
    # 1. Reconnaissance / Port Scanning
    # 2. Initial Access & Exploitation
    # 3. Lateral Movement / Internal Spread
    # 4. Command & Control (C2) Beaconing
    # 5. Data Exfiltration Volume
    # =========================================================================

    # 1. Reconnaissance Signal: elevated scan score, high SYN/ACK ratio, high unique dst ports
    scan_score = get_val("scan_signature_score_mean", 0.0)
    scan_max = get_val("scan_signature_score_max", 0.0)
    syn_ack = get_val("syn_ack_ratio", 0.0)
    unique_ports = get_val("unique_dst_ports", 0.0)

    recon_sig = 0.5 * min(1.0, max(scan_score, scan_max)) + \
                0.3 * min(1.0, max(0.0, (syn_ack - 0.15) / 0.5)) + \
                0.2 * min(1.0, unique_ports / 20.0)
    recon_sig = float(np.clip(recon_sig, 0.0, 1.0))

    # 2. Initial Access Signal: elevated retransmits, sustained flow durations, moderate scan
    flow_dur_mean = get_val("flow_duration_mean", 0.0)
    retrans_rate = get_val("retransmit_rate", 0.0)
    rst_rate = get_val("rst_rate", 0.0)

    access_sig = 0.4 * min(1.0, max(0.0, flow_dur_mean - 0.5) / 2.5) + \
                 0.3 * min(1.0, retrans_rate * 5.0) + \
                 0.3 * min(1.0, rst_rate * 10.0)
    access_sig = float(np.clip(access_sig, 0.0, 1.0))

    # 3. Lateral Movement: internal fan-out, multiple destination IPs
    unique_dst_ips = get_val("unique_dst_ips", 0.0)
    fan_out = get_val("fan_out_ratio", 0.0)
    ttl_var = get_val("ttl_variance_mean", 0.0)

    if unique_dst_ips > 1:
        lateral_sig = 0.5 * min(1.0, (unique_dst_ips - 1.0) / 7.0) + \
                      0.3 * min(1.0, fan_out / 5.0) + \
                      0.2 * min(1.0, max(0.0, 0.1 - ttl_var) / 0.1)
    else:
        lateral_sig = 0.0
    lateral_sig = float(np.clip(lateral_sig, 0.0, 1.0))

    # 4. Command & Control Signal: periodic beaconing (very low IAT variance)
    iat_var = get_val("iat_var_avg", 1.0)
    iat_mean = get_val("iat_mean_avg", 0.0)
    # Low non-zero IAT variance indicates deterministic timing
    c2_beacon_sig = 0.0
    if 0.001 < iat_mean < 0.5 and iat_var < 0.005:
        c2_beacon_sig = min(1.0, 0.005 / max(iat_var, 1e-6)) * 0.8
    c2_sig = float(np.clip(c2_beacon_sig, 0.0, 1.0))

    # 5. Exfiltration Signal: massive outbound bytes and large payload sizes
    byte_rate = get_val("byte_rate", 0.0)
    payload_max = get_val("payload_size_max", 0.0)
    total_bytes = get_val("total_bytes", 0.0)

    exfil_sig = 0.6 * min(1.0, max(0.0, byte_rate - 30000.0) / 150000.0) + \
                0.4 * min(1.0, max(0.0, payload_max - 1000.0) / 450.0)
    exfil_sig = float(np.clip(exfil_sig, 0.0, 1.0))

    # Composite Infiltration Score: soft-max / envelope across attack signals
    stage_signals = {
        "Reconnaissance": recon_sig,
        "Initial Access": access_sig,
        "Lateral Movement": lateral_sig,
        "Command & Control": c2_sig,
        "Exfiltration": exfil_sig,
    }

    # Dominant attack stage characteristics in the predicted state
    max_stage_name = max(stage_signals, key=stage_signals.get)
    max_signal_val = stage_signals[max_stage_name]

    # Overall composite infiltration score
    # Baseline benign assumption: score starts near 0.05
    composite_score = float(np.clip(max_signal_val, 0.0, 1.0))

    # Threat Tier Classification
    if composite_score < 0.20:
        threat_level = "Benign"
        dominant_stage = "Benign"
        explanation = "Forecasted network state conforms to normal operational baselines."
    elif composite_score < 0.45:
        threat_level = "Guarded"
        dominant_stage = max_stage_name
        explanation = f"Subtle anomalies detected in forecasted state consistent with early {max_stage_name}."
    elif composite_score < 0.70:
        threat_level = "Elevated"
        dominant_stage = max_stage_name
        explanation = f"Forecasted network dynamics strongly indicate impending {max_stage_name}."
    elif composite_score < 0.85:
        threat_level = "High"
        dominant_stage = max_stage_name
        explanation = f"High-confidence forecast of active {max_stage_name} progression."
    else:
        threat_level = "Critical"
        dominant_stage = max_stage_name
        explanation = f"Critical threat: forecasted state exhibits severe {max_stage_name} signatures."

    return InfiltrationAssessment(
        infiltration_score=round(composite_score, 4),
        threat_level=threat_level,
        dominant_stage=dominant_stage,
        sub_scores={
            "reconnaissance": round(recon_sig, 4),
            "initial_access": round(access_sig, 4),
            "lateral_movement": round(lateral_sig, 4),
            "command_and_control": round(c2_sig, 4),
            "exfiltration": round(exfil_sig, 4),
        },
        explanation=explanation,
    )

"""Strict Causal Windowing and Network State Aggregation for World Models.

This module converts raw, per-flow network traffic records into a continuous
time-series of multi-dimensional network state vectors s_t.

CRITICAL ARCHITECTURAL REQUIREMENT:
A World Model forecasts the temporal evolution of network state and attack
progression. The state at observation timestamp t represents the network's condition
over the preceding causal window (t - W, t].
NO DATA from timestamp > t (such as t+1 or later) is ever included when aggregating
the state at time t. Violating this rule introduces lookahead bias and fatally breaks
real-time attack forecasting.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.data.config import (
    AttackStage,
    WindowConfig,
    SCHEMA_COLUMNS,
    NUMERICAL_FLOW_COLS,
    validate_csv_schema,
)


# Ordered list of feature names comprising the network state vector s_t
NETWORK_STATE_FEATURES: List[str] = [
    # Traffic Volume & Rates
    "flow_count",
    "total_bytes",
    "total_packets",
    "byte_rate",
    "packet_rate",
    "bytes_per_flow_mean",
    "packets_per_flow_mean",
    # Flow Durations
    "flow_duration_mean",
    "flow_duration_max",
    "flow_duration_std",
    # TCP Flags & Ratios
    "syn_count_sum",
    "ack_count_sum",
    "fin_count_sum",
    "rst_count_sum",
    "syn_ack_ratio",
    "rst_rate",
    # Inter-Arrival Time (IAT) Dynamics
    "iat_mean_avg",
    "iat_var_avg",
    "iat_max_peak",
    # Packet & Header Characteristics
    "ttl_variance_mean",
    "tcp_window_size_mean",
    "tcp_window_size_min",
    "fragment_flag_sum",
    "payload_size_mean_avg",
    "payload_size_max",
    "retransmit_count_sum",
    "retransmit_rate",
    # Topological Fan-out & Scanning Signatures
    "unique_src_ips",
    "unique_dst_ips",
    "unique_dst_ports",
    "fan_out_ratio",
    "scan_signature_score_mean",
    "scan_signature_score_max",
    # Protocol Distributions
    "tcp_flow_ratio",
    "udp_flow_ratio",
    "other_protocol_ratio",
]


class CausalNetworkStateExtractor:
    """Extracts strictly causal, fixed-interval network state vectors from flow data."""

    def __init__(self, config: Optional[WindowConfig] = None):
        self.config = config or WindowConfig()

    def validate_schema(self, df: pd.DataFrame) -> None:
        """Validate that all required columns are present in the input DataFrame."""
        validate_csv_schema(df)

    def extract_features_from_window(
        self,
        window_df: pd.DataFrame,
        window_size_sec: float,
    ) -> Dict[str, float]:
        """Aggregate flow and packet features within a single causal window into a state vector.
        
        Args:
            window_df: DataFrame containing ONLY flows strictly within (t - W, t].
            window_size_sec: Duration W of the window in seconds.

        Returns:
            Dictionary mapping feature names in NETWORK_STATE_FEATURES to float values.
        """
        flow_count = len(window_df)

        if flow_count == 0:
            # Clean idle state representation: preserves temporal continuity in World Models
            idle_state = {feat: 0.0 for feat in NETWORK_STATE_FEATURES}
            return idle_state

        total_bytes = float(window_df["bytes_per_flow"].sum())
        total_packets = float(window_df["packets_per_flow"].sum())
        byte_rate = total_bytes / window_size_sec
        packet_rate = total_packets / window_size_sec
        bytes_per_flow_mean = float(window_df["bytes_per_flow"].mean())
        packets_per_flow_mean = float(window_df["packets_per_flow"].mean())

        # Flow duration
        dur = window_df["flow_duration"]
        flow_duration_mean = float(dur.mean())
        flow_duration_max = float(dur.max())
        flow_duration_std = float(dur.std(ddof=0)) if flow_count > 1 else 0.0

        # TCP Flags & Ratios
        syn_sum = float(window_df["syn_count"].sum())
        ack_sum = float(window_df["ack_count"].sum())
        fin_sum = float(window_df["fin_count"].sum())
        rst_sum = float(window_df["rst_count"].sum())
        syn_ack_ratio = syn_sum / (ack_sum + 1.0)
        rst_rate = rst_sum / (total_packets + 1.0)

        # IAT Dynamics
        iat_mean_avg = float(window_df["iat_mean"].mean())
        iat_var_avg = float(window_df["iat_var"].mean())
        iat_max_peak = float(window_df["iat_max"].max())

        # Packet & Header Characteristics
        ttl_variance_mean = float(window_df["ttl_variance"].mean())
        tcp_window_size_mean = float(window_df["tcp_window_size"].mean())
        tcp_window_size_min = float(window_df["tcp_window_size"].min())
        fragment_flag_sum = float(window_df["fragment_flag"].sum())
        payload_size_mean_avg = float(window_df["payload_size_mean"].mean())
        payload_size_max = float(window_df["payload_size_mean"].max())
        retransmit_sum = float(window_df["retransmit_count"].sum())
        retransmit_rate = retransmit_sum / (total_packets + 1.0)

        # Topological Fan-out & Scanning Signatures
        unique_src_ips = float(window_df["src_ip"].nunique())
        unique_dst_ips = float(window_df["dst_ip"].nunique())
        unique_dst_ports = float(window_df["dst_port"].nunique())
        fan_out_ratio = unique_dst_ports / max(unique_src_ips, 1.0)
        scan_sig_mean = float(window_df["scan_signature_score"].mean())
        scan_sig_max = float(window_df["scan_signature_score"].max())

        # Protocol Distribution (Vectorized for maximum throughput on large datasets)
        proto_vals = window_df["protocol"]
        tcp_mask = (proto_vals == 6) | (proto_vals == "6") | (proto_vals == "tcp") | (proto_vals == "TCP")
        udp_mask = (proto_vals == 17) | (proto_vals == "17") | (proto_vals == "udp") | (proto_vals == "UDP")
        tcp_count = float(tcp_mask.sum())
        udp_count = float(udp_mask.sum())
        other_proto_count = float(max(0.0, flow_count - (tcp_count + udp_count)))

        tcp_flow_ratio = tcp_count / flow_count
        udp_flow_ratio = udp_count / flow_count
        other_proto_ratio = max(0.0, other_proto_count / flow_count)

        return {
            "flow_count": float(flow_count),
            "total_bytes": total_bytes,
            "total_packets": total_packets,
            "byte_rate": byte_rate,
            "packet_rate": packet_rate,
            "bytes_per_flow_mean": bytes_per_flow_mean,
            "packets_per_flow_mean": packets_per_flow_mean,
            "flow_duration_mean": flow_duration_mean,
            "flow_duration_max": flow_duration_max,
            "flow_duration_std": flow_duration_std,
            "syn_count_sum": syn_sum,
            "ack_count_sum": ack_sum,
            "fin_count_sum": fin_sum,
            "rst_count_sum": rst_sum,
            "syn_ack_ratio": syn_ack_ratio,
            "rst_rate": rst_rate,
            "iat_mean_avg": iat_mean_avg,
            "iat_var_avg": iat_var_avg,
            "iat_max_peak": iat_max_peak,
            "ttl_variance_mean": ttl_variance_mean,
            "tcp_window_size_mean": tcp_window_size_mean,
            "tcp_window_size_min": tcp_window_size_min,
            "fragment_flag_sum": fragment_flag_sum,
            "payload_size_mean_avg": payload_size_mean_avg,
            "payload_size_max": payload_size_max,
            "retransmit_count_sum": retransmit_sum,
            "retransmit_rate": retransmit_rate,
            "unique_src_ips": unique_src_ips,
            "unique_dst_ips": unique_dst_ips,
            "unique_dst_ports": unique_dst_ports,
            "fan_out_ratio": fan_out_ratio,
            "scan_signature_score_mean": scan_sig_mean,
            "scan_signature_score_max": scan_sig_max,
            "tcp_flow_ratio": tcp_flow_ratio,
            "udp_flow_ratio": udp_flow_ratio,
            "other_protocol_ratio": other_proto_ratio,
        }

    def determine_window_label(
        self,
        window_df: pd.DataFrame,
    ) -> Tuple[int, str, int, str]:
        """Determine the ground-truth attack progression stage for a window.

        If multiple stages occur within the causal window, we record the maximum
        severity stage reached by timestamp t.

        Returns:
            Tuple of (stage_ordinal, stage_name, is_attack, dominant_label)
        """
        if len(window_df) == 0:
            return (
                int(AttackStage.BENIGN),
                AttackStage.to_display_str(AttackStage.BENIGN),
                0,
                "Benign",
            )

        def _parse_stage(st) -> int:
            if isinstance(st, (int, np.integer)):
                return int(st)
            return int(AttackStage.from_str(str(st)))

        parsed_stages = window_df["attack_stage"].apply(_parse_stage)
        max_stage_val = int(parsed_stages.max())
        stage_name = AttackStage.to_display_str(max_stage_val)
        is_attack = 1 if max_stage_val > AttackStage.BENIGN else 0

        # Dominant attack label name
        if is_attack:
            attack_flows = window_df[parsed_stages > AttackStage.BENIGN]
            dominant_label = str(attack_flows["attack_label"].mode().iloc[0])
        else:
            dominant_label = "Benign"

        return max_stage_val, stage_name, is_attack, dominant_label

    def process_session(self, df_session: pd.DataFrame) -> pd.DataFrame:
        """Process a single session into a sequence of causally windowed state vectors."""
        session_id = df_session["session_id"].iloc[0]

        # Ensure timestamps are numeric (seconds or epoch float) and sorted ascending
        df_sorted = df_session.sort_values(by="timestamp", ascending=True).copy()
        df_sorted["timestamp"] = df_sorted["timestamp"].astype(float)

        t_min = float(df_sorted["timestamp"].min())
        t_max = float(df_sorted["timestamp"].max())

        window_size = float(self.config.window_size_sec)
        step_size = float(self.config.step_size_sec)

        # Build grid of window reference timestamps t_k
        # Window k evaluates the historical interval (t_k - window_size, t_k]
        # We anchor to the nearest step boundary at or before t_min:
        t_anchor = float(np.floor(t_min / step_size) * step_size)
        first_cutoff = t_anchor + window_size

        ts_arr = df_sorted["timestamp"].values

        if self.config.fill_empty_windows:
            window_cutoffs = []
            curr_cutoff = first_cutoff
            while curr_cutoff < t_max:
                window_cutoffs.append(curr_cutoff)
                curr_cutoff += step_size
            window_cutoffs.append(curr_cutoff)
        else:
            # Only keep windows that contain at least one flow (O(log N) searchsorted)
            window_cutoffs = []
            curr_cutoff = first_cutoff
            while curr_cutoff <= t_max + step_size:
                t_start = curr_cutoff - window_size
                side_s = "left" if curr_cutoff == first_cutoff else "right"
                idx_s = int(np.searchsorted(ts_arr, t_start, side=side_s))
                idx_e = int(np.searchsorted(ts_arr, curr_cutoff, side="right"))
                if idx_e > idx_s:
                    window_cutoffs.append(curr_cutoff)
                curr_cutoff += step_size

        records = []
        for step_idx, t_k in enumerate(window_cutoffs):
            t_start = t_k - window_size

            # O(log N) slice preserving strict causal boundaries
            if step_idx == 0:
                idx_start = int(np.searchsorted(ts_arr, t_start, side="left"))
            else:
                idx_start = int(np.searchsorted(ts_arr, t_start, side="right"))
            idx_end = int(np.searchsorted(ts_arr, t_k, side="right"))

            window_flows = df_sorted.iloc[idx_start:idx_end]

            if len(window_flows) == 0 and not self.config.fill_empty_windows:
                continue

            state_features = self.extract_features_from_window(window_flows, window_size)
            stage_val, stage_name, is_attack, dom_label = self.determine_window_label(window_flows)

            row = {
                "session_id": session_id,
                "step_index": step_idx,
                "window_start": t_start,
                "window_end": t_k,
                "timestamp": t_k,  # Reference timestamp of state observation s_t
                **state_features,
                "attack_stage": stage_val,
                "attack_stage_name": stage_name,
                "is_attack": is_attack,
                "attack_label": dom_label,
            }
            records.append(row)

        return pd.DataFrame(records)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process multiple sessions into a unified DataFrame of causal network state vectors.
        
        Args:
            df: Input DataFrame conforming to SCHEMA_COLUMNS.

        Returns:
            DataFrame where each row is a network state vector s_t for a specific session and window.
        """
        self.validate_schema(df)
        session_dfs = []

        for session_id, group in df.groupby("session_id", sort=False):
            session_result = self.process_session(group)
            if not session_result.empty:
                session_dfs.append(session_result)

        if not session_dfs:
            return pd.DataFrame()

        return pd.concat(session_dfs, ignore_index=True)


def load_and_window_csv(
    csv_path_or_df: Union[str, pd.DataFrame],
    config: Optional[WindowConfig] = None,
) -> pd.DataFrame:
    """Convenience function to load a CSV and apply strict causal windowing.

    Args:
        csv_path_or_df: File path to CSV or pre-loaded pandas DataFrame.
        config: Optional WindowConfig instance (default: 30s window, 30s step).

    Returns:
        DataFrame containing windowed network state vectors and attack progression targets.
    """
    if isinstance(csv_path_or_df, str):
        df = pd.read_csv(csv_path_or_df)
    else:
        df = csv_path_or_df.copy()

    extractor = CausalNetworkStateExtractor(config)
    return extractor.process(df)

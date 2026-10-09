"""High-Fidelity Synthetic Multi-Stage Network Attack Data Generator.

Generates realistic network flow traces adhering strictly to the user schema:
session_id, timestamp, src_ip, dst_ip, src_port, dst_port, protocol,
syn_count, ack_count, fin_count, rst_count, bytes_per_flow, packets_per_flow,
flow_duration, iat_mean, iat_var, iat_max,
ttl_variance, tcp_window_size, fragment_flag, payload_size_mean, retransmit_count,
scan_signature_score, attack_label, attack_stage

Simulates realistic multi-stage cyber kill chain progression over time:
Benign -> Reconnaissance -> Initial Access -> Lateral Movement -> Command & Control -> Exfiltration.
"""

import argparse
import random
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from src.data.config import SCHEMA_COLUMNS, AttackStage


def generate_synthetic_attack_scenario(
    session_id: str,
    start_time: float = 0.0,
    stage_duration_sec: float = 120.0,
    benign_ratio: float = 1.0,
    flows_per_minute: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a single session showing kill-chain evolution over time."""
    np.random.seed(seed)
    random.seed(seed)

    benign_duration = stage_duration_sec * max(1.0, benign_ratio)
    stages_sequence = [
        (AttackStage.BENIGN, "Benign", "Normal", benign_duration),
        (AttackStage.RECONNAISSANCE, "Reconnaissance", "Port_Scan", stage_duration_sec),
        (AttackStage.INITIAL_ACCESS, "Initial Access", "SSH_BruteForce", stage_duration_sec),
        (AttackStage.LATERAL_MOVEMENT, "Lateral Movement", "SMB_Propagation", stage_duration_sec),
        (AttackStage.COMMAND_AND_CONTROL, "Command & Control", "C2_Beaconing", stage_duration_sec),
        (AttackStage.EXFILTRATION, "Exfiltration", "Data_Exfiltration", stage_duration_sec),
    ]

    records: List[Dict[str, Any]] = []
    current_time = start_time

    # Enterprise network topology
    internal_hosts = [f"192.168.1.{i}" for i in range(10, 30)]
    external_servers = [f"203.0.113.{i}" for i in range(1, 10)]
    attacker_ip = "198.51.100.44"
    c2_server_ip = "198.51.100.99"
    compromised_host = "192.168.1.15"

    for stage_enum, stage_name, label_name, duration in stages_sequence:
        stage_end_time = current_time + duration
        flow_interval = 60.0 / flows_per_minute

        while current_time < stage_end_time:
            current_time += float(np.random.exponential(flow_interval))
            if current_time >= stage_end_time:
                break

            # Default benign flow template
            src_ip = random.choice(internal_hosts)
            dst_ip = random.choice(external_servers)
            src_port = random.randint(32768, 65535)
            dst_port = random.choice([80, 443, 53, 8080])
            protocol = 6  # TCP
            syn_count = 1
            ack_count = random.randint(5, 25)
            fin_count = 1
            rst_count = 0
            bytes_per_flow = int(np.random.lognormal(mean=8.0, sigma=1.2))
            packets_per_flow = max(4, int(bytes_per_flow / 800) + random.randint(2, 6))
            flow_duration = float(np.random.exponential(1.5))
            iat_mean = float(flow_duration / max(packets_per_flow - 1, 1))
            iat_var = float(iat_mean * 0.4)
            iat_max = float(iat_mean * 2.5)
            ttl_variance = float(np.random.uniform(0.01, 0.15))
            tcp_window_size = int(random.choice([16384, 32768, 65535]))
            fragment_flag = 0
            payload_size_mean = float(bytes_per_flow / max(packets_per_flow, 1))
            retransmit_count = int(np.random.poisson(0.1))
            scan_signature_score = float(np.random.uniform(0.0, 0.08))
            row_label = "Normal"
            row_stage = "Benign"

            # Inject attack characteristics depending on stage
            if stage_enum == AttackStage.RECONNAISSANCE and random.random() < 0.70:
                # High destination port sweep, high SYN, low ACK, high scan score
                src_ip = attacker_ip
                dst_ip = compromised_host
                dst_port = random.randint(20, 1024)
                syn_count = random.randint(1, 3)
                ack_count = 0
                fin_count = 0
                rst_count = random.choice([0, 1])
                bytes_per_flow = random.randint(40, 120)
                packets_per_flow = random.randint(1, 3)
                flow_duration = float(np.random.uniform(0.001, 0.05))
                iat_mean = 0.005
                iat_var = 0.0001
                iat_max = 0.01
                scan_signature_score = float(np.random.uniform(0.75, 0.98))
                row_label = label_name
                row_stage = stage_name

            elif stage_enum == AttackStage.INITIAL_ACCESS and random.random() < 0.60:
                # SSH/RDP brute force attempts
                src_ip = attacker_ip
                dst_ip = compromised_host
                dst_port = 22
                syn_count = 1
                ack_count = random.randint(10, 30)
                rst_count = 1
                bytes_per_flow = random.randint(800, 3500)
                packets_per_flow = random.randint(12, 40)
                flow_duration = float(np.random.uniform(0.5, 3.0))
                payload_size_mean = float(np.random.uniform(60, 150))
                retransmit_count = random.randint(1, 4)
                scan_signature_score = float(np.random.uniform(0.40, 0.70))
                row_label = label_name
                row_stage = stage_name

            elif stage_enum == AttackStage.LATERAL_MOVEMENT and random.random() < 0.65:
                # Internal propagation: SMB / RPC sweep
                src_ip = compromised_host
                dst_ip = random.choice(internal_hosts)
                dst_port = random.choice([445, 135, 3389])
                syn_count = 1
                ack_count = random.randint(8, 20)
                bytes_per_flow = random.randint(1500, 8000)
                packets_per_flow = random.randint(15, 50)
                flow_duration = float(np.random.uniform(0.8, 4.0))
                ttl_variance = float(np.random.uniform(0.01, 0.05))  # LAN uniform TTL
                scan_signature_score = float(np.random.uniform(0.50, 0.85))
                row_label = label_name
                row_stage = stage_name

            elif stage_enum == AttackStage.COMMAND_AND_CONTROL and random.random() < 0.50:
                # Periodic C2 beaconing with very low jitter / IAT variance
                src_ip = compromised_host
                dst_ip = c2_server_ip
                dst_port = 443
                syn_count = 1
                ack_count = random.randint(6, 12)
                bytes_per_flow = random.randint(300, 900)
                packets_per_flow = random.randint(6, 14)
                flow_duration = float(np.random.uniform(0.2, 0.6))
                iat_mean = 0.05
                iat_var = 0.00005  # Highly deterministic timing
                iat_max = 0.08
                scan_signature_score = float(np.random.uniform(0.20, 0.45))
                row_label = label_name
                row_stage = stage_name

            elif stage_enum == AttackStage.EXFILTRATION and random.random() < 0.75:
                # Massive outbound transfer
                src_ip = compromised_host
                dst_ip = c2_server_ip
                dst_port = 443
                syn_count = 1
                ack_count = random.randint(50, 200)
                bytes_per_flow = int(np.random.uniform(50000, 500000))
                packets_per_flow = max(50, int(bytes_per_flow / 1400))
                flow_duration = float(np.random.uniform(3.0, 15.0))
                payload_size_mean = float(np.random.uniform(1100, 1450))
                tcp_window_size = 65535
                retransmit_count = random.randint(2, 10)
                scan_signature_score = float(np.random.uniform(0.30, 0.60))
                row_label = label_name
                row_stage = stage_name

            records.append({
                "session_id": session_id,
                "timestamp": round(current_time, 4),
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_port": src_port,
                "dst_port": dst_port,
                "protocol": protocol,
                "syn_count": syn_count,
                "ack_count": ack_count,
                "fin_count": fin_count,
                "rst_count": rst_count,
                "bytes_per_flow": bytes_per_flow,
                "packets_per_flow": packets_per_flow,
                "flow_duration": round(flow_duration, 4),
                "iat_mean": round(iat_mean, 6),
                "iat_var": round(iat_var, 6),
                "iat_max": round(iat_max, 6),
                "ttl_variance": round(ttl_variance, 4),
                "tcp_window_size": tcp_window_size,
                "fragment_flag": fragment_flag,
                "payload_size_mean": round(payload_size_mean, 2),
                "retransmit_count": retransmit_count,
                "scan_signature_score": round(scan_signature_score, 4),
                "attack_label": row_label,
                "attack_stage": row_stage,
            })

    df = pd.DataFrame(records)
    return df[SCHEMA_COLUMNS]


def generate_synthetic_dataset(
    num_sessions: int = 3,
    output_path: Optional[str] = None,
    stage_duration_sec: float = 90.0,
    benign_ratio: float = 1.0,
    seed: int = 100,
) -> pd.DataFrame:
    """Generate a multi-session synthetic attack dataset and optionally save to CSV."""
    all_sessions = []
    for s_idx in range(num_sessions):
        session_id = f"session_{s_idx + 1:03d}"
        df_session = generate_synthetic_attack_scenario(
            session_id=session_id,
            start_time=0.0,
            stage_duration_sec=stage_duration_sec,
            benign_ratio=benign_ratio,
            flows_per_minute=120,
            seed=seed + (s_idx * 37),
        )
        all_sessions.append(df_session)

    full_df = pd.concat(all_sessions, ignore_index=True)
    if output_path:
        full_df.to_csv(output_path, index=False)
        print(f"Saved {len(full_df)} flows across {num_sessions} sessions to {output_path}")

    return full_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic network attack flows")
    parser.add_argument("--output", type=str, default="data/synthetic_attacks.csv")
    parser.add_argument("--num-sessions", type=int, default=3)
    parser.add_argument("--stage-duration", type=float, default=90.0)
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    generate_synthetic_dataset(
        num_sessions=args.num_sessions,
        output_path=args.output,
        stage_duration_sec=args.stage_duration,
    )

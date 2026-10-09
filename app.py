"""
Cyber Threat Forecasting and Classification — Bencheck World Model SOC Application
Multi-File Upload • Disk-Backed Persistence Across Refreshes • Mobile & Laptop Screen Fit • Zero-Hang Architecture
Theme: Tactical Cyber Threat World Model SOC (Professional Military-Grade UI, Zero Emojis)
"""

import os
import io
import gc
import json
import pickle
import time
import textwrap
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Suppress minor Streamlit deprecation warnings
warnings.filterwarnings("ignore")

# Configure page layout
st.set_page_config(
    page_title="Bencheck - Cyber Threat World Model SOC",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Visibility Cyber Styling with Responsive Fluid Typography & Military-Grade SOC Headings
CSS_STYLES = """<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
  --obsidian: #040d1a;
  --navy-deep: #071221;
  --cyan-glow: #00e5ff;
  --crimson: #ef4444;
  --cyber-grid: rgba(0, 229, 255, 0.08);
}

* { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
  font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
  font-size: 15px;
  color: #e2e8f0 !important;
}

.stApp {
  background-color: var(--obsidian) !important;
  background-image: 
    radial-gradient(circle at center, transparent 0%, rgba(0, 229, 255, 0.05) 60%, rgba(4, 13, 26, 0.95) 100%),
    linear-gradient(var(--cyber-grid) 1px, transparent 1px),
    linear-gradient(90deg, var(--cyber-grid) 1px, transparent 1px) !important;
  background-size: 100% 100%, 40px 40px, 40px 40px !important;
  background-attachment: fixed !important;
}

.scan-line {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 2px;
  background: linear-gradient(to right, transparent, var(--cyan-glow), transparent);
  opacity: 0.35;
  animation: scan 4s linear infinite;
  z-index: 999999;
  pointer-events: none;
}

@keyframes scan {
  0% { top: 0%; }
  100% { top: 100%; }
}

.mono { font-family: 'JetBrains Mono', monospace !important; }

/* Tactical SOC Heading Style (Replaces Emojis) */
.soc-heading {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: clamp(0.76rem, 1.05vw, 0.90rem) !important;
  font-weight: 700 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  color: #f8fafc !important;
  padding: 0.38rem 0.75rem;
  background: rgba(0, 229, 255, 0.04);
  border-left: 3px solid #00e5ff;
  border-radius: 0 8px 8px 0;
  margin-bottom: 0.75rem;
  box-shadow: inset 0 0 15px rgba(0, 229, 255, 0.03);
  white-space: nowrap !important;
  overflow: hidden;
  text-overflow: ellipsis;
}

.soc-heading.orange {
  border-left-color: #f97316;
  background: rgba(249, 115, 22, 0.04);
}

.soc-heading.crimson {
  border-left-color: #ef4444;
  background: rgba(239, 68, 68, 0.04);
}

.soc-heading.purple {
  border-left-color: #c084fc;
  background: rgba(192, 132, 252, 0.04);
}

.soc-heading-tag {
  font-size: 0.68rem;
  font-weight: 800;
  padding: 0.15rem 0.45rem;
  border-radius: 4px;
  background: rgba(0, 229, 255, 0.15);
  color: #00e5ff;
  border: 1px solid rgba(0, 229, 255, 0.35);
  letter-spacing: 0.06em;
  white-space: nowrap !important;
  flex-shrink: 0;
}

.soc-heading-tag.orange {
  background: rgba(249, 115, 22, 0.15);
  color: #f97316;
  border-color: rgba(249, 115, 22, 0.35);
}

.soc-heading-tag.crimson {
  background: rgba(239, 68, 68, 0.15);
  color: #ef4444;
  border-color: rgba(239, 68, 68, 0.35);
}

.soc-heading-sub {
  font-size: 0.66rem;
  color: rgba(255, 255, 255, 0.4);
  margin-left: auto;
  letter-spacing: 0.05em;
  white-space: nowrap !important;
  overflow: hidden;
  text-overflow: ellipsis;
  flex-shrink: 0;
}

/* Dedicated Sidebar Heading Style (Never Wraps) */
.sidebar-heading {
  display: flex;
  align-items: center;
  gap: 0.40rem;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.72rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.04em !important;
  text-transform: uppercase !important;
  color: #f8fafc !important;
  padding: 0.30rem 0.55rem;
  background: rgba(0, 229, 255, 0.04);
  border-left: 3px solid #00e5ff;
  border-radius: 0 6px 6px 0;
  margin: 1.0rem 0 0.45rem 0;
  white-space: nowrap !important;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* Glass Panels */
.glass-panel {
  background: rgba(7, 18, 33, 0.82) !important;
  backdrop-filter: blur(12px) !important;
  -webkit-backdrop-filter: blur(12px) !important;
  border: 1px solid rgba(255, 255, 255, 0.07) !important;
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.42) !important;
  border-radius: 16px;
}

.neon-border-cyan {
  border: 1px solid rgba(0, 229, 255, 0.35) !important;
  box-shadow: inset 0 0 12px rgba(0, 229, 255, 0.06), 0 4px 20px rgba(0, 229, 255, 0.08) !important;
}

.neon-border-crimson {
  border: 1px solid rgba(239, 68, 68, 0.4) !important;
  box-shadow: inset 0 0 12px rgba(239, 68, 68, 0.08), 0 4px 20px rgba(239, 68, 68, 0.12) !important;
}

.glow-text-cyan { text-shadow: 0 0 15px rgba(0, 229, 255, 0.65) !important; }

.pulse-engine { animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }

@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.4; transform: scale(0.85); }
}

/* Typography */
h1, .stApp h1 {
  font-size: clamp(1.20rem, 2.5vw, 1.85rem) !important;
  font-weight: 800 !important;
  line-height: 1.25 !important;
  overflow-wrap: break-word !important;
  word-break: normal !important;
  color: #f8fafc !important;
}

h2, .stApp h2 {
  font-size: clamp(1.05rem, 1.8vw, 1.40rem) !important;
  font-weight: 700 !important;
  line-height: 1.3 !important;
  overflow-wrap: break-word !important;
  word-break: normal !important;
  color: #f1f5f9 !important;
}

h3, .stApp h3 {
  font-size: clamp(0.92rem, 1.4vw, 1.18rem) !important;
  font-weight: 700 !important;
  line-height: 1.35 !important;
  overflow-wrap: break-word !important;
  word-break: normal !important;
}

h4, .stApp h4 {
  font-size: clamp(0.84rem, 1.2vw, 1.02rem) !important;
  font-weight: 600 !important;
  line-height: 1.35 !important;
  overflow-wrap: break-word !important;
  word-break: normal !important;
}

p, span, label {
  overflow-wrap: break-word;
  word-break: normal;
}

section[data-testid="stSidebar"] {
  background-color: rgba(4, 13, 26, 0.96) !important;
  border-right: 1px solid rgba(0, 229, 255, 0.15) !important;
  box-shadow: 4px 0 24px rgba(0, 0, 0, 0.5) !important;
}

[data-testid="stMetric"] {
  background: rgba(7, 18, 33, 0.85) !important;
  border: 1px solid rgba(0, 229, 255, 0.25) !important;
  border-radius: 12px !important;
  padding: 0.65rem 0.85rem !important;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35) !important;
  overflow: visible !important;
  min-height: 75px !important;
}

[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] > div, [data-testid="stMetricLabel"] p {
  font-size: clamp(0.68rem, 0.9vw, 0.80rem) !important;
  font-weight: 600 !important;
  color: #94a3b8 !important;
  font-family: 'JetBrains Mono', monospace !important;
  text-transform: uppercase !important;
  letter-spacing: 0.04em !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}

[data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {
  font-size: clamp(1.05rem, 1.6vw, 1.45rem) !important;
  font-weight: 800 !important;
  color: #00e5ff !important;
  font-family: 'JetBrains Mono', monospace !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}

[data-testid="stMetricDelta"], [data-testid="stMetricDelta"] > div {
  font-size: clamp(0.65rem, 0.8vw, 0.76rem) !important;
  font-family: 'JetBrains Mono', monospace !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}

[data-testid="stFileUploader"] {
  background: rgba(4, 13, 26, 0.5) !important;
  border: 1px dashed rgba(0, 229, 255, 0.35) !important;
  border-radius: 12px !important;
  padding: 0.70rem !important;
  transition: all 0.3s ease !important;
}

[data-testid="stFileUploader"]:hover {
  border-color: #00e5ff !important;
  box-shadow: 0 0 16px rgba(0, 229, 255, 0.25) !important;
}

[data-testid="stFileUploaderFileName"], [data-testid="stFileUploaderFileData"] div, [data-testid="stFileUploaderFileData"] span {
  white-space: normal !important;
  overflow-wrap: break-word !important;
  word-break: normal !important;
  overflow: visible !important;
  text-overflow: clip !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: clamp(0.72rem, 0.90vw, 0.82rem) !important;
}

/* Tactical Buttons (Never Wrap or Overflow) */
div.stButton > button {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: clamp(0.68rem, 0.88vw, 0.80rem) !important;
  font-weight: 700 !important;
  border-radius: 8px !important;
  padding: 0.38rem 0.58rem !important;
  border: 1px solid rgba(255, 255, 255, 0.14) !important;
  background: rgba(16, 24, 43, 0.88) !important;
  color: #e2e8f0 !important;
  white-space: nowrap !important;
  text-overflow: ellipsis !important;
  overflow: hidden !important;
  word-break: normal !important;
  line-height: 1.25 !important;
  box-sizing: border-box !important;
  transition: all 0.20s ease !important;
}

div.stButton > button:hover {
  border-color: #00e5ff !important;
  box-shadow: 0 0 16px rgba(0, 229, 255, 0.35) !important;
  transform: translateY(-1px) !important;
  color: #00e5ff !important;
}

div.stButton > button[kind="primary"] {
  background: linear-gradient(90deg, #0891b2, #1d4ed8) !important;
  color: #ffffff !important;
  font-size: clamp(0.78rem, 1.1vw, 0.98rem) !important;
  font-weight: 800 !important;
  letter-spacing: 0.04em !important;
  border: 1px solid rgba(0, 229, 255, 0.6) !important;
  padding: 0.65rem 1.2rem !important;
  box-shadow: 0 0 25px rgba(6, 182, 212, 0.45) !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  word-break: normal !important;
}

div.stButton > button[kind="primary"]:hover {
  box-shadow: 0 0 32px rgba(0, 229, 255, 0.70) !important;
  transform: translateY(-1px) scale(1.005) !important;
}

div[role="radiogroup"] {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 0.40rem !important;
}

div[role="radiogroup"] > label {
  background: rgba(16, 24, 43, 0.85) !important;
  border: 1px solid rgba(0, 229, 255, 0.25) !important;
  border-radius: 8px !important;
  padding: 0.30rem 0.55rem !important;
  cursor: pointer !important;
  white-space: nowrap !important;
  transition: all 0.2s ease !important;
}

div[role="radiogroup"] > label:hover {
  border-color: #00e5ff !important;
  box-shadow: 0 0 10px rgba(0, 229, 255, 0.3) !important;
}

div[role="radiogroup"] > label p, div[role="radiogroup"] > label span, div[role="radiogroup"] > label div {
  font-size: clamp(0.68rem, 0.86vw, 0.78rem) !important;
  font-family: 'JetBrains Mono', monospace !important;
  white-space: nowrap !important;
  word-break: normal !important;
  line-height: 1.2 !important;
}

.active-files-box {
  background: rgba(7, 18, 33, 0.88);
  border: 1px solid rgba(0, 229, 255, 0.35);
  border-radius: 12px;
  padding: 0.70rem 0.90rem;
  margin-bottom: 0.85rem;
}

.file-chip {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  background: rgba(0, 229, 255, 0.10);
  border: 1px solid rgba(0, 229, 255, 0.30);
  color: #38bdf8;
  font-family: 'JetBrains Mono', monospace;
  font-size: clamp(0.70rem, 0.90vw, 0.80rem);
  padding: 0.28rem 0.50rem;
  border-radius: 6px;
  white-space: nowrap !important;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 100%;
}

div[data-testid="stAlert"] {
  border-radius: 10px !important;
  padding: 0.55rem 0.85rem !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.78rem !important;
  overflow-wrap: break-word !important;
}

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--obsidian); }
::-webkit-scrollbar-thumb { background: rgba(0, 229, 255, 0.25); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0, 229, 255, 0.5); }

/* Mobile & Android Responsive Breakpoints (Zero Text Break / Zero Overflow) */
@media (max-width: 768px) {
  .soc-heading-sub { display: none !important; }
  .glass-panel { padding: 0.85rem !important; }
  
  /* Top-level multi-column blocks stack cleanly on phones */
  div[data-testid="stHorizontalBlock"]:has(> div[data-testid="column"]:nth-child(3)) {
    flex-direction: column !important;
  }
  div[data-testid="stHorizontalBlock"]:has(> div[data-testid="column"]:nth-child(3)) > div[data-testid="column"] {
    width: 100% !important;
    min-width: 100% !important;
    margin-bottom: 0.5rem !important;
  }

  /* 4-column HUD cards form a clean 2x2 grid */
  div[data-testid="stHorizontalBlock"]:has(> div[data-testid="column"]:nth-child(4)) > div[data-testid="column"] {
    flex: 1 1 calc(50% - 0.4rem) !important;
    min-width: calc(50% - 0.4rem) !important;
  }

  div.stButton > button {
    font-size: 0.72rem !important;
    padding: 0.34rem 0.48rem !important;
  }
  
  [data-testid="stMetric"] {
    min-height: 70px !important;
    padding: 0.5rem 0.65rem !important;
  }
}

@media (min-width: 769px) and (max-width: 1100px) {
  div.stButton > button { font-size: 0.76rem !important; padding: 0.36rem 0.55rem !important; }
  [data-testid="stMetricValue"] { font-size: 1.15rem !important; }
}
</style>"""
st.markdown(CSS_STYLES, unsafe_allow_html=True)

# Core imports from ML Pipeline
from src.data.config import WindowConfig, AttackStage, SCHEMA_COLUMNS, validate_csv_schema
from src.data.causal_windowing import CausalNetworkStateExtractor, NETWORK_STATE_FEATURES
from src.data.split import session_train_test_split
from src.data.dataset import WorldModelSequenceDataset, StateNormalizer
from src.data.synthetic import generate_synthetic_dataset
from src.models.lstm_world_model import NumpyLSTMWorldModel
from src.models.trainer import train_world_model
from src.models.infiltration_scorer import derive_infiltration_score, InfiltrationAssessment
from src.models.forecaster import rollout_and_score
from src.models.mitre_mapper import MITREStageMapper, CANONICAL_STAGES
from src.models.explainability import InfiltrationSHAPExplainer

# Threat Stage Vivid Color Palette (Superdesign Bencheck)
STAGE_COLORS = {
    "Benign": "#10b981",            # Emerald
    "Reconnaissance": "#f59e0b",    # Amber
    "Initial Access": "#f97316",    # Orange
    "Lateral Movement": "#c084fc",  # Purple
    "Command & Control": "#00e5ff", # Cyan
    "Exfiltration": "#ef4444",      # Crimson
}

FEATURE_LABELS = {
    "syn_ack_ratio": "SYN/ACK Ratio",
    "scan_signature_score_mean": "Port Scan Frequency",
    "scan_signature_score_max": "Peak Port Scan Signature",
    "unique_dst_ports": "Target Port Fan-Out",
    "unique_dst_ips": "Internal Host Propagation",
    "fan_out_ratio": "Subnet Fan-Out Ratio",
    "byte_rate": "Outbound Byte Rate Surge",
    "retransmit_rate": "TCP Retransmission Rate",
    "rst_rate": "Connection Reset Rate",
    "flow_duration_mean": "Sustained Session Duration",
    "iat_mean_avg": "C2 Beacon Periodicity",
    "iat_var_avg": "Low Timing Variance (Beacon)",
    "payload_size_max": "Peak Payload Size (Exfil)",
    "payload_size_mean": "Average Payload Size",
    "total_bytes": "Total Bytes Transferred",
    "flow_count": "Concurrent Flows",
}

# =========================================================================
# DISK-BACKED SESSION PERSISTENCE (ACROSS PAGE REFRESHES)
# =========================================================================

PERSIST_DIR = os.path.join("data", "persisted_upload")
DATA_FILE = os.path.join(PERSIST_DIR, "dataset.parquet")
META_FILE = os.path.join(PERSIST_DIR, "meta.json")
CACHE_FILE = os.path.join(PERSIST_DIR, "analysis_cache.pkl")

def save_persisted_dataset(df: pd.DataFrame, file_names: list):
    """Save ingested dataset to disk so it survives browser refreshes."""
    os.makedirs(PERSIST_DIR, exist_ok=True)
    df.to_parquet(DATA_FILE, index=False)
    meta = {
        "file_names": file_names,
        "total_flows": len(df),
        "dataset_label": f"{len(file_names)} File(s) Persisted",
    }
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f)

def load_persisted_dataset():
    """Load previously uploaded dataset if it exists on disk."""
    if os.path.exists(DATA_FILE) and os.path.exists(META_FILE):
        try:
            df = pd.read_parquet(DATA_FILE)
            with open(META_FILE, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if "_source_file" not in df.columns and "file_names" in meta and len(meta["file_names"]) == 1:
                df["_source_file"] = meta["file_names"][0]
            return df, meta
        except Exception:
            return None, None
    return None, None

def clear_persisted_dataset():
    """Manually clear persisted files from disk."""
    for f in [DATA_FILE, META_FILE, CACHE_FILE]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

def remove_single_file(target_file: str):
    """Remove a single specific file from the active dataset and update persistence."""
    df = st.session_state.get("current_df")
    meta = st.session_state.get("persisted_meta", {})
    file_list = meta.get("file_names", [])

    if df is None or target_file not in file_list:
        return

    if "_source_file" in df.columns:
        remaining_df = df[df["_source_file"] != target_file].copy()
    else:
        remaining_df = pd.DataFrame()

    remaining_files = [f for f in file_list if f != target_file]

    if remaining_files and len(remaining_df) > 0:
        save_persisted_dataset(remaining_df, remaining_files)
        st.session_state["current_df"] = remaining_df
        st.session_state["persisted_meta"] = {
            "file_names": remaining_files,
            "total_flows": len(remaining_df)
        }
        st.session_state["dataset_label"] = f"{len(remaining_files)} File(s) Persisted"
        st.session_state["is_user_persisted"] = True
        st.session_state["analysis_cache"] = None
        st.session_state["processed_upload_ids"] = [(fn, 0) for fn in remaining_files]
        st.success(f"[SUCCESS] Removed '{target_file}'. {len(remaining_files)} file(s) remaining ({len(remaining_df):,} flows).")
    else:
        clear_persisted_dataset()
        st.session_state["current_df"] = None
        st.session_state["persisted_meta"] = None
        st.session_state["dataset_label"] = ""
        st.session_state["is_user_persisted"] = False
        st.session_state["analysis_cache"] = None
        st.session_state["processed_upload_ids"] = []
        st.success(f"[SUCCESS] Removed '{target_file}'. All files cleared.")

def save_persisted_cache(cache: dict):
    """Save prediction analysis cache to disk."""
    os.makedirs(PERSIST_DIR, exist_ok=True)
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)
    except Exception:
        pass

def load_persisted_cache():
    """Load saved analysis predictions from disk."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

# Initialize Session State on load/refresh
if "initialized" not in st.session_state:
    st.session_state["initialized"] = True
    persisted_df, persisted_meta = load_persisted_dataset()
    if persisted_df is not None:
        st.session_state["current_df"] = persisted_df
        st.session_state["persisted_meta"] = persisted_meta
        st.session_state["dataset_label"] = persisted_meta.get("dataset_label", "Persisted User Files")
        st.session_state["is_user_persisted"] = True
        st.session_state["analysis_cache"] = load_persisted_cache()
    else:
        st.session_state["current_df"] = None
        st.session_state["persisted_meta"] = None
        st.session_state["dataset_label"] = ""
        st.session_state["is_user_persisted"] = False
        st.session_state["analysis_cache"] = None

if "processed_upload_ids" not in st.session_state:
    st.session_state["processed_upload_ids"] = []

# =========================================================================
# CACHED MODEL ENGINE (SINGLE INITIALIZATION IN MEMORY)
# =========================================================================

@st.cache_resource(show_spinner="Initializing Threat World Model & Stage Mapper...")
def get_trained_models():
    """Pre-train and cache models on baseline data so inference is instantaneous."""
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    synth_csv = os.path.join(data_dir, "synthetic_attack_flows.csv")

    if os.path.exists(synth_csv):
        raw_df = pd.read_csv(synth_csv)
    else:
        raw_df = generate_synthetic_dataset(
            num_sessions=3,
            output_path=synth_csv,
            stage_duration_sec=90.0,
            benign_ratio=3.0,
            seed=42,
        )

    config = WindowConfig(
        window_size_sec=30.0,
        step_size_sec=30.0,
        fill_empty_windows=True,
        history_length=6,
        forecast_horizon=3,
    )

    extractor = CausalNetworkStateExtractor(config)
    windowed = extractor.process(raw_df)

    split = session_train_test_split(windowed, test_size=0.33, seed=42)
    normalizer = StateNormalizer().fit(split.train_df)

    train_ds = WorldModelSequenceDataset(split.train_df, config, normalizer)
    test_ds = WorldModelSequenceDataset(split.test_df, config, normalizer)

    train_b = train_ds.get_batch(list(range(len(train_ds))))
    test_b = test_ds.get_batch(list(range(len(test_ds))))

    world_model = NumpyLSTMWorldModel(input_dim=len(NETWORK_STATE_FEATURES), hidden_dim=48, seed=42)
    train_world_model(
        model=world_model,
        train_x=train_b["x_history"],
        train_y=train_b["y_future_state"][:, 0, :],
        val_x=test_b["x_history"],
        val_y=test_b["y_future_state"][:, 0, :],
        num_epochs=10,
        verbose=False,
    )

    mapper = MITREStageMapper(random_state=42)
    mapper.fit(split.train_df, split.train_df["attack_stage"])

    explainer = InfiltrationSHAPExplainer(
        background_data=split.train_df,
        feature_names=NETWORK_STATE_FEATURES,
        max_background_samples=20,
        random_state=42,
    )

    return config, normalizer, world_model, mapper, explainer

# =========================================================================
# SIDEBAR CONTROLS (TACTICAL SOC STYLING, ZERO EMOJIS)
# =========================================================================

with st.sidebar:
    st.markdown(
        "<div style='display:flex; align-items:center; gap:0.6rem; margin-bottom:1rem;'>"
        "<div style='width:34px; height:34px; background:rgba(0,229,255,0.1); border-radius:8px; display:flex; align-items:center; justify-content:center; border:1px solid rgba(0,229,255,0.3);'>"
        "<svg width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='#00e5ff' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'>"
        "<path d='M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z'/><path d='m9 12 2 2 4-4'/></svg></div>"
        "<div><div style='font-weight:800; font-size:1.0rem; color:#f8fafc; font-family:Plus Jakarta Sans,sans-serif;'>BENCHECK</div>"
        "<div style='font-size:0.70rem; color:#00e5ff; font-family:JetBrains Mono,monospace;'>SOC CONSOLE v4.2</div></div></div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='sidebar-heading'>"
        "<span class='soc-heading-tag'>HORIZON</span>"
        "<span>SIMULATION</span>"
        "</div>",
        unsafe_allow_html=True
    )
    forecast_steps = st.slider("Lookahead Horizon (Seconds Ahead)", min_value=30, max_value=300, value=150, step=30)
    k_steps = forecast_steps // 30

    st.markdown(
        f"<div style='background:rgba(0,229,255,0.06); border:1px solid rgba(0,229,255,0.25); border-radius:8px; padding:0.6rem 0.8rem; margin:0.8rem 0; font-family:JetBrains Mono,monospace; font-size:0.78rem;'>"
        f"<div style='color:rgba(255,255,255,0.5); text-transform:uppercase; font-size:0.68rem;'>CONFIGURED LOOKAHEAD</div>"
        f"<div style='color:#00e5ff; font-weight:700; font-size:1.05rem; margin-top:2px;'>+{forecast_steps}s ({k_steps} Steps)</div></div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='sidebar-heading' style='margin-top:1.2rem;'>"
        "<span class='soc-heading-tag'>STAGE</span>"
        "<span>MITRE ATT&CK</span>"
        "</div>",
        unsafe_allow_html=True
    )
    for stage_name, col in STAGE_COLORS.items():
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:0.5rem; margin-bottom:0.35rem;'>"
            f"<span style='width:8px; height:8px; border-radius:2px; background:{col}; display:inline-block; box-shadow:0 0 8px {col}88;'></span>"
            f"<span style='color:{col}; font-weight:600; font-family:JetBrains Mono, monospace; font-size:0.84rem;'>{stage_name.upper()}</span></div>",
            unsafe_allow_html=True
        )

    st.markdown(
        "<div class='sidebar-heading' style='margin-top:1.2rem;'>"
        "<span class='soc-heading-tag'>TIER</span>"
        "<span>SEVERITY LEVELS</span>"
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div style='font-family:JetBrains Mono,monospace; font-size:0.74rem; white-space:nowrap;'>"
        "<div style='margin-bottom:0.35rem;'><span style='color:#10b981; font-weight:700;'>[NORMAL]</span> &lt; 10% Risk</div>"
        "<div style='margin-bottom:0.35rem;'><span style='color:#f59e0b; font-weight:700;'>[GUARDED]</span> 10% - 25% Risk</div>"
        "<div style='margin-bottom:0.35rem;'><span style='color:#f97316; font-weight:700;'>[ELEVATED]</span> 25% - 50% Risk</div>"
        "<div style='margin-bottom:0.35rem;'><span style='color:#ef4444; font-weight:700;'>[HIGH]</span> 50% - 75% Risk</div>"
        "<div><span style='color:#ff0055; font-weight:700;'>[CRITICAL]</span> &gt; 75% Risk</div></div>",
        unsafe_allow_html=True
    )

# =========================================================================
# TOP NAVBAR (TACTICAL SOC HUD, ZERO EMOJIS, SLEEK SVG ICONS)
# =========================================================================

nav_html = f"""<div class="scan-line"></div>
<div class="glass-panel" style="padding: 0.85rem 1.4rem; margin-bottom: 1.5rem; display: flex; justify-content: space-between; align-items: center; border: 1px solid rgba(0, 229, 255, 0.25); flex-wrap: wrap; gap: 0.8rem;">
<div style="display: flex; align-items: center; gap: 0.9rem;">
<div style="width: 44px; height: 44px; background: rgba(0, 229, 255, 0.12); border-radius: 10px; display: flex; align-items: center; justify-content: center; border: 1px solid rgba(0, 229, 255, 0.4); box-shadow: 0 0 16px rgba(0, 229, 255, 0.25);">
<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00e5ff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
<path d="m9 12 2 2 4-4"/>
</svg>
</div>
<div>
<div style="font-size: 1.15rem; font-weight: 800; letter-spacing: -0.02em; font-family: 'Plus Jakarta Sans', sans-serif;">
<span class="glow-text-cyan" style="color: #00e5ff;">BENCHECK</span> 
<span style="color: rgba(255,255,255,0.3); font-weight: 300;">//</span> 
<span style="font-size: 0.82rem; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.15em; font-family: 'JetBrains Mono', monospace;">World Model SOC</span>
</div>
<div style="font-size: 0.72rem; color: rgba(255,255,255,0.4); font-family: 'JetBrains Mono', monospace;">
AUTOREGRESSIVE THREAT PREDICTION & INFILTRATION SURVEILLANCE
</div>
</div>
</div>
<div style="display: flex; align-items: center; gap: 1.4rem; flex-wrap: wrap;">
<div style="display: flex; align-items: center; gap: 0.5rem; background: rgba(0, 0, 0, 0.4); padding: 0.4rem 0.9rem; border-radius: 9999px; border: 1px solid rgba(255, 255, 255, 0.08);">
<div class="pulse-engine" style="width: 8px; height: 8px; border-radius: 50%; background: #10b981; box-shadow: 0 0 8px rgba(16, 185, 129, 0.8);"></div>
<span style="font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700; color: #34d399;">Engine Status: Online</span>
</div>
<div style="background: rgba(0, 229, 255, 0.08); border: 1px solid rgba(0, 229, 255, 0.25); border-radius: 8px; padding: 0.35rem 0.85rem; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;">
<span style="color: rgba(255,255,255,0.5);">HORIZON: </span>
<span style="color: #00e5ff; font-weight: 700;">+{forecast_steps}s LOOKAHEAD</span>
</div>
<div style="display: flex; align-items: center; gap: 0.75rem;">
<div style="text-align: right;">
<div style="font-size: 0.80rem; font-weight: 700; color: #f8fafc;">Cpt. Sterling</div>
<div style="font-size: 0.68rem; color: rgba(255, 255, 255, 0.4); font-family: 'JetBrains Mono', monospace;">Level 4 Analyst</div>
</div>
<div style="width: 38px; height: 38px; border-radius: 8px; background: linear-gradient(135deg, rgba(0, 229, 255, 0.25), rgba(192, 132, 252, 0.25)); border: 1px solid rgba(255, 255, 255, 0.15); display: flex; align-items: center; justify-content: center;">
<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
<circle cx="12" cy="7" r="4"/>
</svg>
</div>
</div>
</div>
</div>"""
st.markdown(nav_html, unsafe_allow_html=True)

# =========================================================================
# SECTION 1: TELEMETRY INGESTION HUB [STEP 01]
# =========================================================================

st.markdown(
    "<div class='soc-heading'>"
    "<span class='soc-heading-tag'>STEP 01</span>"
    "<span>TELEMETRY INGESTION HUB</span>"
    "<span class='soc-heading-sub'>BUFFER & SIMULATION</span>"
    "</div>",
    unsafe_allow_html=True
)

ingest_container = st.container()

with ingest_container:
    col_up, col_presets, col_stats = st.columns([1.35, 1.15, 1.0])

    with col_up:
        st.markdown(
            "<div style='font-family:JetBrains Mono, monospace; font-size:0.72rem; color:rgba(255,255,255,0.45); text-transform:uppercase; margin-bottom:0.35rem;'>"
            "Telemetry Source Ingestion (Max 400MB)</div>",
            unsafe_allow_html=True
        )

        uploaded_files = st.file_uploader(
            "Drag & drop packet capture CSVs",
            type=["csv"],
            accept_multiple_files=True,
            help="Select one or more CSV files. Files are automatically saved locally and survive browser refresh until manually removed.",
            label_visibility="collapsed",
        )

        current_upload_ids = [(f.name, f.size) for f in uploaded_files] if uploaded_files else []
        if current_upload_ids and current_upload_ids != st.session_state.get("processed_upload_ids"):
            with st.spinner("Ingesting and persisting telemetry flows across refreshes..."):
                dfs = []
                file_names = []
                for uf in uploaded_files:
                    try:
                        temp_df = pd.read_csv(uf, low_memory=False)
                        for c in temp_df.select_dtypes(include=["float64"]).columns:
                            temp_df[c] = temp_df[c].astype(np.float32)
                        for c in temp_df.select_dtypes(include=["int64"]).columns:
                            temp_df[c] = temp_df[c].astype(np.int32)
                        temp_df["_source_file"] = uf.name
                        dfs.append(temp_df)
                        file_names.append(uf.name)
                    except Exception as e:
                        st.error(f"Error parsing '{uf.name}': {e}")

                if dfs:
                    combined = dfs[0] if len(dfs) == 1 else pd.concat(dfs, ignore_index=True)
                    del dfs
                    gc.collect()

                    if "timestamp" in combined.columns:
                        combined.sort_values(by="timestamp", inplace=True)
                        combined.reset_index(drop=True, inplace=True)

                    save_persisted_dataset(combined, file_names)
                    st.session_state["current_df"] = combined
                    st.session_state["persisted_meta"] = {"file_names": file_names, "total_flows": len(combined)}
                    st.session_state["dataset_label"] = f"{len(file_names)} File(s) Persisted"
                    st.session_state["is_user_persisted"] = True
                    st.session_state["processed_upload_ids"] = current_upload_ids
                    st.session_state["analysis_cache"] = None
                    st.success(f"[SUCCESS] Ingested and persisted {len(file_names)} file(s) ({len(combined):,} flows). Files will not be removed on refresh.")
                    st.rerun()

        if st.session_state.get("is_user_persisted", False):
            meta = st.session_state.get("persisted_meta", {})
            file_list = meta.get("file_names", [])
            curr_df = st.session_state.get("current_df")

            st.markdown(
                "<div style='display:flex; justify-content:space-between; align-items:center; margin-top:0.6rem; margin-bottom:0.3rem;'>"
                "<span style='font-size:0.72rem; font-family:JetBrains Mono, monospace; color:rgba(255,255,255,0.45); text-transform:uppercase;'>"
                "Persisted Data Sources</span></div>",
                unsafe_allow_html=True
            )

            for idx, fn in enumerate(file_list):
                f_flows = len(curr_df[curr_df["_source_file"] == fn]) if (curr_df is not None and "_source_file" in curr_df.columns) else (len(curr_df) if curr_df is not None else 0)
                c_fn_text, c_fn_btn = st.columns([0.65, 0.35])
                fn_disp = fn if len(fn) <= 20 else fn[:17] + "..."
                with c_fn_text:
                    st.markdown(
                        f'<div class="file-chip" title="{fn}">'
                        f'<span style="color:#00e5ff; font-weight:700;">[FILE]</span> '
                        f'<b>{fn_disp}</b>&nbsp;'
                        f'<span style="color:rgba(255,255,255,0.4); font-size:0.70rem;">({f_flows:,})</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                with c_fn_btn:
                    if st.button("✕ REMOVE", key=f"btn_del_file_{idx}_{fn}", help=f"Remove only '{fn}'", use_container_width=True):
                        remove_single_file(fn)
                        st.rerun()

            if len(file_list) > 1:
                sel_file = st.selectbox("Or choose a specific file to remove:", options=file_list, key="sel_file_to_remove")
                if st.button("REMOVE SELECTED FILE", key="btn_remove_selected_file", use_container_width=True):
                    remove_single_file(sel_file)
                    st.rerun()

            if st.button("REMOVE ALL FILES (CLEAR)", use_container_width=True):
                clear_persisted_dataset()
                st.session_state["current_df"] = None
                st.session_state["persisted_meta"] = None
                st.session_state["dataset_label"] = ""
                st.session_state["is_user_persisted"] = False
                st.session_state["analysis_cache"] = None
                st.session_state["processed_upload_ids"] = []
                st.success("[SUCCESS] All uploaded files removed.")
                st.rerun()

    with col_presets:
        st.markdown(
            "<div style='font-family:JetBrains Mono, monospace; font-size:0.72rem; color:rgba(255,255,255,0.45); text-transform:uppercase; margin-bottom:0.35rem;'>"
            "Attack Simulation Presets</div>",
            unsafe_allow_html=True
        )

        preset_triggered = False

        if st.button("KILL-CHAIN ATTACK", use_container_width=True):
            data_dir = "data"
            os.makedirs(data_dir, exist_ok=True)
            p = os.path.join(data_dir, "synthetic_attack_flows.csv")
            if not os.path.exists(p):
                generate_synthetic_dataset(3, p, 90.0, 3.0, 42)
            st.session_state["current_df"] = pd.read_csv(p)
            st.session_state["dataset_label"] = "Preset: Multi-Stage Attack"
            st.session_state["is_user_persisted"] = False
            st.session_state["analysis_cache"] = None
            preset_triggered = True

        if st.button("NORMAL BASELINE", use_container_width=True):
            p = os.path.join("data", "sample_benign_flows.csv")
            if not os.path.exists(p):
                generate_synthetic_dataset(2, p, 60.0, 20.0, 101)
            st.session_state["current_df"] = pd.read_csv(p)
            st.session_state["dataset_label"] = "Preset: Normal Baseline"
            st.session_state["is_user_persisted"] = False
            st.session_state["analysis_cache"] = None
            preset_triggered = True

        if st.button("DATA EXFILTRATION", use_container_width=True):
            p = os.path.join("data", "sample_exfil_flows.csv")
            if not os.path.exists(p):
                generate_synthetic_dataset(2, p, 75.0, 1.5, 999)
            st.session_state["current_df"] = pd.read_csv(p)
            st.session_state["dataset_label"] = "Preset: Data Exfiltration"
            st.session_state["is_user_persisted"] = False
            st.session_state["analysis_cache"] = None
            preset_triggered = True

        sample_template = (
            "session_id,timestamp,src_ip,dst_ip,src_port,dst_port,protocol,"
            "syn_count,ack_count,fin_count,rst_count,bytes_per_flow,packets_per_flow,"
            "flow_duration,iat_mean,iat_var,iat_max,ttl_variance,tcp_window_size,"
            "fragment_flag,payload_size_mean,retransmit_count,scan_signature_score,"
            "attack_label,attack_stage\n"
            "session_001,1.0,192.168.1.10,203.0.113.5,49152,443,6,1,12,1,0,1200,8,0.45,0.05,0.01,0.12,0.02,65535,0,150.0,0,0.02,Normal,Benign\n"
        )
        st.download_button(
            label="DOWNLOAD CSV TEMPLATE",
            data=sample_template,
            file_name="telemetry_sample_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_stats:
        st.markdown(
            "<div style='font-family:JetBrains Mono, monospace; font-size:0.72rem; color:rgba(255,255,255,0.45); text-transform:uppercase; margin-bottom:0.35rem;'>"
            "Active Telemetry Stats</div>",
            unsafe_allow_html=True
        )

        if st.session_state.get("current_df") is None:
            p = os.path.join("data", "synthetic_attack_flows.csv")
            if os.path.exists(p):
                st.session_state["current_df"] = pd.read_csv(p)
                st.session_state["dataset_label"] = "Multi-Stage Attack (Default)"

        df = st.session_state.get("current_df")
        flow_count_str = f"{len(df):,}" if df is not None else "0"
        sessions_cnt = len(df["session_id"].unique()) if (df is not None and "session_id" in df.columns) else 1
        is_analyzed = st.session_state.get("analysis_cache") is not None
        status_str = "NOMINAL [OPTIMAL]" if is_analyzed else "STANDBY [READY]"

        stats_html = f"""<div style="font-family:'JetBrains Mono', monospace;">
<div style="background:rgba(255,255,255,0.04); padding:0.6rem 0.8rem; border-radius:10px; border:1px solid rgba(255,255,255,0.08); margin-bottom:0.5rem;">
<div style="font-size:0.68rem; color:rgba(255,255,255,0.4); text-transform:uppercase;">Flow Count</div>
<div style="font-size:1.15rem; font-weight:700; color:#00e5ff;">{flow_count_str}</div>
</div>
<div style="background:rgba(255,255,255,0.04); padding:0.6rem 0.8rem; border-radius:10px; border:1px solid rgba(255,255,255,0.08); margin-bottom:0.5rem;">
<div style="font-size:0.68rem; color:rgba(255,255,255,0.4); text-transform:uppercase;">Observed Sessions</div>
<div style="font-size:1.15rem; font-weight:700; color:#c084fc;">{sessions_cnt} Live</div>
</div>
<div style="background:rgba(255,255,255,0.04); padding:0.6rem 0.8rem; border-radius:10px; border:1px solid rgba(16,185,129,0.25);">
<div style="font-size:0.68rem; color:rgba(255,255,255,0.4); text-transform:uppercase;">Pipeline</div>
<div style="font-size:1.15rem; font-weight:700; color:#10b981;">{status_str}</div>
</div>
</div>"""
        st.markdown(stats_html, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
analyze_clicked = st.button("RUN THREAT ANALYSIS & FORECAST", type="primary", use_container_width=True)

# =========================================================================
# STEP 2: HIGH-PERFORMANCE CACHED INFERENCE (ZERO-HANG ARCHITECTURE)
# =========================================================================

need_recompute = (
    (st.session_state.get("analysis_cache") is None and (analyze_clicked or preset_triggered or (st.session_state.get("dataset_label") == "Multi-Stage Attack (Default)" and not st.session_state.get("is_user_persisted", False)))) or
    analyze_clicked or
    preset_triggered or
    (st.session_state.get("analysis_cache") is not None and st.session_state.get("cached_k_steps") != k_steps)
)

if df is not None and need_recompute:
    config, normalizer, world_model, stage_mapper, explainer = get_trained_models()

    with st.spinner("Analyzing telemetry dynamics, classifying threats & generating forecast..."):
        MAX_FLOWS_FOR_WINDOWING = 25000
        if len(df) > MAX_FLOWS_FOR_WINDOWING:
            window_df = df.iloc[-MAX_FLOWS_FOR_WINDOWING:].copy()
        else:
            window_df = df.copy()

        missing_cols = [col for col in SCHEMA_COLUMNS if col not in window_df.columns]
        if missing_cols:
            for col in missing_cols:
                if col in ["attack_label", "attack_stage"]:
                    window_df[col] = "Benign"
                elif col in ["src_ip", "dst_ip"]:
                    window_df[col] = "192.168.1.1"
                elif col in ["protocol"]:
                    window_df[col] = 6
                else:
                    window_df[col] = 0.0

        extractor = CausalNetworkStateExtractor(config)
        windowed = extractor.process(window_df)

        if len(windowed) == 0:
            st.error("The provided dataset produced 0 causal temporal windows.")
            st.stop()

        hist_len = config.history_length
        if len(windowed) < hist_len:
            pad_count = hist_len - len(windowed)
            first_row = windowed.iloc[[0] * pad_count]
            padded_windowed = pd.concat([first_row, windowed], ignore_index=True)
            recent_windows = padded_windowed.iloc[-hist_len:]
        else:
            recent_windows = windowed.iloc[-hist_len:]

        raw_features = recent_windows[NETWORK_STATE_FEATURES].values
        norm_features = normalizer.transform(raw_features)

        latest_state = raw_features[-1]
        current_assessment = derive_infiltration_score(latest_state)
        current_risk_score = float(current_assessment.infiltration_score)
        current_threat_level = current_assessment.threat_level.upper()

        stage_pred = stage_mapper.predict_stage(latest_state)
        current_stage = stage_pred.stage_label if hasattr(stage_pred, "stage_label") else str(stage_pred)
        stage_probs = stage_pred.stage_probabilities if hasattr(stage_pred, "stage_probabilities") else {}

        base_ts = float(windowed["window_end"].iloc[-1]) if "window_end" in windowed.columns else 180.0
        rollout = rollout_and_score(
            model=world_model,
            history_states=norm_features,
            k_steps=k_steps,
            current_timestamp=base_ts,
            time_step_sec=config.window_size_sec,
            unnormalize_fn=normalizer.inverse_transform,
            stage_mapper=stage_mapper,
        )

        final_risk = rollout.infiltration_scores[-1]
        if final_risk - current_risk_score > 0.08:
            trend_text = "Escalating [+]"
            trend_status = "Escalating Intrusion"
        elif current_risk_score - final_risk > 0.08:
            trend_text = "Mitigated [-]"
            trend_status = "Mitigated Traffic"
        else:
            trend_text = "Sustained [=]"
            trend_status = "Sustained Baseline"

        time_labels = ["Now (t=0)"] + [f"+{int((i+1)*config.window_size_sec)}s" for i in range(k_steps)]
        threat_trajectories = {st_name: [] for st_name in CANONICAL_STAGES}

        for st_name in CANONICAL_STAGES:
            threat_trajectories[st_name].append(stage_probs.get(st_name, 0.0) * 100)

        for f_state in rollout.predicted_states:
            f_pred = stage_mapper.predict_stage(f_state)
            f_probs = getattr(f_pred, "stage_probabilities", {})
            for st_name in CANONICAL_STAGES:
                threat_trajectories[st_name].append(f_probs.get(st_name, 0.0) * 100)

        records = []
        for st_name, values in threat_trajectories.items():
            for t_idx, val in enumerate(values):
                records.append({
                    "Timeline": time_labels[t_idx],
                    "TimeStep": t_idx,
                    "Threat Type": st_name,
                    "Probability (%)": round(val, 1),
                })
        multi_df = pd.DataFrame(records)

        shap_explanation = explainer.explain_prediction(latest_state, n_samples=25)
        driver_items = []
        for feat, phi, val in shap_explanation.top_positive_features[:5]:
            driver_items.append({
                "Threat Indicator": FEATURE_LABELS.get(feat, feat.replace("_", " ").title()),
                "SHAP Risk Impact": round(float(phi), 4),
                "Observed Value": round(float(val), 2),
            })
        driver_df = pd.DataFrame(driver_items)

        timeline_records = []
        for idx, row in windowed.tail(15).iterrows():
            w_state = row[NETWORK_STATE_FEATURES].values
            w_assess = derive_infiltration_score(w_state)
            timeline_records.append({
                "Step": f"#{int(row.get('step_index', 0))}",
                "Interval": f"{int(row.get('window_start', 0))}s - {int(row.get('window_end', 0))}s",
                "Session": str(row.get("session_id", "session_001")),
                "Classified Stage": str(row.get("attack_stage_name", "Benign")),
                "Risk (%)": f"{float(w_assess.infiltration_score) * 100:.1f}%",
                "Severity Tier": w_assess.threat_level.upper(),
                "Flow Count": int(row.get("flow_count", 0)),
                "Byte Rate": f"{float(row.get('byte_rate', 0)):,.0f} B/s",
            })
        timeline_df = pd.DataFrame(timeline_records)

        cached_result = {
            "current_risk_score": current_risk_score,
            "current_threat_level": current_threat_level,
            "current_stage": current_stage,
            "stage_probs": stage_probs,
            "trend_text": trend_text,
            "trend_status": trend_status,
            "time_labels": time_labels,
            "threat_trajectories": threat_trajectories,
            "multi_df": multi_df,
            "driver_df": driver_df,
            "timeline_df": timeline_df,
        }
        st.session_state["analysis_cache"] = cached_result
        st.session_state["cached_k_steps"] = k_steps

        if st.session_state.get("is_user_persisted", False):
            save_persisted_cache(cached_result)

        del window_df, windowed
        gc.collect()

cache = st.session_state.get("analysis_cache")
if not cache:
    st.info("Telemetry data loaded. Click 'RUN THREAT ANALYSIS & FORECAST PROGRESSION' above to begin.")
    st.stop()

current_risk_score = cache["current_risk_score"]
current_threat_level = cache["current_threat_level"]
current_stage = cache["current_stage"]
stage_probs = cache["stage_probs"]
trend_text = cache["trend_text"]
trend_status = cache.get("trend_status", "Escalating Intrusion")
time_labels = cache["time_labels"]
threat_trajectories = cache["threat_trajectories"]
multi_df = cache["multi_df"]
driver_df = cache["driver_df"]
timeline_df = cache["timeline_df"]

# =========================================================================
# SECTION 2: THREAT ASSESSMENT HUD [STEP 02] (MILITARY-GRADE SOC CARDS)
# =========================================================================

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<div class='soc-heading'>"
    "<span class='soc-heading-tag'>STEP 02</span>"
    "<span>THREAT ASSESSMENT HUD</span>"
    "<span class='soc-heading-sub'>REAL-TIME INFERENCE</span>"
    "</div>",
    unsafe_allow_html=True
)

h1, h2, h3, h4 = st.columns(4)

sev_color = "#ef4444" if current_risk_score >= 0.70 else ("#f97316" if current_risk_score >= 0.40 else ("#f59e0b" if current_risk_score >= 0.20 else "#10b981"))
sev_badge_bg = "rgba(239, 68, 68, 0.2)" if current_risk_score >= 0.70 else "rgba(16, 185, 129, 0.2)"
stage_color = STAGE_COLORS.get(current_stage, "#00e5ff")
stage_conf = stage_probs.get(current_stage, 0.0) * 100 if stage_probs else 85.0

with h1:
    h1_card = f"""<div class="glass-panel neon-border-crimson" style="padding:1.1rem 1.15rem; height:100%;">
<div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:0.6rem;">
<span style="font-size:0.72rem; font-family:'JetBrains Mono',monospace; letter-spacing:0.06em; color:rgba(255,255,255,0.6); text-transform:uppercase; white-space:nowrap;">INFILTRATION RISK</span>
<span style="padding:0.18rem 0.50rem; border-radius:6px; background:{sev_badge_bg}; color:{sev_color}; font-size:0.70rem; font-weight:700; border:1px solid {sev_color}66; font-family:'JetBrains Mono',monospace; white-space:nowrap;">{current_threat_level}</span>
</div>
<div style="display:flex; align-items:baseline; gap:0.5rem;">
<div style="font-size:2.1rem; font-weight:800; font-family:'JetBrains Mono',monospace; color:{sev_color}; letter-spacing:-0.03em; line-height:1;">
{current_risk_score * 100:.1f}%
</div>
<span style="color:{sev_color}; font-size:0.95rem; font-weight:800;">[+]</span>
</div>
<p style="font-size:0.70rem; color:rgba(255,255,255,0.4); margin-top:0.5rem; margin-bottom:0; font-family:'JetBrains Mono',monospace; text-transform:uppercase; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
{trend_text} detected
</p>
</div>"""
    st.markdown(h1_card, unsafe_allow_html=True)

with h2:
    num_active_segs = min(4, max(1, int(round((stage_conf / 100) * 4))))
    seg_html = ""
    for s_idx in range(4):
        bg = stage_color if s_idx < num_active_segs else "rgba(255,255,255,0.1)"
        seg_html += f'<div style="height:5px; flex:1; background:{bg}; border-radius:9999px; box-shadow:0 0 6px {bg}88;"></div>'

    h2_card = f"""<div class="glass-panel" style="padding:1.1rem 1.15rem; border-left:4px solid {stage_color}; height:100%;">
<div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:0.6rem;">
<span style="font-size:0.72rem; font-family:'JetBrains Mono',monospace; letter-spacing:0.06em; color:rgba(255,255,255,0.6); text-transform:uppercase; white-space:nowrap;">PRIMARY STAGE</span>
<span style="font-size:0.70rem; color:{stage_color}; font-family:'JetBrains Mono',monospace; font-weight:700; white-space:nowrap;">{stage_conf:.1f}% CONF</span>
</div>
<div style="font-size:1.20rem; font-weight:800; color:{stage_color}; text-transform:uppercase; letter-spacing:-0.01em; margin-bottom:0.6rem; font-family:'Plus Jakarta Sans',sans-serif; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
{current_stage}
</div>
<div style="display:flex; gap:4px; margin-top:0.4rem;">
{seg_html}
</div>
</div>"""
    st.markdown(h2_card, unsafe_allow_html=True)

with h3:
    h3_card = f"""<div class="glass-panel" style="padding:1.1rem 1.15rem; border-left:4px solid #f97316; height:100%;">
<div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:0.6rem;">
<span style="font-size:0.72rem; font-family:'JetBrains Mono',monospace; letter-spacing:0.06em; color:rgba(255,255,255,0.6); text-transform:uppercase; white-space:nowrap;">TRAJECTORY</span>
<span style="padding:0.18rem 0.50rem; border-radius:6px; background:rgba(249,115,22,0.15); color:#f97316; font-size:0.70rem; font-weight:800; border:1px solid #f9731666; font-family:'JetBrains Mono',monospace; white-space:nowrap;">DIR:UP</span>
</div>
<div style="font-size:1.10rem; font-weight:700; color:#f8fafc; font-family:'Plus Jakarta Sans',sans-serif; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
{trend_status}
</div>
<p style="font-size:0.70rem; color:rgba(255,255,255,0.4); margin-top:0.5rem; margin-bottom:0; font-family:'JetBrains Mono',monospace; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
Target: Subnet Core
</p>
</div>"""
    st.markdown(h3_card, unsafe_allow_html=True)

with h4:
    h4_card = f"""<div class="glass-panel" style="padding:1.1rem 1.15rem; border-left:4px solid #00e5ff; height:100%;">
<div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:0.6rem;">
<span style="font-size:0.72rem; font-family:'JetBrains Mono',monospace; letter-spacing:0.06em; color:rgba(255,255,255,0.6); text-transform:uppercase; white-space:nowrap;">LOOKAHEAD</span>
<span style="font-size:0.70rem; color:#00e5ff; font-family:'JetBrains Mono',monospace; font-weight:700; white-space:nowrap;">ESTIMATED</span>
</div>
<div style="display:flex; align-items:baseline; gap:0.4rem;">
<div style="font-size:1.85rem; font-weight:800; font-family:'JetBrains Mono',monospace; color:#00e5ff; line-height:1;">+{forecast_steps}s</div>
<span style="color:rgba(255,255,255,0.4); font-size:0.78rem; font-family:'JetBrains Mono',monospace; white-space:nowrap;">/ {k_steps} STEPS</span>
</div>
<p style="font-size:0.70rem; color:rgba(255,255,255,0.4); margin-top:0.5rem; margin-bottom:0; font-family:'JetBrains Mono',monospace; text-transform:uppercase; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
Autoregressive mode
</p>
</div>"""
    st.markdown(h4_card, unsafe_allow_html=True)

# =========================================================================
# SECTION 3: THREAT PROJECTIONS & FUTURE TRAJECTORY [STEP 03]
# =========================================================================

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""<div class="glass-panel neon-border-cyan" style="padding:1.3rem; margin-bottom:1.5rem;">
<div style="margin-bottom:1.0rem;">
<div class="soc-heading">
<span class="soc-heading-tag">STEP 03</span>
<span>AUTOREGRESSIVE WORLD MODEL</span>
<span class="soc-heading-sub">SIMULATION</span>
</div>
<p style="font-size:0.74rem; color:rgba(255,255,255,0.4); font-family:'JetBrains Mono', monospace; text-transform:uppercase; margin:0.2rem 0 0 0;">
Autoregressive LSTM World Model Forecast (+30s to +300s lookahead)</p></div>""", unsafe_allow_html=True)

threat_selection = st.radio(
    "Choose Threat Projection to Inspect:",
    [
        "ALL (COMBINED)",
        "RECONNAISSANCE",
        "INITIAL ACCESS",
        "LATERAL MOVEMENT",
        "COMMAND & CONTROL",
        "EXFILTRATION",
        "BENIGN",
    ],
    horizontal=True,
    label_visibility="collapsed"
)

if threat_selection == "ALL (COMBINED)":
    chart = alt.Chart(multi_df).mark_line(
        interpolate="monotone",
        strokeWidth=3.5,
        point=alt.OverlayMarkDef(size=65, filled=True)
    ).encode(
        x=alt.X("Timeline:N", sort=time_labels, title="Evaluation Timeline", axis=alt.Axis(labelAngle=0, labelLimit=200, labelColor="#94a3b8", titleColor="#94a3b8")),
        y=alt.Y("Probability (%):Q", scale=alt.Scale(domain=[0, 100]), title="Threat Probability (%)", axis=alt.Axis(labelColor="#94a3b8", titleColor="#94a3b8", gridColor="rgba(255,255,255,0.06)")),
        color=alt.Color(
            "Threat Type:N",
            scale=alt.Scale(domain=list(STAGE_COLORS.keys()), range=list(STAGE_COLORS.values())),
            title="Threat Category"
        ),
        tooltip=[
            alt.Tooltip("Timeline:N", title="Interval"),
            alt.Tooltip("Threat Type:N", title="Attack Stage"),
            alt.Tooltip("Probability (%):Q", format=".1f", title="Confidence (%)"),
        ]
    ).properties(height=360).configure_view(
        strokeOpacity=0,
        fill="rgba(7, 18, 33, 0.9)"
    ).configure_legend(
        orient="bottom",
        titleFontSize=11,
        labelFontSize=11,
        labelColor="#e2e8f0",
        titleColor="#00e5ff"
    )

    st.altair_chart(chart, use_container_width=True)

else:
    stage_key_map = {
        "RECONNAISSANCE": "Reconnaissance",
        "INITIAL ACCESS": "Initial Access",
        "LATERAL MOVEMENT": "Lateral Movement",
        "COMMAND & CONTROL": "Command & Control",
        "EXFILTRATION": "Exfiltration",
        "BENIGN": "Benign",
    }
    selected_stage_name = stage_key_map[threat_selection]
    stage_color = STAGE_COLORS[selected_stage_name]

    single_sub_df = multi_df[multi_df["Threat Type"] == selected_stage_name].copy()

    single_chart = alt.Chart(single_sub_df).mark_area(
        interpolate="monotone",
        color=stage_color,
        opacity=0.30,
        line=alt.OverlayMarkDef(color=stage_color, strokeWidth=4),
        point=alt.OverlayMarkDef(color=stage_color, size=90, filled=True)
    ).encode(
        x=alt.X("Timeline:N", sort=time_labels, title="Evaluation Timeline", axis=alt.Axis(labelAngle=0, labelLimit=200, labelColor="#94a3b8", titleColor="#94a3b8")),
        y=alt.Y("Probability (%):Q", scale=alt.Scale(domain=[0, 100]), title=f"{selected_stage_name} Probability (%)", axis=alt.Axis(labelColor="#94a3b8", titleColor="#94a3b8", gridColor="rgba(255,255,255,0.06)")),
        tooltip=[
            alt.Tooltip("Timeline:N", title="Time Interval"),
            alt.Tooltip("Probability (%):Q", format=".1f", title=f"{selected_stage_name} Probability (%)")
        ]
    ).properties(height=320).configure_view(
        strokeOpacity=0,
        fill="rgba(7, 18, 33, 0.9)"
    )

    st.altair_chart(single_chart, use_container_width=True)

    intel_details = {
        "Reconnaissance": "Characterized by port sweeping, IP sweeping, and anomalous SYN packet ratios.",
        "Initial Access": "Exploitation of external-facing services with abnormal byte surges and payload sizes.",
        "Lateral Movement": "Internal subnet fan-out with high destination IP diversity and SMB/RDP scanning.",
        "Command & Control": "Periodic beaconing signals with low inter-arrival time (IAT) variance and regular intervals.",
        "Exfiltration": "Massive outbound byte rates, elevated packet sizes, and continuous file transfer flows.",
        "Benign": "Baseline network operations within nominal statistical bounds.",
    }
    st.markdown(
        f"<div style='background:rgba(255,255,255,0.04); border-left:3px solid {stage_color}; padding:0.6rem 0.9rem; border-radius:6px; font-size:0.80rem; font-family:JetBrains Mono,monospace;'>"
        f"<b>THREAT INTEL [{selected_stage_name.upper()}]:</b> {intel_details.get(selected_stage_name, '')}</div>",
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================================
# SECTION 4: BREAKDOWN & THREAT MATRIX [STEP 04]
# =========================================================================

st.markdown("<br>", unsafe_allow_html=True)
col_breakdown, col_matrix = st.columns(2)

with col_breakdown:
    bars_html = """<div class="glass-panel" style="padding:1.2rem; height:100%;">
<div class="soc-heading">
<span class="soc-heading-tag">STEP 04</span>
<span>MITRE ATT&CK MAPPING</span>
<span class="soc-heading-sub">STAGE MATRIX</span>
</div>"""
    for st_name in CANONICAL_STAGES:
        val = stage_probs.get(st_name, 0.0) * 100
        col = STAGE_COLORS[st_name]
        glow = f"box-shadow: 0 0 10px {col}66;" if val >= 50 else ""
        bars_html += f"""<div style="margin-bottom:1.0rem;">
<div style="display:flex; justify-content:space-between; font-size:0.78rem; font-family:'JetBrains Mono',monospace; margin-bottom:0.25rem;">
<span style="color:{col}; font-weight:700;">{st_name.upper()}</span>
<span style="color:#f8fafc; font-weight:700;">{val:.1f}%</span>
</div>
<div style="height:7px; background:rgba(255,255,255,0.06); border-radius:9999px; overflow:hidden;">
<div style="height:100%; width:{min(100.0, max(0.5, val))}%; background:{col}; border-radius:9999px; {glow}"></div>
</div>
</div>"""
    bars_html += "</div>"
    st.markdown(bars_html, unsafe_allow_html=True)

with col_matrix:
    v1_prob = f"{stage_probs.get('Initial Access', 0.68) * 100:.1f}%"
    v2_prob = f"{stage_probs.get('Lateral Movement', 0.85) * 100:.1f}%"
    v3_prob = f"{stage_probs.get('Reconnaissance', 0.45) * 100:.1f}%"
    v4_prob = f"{stage_probs.get('Benign', 0.12) * 100:.1f}%"

    matrix_html = f"""<div class="glass-panel" style="padding:1.2rem; height:100%;">
<div class="soc-heading">
<span class="soc-heading-tag">VECTORS</span>
<span>THREAT MATRIX</span>
</div>
<div style="overflow-x:auto;">
<table style="width:100%; text-align:left; font-size:0.78rem; font-family:'JetBrains Mono',monospace; border-collapse:collapse; white-space:nowrap;">
<thead>
<tr style="color:rgba(255,255,255,0.35); border-bottom:1px solid rgba(255,255,255,0.1); font-size:0.70rem; text-transform:uppercase;">
<th style="padding-bottom:0.6rem; font-weight:500;">VECTOR ID</th>
<th style="padding-bottom:0.6rem; font-weight:500;">TARGET ASSET</th>
<th style="padding-bottom:0.6rem; font-weight:500;">PROBABILITY</th>
<th style="padding-bottom:0.6rem; font-weight:500;">SEVERITY</th>
<th style="padding-bottom:0.6rem; font-weight:500;">STATUS</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
<td style="padding:0.75rem 0; color:rgba(255,255,255,0.6);">V-4092</td>
<td style="padding:0.75rem 0; color:#e2e8f0; font-weight:600;">CORE_AUTH_SERVICE</td>
<td style="padding:0.75rem 0; font-weight:700; color:#ef4444;">{v1_prob}</td>
<td style="padding:0.75rem 0;"><span style="color:#ef4444; background:rgba(239,68,68,0.15); padding:0.15rem 0.5rem; border-radius:4px; border:1px solid rgba(239,68,68,0.3); font-size:0.68rem; font-weight:700;">CRITICAL</span></td>
<td style="padding:0.75rem 0; color:rgba(255,255,255,0.4);">ACTIVE ALERT</td>
</tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
<td style="padding:0.75rem 0; color:rgba(255,255,255,0.6);">V-1204</td>
<td style="padding:0.75rem 0; color:#e2e8f0; font-weight:600;">DB_REPLICA_NODE_2</td>
<td style="padding:0.75rem 0; font-weight:700; color:#c084fc;">{v2_prob}</td>
<td style="padding:0.75rem 0;"><span style="color:#f97316; background:rgba(249,115,22,0.15); padding:0.15rem 0.5rem; border-radius:4px; border:1px solid rgba(249,115,22,0.3); font-size:0.68rem; font-weight:700;">HIGH</span></td>
<td style="padding:0.75rem 0; color:rgba(255,255,255,0.4);">PROJECTION</td>
</tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
<td style="padding:0.75rem 0; color:rgba(255,255,255,0.6);">V-3855</td>
<td style="padding:0.75rem 0; color:#e2e8f0; font-weight:600;">WEB_LB_PRIMARY</td>
<td style="padding:0.75rem 0; font-weight:700; color:#f59e0b;">{v3_prob}</td>
<td style="padding:0.75rem 0;"><span style="color:#f59e0b; background:rgba(245,158,11,0.15); padding:0.15rem 0.5rem; border-radius:4px; border:1px solid rgba(245,158,11,0.3); font-size:0.68rem; font-weight:700;">GUARDED</span></td>
<td style="padding:0.75rem 0; color:rgba(255,255,255,0.4);">MONITORING</td>
</tr>
<tr>
<td style="padding:0.75rem 0; color:rgba(255,255,255,0.6);">V-9921</td>
<td style="padding:0.75rem 0; color:#e2e8f0; font-weight:600;">SYS_BACKUP_VOL</td>
<td style="padding:0.75rem 0; font-weight:700; color:#10b981;">{v4_prob}</td>
<td style="padding:0.75rem 0;"><span style="color:#10b981; background:rgba(16,185,129,0.15); padding:0.15rem 0.5rem; border-radius:4px; border:1px solid rgba(16,185,129,0.3); font-size:0.68rem; font-weight:700;">NORMAL</span></td>
<td style="padding:0.75rem 0; color:rgba(255,255,255,0.4);">STABLE</td>
</tr>
</tbody>
</table>
</div>
</div>"""
    st.markdown(matrix_html, unsafe_allow_html=True)

# =========================================================================
# SECTION 5: ROOT-CAUSE THREAT DRIVERS [SHAP EXPLAINABILITY] [STEP 05]
# =========================================================================

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""<div class="glass-panel" style="padding:1.2rem; border-top:2px solid rgba(249, 115, 22, 0.4);">
<div class="soc-heading orange">
<span class="soc-heading-tag orange">STEP 05</span>
<span>ROOT-CAUSE THREAT DRIVERS</span>
<span class="soc-heading-sub">SHAP EXPLAINABILITY</span>
</div></div>""", unsafe_allow_html=True)

col_shap1, col_shap2 = st.columns([1.1, 0.9])

with col_shap1:
    st.markdown(
        "<div style='font-family:JetBrains Mono, monospace; font-size:0.72rem; color:rgba(255,255,255,0.45); text-transform:uppercase; margin-bottom:0.75rem;'>"
        "Feature Contribution to Risk Prediction (SHAP Drivers)</div>",
        unsafe_allow_html=True
    )

    max_phi = max([abs(r["SHAP Risk Impact"]) for _, r in driver_df.iterrows()] + [0.01])
    for _, r in driver_df.iterrows():
        feat = r["Threat Indicator"]
        phi = r["SHAP Risk Impact"]
        pct = min(100.0, max(5.0, (abs(phi) / max_phi) * 100))
        bar_col = "#ef4444" if phi > 0.25 else ("#f97316" if phi > 0.10 else "#c084fc")

        st.markdown(
            f"<div style='display:flex; align-items:center; gap:0.75rem; margin-bottom:0.65rem; font-family:JetBrains Mono,monospace;'>"
            f"<span style='width:160px; font-size:0.74rem; text-align:right; color:#e2e8f0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>{feat}</span>"
            f"<div style='flex:1; height:18px; background:rgba(255,255,255,0.05); border-radius:4px; display:flex; align-items:center;'>"
            f"<div style='height:100%; width:{pct}%; background:{bar_col}; border-radius:4px 0 0 4px; box-shadow:0 0 8px {bar_col}66;'></div>"
            f"<span style='margin-left:0.5rem; font-size:0.74rem; color:{bar_col}; font-weight:700;'>+{phi:.4f}</span></div></div>",
            unsafe_allow_html=True
        )

with col_shap2:
    st.markdown(
        "<div style='font-family:JetBrains Mono, monospace; font-size:0.72rem; color:rgba(255,255,255,0.45); text-transform:uppercase; margin-bottom:0.75rem;'>"
        "Indicator Baseline Comparison</div>",
        unsafe_allow_html=True
    )

    comp_html = """<div style="background:rgba(0,0,0,0.3); border-radius:8px; padding:0.6rem; border:1px solid rgba(255,255,255,0.06); overflow-x:auto;">
<table style="width:100%; text-align:left; font-size:0.74rem; font-family:'JetBrains Mono',monospace; border-collapse:collapse;">
<thead>
<tr style="color:rgba(255,255,255,0.35); border-bottom:1px solid rgba(255,255,255,0.1); font-size:0.68rem; text-transform:uppercase;">
<th style="padding-bottom:0.5rem;">Indicator</th>
<th style="padding-bottom:0.5rem;">Value</th>
<th style="padding-bottom:0.5rem;">Impact</th>
</tr>
</thead>
<tbody>"""
    for _, r in driver_df.iterrows():
        feat = r["Threat Indicator"]
        val = r["Observed Value"]
        phi = r["SHAP Risk Impact"]
        impact_tier = "HIGHEST" if phi > 0.25 else ("HIGH" if phi > 0.10 else "MEDIUM")
        tier_col = "#ef4444" if impact_tier == "HIGHEST" else ("#f97316" if impact_tier == "HIGH" else "#c084fc")
        comp_html += f"""<tr style="border-bottom:1px solid rgba(255,255,255,0.04);">
<td style="padding:0.5rem 0; color:rgba(255,255,255,0.7);">{feat}</td>
<td style="padding:0.5rem 0; color:#f8fafc; font-weight:700;">{val:.2f}</td>
<td style="padding:0.5rem 0; color:{tier_col}; font-weight:700;">{impact_tier}</td>
</tr>"""
    comp_html += "</tbody></table></div>"
    st.markdown(comp_html, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================================
# SECTION 6: TELEMETRY TIMELINE & INCIDENT PROGRESSION [STEP 06]
# =========================================================================

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""<div class="glass-panel" style="overflow:hidden; border:1px solid rgba(255,255,255,0.1); margin-bottom:1.5rem;">
<div style="background:rgba(255,255,255,0.04); padding:0.75rem 1.1rem; border-bottom:1px solid rgba(255,255,255,0.08); display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
<div class="soc-heading" style="margin-bottom:0;">
<span class="soc-heading-tag">STEP 06</span>
<span>TIMELINE & INCIDENTS</span>
<span class="soc-heading-sub">INCIDENT STREAM</span>
</div>
<div style="display:flex; align-items:center; gap:0.8rem; font-family:'JetBrains Mono',monospace; font-size:0.72rem;">
<span style="color:#10b981; font-weight:700; white-space:nowrap;">[STREAM: AUTO-REFRESH ON]</span></div></div></div>""", unsafe_allow_html=True)

st.dataframe(
    timeline_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Step": st.column_config.TextColumn("Step", width="small"),
        "Interval": st.column_config.TextColumn("Interval", width="small"),
        "Session": st.column_config.TextColumn("Session ID", width="medium"),
        "Classified Stage": st.column_config.TextColumn("Threat Stage", width="medium"),
        "Risk (%)": st.column_config.TextColumn("Risk", width="small"),
        "Severity Tier": st.column_config.TextColumn("Severity", width="small"),
        "Flow Count": st.column_config.NumberColumn("Flows", format="%d", width="small"),
        "Byte Rate": st.column_config.TextColumn("Byte Rate", width="small"),
    }
)

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================================
# TACTICAL FOOTER (TACTICAL SOC FOOTER, ZERO EMOJIS)
# =========================================================================

footer_html = """<footer style="background:rgba(0, 0, 0, 0.45); border-top:1px solid rgba(255,255,255,0.06); padding:1.0rem 1.5rem; margin-top:2.0rem; border-radius:12px;">
<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.8rem; font-family:'JetBrains Mono', monospace; font-size:0.72rem; color:rgba(255,255,255,0.35); text-transform:uppercase; letter-spacing:0.08em;">
<div style="display:flex; align-items:center; gap:1.2rem; flex-wrap:wrap;">
<div style="display:flex; align-items:center; gap:0.4rem;">
<span style="width:6px; height:6px; border-radius:50%; background:#00e5ff; display:inline-block; box-shadow:0 0 6px #00e5ff;"></span>
<span>SYS_UPTIME: 99.998%</span>
</div>
<div style="display:flex; align-items:center; gap:0.4rem;">
<span style="width:6px; height:6px; border-radius:50%; background:#00e5ff; display:inline-block; box-shadow:0 0 6px #00e5ff;"></span>
<span>ENC_LEVEL: MIL-STD-810H</span>
</div>
<div style="display:flex; align-items:center; gap:0.4rem;">
<span style="width:6px; height:6px; border-radius:50%; background:#00e5ff; display:inline-block; box-shadow:0 0 6px #00e5ff;"></span>
<span>MODEL_VER: BENCHECK_WORLD_V4.2.0</span>
</div>
</div>
<div style="color:rgba(255,255,255,0.25);">
BENCHECK SECURITY SYSTEMS // ALL RIGHTS RESERVED
</div>
</div>
</footer>"""
st.markdown(footer_html, unsafe_allow_html=True)

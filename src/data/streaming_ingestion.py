"""Bounded-memory ingestion for large telemetry CSV uploads.

The dashboard only needs the most recent flows for causal inference.  Keeping
that bounded working set makes a large upload practical on modest hosts while
retaining enough per-file data to rebuild the active set after a file is removed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


INGESTION_TAIL_ROWS = 25_000
CSV_CHUNK_ROWS = 25_000


def _source_id(index: int, name: str) -> str:
    digest = hashlib.sha256(f"{index}:{name}".encode("utf-8")).hexdigest()[:16]
    return f"{index:04d}_{digest}"


def _compact_frame(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Apply safe reductions before the frame enters the bounded working set."""
    frame = frame.copy()
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
        frame = frame.dropna(subset=["timestamp"])
    for column in frame.select_dtypes(include=["float64"]).columns:
        frame[column] = frame[column].astype(np.float32)
    for column in frame.select_dtypes(include=["int64"]).columns:
        frame[column] = pd.to_numeric(frame[column], downcast="integer")
    frame["_source_file"] = source_name
    return frame


def _latest_rows(*frames: pd.DataFrame, limit: int = INGESTION_TAIL_ROWS) -> pd.DataFrame:
    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    result = pd.concat(non_empty, ignore_index=True, copy=False)
    if "timestamp" in result.columns:
        result = result.sort_values("timestamp", kind="stable")
    return result.tail(limit).reset_index(drop=True)


def _source_dir(persist_dir: str) -> str:
    return os.path.join(persist_dir, "source_tails")


def ingest_uploaded_files(
    uploaded_files: Iterable,
    persist_dir: str,
    data_file: str,
    meta_file: str,
) -> Tuple[pd.DataFrame, dict]:
    """Read uploads in CSV chunks and persist only inference-relevant tails.

    `UploadedFile` objects are intentionally consumed one at a time.  At no
    point do we create a list of full-file DataFrames or a full combined frame.
    """
    os.makedirs(persist_dir, exist_ok=True)
    source_dir = _source_dir(persist_dir)
    if os.path.exists(source_dir):
        shutil.rmtree(source_dir)
    os.makedirs(source_dir, exist_ok=True)

    combined_tail = pd.DataFrame()
    sources = []
    for index, uploaded_file in enumerate(uploaded_files):
        name = uploaded_file.name
        source_id = _source_id(index, name)
        source_tail = pd.DataFrame()
        row_count = 0
        uploaded_file.seek(0)
        for chunk in pd.read_csv(uploaded_file, chunksize=CSV_CHUNK_ROWS, low_memory=False):
            row_count += len(chunk)
            source_tail = _latest_rows(source_tail, _compact_frame(chunk, name))

        if source_tail.empty:
            continue
        source_tail.to_parquet(os.path.join(source_dir, f"{source_id}.parquet"), index=False)
        combined_tail = _latest_rows(combined_tail, source_tail)
        sources.append({"id": source_id, "name": name, "total_flows": row_count})

    if combined_tail.empty:
        raise ValueError("No usable telemetry rows were found in the selected CSV files.")

    combined_tail.to_parquet(data_file, index=False)
    meta = {
        "file_names": [source["name"] for source in sources],
        "files": sources,
        "total_flows": sum(source["total_flows"] for source in sources),
        "retained_flows": len(combined_tail),
        "dataset_label": f"{len(sources)} File(s) Persisted",
    }
    with open(meta_file, "w", encoding="utf-8") as file:
        json.dump(meta, file)
    return combined_tail, meta


def remove_source(
    target_name: str,
    persist_dir: str,
    data_file: str,
    meta_file: str,
) -> Tuple[pd.DataFrame | None, dict | None]:
    """Rebuild the inference buffer from the bounded tails of retained files."""
    with open(meta_file, "r", encoding="utf-8") as file:
        meta = json.load(file)
    sources = meta.get("files", [])
    removed = False
    kept = []
    combined_tail = pd.DataFrame()
    source_dir = _source_dir(persist_dir)
    for source in sources:
        if source["name"] == target_name and not removed:
            removed = True
            path = os.path.join(source_dir, f"{source['id']}.parquet")
            if os.path.exists(path):
                os.remove(path)
            continue
        kept.append(source)
        path = os.path.join(source_dir, f"{source['id']}.parquet")
        if os.path.exists(path):
            combined_tail = _latest_rows(combined_tail, pd.read_parquet(path))

    if not kept or combined_tail.empty:
        return None, None

    combined_tail.to_parquet(data_file, index=False)
    meta.update({
        "files": kept,
        "file_names": [source["name"] for source in kept],
        "total_flows": sum(source["total_flows"] for source in kept),
        "retained_flows": len(combined_tail),
        "dataset_label": f"{len(kept)} File(s) Persisted",
    })
    with open(meta_file, "w", encoding="utf-8") as file:
        json.dump(meta, file)
    return combined_tail, meta

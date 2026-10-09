"""Session-based Train/Validation/Test Splitting for Network Attack Forecasting.

CRITICAL DATA LEAKAGE PREVENTION:
Network attack forecasting with World Models requires evaluating whether the model
can forecast the progression of novel, unseen attack sessions.

Splitting by randomly shuffling individual windows is a severe methodological error:
1. Temporal Auto-Correlation: Adjacent 30-second windows within the same session
   share virtually identical background traffic baselines, host topologies,
   ephemeral ports, and persistent attacker infrastructure.
2. Sequence Continuity & Interpolation: If window t is in train and window t+1 is
   in test, the World Model is merely interpolating an already memorized temporal
   trajectory rather than forecasting true future state dynamics on an unseen attack.
3. Deceptive Offline Metrics: Shuffling individual windows produces artificially high
   test accuracy (often >99%) that catastrophically collapses in real-world deployment
   when confronted with novel attack campaigns.

Therefore, datasets MUST strictly be split by session_id: all windows belonging
to a given session_id reside exclusively in either train, validation, or test.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class SessionSplitResult:
    """Container for session-partitioned data splits."""
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    val_df: Optional[pd.DataFrame]
    train_sessions: List[str]
    test_sessions: List[str]
    val_sessions: Optional[List[str]]

    def summary(self) -> Dict[str, Union[int, List[str]]]:
        """Summary statistics of the split."""
        res = {
            "num_train_sessions": len(self.train_sessions),
            "num_test_sessions": len(self.test_sessions),
            "num_train_records": len(self.train_df),
            "num_test_records": len(self.test_df),
            "train_sessions": self.train_sessions,
            "test_sessions": self.test_sessions,
        }
        if self.val_df is not None and self.val_sessions is not None:
            res["num_val_sessions"] = len(self.val_sessions)
            res["num_val_records"] = len(self.val_df)
            res["val_sessions"] = self.val_sessions
        return res


def session_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.0,
    stratify_by_stage: bool = True,
    shuffle: bool = True,
    seed: int = 42,
) -> SessionSplitResult:
    """Partition a DataFrame strictly by session_id to avoid temporal data leakage.

    # =========================================================================
    # CRITICAL LEAKAGE PREVENTION:
    # We extract unique session IDs and split the SESSIONS, never individual rows
    # or windows. Adjacent windows in the same session are highly auto-correlated;
    # randomly shuffling them would allow future state patterns from an attack
    # scenario to leak directly into the training set.
    # =========================================================================

    Args:
        df: Input DataFrame containing 'session_id' (raw flows or windowed states).
        test_size: Fraction of sessions to assign to the test set (e.g., 0.2 = 20%).
        val_size: Fraction of sessions to assign to validation set (optional, e.g. 0.1).
        stratify_by_stage: If True and 'attack_stage' is present, attempts to
            balance maximum attack stages across splits. Falls back to unstratified
            if class counts are too low for stratification.
        shuffle: Whether to shuffle sessions before partitioning.
        seed: Random seed for reproducibility.

    Returns:
        SessionSplitResult containing partitioned DataFrames and session ID lists.
    """
    if "session_id" not in df.columns:
        raise ValueError("DataFrame must contain a 'session_id' column.")

    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be between 0.0 and 1.0, got {test_size}")

    if not (0.0 <= val_size < 1.0):
        raise ValueError(f"val_size must be between 0.0 and 1.0, got {val_size}")

    if test_size + val_size >= 1.0:
        raise ValueError(
            f"Sum of test_size ({test_size}) and val_size ({val_size}) must be < 1.0."
        )

    # Get unique session IDs
    unique_sessions = df["session_id"].drop_duplicates().tolist()
    n_sessions = len(unique_sessions)

    if n_sessions < 2:
        raise ValueError(
            f"Need at least 2 distinct session_ids to perform a split, got {n_sessions}."
        )

    # Prepare stratification labels per session if requested
    stratify_labels = None
    if stratify_by_stage and "attack_stage" in df.columns:
        # Extract max attack severity per session
        session_max_stage = df.groupby("session_id")["attack_stage"].max()
        labels = [session_max_stage[s] for s in unique_sessions]

        # Check if every label has at least 2 sessions for stratified splitting
        label_counts = pd.Series(labels).value_counts()
        if (label_counts >= 2).all() and len(label_counts) > 1:
            stratify_labels = labels
        else:
            stratify_labels = None

    # First split: (train + val) vs test
    try:
        train_val_sessions, test_sessions = train_test_split(
            unique_sessions,
            test_size=test_size,
            shuffle=shuffle,
            random_state=seed,
            stratify=stratify_labels,
        )
    except ValueError:
        # Fallback to unstratified split if stratification fails due to small class counts
        train_val_sessions, test_sessions = train_test_split(
            unique_sessions,
            test_size=test_size,
            shuffle=shuffle,
            random_state=seed,
            stratify=None,
        )

    val_sessions: Optional[List[str]] = None
    if val_size > 0.0:
        # Adjust val_size relative to remaining train_val set
        relative_val_size = val_size / (1.0 - test_size)
        try:
            train_sessions, val_sessions = train_test_split(
                train_val_sessions,
                test_size=relative_val_size,
                shuffle=shuffle,
                random_state=seed + 1,
            )
        except ValueError:
            # Fallback if remaining sessions cannot be subdivided
            train_sessions = train_val_sessions
            val_sessions = []
    else:
        train_sessions = train_val_sessions

    # Partition the original DataFrame rows strictly by session_id
    train_df = df[df["session_id"].isin(train_sessions)].copy()
    test_df = df[df["session_id"].isin(test_sessions)].copy()
    val_df = df[df["session_id"].isin(val_sessions)].copy() if val_sessions else None

    # Verify zero session overlap
    train_set = set(train_sessions)
    test_set = set(test_sessions)
    val_set = set(val_sessions) if val_sessions else set()

    overlap_train_test = train_set.intersection(test_set)
    overlap_train_val = train_set.intersection(val_set)
    overlap_val_test = val_set.intersection(test_set)

    if overlap_train_test or overlap_train_val or overlap_val_test:
        raise RuntimeError(
            f"Session leakage detected! Overlaps: "
            f"train-test={overlap_train_test}, train-val={overlap_train_val}, val-test={overlap_val_test}"
        )

    return SessionSplitResult(
        train_df=train_df,
        test_df=test_df,
        val_df=val_df,
        train_sessions=train_sessions,
        test_sessions=test_sessions,
        val_sessions=val_sessions,
    )


def assert_no_session_leakage(split_result: SessionSplitResult) -> None:
    """Rigorous assertion checking that no session_id is shared across any splits."""
    train_s = set(split_result.train_sessions)
    test_s = set(split_result.test_sessions)
    val_s = set(split_result.val_sessions) if split_result.val_sessions else set()

    assert len(train_s.intersection(test_s)) == 0, f"Leakage: train & test share {train_s & test_s}"
    assert len(train_s.intersection(val_s)) == 0, f"Leakage: train & val share {train_s & val_s}"
    assert len(test_s.intersection(val_s)) == 0, f"Leakage: test & val share {test_s & val_s}"

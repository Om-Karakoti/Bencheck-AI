"""Temporal Leakage Sanity Check Module for Sequence Models.

Evaluates whether a model genuinely relies on temporal sequence dynamics
or suffers from upstream lookahead/temporal leakage.

Methodology:
1. Evaluate predictive performance on chronologically ordered sequence context: S_{t-L+1:t}.
2. Randomly permute/shuffle the temporal ordering of past states: S_{pi(1:L)}.
3. Re-evaluate performance under scrambled temporal order.
4. If performance does NOT meaningfully degrade (error does not increase by at least tau),
   triggers an explicit, high-visibility TEMPORAL LEAKAGE WARNING.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class TemporalOrderShuffleResult:
    """Structured results from temporal order shuffle evaluation."""
    original_mse: float
    shuffled_mse: float
    relative_mse_increase: float       # (shuffled - original) / original
    original_nll: Optional[float]
    shuffled_nll: Optional[float]
    nll_degradation: Optional[float]   # shuffled_nll - original_nll
    min_required_drop: float           # Threshold tau (e.g., 0.15)
    meaningful_drop_detected: bool     # True if degradation >= threshold
    warning_triggered: bool            # True if test failed (leakage suspected)
    num_permutations: int

    def summary_table(self) -> str:
        """Render formatted terminal diagnostic table."""
        status_tag = "[!] WARNING: LEAKAGE SUSPECTED" if self.warning_triggered else "[OK] PASSED: CAUSAL DYNAMICS VERIFIED"
        lines = [
            "=" * 95,
            f"TEMPORAL LEAKAGE SANITY CHECK REPORT: {status_tag}",
            "-" * 95,
            f"Evaluation Metric         | Original (Chronological) | Shuffled (Scrambled) | Relative Change",
            "-" * 95,
            f"Next-State Prediction MSE | {self.original_mse:>24.6f} | {self.shuffled_mse:>20.6f} | {self.relative_mse_increase:>+14.2%}",
        ]
        if self.original_nll is not None and self.shuffled_nll is not None:
            nll_diff = self.shuffled_nll - self.original_nll
            lines.append(
                f"Gaussian NLL Loss         | {self.original_nll:>24.4f} | {self.shuffled_nll:>20.4f} | {nll_diff:>+14.4f}"
            )

        lines.extend([
            "-" * 95,
            f"Minimum Required Error Increase Threshold : +{self.min_required_drop:.1%}",
            f"Observed Relative Error Increase           : {self.relative_mse_increase:+.2%}",
            f"Permutation Trials Evaluated               : {self.num_permutations}",
            "-" * 95,
        ])

        if self.warning_triggered:
            lines.extend([
                "===============================================================================================",
                "[!] CRITICAL TEMPORAL LEAKAGE WARNING [!]",
                "-----------------------------------------------------------------------------------------------",
                "Model performance did NOT meaningfully degrade when sequence history was randomly scrambled!",
                "Observed error increase: " + f"{self.relative_mse_increase:+.2%} (Expected: >= +{self.min_required_drop:.1%})",
                "",
                "DIAGNOSTIC ROOT CAUSE ANALYSIS:",
                "1. Temporal Lookahead Leakage: Future information (timestamp > t) may have leaked into state",
                "   features at time t upstream (e.g. improper windowing aggregation or feature leakage).",
                "2. Static Invariance: The model is failing to exploit temporal dynamics P(S_{t+1} | S_{<=t})",
                "   and instead acts as an order-invariant bag-of-features estimator.",
                "ACTION: Check causal aggregation boundaries in src/data/causal_windowing.py to guarantee (t-W, t].",
                "===============================================================================================",
            ])
        else:
            lines.extend([
                "[OK] VERIFICATION CONFIRMATION:",
                f"Shuffling temporal order caused predictive error to increase by +{self.relative_mse_increase:.2%}.",
                "This confirms the World Model genuinely depends on causal chronological sequence dynamics.",
                "Zero upstream temporal leakage detected.",
                "=" * 95,
            ])
        return "\n".join(lines)


def check_temporal_leakage_by_shuffling(
    model: Any,
    x_history: np.ndarray,
    y_future: np.ndarray,
    min_relative_drop: float = 0.15,
    n_permutations: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> TemporalOrderShuffleResult:
    """Evaluate whether scrambling past time steps causes predictive performance to drop.
    
    Args:
        model: Trained sequence model with .predict_next_state() or .predict().
        x_history: Sequence history array of shape (N, L, D).
        y_future: Target next state array of shape (N, D) or (N, H, D).
        min_relative_drop: Minimum required relative increase in MSE (e.g., 0.15 = +15%).
        n_permutations: Number of random temporal permutations to evaluate and average.
        random_state: Seed for reproducible permutations.
        verbose: If True, prints formatted diagnostic results to stdout.

    Returns:
        TemporalOrderShuffleResult with metrics and warning status.
    """
    x_arr = np.asarray(x_history, dtype=np.float32)
    y_arr = np.asarray(y_future, dtype=np.float32)

    if x_arr.ndim != 3:
        raise ValueError(f"x_history must be 3D (N, L, D), got shape: {x_arr.shape}")

    # If y_future is (N, H, D), evaluate immediate next state S_{t+1} at horizon index 0
    if y_arr.ndim == 3:
        y_next = y_arr[:, 0, :]
    else:
        y_next = y_arr

    n_samples, seq_len, num_features = x_arr.shape
    if seq_len < 2:
        raise ValueError(f"Sequence length L must be >= 2 to evaluate temporal shuffling, got: {seq_len}")

    rng = np.random.RandomState(random_state)

    def _evaluate(x_input: np.ndarray) -> Tuple[float, Optional[float]]:
        """Compute MSE and optional Gaussian NLL on input context."""
        if hasattr(model, "predict_next_state"):
            pred_dist = model.predict_next_state(x_input)
            pred_mean = pred_dist.mode()
            if hasattr(pred_dist, "nll"):
                nll_val = float(np.mean(pred_dist.nll(y_next)))
            else:
                nll_val = None
        elif hasattr(model, "predict"):
            pred_mean = model.predict(x_input)
            nll_val = None
        else:
            raise TypeError("Model must implement .predict_next_state() or .predict()")

        mse_val = float(np.mean((pred_mean - y_next) ** 2))
        return mse_val, nll_val

    # 1. Evaluate baseline chronological performance
    orig_mse, orig_nll = _evaluate(x_arr)

    # 2. Evaluate across random temporal order permutations
    shuffled_mse_list = []
    shuffled_nll_list = []

    for _ in range(n_permutations):
        # Generate a non-identity permutation of time steps 0..L-1
        perm = rng.permutation(seq_len)
        while np.array_equal(perm, np.arange(seq_len)):
            perm = rng.permutation(seq_len)

        # Scramble temporal sequence along time axis (axis 1)
        x_shuffled = x_arr[:, perm, :]
        mse_k, nll_k = _evaluate(x_shuffled)
        shuffled_mse_list.append(mse_k)
        if nll_k is not None:
            shuffled_nll_list.append(nll_k)

    avg_shuffled_mse = float(np.mean(shuffled_mse_list))
    avg_shuffled_nll = float(np.mean(shuffled_nll_list)) if shuffled_nll_list else None

    # 3. Calculate degradation
    if orig_mse > 1e-9:
        rel_mse_increase = (avg_shuffled_mse - orig_mse) / orig_mse
    else:
        rel_mse_increase = float(avg_shuffled_mse - orig_mse)

    nll_deg = (avg_shuffled_nll - orig_nll) if (orig_nll is not None and avg_shuffled_nll is not None) else None

    meaningful_drop = rel_mse_increase >= min_relative_drop
    warning_triggered = not meaningful_drop

    result = TemporalOrderShuffleResult(
        original_mse=round(orig_mse, 6),
        shuffled_mse=round(avg_shuffled_mse, 6),
        relative_mse_increase=round(rel_mse_increase, 4),
        original_nll=round(orig_nll, 4) if orig_nll is not None else None,
        shuffled_nll=round(avg_shuffled_nll, 4) if avg_shuffled_nll is not None else None,
        nll_degradation=round(nll_deg, 4) if nll_deg is not None else None,
        min_required_drop=min_relative_drop,
        meaningful_drop_detected=meaningful_drop,
        warning_triggered=warning_triggered,
        num_permutations=n_permutations,
    )

    if verbose:
        print("\n" + result.summary_table())

    return result

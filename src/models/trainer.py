"""Training Loop and Validation Pipeline for World Model Sequence Dynamics.

Trains the LSTM sequence model using Gaussian Negative Log-Likelihood (NLL)
to predict the distribution over the next state P(S_{t+1} | S_{<=t}).

Logs train and validation losses per epoch.
"""

from dataclasses import dataclass, field
import os
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from src.models.lstm_world_model import (
    LSTMWorldModel,
    NumpyLSTMWorldModel,
    GaussianNLLLoss,
    HAS_TORCH,
)

if HAS_TORCH:
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainingHistory:
    """Stores epoch-by-epoch loss metrics for training and validation."""
    epochs: List[int] = field(default_factory=list)
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    elapsed_sec: float = 0.0

    def summary(self) -> Dict[str, Union[int, float]]:
        return {
            "total_epochs": len(self.epochs),
            "best_epoch": self.best_epoch,
            "best_val_loss": round(self.best_val_loss, 4),
            "final_train_loss": round(self.train_losses[-1], 4) if self.train_losses else None,
            "final_val_loss": round(self.val_losses[-1], 4) if self.val_losses else None,
            "elapsed_seconds": round(self.elapsed_sec, 2),
        }


def _train_torch_model(
    model,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: Optional[np.ndarray],
    val_y: Optional[np.ndarray],
    num_epochs: int = 20,
    batch_size: int = 16,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    checkpoint_path: Optional[str] = None,
    verbose: bool = True,
) -> TrainingHistory:
    """Train PyTorch LSTM World Model using Gaussian NLL."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Prepare PyTorch Tensors & Loaders
    tx = torch.tensor(train_x, dtype=torch.float32)
    ty = torch.tensor(train_y, dtype=torch.float32)
    train_dataset = TensorDataset(tx, ty)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    has_val = val_x is not None and val_y is not None and len(val_x) > 0
    if has_val:
        vx = torch.tensor(val_x, dtype=torch.float32).to(device)
        vy = torch.tensor(val_y, dtype=torch.float32).to(device)

    history = TrainingHistory()
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_batch_losses = []

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()

            dist = model(bx)
            loss = GaussianNLLLoss(by, dist)
            loss.backward()

            # Gradient clipping to prevent exploding gradients in recurrent models
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_batch_losses.append(loss.item())

        epoch_train_loss = float(np.mean(train_batch_losses))
        history.epochs.append(epoch)
        history.train_losses.append(epoch_train_loss)

        # Validation phase
        if has_val:
            model.eval()
            with torch.no_grad():
                val_dist = model(vx)
                val_loss = float(GaussianNLLLoss(vy, val_dist).item())
        else:
            val_loss = epoch_train_loss

        history.val_losses.append(val_loss)

        # Track best model checkpoint
        if val_loss < history.best_val_loss:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            if checkpoint_path:
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)

        if verbose:
            print(
                f"Epoch [{epoch:02d}/{num_epochs:02d}] | "
                f"Train Loss (NLL): {epoch_train_loss:>9.4f} | "
                f"Val Loss (NLL): {val_loss:>9.4f}" +
                (" [BEST]" if epoch == history.best_epoch and has_val else "")
            )

    history.elapsed_sec = time.time() - start_time
    return history


def _train_numpy_model(
    model: NumpyLSTMWorldModel,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: Optional[np.ndarray],
    val_y: Optional[np.ndarray],
    num_epochs: int = 20,
    batch_size: int = 16,
    lr: float = 0.005,
    checkpoint_path: Optional[str] = None,
    verbose: bool = True,
) -> TrainingHistory:
    """Train NumPy LSTM World Model using Adam optimizer on Gaussian NLL."""
    # Initialize Adam state moments
    params = [model.W_mean, model.b_mean, model.W_var, model.b_var]
    m_moments = [np.zeros_like(p) for p in params]
    v_moments = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t_step = 0

    n_samples = len(train_x)
    has_val = val_x is not None and val_y is not None and len(val_x) > 0

    history = TrainingHistory()
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        indices = np.random.permutation(n_samples)
        batch_losses = []

        for start_idx in range(0, n_samples, batch_size):
            batch_idx = indices[start_idx:start_idx + batch_size]
            bx = train_x[batch_idx]
            by = train_y[batch_idx]

            # Forward pass
            dist = model.forward(bx)
            loss = GaussianNLLLoss(by, dist)
            batch_losses.append(loss)

            # Gradient calculation for Gaussian NLL:
            # L = 0.5 * sum (log_var + (y - mu)^2 / exp(log_var))
            var = np.exp(dist.log_var) + 1e-6
            diff = dist.mean - by  # (B, D)

            # dL / d_mu = diff / var
            d_mu = diff / var  # (B, D)
            # dL / d_log_var = 0.5 * (1 - (diff^2 / var))
            d_log_var = 0.5 * (1.0 - (diff ** 2) / var)  # (B, D)

            # Get hidden representations h_t
            # Forward h computation
            B, L, _ = bx.shape
            H = model.hidden_dim
            h = np.zeros((B, H), dtype=np.float32)
            c = np.zeros((B, H), dtype=np.float32)
            for t_idx in range(L):
                xt = bx[:, t_idx, :]
                gates = xt @ model.W_x.T + h @ model.W_h.T + model.b_lstm
                i = model._sigmoid(gates[:, :H])
                f = model._sigmoid(gates[:, H:2 * H])
                g = np.tanh(gates[:, 2 * H:3 * H])
                o = model._sigmoid(gates[:, 3 * H:])
                c = f * c + i * g
                h = o * np.tanh(c)

            # Head gradients
            grad_W_mean = (d_mu.T @ h) / len(bx)
            grad_b_mean = np.mean(d_mu, axis=0)
            grad_W_var = (d_log_var.T @ h) / len(bx)
            grad_b_var = np.mean(d_log_var, axis=0)

            grads = [grad_W_mean, grad_b_mean, grad_W_var, grad_b_var]

            # Adam update
            t_step += 1
            for i_p in range(len(params)):
                g_clipped = np.clip(grads[i_p], -1.0, 1.0)
                m_moments[i_p] = beta1 * m_moments[i_p] + (1.0 - beta1) * g_clipped
                v_moments[i_p] = beta2 * v_moments[i_p] + (1.0 - beta2) * (g_clipped ** 2)
                m_hat = m_moments[i_p] / (1.0 - (beta1 ** t_step))
                v_hat = v_moments[i_p] / (1.0 - (beta2 ** t_step))
                params[i_p] -= lr * m_hat / (np.sqrt(v_hat) + eps)

        epoch_train_loss = float(np.mean(batch_losses))
        history.epochs.append(epoch)
        history.train_losses.append(epoch_train_loss)

        # Validation
        if has_val:
            val_dist = model.forward(val_x)
            val_loss = GaussianNLLLoss(val_y, val_dist)
        else:
            val_loss = epoch_train_loss

        history.val_losses.append(val_loss)

        if val_loss < history.best_val_loss:
            history.best_val_loss = val_loss
            history.best_epoch = epoch

        if verbose:
            print(
                f"Epoch [{epoch:02d}/{num_epochs:02d}] | "
                f"Train Loss (NLL): {epoch_train_loss:>9.4f} | "
                f"Val Loss (NLL): {val_loss:>9.4f}" +
                (" [BEST]" if epoch == history.best_epoch and has_val else "")
            )

    history.elapsed_sec = time.time() - start_time
    return history


def train_world_model(
    model: Union[LSTMWorldModel, NumpyLSTMWorldModel],
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: Optional[np.ndarray] = None,
    val_y: Optional[np.ndarray] = None,
    num_epochs: int = 15,
    batch_size: int = 16,
    lr: float = 0.002,
    checkpoint_path: Optional[str] = "checkpoints/best_world_model.pt",
    verbose: bool = True,
) -> TrainingHistory:
    """Universal training loop for World Models with train/val loss logging per epoch.

    Args:
        model: LSTMWorldModel instance.
        train_x: Training input sequences of shape (N_train, L, D).
        train_y: Training target states of shape (N_train, D) representing S_{t+1}.
        val_x: Optional validation input sequences (N_val, L, D).
        val_y: Optional validation target states (N_val, D).
        num_epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate.
        checkpoint_path: Path to save the best model weights.
        verbose: If True, prints per-epoch loss logs.

    Returns:
        TrainingHistory containing train_losses and val_losses per epoch.
    """
    if HAS_TORCH and isinstance(model, torch.nn.Module):
        return _train_torch_model(
            model=model,
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            checkpoint_path=checkpoint_path,
            verbose=verbose,
        )
    else:
        return _train_numpy_model(
            model=model,
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            checkpoint_path=checkpoint_path,
            verbose=verbose,
        )

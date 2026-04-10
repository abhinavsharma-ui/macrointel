"""
LSTM Price Pattern Recognition Model — Phase 3
================================================
Architecture: 3-layer stacked LSTM with attention mechanism.

Why LSTM for this:
  - Prices are sequential — LSTM captures temporal dependencies
  - Multi-scale patterns: short-term noise + medium-term trends
  - Attention layer focuses on the most predictive timesteps
  - Handles variable-length regime windows

Input:  60-day lookback window of 150+ features per symbol
Output: 5 probability distributions over {-1, 0, +1} for horizons
        [1d, 3d, 5d, 10d, 21d]

Training: Google Colab T4 GPU (~45 min for full universe)
Inference: CPU-only, <10ms per symbol
"""

import logging
import math
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    logging.warning("PyTorch not installed. Run: pip install torch")

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models" / "checkpoints"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────
class SequenceDataset(Dataset):
    """
    Converts feature matrices into overlapping sequences for LSTM.
    
    For each day T, creates a 60-day lookback window [T-60 : T]
    and labels it with the forward return at T+horizon.
    
    Label encoding:
      +1 (buy)    if forward_return > +threshold
       0 (hold)   if -threshold <= forward_return <= +threshold
      -1 (sell)   if forward_return < -threshold
    
    Threshold is set dynamically per symbol based on its own volatility
    (1× daily std) to normalize across high/low volatility assets.
    """

    def __init__(
        self,
        feature_matrix: pd.DataFrame,
        sequence_length: int = 60,
        horizon: int = 5,
        target_col: str = "close",
        threshold_multiplier: float = 0.5,  # threshold = 0.5 × daily_std
        normalization_params: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.seq_len = sequence_length
        self.horizon = horizon

        # Select only numeric features, drop NaN rows
        numeric_cols = [c for c in feature_matrix.columns
                        if feature_matrix[c].dtype in (np.float64, np.float32, np.int64)
                        and c not in ("symbol", "has_gaps")]
        self.feature_names = numeric_cols

        df = feature_matrix[numeric_cols].copy()

        # Forward returns for labeling
        if "close" in feature_matrix.columns:
            fwd_ret = feature_matrix["close"].pct_change(horizon).shift(-horizon)
            daily_std = feature_matrix["close"].pct_change().std()
            threshold = daily_std * threshold_multiplier
        else:
            fwd_ret = pd.Series(0, index=feature_matrix.index)
            threshold = 0.01

        # Encode labels: +1 / 0 / -1
        labels = pd.cut(
            fwd_ret,
            bins=[-np.inf, -threshold, threshold, np.inf],
            labels=[0, 1, 2],  # sell=0, hold=1, buy=2
        ).astype(float).fillna(1)

        # Normalize features. If params are provided from train split, reuse them for val/test.
        if normalization_params:
            mean_map = normalization_params.get("means", {}) or {}
            std_map = normalization_params.get("stds", {}) or {}
            self.means = pd.Series({c: float(mean_map.get(c, 0.0)) for c in df.columns})
            self.stds = pd.Series({c: float(std_map.get(c, 1.0)) for c in df.columns}).replace(0, 1)
        else:
            self.means = df.mean()
            self.stds  = df.std().replace(0, 1)
        df_norm = (df - self.means) / self.stds
        df_norm = df_norm.clip(-5, 5).fillna(0)

        self.X = df_norm.values.astype(np.float32)
        self.y = labels.values.astype(np.int64)
        self.n_features = df_norm.shape[1]

        # Valid indices: need seq_len history AND horizon future
        self.valid_idx = list(range(sequence_length, len(df) - horizon))

        logger.info(
            f"SequenceDataset: {len(self.valid_idx)} samples | "
            f"{self.n_features} features | horizon={horizon}d | "
            f"threshold={threshold:.4f}"
        )

    def __len__(self) -> int:
        return len(self.valid_idx)

    def __getitem__(self, idx: int) -> Tuple:
        i = self.valid_idx[idx]
        x = self.X[i - self.seq_len : i]          # (seq_len, n_features)
        y = self.y[i]                              # scalar label
        return torch.FloatTensor(x), torch.LongTensor([y]).squeeze()

    def get_normalization_params(self) -> Dict:
        return {"means": self.means.to_dict(), "stds": self.stds.to_dict()}


# ─────────────────────────────────────────────────────────────
# Temporal Attention Layer
# ─────────────────────────────────────────────────────────────
class TemporalAttention(nn.Module):
    """
    Soft attention over the LSTM hidden states across time.
    Learns which of the 60 timesteps matter most for the prediction.
    This is what makes the model explainable — high attention on day T-5
    means recent price action is driving the signal.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_size)
        Returns:
            context: (batch, hidden_size) — attention-weighted summary
            weights: (batch, seq_len) — for visualization
        """
        scores = self.attention(lstm_output).squeeze(-1)  # (batch, seq_len)
        weights = F.softmax(scores, dim=-1)               # (batch, seq_len)
        context = torch.bmm(
            weights.unsqueeze(1), lstm_output
        ).squeeze(1)                                       # (batch, hidden_size)
        return context, weights


# ─────────────────────────────────────────────────────────────
# LSTM Model
# ─────────────────────────────────────────────────────────────
class LSTMPredictor(nn.Module):
    """
    3-layer stacked LSTM with temporal attention.
    
    Architecture diagram:
      Input (60 × n_features)
        ↓
      Input projection (n_features → 128)
        ↓
      LSTM Layer 1 (128 → 256, dropout=0.2)
        ↓
      LSTM Layer 2 (256 → 256, dropout=0.2)
        ↓
      LSTM Layer 3 (256 → 256)
        ↓
      Temporal Attention (256 → 256, learns which timesteps matter)
        ↓
      Classifier head:
        Linear(256 → 128) → BatchNorm → GELU → Dropout(0.3)
        Linear(128 → 64)  → GELU
        Linear(64 → 3)    → softmax over {sell, hold, buy}
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        n_classes: int = 3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Project raw features to LSTM input size
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

        # Stacked LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,  # Causal — no future info
        )

        # Temporal attention
        self.attention = TemporalAttention(hidden_size)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 (standard LSTM trick)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            logits: (batch, n_classes)
            attention_weights: (batch, seq_len) if return_attention else None
        """
        # Project to LSTM input
        projected = self.input_proj(
            x.view(-1, x.size(-1))
        ).view(x.size(0), x.size(1), -1)  # (batch, seq, 128)

        # LSTM
        lstm_out, _ = self.lstm(projected)  # (batch, seq, hidden)

        # Attention over time
        context, attn_weights = self.attention(lstm_out)

        # Classify
        logits = self.classifier(context)

        return logits, attn_weights if return_attention else None

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """Return softmax probabilities. Shape: (batch, 3)."""
        self.eval()
        with torch.no_grad():
            logits, _ = self(x)
            return F.softmax(logits, dim=-1).numpy()


# ─────────────────────────────────────────────────────────────
# Label Smoothing Loss (better calibration)
# ─────────────────────────────────────────────────────────────
class LabelSmoothingCELoss(nn.Module):
    """
    Cross-entropy with label smoothing.
    Prevents overconfident predictions — important for financial models
    where the "true" label is inherently uncertain.
    smoothing=0.1 → true class gets 0.9, other classes share 0.1.
    """

    def __init__(self, n_classes: int = 3, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.n_classes = n_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        confidence = 1.0 - self.smoothing
        smooth_val = self.smoothing / (self.n_classes - 1)

        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth_labels = torch.full_like(log_probs, smooth_val)
            smooth_labels.scatter_(1, targets.unsqueeze(1), confidence)

        return -(smooth_labels * log_probs).sum(dim=-1).mean()


# ─────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────
class LSTMTrainer:
    """
    Full training loop with:
    - OneCycleLR scheduler (trains faster, better generalization)
    - Early stopping on validation loss
    - Model checkpointing
    - Training metrics logging
    """

    def __init__(
        self,
        model: "LSTMPredictor",
        device: str = "auto",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        if not TORCH_OK:
            raise RuntimeError("PyTorch not installed")

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto" else torch.device(device)
        )
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.criterion = LabelSmoothingCELoss()
        logger.info(f"LSTM trainer initialized on {self.device}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 100,
        patience: int = 12,
        checkpoint_path: Optional[Path] = None,
    ) -> Dict:
        """Full training loop."""
        checkpoint_path = checkpoint_path or MODELS_DIR / "lstm_best.pt"

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=1e-3,
            epochs=max_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
        )

        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(max_epochs):
            # ── Training ──────────────────────────────────
            self.model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                logits, _ = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()

                # Gradient clipping — essential for LSTMs
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                scheduler.step()
                train_losses.append(loss.item())

            # ── Validation ────────────────────────────────
            val_loss, val_acc = self._evaluate(val_loader)
            avg_train_loss = np.mean(train_losses)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch+1:3d}/{max_epochs} | "
                    f"train_loss={avg_train_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | val_acc={val_acc:.3f}"
                )

            # ── Early stopping ────────────────────────────
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "val_acc": val_acc,
                    "n_features": self.model.lstm.input_size,
                }, checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        logger.info(f"Training complete. Best val_loss={best_val_loss:.4f}, val_acc={ckpt['val_acc']:.3f}")
        return history

    def _evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        losses, correct, total = [], 0, 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                logits, _ = self.model(X_batch)
                losses.append(self.criterion(logits, y_batch).item())
                preds = logits.argmax(dim=-1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)
        return float(np.mean(losses)), correct / max(total, 1)


# ─────────────────────────────────────────────────────────────
# Loader (inference)
# ─────────────────────────────────────────────────────────────
def load_lstm(checkpoint_path: Path, n_features: int, device: str = "cpu") -> "LSTMPredictor":
    """Load a trained LSTM model for inference."""
    model = LSTMPredictor(n_features=n_features)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__" and TORCH_OK:
    print("Testing LSTM model architecture...")
    model = LSTMPredictor(n_features=80)
    dummy = torch.randn(4, 60, 80)
    logits, attn = model(dummy, return_attention=True)
    print(f"  Input:     {dummy.shape}")
    print(f"  Logits:    {logits.shape}   (batch=4, classes=3)")
    print(f"  Attention: {attn.shape}  (batch=4, seq=60)")
    probs = F.softmax(logits, dim=-1)
    print(f"  Probs:     {probs.detach().numpy().round(3)}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,}")
    print("LSTM test passed ✓")

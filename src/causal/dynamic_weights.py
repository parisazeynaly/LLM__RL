"""
DynamicCausalWeights: periodically re-estimates per-factor importance from a
rolling buffer of (causal_state, outcome) pairs, via OLS with a correlation
fallback, and pushes the result into CausalBreakerSCM.

Extracted from causal_rl.py, lines 469-617 (excluding EpsilonSchedule,
which moved to src/utils.py — it's an RL exploration schedule, not a causal
component, and doesn't belong in this package).
"""

import threading

import numpy as np
import pandas as pd

from src.causal.scm import SCM
from src.config import OBS_KEYS


class DynamicCausalWeights:
    def __init__(self, min_samples=100, update_interval=500, max_buffer_size=5000):
        self.min_samples = min_samples
        self.update_interval = update_interval
        self.max_buffer_size = max_buffer_size

        self.buffer = []
        self.weight_history = []
        self.last_update_step = 0

        self.weights = np.ones(len(OBS_KEYS), dtype=np.float32) / len(OBS_KEYS)
        self._lock = threading.Lock()

    def add(self, obs, y, step=None):
        obs = np.asarray(obs, dtype=np.float32)

        row = {k: float(obs[i]) for i, k in enumerate(OBS_KEYS)}
        row["y"] = float(y)
        if step is not None:
            row["step"] = int(step)

        with self._lock:
            self.buffer.append(row)
            if len(self.buffer) > self.max_buffer_size:
                self.buffer = self.buffer[-self.max_buffer_size:]

    def maybe_update(self, step: int) -> bool:
        if step - self.last_update_step < self.update_interval:
            return False

        with self._lock:
            if len(self.buffer) < self.min_samples:
                print(f"[Step {step}] Skipped causal weight update: only {len(self.buffer)} samples.")
                return False
            buffer_snapshot = list(self.buffer)

        df = pd.DataFrame(buffer_snapshot)
        required_cols = OBS_KEYS + ["y"]

        if not all(col in df.columns for col in required_cols):
            print(f"[Step {step}] Skipped causal weight update: missing required columns.")
            return False

        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required_cols)

        if len(df) < self.min_samples:
            print(f"[Step {step}] Skipped causal weight update: only {len(df)} clean samples.")
            return False

        if df["y"].std() < 0.01:
            print(f"[Step {step}] Skipped causal weight update: insufficient outcome variation.")
            return False

        X = df[OBS_KEYS].values.astype(np.float32)
        y = df["y"].values.astype(np.float32).reshape(-1)

        try:
            X_aug = np.column_stack([X, np.ones(len(X), dtype=np.float32)])
            coef_with_intercept, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
            coef = coef_with_intercept[:-1]

            # Use absolute partial effects for stable factor importance
            w = np.abs(coef).astype(np.float32) + 1e-6

            if not np.all(np.isfinite(w)) or w.sum() <= 0:
                raise ValueError("Invalid OLS-derived weights.")

        except Exception as e:
            print(f"[Step {step}] OLS update failed, using correlation fallback. Reason: {e}")
            w = np.zeros(len(OBS_KEYS), dtype=np.float32)
            for i, factor in enumerate(OBS_KEYS):
                corr = df[factor].corr(df["y"])
                if np.isnan(corr):
                    corr = 0.0
                w[i] = abs(corr) + 1e-6

        w = w / w.sum()

        with self._lock:
            self.weights = w.astype(np.float32)
            self.last_update_step = int(step)

            hist_row = {"step": int(step)}
            for factor, weight in zip(OBS_KEYS, self.weights):
                hist_row[factor] = float(weight)
            self.weight_history.append(hist_row)

        try:
            scm_updated = SCM.update(df.to_dict(orient="records"))
        except Exception as e:
            print(f"[Step {step}] SCM.update failed: {e}")
            scm_updated = False

        print(f"\n[Step {step}] Updated dynamic causal weights")
        for factor, weight in zip(OBS_KEYS, self.weights):
            print(f"  {factor:35s}: {weight:.3f}")
        print(f"[Step {step}] SCM y_weights {'also updated' if scm_updated else 'not updated'}.")

        return True

    def get(self) -> np.ndarray:
        with self._lock:
            return self.weights.copy()

    def export_buffer(self, path: str) -> None:
        with self._lock:
            df = pd.DataFrame(self.buffer)
        df.to_csv(path, index=False)
        print(f"Causal buffer exported -> {path}")

    def export_weight_history(self, path: str) -> None:
        with self._lock:
            df = pd.DataFrame(self.weight_history)
        df.to_csv(path, index=False)
        print(f"Weight history exported -> {path}")


# Module-level singleton, matching the original notebook's usage pattern.
DYNAMIC_WEIGHTS = DynamicCausalWeights(
    min_samples=100,
    update_interval=500,
    max_buffer_size=5000,
)

"""Synthetic staking dataset as float32 arrays for PyTorch."""

import numpy as np

RATE = 0.0002  # reward per token per day


def make_dataset(n: int = 4000, seed: int = 42):
    """Return (X, y): features [amount, duration_days] and the reward target.

    Reward grows with amount * duration (plus noise), mirroring the on-chain vault.
    """
    rng = np.random.default_rng(seed)
    amount = rng.uniform(1, 1000, n)
    duration = rng.uniform(1, 365, n)
    noise = rng.normal(0, 2.0, n)
    reward = np.clip(amount * duration * RATE + noise, 0, None)

    X = np.stack([amount, duration], axis=1).astype("float32")
    y = reward.astype("float32").reshape(-1, 1)
    return X, y

# Torch Staking Net

A small **PyTorch** neural network that predicts staking rewards from stake size and duration, trained with a standard `nn.Module` + training loop.

It's the deep-learning counterpart to my scikit-learn version ([`staking-rewards-ml`](https://github.com/liander-ai/staking-rewards-ml)): same synthetic staking data (reward grows with `amount * duration`, echoing the on-chain vault), but here an MLP is trained by hand with Adam + MSE loss and backprop.

## Model

```
RewardNet: Linear(2, 32) -> ReLU -> Linear(32, 32) -> ReLU -> Linear(32, 1)
```

Features are standardized, then the network trains for 300 epochs with `Adam(lr=0.01)` and `MSELoss`, reaching a strong holdout R² (loss drops sharply as it learns the amount×duration signal).

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch (CPU), numpy, scikit-learn
python -m src.main
```

```
first-epoch loss: ...
final-epoch loss: ...
holdout R^2     : ...
```

## Tests

```bash
pytest -q
```

Tests check the network's output shape and that training both **reduces the loss** and **learns the relationship** (R² on a holdout split). The suite runs in CI on every push.

## Layout

```
src/data.py     synthetic staking data as float32 arrays
src/model.py    RewardNet (nn.Module MLP)
src/train.py    training loop (Adam + MSELoss) + holdout R^2
src/main.py     CLI
tests/          pytest suite
```

## Stack

- **PyTorch** (`torch`, `nn.Module`, `Adam`, `MSELoss`)
- NumPy, scikit-learn (data split + R²)
- pytest

## License

MIT

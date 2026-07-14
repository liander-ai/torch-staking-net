"""Train the RewardNet with a standard PyTorch training loop."""

import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from src.data import make_dataset
from src.model import RewardNet


def train(epochs: int = 300, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    X, y = make_dataset(seed=seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Standardize features so the MLP converges cleanly.
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0) + 1e-8
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std

    xt, yt = torch.tensor(X_tr), torch.tensor(y_tr)
    xv = torch.tensor(X_te)

    model = RewardNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    first_loss = None
    last_loss = None
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        optimizer.step()
        if first_loss is None:
            first_loss = loss.item()
        last_loss = loss.item()

    model.eval()
    with torch.no_grad():
        preds = model(xv).numpy()

    return {
        "model": model,
        "r2": float(r2_score(y_te, preds)),
        "first_loss": float(first_loss),
        "last_loss": float(last_loss),
    }

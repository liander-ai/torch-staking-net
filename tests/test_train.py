import torch

from src.model import RewardNet
from src.train import train


def test_model_output_shape():
    model = RewardNet()
    out = model(torch.zeros(5, 2))
    assert out.shape == (5, 1)


def test_training_reduces_loss_and_learns():
    result = train(epochs=300)
    # loss goes down substantially over training...
    assert result["last_loss"] < result["first_loss"] * 0.5
    # ...and the model fits the held-out data well.
    assert result["r2"] > 0.8

"""CLI: train the network and report the loss drop + holdout R^2."""

from src.train import train


def main() -> None:
    result = train()
    print(f"first-epoch loss: {result['first_loss']:.4f}")
    print(f"final-epoch loss: {result['last_loss']:.4f}")
    print(f"holdout R^2     : {result['r2']:.4f}")


if __name__ == "__main__":
    main()

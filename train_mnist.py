"""Two affine layers: 784 -> 128 -> 10, with one hidden ReLU."""

import argparse
from pathlib import Path
import numpy as np
from nn import Linear, ReLU, SGD, SoftmaxCrossEntropy
from nn.data import iter_minibatches, load_mnist


def build_model(hidden_size: int = 128, seed: int = 42) -> list:
    """Initialization is supplied; model forward/backward remain exercises."""
    rng = np.random.default_rng(seed)
    return [Linear(784, hidden_size, rng=rng, dtype=np.float32), ReLU(),
            Linear(hidden_size, 10, rng=rng, dtype=np.float32)]


def forward(model: list, x: np.ndarray) -> np.ndarray:
    """Return raw logits by applying layers in order."""
    # TODO: Pass each layer's output to the next and return the final logits.


def backward(model: list, grad_logits: np.ndarray) -> np.ndarray:
    """Propagate through layers in reverse order; return the input gradient."""
    # TODO: Chain each backward in reverse order; no SGD update inside this function.


def evaluate(model: list, x: np.ndarray, y: np.ndarray,
             batch_size: int = 1000) -> dict[str, float]:
    """Return {'loss': mean CE, 'accuracy': fraction in [0,1]} without updates.

    Weight batch losses by sample count, including the final partial batch.
    Use raw logits' argmax for predicted class. Do not run backward or SGD.
    """
    # TODO: Iterate non-shuffled batches, forward, compute loss and correct counts.
    # TODO: Divide total weighted loss and correct count by len(y); return metrics.


def train(model: list, train_x: np.ndarray, train_y: np.ndarray, *,
          epochs: int = 20, batch_size: int = 128, lr: float = 0.1,
          seed: int = 42, verbose: bool = True) -> list[float]:
    """Return one sample-weighted pre-update mean training loss per epoch.

    Shuffle each epoch with ONE seeded Generator; include every training sample.
    Use CE on logits, explicit backward, and plain SGD. Print epoch/loss when
    verbose=True. Test data is deliberately excluded from this function.
    """
    # TODO: Build RNG, loss and SGD from Linear parameter/gradient references.
    # TODO: For each epoch, use iter_minibatches with shuffle=True and shared RNG.
    # TODO: For each batch: zero_grad -> forward -> CE -> backward -> SGD.step.
    # TODO: Accumulate loss * actual batch size; append/print the epoch mean.
    # TODO: Return the epoch loss history.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).parent / "data" / "mnist")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.hidden_size) <= 0 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("epochs, batch-size, hidden-size and lr must be positive and finite")
    train_x, train_y, test_x, test_y = load_mnist(
        args.data_dir, download=not args.no_download)
    print(f"MNIST loaded: train={len(train_y)} test={len(test_y)}")
    if args.download_only:
        return
    model = build_model(args.hidden_size, args.seed)
    history = train(model, train_x, train_y, epochs=args.epochs,
                    batch_size=args.batch_size, lr=args.lr, seed=args.seed)
    if history is None:
        raise SystemExit("Exercise unfinished: implement train/evaluate and nn TODOs first.")
    metrics = evaluate(model, test_x, test_y)
    print(f"test_loss={metrics['loss']:.6f} test_accuracy={metrics['accuracy']:.2%}")
    if metrics["accuracy"] <= 0.95:
        raise SystemExit("Target not reached: test accuracy must be strictly >95%.")


if __name__ == "__main__":
    main()

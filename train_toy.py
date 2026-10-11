"""Single hidden neuron: y_hat = v * ReLU(w*x+b); all w,b,v are trainable."""

import numpy as np
from nn import Linear, MSE, ReLU, SGD


def train_toy(steps: int = 50, lr: float = 0.01, *, verbose: bool = True) -> dict:
    """Train x=[[2]], y=[[10]], w=1, b=1, v=2 for exactly steps updates.

    Return {'losses': list[float], 'final_loss': float, 'w': float,
            'b': float, 'v': float}.
    losses records pre-update MSE, one entry per step; final_loss is after
    the last update. Print 'step=001 loss=...' per step when verbose=True.
    The initial prediction is 6 and initial MSE is 16.
    """
    # TODO: Set x/y; create Linear(1,1), ReLU(), Linear(1,1,bias=False), MSE.
    # TODO: Set first-layer W=1, b=1 and output-layer W=2 in place.
    # TODO: Build SGD over both layers' parameters with the provided lr.
    # TODO: Repeat steps times: zero gradients, forward, loss, record/print loss.
    # TODO: Backpropagate loss -> output Linear -> ReLU -> input Linear.
    # TODO: Update parameters only AFTER all backward calls finish.
    # TODO: Recompute final loss and return history and final scalar parameters.


if __name__ == "__main__":
    result = train_toy()
    if result is None:
        raise SystemExit("Exercise unfinished: implement train_toy and nn TODOs first.")

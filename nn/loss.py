"""Scalar mean losses; backward includes the reduction's normalization."""

import numpy as np


class MSE:
    """Mean squared error over ALL N*K elements, without a factor of 1/2."""

    def __init__(self):
        self._prediction = None
        self._target = None

    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """Require identical nonempty shapes, cache inputs, return scalar MSE."""
        # TODO: Validate shapes (ValueError), cache inputs, and compute mean loss.

    def backward(self) -> np.ndarray:
        """Return dloss/dprediction; raise RuntimeError without forward."""
        # TODO: Differentiate the square and the mean over every element.


class SoftmaxCrossEntropy:
    """Mean cross-entropy from raw logits (N, C) and integer labels (N,).

    Labels must be in [0, C); normalize by N, not N*C. Avoid log(softmax)
    underflow: use shifted logits and log-sum-exp. Never modify caller arrays.
    """

    def __init__(self):
        self._probabilities = None
        self._labels = None

    def forward(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Validate shapes/labels, cache probabilities/labels, return scalar loss."""
        # TODO: Reject empty/invalid shapes, noninteger/out-of-range labels (ValueError).
        # TODO: Shift each row, compute stable log-sum-exp and mean cross-entropy.
        # TODO: Cache probabilities and labels for backward.

    def backward(self) -> np.ndarray:
        """Return dloss/dlogits, shape (N, C); require forward first."""
        # TODO: Raise RuntimeError without cache; apply the fused mean-loss gradient.
        # TODO: Work on a copy so repeated backward calls give the same result.

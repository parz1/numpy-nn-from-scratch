"""Plain stochastic gradient descent, without momentum or autodiff."""

from collections.abc import Iterable
import numpy as np


class SGD:
    """Hold live parameter/gradient references. Updates must happen in place."""

    def __init__(self, parameters: Iterable[tuple[np.ndarray, np.ndarray]],
                 lr: float = 0.01):
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be positive and finite")
        self.parameters = list(parameters)
        if any(p.shape != g.shape for p, g in self.parameters):
            raise ValueError("Each gradient must match its parameter shape")
        self.lr = lr

    def step(self) -> None:
        """Apply one in-place gradient-descent update to every parameter."""
        # TODO: Update each parameter using its current gradient and self.lr.

    def zero_grad(self) -> None:
        """Reset gradient buffers in place, preserving optimizer references."""
        # TODO: Fill each gradient array with zeros.

"""Layers use row-major batches and explicit cached backward passes."""

import numpy as np


class Linear:
    """Affine layer: (N, in_features) -> (N, out_features).

    W has shape (in_features, out_features), b has shape (out_features,).
    Backward overwrites dW/db on each call; the loss handles batch averaging.
    bias=False is useful for the toy network's trainable output weight v.
    """

    def __init__(self, in_features: int, out_features: int, *,
                 rng: np.random.Generator | None = None, bias: bool = True,
                 dtype: np.dtype = np.float64):
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Feature counts must be positive")
        rng = np.random.default_rng() if rng is None else rng
        self.W = (rng.standard_normal((in_features, out_features))
                  * np.sqrt(2.0 / in_features)).astype(dtype)
        self.b = np.zeros(out_features, dtype=dtype) if bias else None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b) if bias else None
        self._x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Cache x, then return affine output; omit the bias if self.b is None."""
        self._x = x
        z = x @ self.W
        if self.b is not None:
            z = z + self.b
        return z

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Store dW/db and return dx with the cached input's shape.

        Require a preceding forward (raise RuntimeError otherwise).
        Sum bias gradients over samples; do not average a second time.
        Write dW/db in place so existing optimizer references remain valid.
        Compute all gradients with the weights used in forward, before SGD.
        """
        # TODO: Validate the cache, overwrite parameter gradients, and return dx.

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return live (parameter, gradient) references for SGD."""
        pairs = [(self.W, self.dW)]
        if self.b is not None:
            pairs.append((self.b, self.db))
        return pairs


class ReLU:
    """Elementwise ReLU; use derivative zero at x == 0."""

    def __init__(self):
        self._x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Cache x and return an array of the same shape."""
        # TODO: Cache input and apply ReLU without mutating x.

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Apply the cached ReLU derivative; require forward first."""
        # TODO: Raise RuntimeError without a cache; mask the upstream gradient.

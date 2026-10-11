"""Central differences: intentionally independent of analytical backward code."""

import numpy as np


def numerical_gradient(function, value: np.ndarray, epsilon: float = 1e-6):
    """Differentiate a scalar function, perturbing/restoring one element at a time."""
    gradient = np.empty_like(value, dtype=np.float64)
    for index in np.ndindex(value.shape):
        original = value[index].copy()
        try:
            value[index] = original + epsilon
            positive = float(function())
            value[index] = original - epsilon
            negative = float(function())
            gradient[index] = (positive - negative) / (2 * epsilon)
        finally:
            value[index] = original
    return gradient


def array_result(value):
    assert isinstance(value, np.ndarray), "Implement the corresponding TODO: expected ndarray"
    assert np.all(np.isfinite(value)), "Output contains NaN/Inf"
    return value


def scalar_result(value):
    assert value is not None, "Implement the corresponding forward TODO: expected scalar loss"
    assert np.ndim(value) == 0 and np.isfinite(value)
    return float(value)


def assert_gradient(actual, expected):
    actual = array_result(actual)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-7)

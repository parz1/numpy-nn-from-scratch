import numpy as np
import pytest
from nn import Linear, SGD


def test_sgd_update_and_zero_grad_in_place():
    parameter = np.array([1., -2.])
    gradient = np.array([3., -4.])
    optimizer = SGD(iter([(parameter, gradient)]), lr=0.1)
    optimizer.step()
    np.testing.assert_allclose(parameter, [0.7, -1.6])
    np.testing.assert_array_equal(gradient, [3., -4.])
    optimizer.zero_grad()
    np.testing.assert_array_equal(gradient, [0., 0.])
    assert optimizer.parameters[0][0] is parameter
    assert optimizer.parameters[0][1] is gradient
    optimizer.step()
    np.testing.assert_allclose(parameter, [0.7, -1.6])


def test_optimizer_uses_latest_layer_gradients():
    layer = Linear(1, 1, bias=False)
    layer.W[:] = 1.
    optimizer = SGD(layer.parameters(), lr=0.1)
    for _ in range(2):
        layer.forward(np.array([[2.]]))
        layer.backward(np.array([[3.]]))
        optimizer.step()
    np.testing.assert_allclose(layer.W, [[-0.2]])


@pytest.mark.parametrize("lr", [0., -1., np.nan, np.inf])
def test_invalid_learning_rate(lr):
    with pytest.raises(ValueError):
        SGD([], lr=lr)

import numpy as np
import pytest
from nn import Linear, ReLU
from tests.gradcheck import array_result, assert_gradient, numerical_gradient


@pytest.mark.parametrize("batch,in_features,out_features,bias", [(1, 3, 2, True),
                         (4, 2, 3, True), (3, 2, 2, False)])
def test_linear_gradients(batch, in_features, out_features, bias):
    rng = np.random.default_rng(7)
    layer = Linear(in_features, out_features, rng=rng, bias=bias)
    x = rng.normal(size=(batch, in_features))
    upstream = rng.normal(size=(batch, out_features))
    array_result(layer.forward(x))
    dx = array_result(layer.backward(upstream)).copy()
    dw = layer.dW.copy()
    db = layer.db.copy() if bias else None
    # Arbitrary upstream gradient checks the vector-Jacobian product.
    objective = lambda: np.sum(array_result(layer.forward(x)) * upstream)
    assert_gradient(dx, numerical_gradient(objective, x))
    assert_gradient(dw, numerical_gradient(objective, layer.W))
    if bias:
        assert_gradient(db, numerical_gradient(objective, layer.b))


def test_linear_forward_and_live_gradients():
    layer = Linear(2, 2)
    layer.W[:] = [[1, 2], [3, 4]]
    layer.b[:] = [0.5, -0.5]
    references = layer.parameters()
    x = np.array([[2., -1.], [0., 3.]])
    original = x.copy()
    np.testing.assert_allclose(array_result(layer.forward(x)), [[-0.5, -0.5], [9.5, 11.5]])
    for _ in range(2):
        array_result(layer.backward(np.ones((2, 2))))
        np.testing.assert_allclose(layer.dW, [[2, 2], [2, 2]])
        np.testing.assert_allclose(layer.db, [2, 2])
        assert references[0][1] is layer.dW, "Overwrite gradient buffers in place"
        assert references[1][1] is layer.db
    np.testing.assert_array_equal(x, original)


def test_relu_gradient_away_from_kink():
    layer = ReLU()
    x = np.array([[-2., 0.7, 3.], [1.2, -0.8, -3.]])
    upstream = np.array([[0.3, -1., 2.], [-0.7, 1.3, 0.5]])
    array_result(layer.forward(x))
    actual = array_result(layer.backward(upstream)).copy()
    objective = lambda: np.sum(array_result(layer.forward(x)) * upstream)
    assert_gradient(actual, numerical_gradient(objective, x))


def test_relu_zero_convention_and_no_mutation():
    layer = ReLU()
    x = np.array([[-1., 0., 2.]])
    upstream = np.array([[4., 5., 6.]])
    np.testing.assert_array_equal(array_result(layer.forward(x)), [[0., 0., 2.]])
    np.testing.assert_array_equal(array_result(layer.backward(upstream)), [[0., 0., 6.]])
    np.testing.assert_array_equal(x, [[-1., 0., 2.]])
    np.testing.assert_array_equal(upstream, [[4., 5., 6.]])


@pytest.mark.parametrize("layer", [Linear(2, 2), ReLU()])
def test_backward_requires_forward(layer):
    with pytest.raises(RuntimeError):
        layer.backward(np.ones((1, 2)))

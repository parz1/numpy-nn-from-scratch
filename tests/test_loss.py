import numpy as np
import pytest
from nn import MSE, SoftmaxCrossEntropy
from tests.gradcheck import array_result, assert_gradient, numerical_gradient, scalar_result


@pytest.mark.parametrize("shape", [(1, 1), (3, 2), (2, 3, 2)])
def test_mse_gradient(shape):
    rng = np.random.default_rng(11)
    prediction, target = rng.normal(size=shape), rng.normal(size=shape)
    loss = MSE()
    scalar_result(loss.forward(prediction, target))
    actual = array_result(loss.backward()).copy()
    expected = numerical_gradient(lambda: scalar_result(loss.forward(prediction, target)), prediction)
    assert_gradient(actual, expected)


def test_mse_reduction():
    loss = MSE()
    assert scalar_result(loss.forward(np.array([[1., 3.]]), np.zeros((1, 2)))) == 5.0
    np.testing.assert_allclose(array_result(loss.backward()), [[1., 3.]])


@pytest.mark.parametrize("batch,classes", [(1, 2), (4, 3), (2, 10)])
def test_ce_gradient(batch, classes):
    rng = np.random.default_rng(12)
    logits = rng.normal(size=(batch, classes))
    labels = rng.integers(classes, size=batch)
    loss = SoftmaxCrossEntropy()
    scalar_result(loss.forward(logits, labels))
    actual = array_result(loss.backward()).copy()
    np.testing.assert_allclose(actual.sum(axis=1), 0., atol=1e-12)
    expected = numerical_gradient(lambda: scalar_result(loss.forward(logits, labels)), logits)
    assert_gradient(actual, expected)


def test_ce_uniform_logits():
    loss = SoftmaxCrossEntropy()
    assert scalar_result(loss.forward(np.zeros((2, 4)), np.array([0, 3]))) == pytest.approx(np.log(4))
    np.testing.assert_allclose(array_result(loss.backward()),
                               [[-0.375, 0.125, 0.125, 0.125], [0.125, 0.125, 0.125, -0.375]])


def test_ce_extreme_logits_shift_and_repeat():
    logits = np.array([[1000., -1000., 0.], [-1000., 1000., 0.]])
    original = logits.copy()
    labels = np.array([1, 0])  # Tiny target probabilities must not produce log(0).
    loss = SoftmaxCrossEntropy()
    value = scalar_result(loss.forward(logits, labels))
    assert value == pytest.approx(2000.)
    first = array_result(loss.backward()).copy()
    np.testing.assert_array_equal(array_result(loss.backward()), first)
    shifted = scalar_result(loss.forward(logits + np.array([[10000.], [-10000.]]), labels))
    assert shifted == pytest.approx(value)
    np.testing.assert_array_equal(array_result(loss.backward()), first)
    np.testing.assert_array_equal(logits, original)


@pytest.mark.parametrize("loss", [MSE(), SoftmaxCrossEntropy()])
def test_loss_backward_requires_forward(loss):
    with pytest.raises(RuntimeError):
        loss.backward()


@pytest.mark.parametrize("prediction,target", [(np.zeros((2, 2)), np.zeros((2, 1))),
                                               (np.empty((0, 2)), np.empty((0, 2)))])
def test_mse_invalid_inputs(prediction, target):
    with pytest.raises(ValueError):
        MSE().forward(prediction, target)


@pytest.mark.parametrize("logits,labels", [(np.zeros((2, 3)), np.array([0, 3])),
    (np.zeros((2, 3)), np.array([-1, 0])), (np.zeros((2, 3)), np.array([0., 1.])),
    (np.zeros((2, 3)), np.array([[0], [1]])), (np.zeros(3), np.array([0])),
    (np.empty((0, 3)), np.array([], dtype=int)), (np.empty((2, 0)), np.array([0, 0]))])
def test_ce_invalid_inputs(logits, labels):
    with pytest.raises(ValueError):
        SoftmaxCrossEntropy().forward(logits, labels)

import os
from pathlib import Path
import numpy as np
import pytest
from nn import Linear, ReLU, SoftmaxCrossEntropy
from nn.data import load_mnist
from tests.gradcheck import array_result, assert_gradient, numerical_gradient, scalar_result
from train_toy import train_toy
from train_mnist import backward, build_model, evaluate, forward, train


def test_composed_network_gradients():
    rng = np.random.default_rng(22)
    model = [Linear(2, 3, rng=rng), ReLU(), Linear(3, 2, rng=rng)]
    x = np.array([[0.5, -0.3], [1.1, 0.2]])
    model[0].W[:] = [[0.2, -0.1, 0.3], [0.1, 0.2, -0.1]]
    model[0].b[:] = [1., -1., 0.8]  # Keep ReLU inputs away from zero.
    labels = np.array([0, 1])
    loss = SoftmaxCrossEntropy()
    scalar_result(loss.forward(array_result(forward(model, x)), labels))
    dx = array_result(backward(model, array_result(loss.backward()))).copy()
    gradients = [(p, g.copy()) for layer in (model[0], model[2]) for p, g in layer.parameters()]
    objective = lambda: scalar_result(loss.forward(array_result(forward(model, x)), labels))
    assert_gradient(dx, numerical_gradient(objective, x))
    for parameter, gradient in gradients:
        assert_gradient(gradient, numerical_gradient(objective, parameter))


def test_toy_first_update():
    result = train_toy(steps=1, verbose=False)
    assert isinstance(result, dict), "Implement train_toy TODOs"
    assert result["losses"] == pytest.approx([16.])
    assert result["w"] == pytest.approx(1.32)
    assert result["b"] == pytest.approx(1.16)
    assert result["v"] == pytest.approx(2.24)
    assert result["final_loss"] == pytest.approx((2.24 * (1.32 * 2 + 1.16) - 10) ** 2)


def test_toy_50_steps_and_printing(capsys):
    result = train_toy()
    assert isinstance(result, dict), "Implement train_toy TODOs"
    assert len(result["losses"]) == 50
    assert result["losses"][0] == pytest.approx(16.)
    assert result["final_loss"] < 1e-6
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 50
    assert lines[0].startswith("step=001 loss=")
    assert lines[-1].startswith("step=050 loss=")


def test_evaluate_weights_partial_batch_and_does_not_update():
    model = [Linear(2, 2, bias=False)]
    model[0].W[:] = np.eye(2)
    before = model[0].W.copy()
    x = np.array([[3., 0.], [0., 3.], [0., 3.]])
    labels = np.array([0, 1, 0])
    metrics = evaluate(model, x, labels, batch_size=2)
    assert isinstance(metrics, dict), "Implement evaluate TODOs"
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["loss"] == pytest.approx(np.log1p(np.exp(-3)) + 1.)
    np.testing.assert_array_equal(model[0].W, before)


def test_train_small_separable_problem():
    rng = np.random.default_rng(42)
    model = [Linear(2, 8, rng=rng), ReLU(), Linear(8, 2, rng=rng)]
    x = np.array([[-1., -1.], [-1., -0.5], [1., 1.], [1., 0.5]], dtype=np.float64)
    labels = np.array([0, 0, 1, 1])
    history = train(model, x, labels, epochs=40, batch_size=3, lr=0.1, verbose=False)
    assert isinstance(history, list), "Implement train TODOs"
    assert len(history) == 40 and np.all(np.isfinite(history))
    assert history[-1] < history[0]
    assert evaluate(model, x, labels)["accuracy"] == 1.0


@pytest.mark.mnist
@pytest.mark.skipif(os.environ.get("RUN_MNIST") != "1", reason="Set RUN_MNIST=1 for full MNIST acceptance")
def test_full_mnist_accuracy():
    data_dir = Path(os.environ.get("MNIST_DATA_DIR", Path(__file__).resolve().parents[1] / "data" / "mnist"))
    train_x, train_y, test_x, test_y = load_mnist(data_dir)
    model = build_model(hidden_size=128, seed=42)
    history = train(model, train_x, train_y, epochs=20, batch_size=128, lr=0.1, seed=42, verbose=False)
    assert isinstance(history, list) and len(history) == 20
    assert evaluate(model, test_x, test_y)["accuracy"] > 0.95

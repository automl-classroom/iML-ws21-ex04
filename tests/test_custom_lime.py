import sys
import os  # noqa
sys.path.insert(0, ".")  # noqa

from classifiers.mlp import MLP
from tests.config import WORKING_DIR
from utils.dataset import Dataset
import numpy as np
import matplotlib
import warnings

warnings.filterwarnings("ignore")
matplotlib.use('Agg')


module = __import__(f"{WORKING_DIR}.custom_lime", fromlist=[
    'get_mesh', 'plot_mesh', 'sample_points', 'weight_points', 'plot_points_in_mesh', 'fit_explainer_model'])

dataset = Dataset(
    "wheat_seeds", [1, 5], [7], normalize=True, categorical=True)
(X_train, y_train), (X_test, y_test) = dataset.get_data()

# Unnormalized dataset
unnormalized_dataset = Dataset(
    "wheat_seeds", [1, 5], [7], normalize=False, categorical=True)

input_units = X_train.shape[1]
output_units = len(dataset.get_classes())

model = MLP(3, input_units*2, input_units, output_units, lr=0.01)

filename = "weights.pth"
if not os.path.isfile(filename):
    model.fit(X_train, y_train, num_epochs=50, batch_size=16)
    model.save(filename)
else:
    model.load(filename)


def test_get_mesh():
    u, v, z = module.get_mesh(
        model, unnormalized_dataset, points_per_feature=3)

    assert u.shape[0] == 3
    assert v.shape[0] == 3
    assert z.shape[0] == z.shape[1]
    assert z.shape[0] == 3

    # Check if configspace was used
    assert u[0] == 12.41

    u, v, z = module.get_mesh(model, dataset, points_per_feature=10)

    # Check if matrix has the right symmetry
    assert z[0, -1] == 2
    assert z[5, -1] == 2
    assert z[5, 2] == 0
    assert z[0, 0] == 1
    assert z[-1, -1] == 2
    assert z[-1, 0] == 0


def test_plot_mesh():
    u, v, z = module.get_mesh(model, dataset, points_per_feature=15)
    plt = module.plot_mesh(u, v, z, dataset.get_input_labels(), title="Test")

    ax = plt.gca()

    assert ax.get_xlabel() != ""
    assert ax.get_ylabel() != ""
    assert ax.get_title() != ""
    assert ax.pcolormesh


def test_sample_points():
    num_points = 50
    X, y = module.sample_points(model, dataset, num_points, seed=15)

    assert X.shape[0] == num_points
    assert y.shape[0] == num_points
    assert y[0] == 2
    assert y[-4] == 0
    assert y[-5] == 1

    X, y = module.sample_points(model, dataset, 2, seed=10)
    assert np.round(X[0, 0], 2) == 0.57
    assert np.round(X[1, 1], 2) == 0.21

    # Also test if configspace was used
    X, y = module.sample_points(model, unnormalized_dataset, 2, seed=10)
    assert np.round(X[0, 0], 2) == 15.18


def test_weight_points():
    selected_x = np.array([0.2, 0.3])
    weights = module.weight_points(selected_x, X_test, sigma=0.1)

    assert weights.shape[0] == 80
    assert min(weights) == 0.0
    assert max(weights) == 1.0
    assert weights[14] == max(weights)


def test_plot_points_in_mesh():
    from utils.styled_plot import plt
    plt.figure()

    selected_x = np.array([0.5, 0.5])
    X = np.array([[0.2, 0.2], [0.4, 0.7]])
    y = np.array([2, 1])
    weights = np.array([0.4, 0.5])

    module.plot_points_in_mesh(
        plt,
        X,
        y,
        weights,
        {2: "blue", 1: "yellow"},
        selected_x,
        size=8
    )

    ax = plt.gca()

    assert ax.collections[0].get_offsets()[0][0] == [0.4]
    assert ax.collections[1].get_offsets()[0][1] == [0.2]
    assert ax.collections[2].get_offsets()[0][0] == [0.5]
    assert ax.collections[0].get_sizes()[0] == 4.
    assert ax.collections[0].get_label() == "1"
    assert ax.collections[0].get_fc()[0][0] == 1.
    assert ax.collections[0].get_fc()[0][1] == 1.
    assert ax.collections[0].get_fc()[0][2] == 0.
    assert ax.collections[0].get_fc()[0][3] == 1.


def test_fit_explainer_model():
    X_test = np.array([[0.25, 0.21], [0.5, 0.5], [0.5, 0.1], [0.99, 0.001]])
    X = np.array([[0.2, 0.2], [0.4, 0.7]])
    y = np.array([2, 1])
    weights = np.array([0.4, 0.5])

    model = module.fit_explainer_model(X, y, weights, seed=5)
    p = model.predict(X_test)

    assert p[0] == 2.
    assert p[1] == 1.
    assert p[2] == 2.
    assert p[3] == 2.


if __name__ == "__main__":
    test_get_mesh()
    test_plot_mesh()
    test_sample_points()
    test_weight_points()
    test_plot_points_in_mesh()
    test_fit_explainer_model()

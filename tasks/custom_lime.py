import sys
import os
import random  # noqa
sys.path.insert(0, ".")  # noqa

from utils.styled_plot import plt
from utils.dataset import Dataset
from classifiers.mlp import MLP
import numpy as np
import itertools
from sklearn import tree


def get_mesh(model, dataset, points_per_feature=50):
    """
    Retrieve mesh data for plotting a two-dimensional graph with `points_per_feature` for each axis.
    The space is created by the hyperparameters' lower and upper values. Only the first two input
    labels are used.

    Parameters:
        model: Classifier which can call a predict method.
        dataset (utils.Dataset): Dataset with access to configspace and input labels.
        points_per_feature: How many points in each dimension.

    Returns:
        u (np.ndarray): x-axis points with shape (`points_per_feature`,).
        v (np.ndarray): y-axis points with shape (`points_per_feature`,).
        z (np.ndarray): Values with shape (`points_per_feature`, `points_per_feature`).
    """

    return None


def plot_mesh(u, v, z, labels, title=None, embedded=False):
    """
    Uses the mesh data to add a color mesh to the plot.

    Parameters:
        u (np.ndarray): x-axis points with shape (N,).
        v (np.ndarray): y-axis points with shape (N,).
        z (np.ndarray): Color values with shape (N, N).
        labels (list): Labels for x and y axis.
        title (str): Title for the plot.
        embedded (bool): Whether a new figure should be created or not.

    Returns: 
        plt (matplotlib.pyplot or utils.styled_plot.plt): Plot with applied color mesh.
    """

    if not embedded:
        plt.figure()

    return plt


def sample_points(model, dataset, num_points, seed=0):
    """
    Samples points for the two first features. Uses the bounds from configspace again.

    Parameters:
        model: Classifier which can call a predict method.
        dataset (utils.Dataset): Dataset with access to configspace and input labels.
        num_points (int): How many points should be sampled.
        seed (int): Seed to feed random.

    Returns:
        X (np.ndarray): Data with shape (`num_points`, 2)
        y (np.ndarray): Target values with shape (`num_points`,)
    """

    random.seed(seed)

    return None


def weight_points(selected_x, X, sigma=0.2):
    """
    For every x in `X` returns a weight depending on the distance to `selected_x`.

    Parameters:
        selected_x (np.ndarray): Single point with shape (2,).
        X (np.ndarray): Points with shape (?, 2).
        sigma (float): Sigma value to calculate distance.

    Returns:
        weights (np.ndarray): Normalized weights between 0..1 with shape (?,).
    """

    return None


def plot_points_in_mesh(plt, X=[], y=[], weights=None, colors={}, selected_x=None, size=8):
    """
    Given a plot, add scatter points from `X` and `selected_x`.

    Parameters:
        plt (matplotlib.pyplot or utils.styled_plot.plt): Plot with color mesh.
        X (np.ndarray): Points with shape (?, 2).
        y (np.ndarray): Target values with shape (?,).
        weights (np.ndarray): Normalized weights with shape (?,).
        colors (dict): Returns the color for an y value.
        selected_x (np.ndarray): Single point with shape (2,).
        size (int): Default size of the markers.
    """

    return


def fit_explainer_model(X, y, weights=None, seed=0):
    """
    Fits a decision tree.

    Parameters:
        X (np.ndarray): Points with shape (?, 2).
        y (np.ndarray): Target values with shape (?,).
        weights (np.ndarray): Normalized weights with shape (?,).
        seed (int): Seed for the decision tree.

    Returns:
        model (DecisionTreeRegressor): Fitted explainer model.
    """

    return None


if __name__ == "__main__":
    dataset = Dataset(
        "wheat_seeds", [1, 5], [7], normalize=True, categorical=True)
    (X_train, y_train), (X_test, y_test) = dataset.get_data()

    input_units = X_train.shape[1]
    output_units = len(dataset.get_classes())

    model = MLP(3, input_units*2, input_units, output_units, lr=0.01)

    filename = "weights.pth"
    if not os.path.isfile(filename):
        model.fit(X_train, y_train, num_epochs=50, batch_size=16)
        model.save(filename)
    else:
        model.load(filename)

    selected_x = np.array([0.31, 0.37])
    points_per_feature = 50
    n_points = 1000
    labels = dataset.get_input_labels()
    colors = {
        0: "purple",
        1: "green",
        2: "orange",
    }

    print("Run `get_mesh` ...")
    u, v, z = get_mesh(
        model, dataset, points_per_feature=points_per_feature)

    print("Run `plot_mesh` ...")
    plt = plot_mesh(u, v, z, labels=labels, title="MLP")
    plt.show()

    print("Run `sample_points` ...")
    X_sampled, y_sampled = sample_points(model, dataset, n_points)

    print("Run `plot_points_in_mesh` ...")
    plt = plot_mesh(u, v, z, labels=labels, title="MLP + Sampled Points")
    plot_points_in_mesh(plt, X_sampled, y_sampled, colors=colors)
    plt.legend()
    plt.show()

    print("Run `weight_points` ...")
    weights = weight_points(selected_x, X_sampled)

    print("Run `plot_points_in_mesh` ...")
    plt = plot_mesh(u, v, z, labels=labels,
                    title="MLP + Weighted Sampled Points")
    plot_points_in_mesh(plt, X_sampled, y_sampled, weights, colors, selected_x)
    plt.legend()
    plt.show()

    print("Run `fit_explainer_model` ...")
    explainer = fit_explainer_model(X_sampled, y_sampled, weights)

    print("Compare models ...")
    plt.subplots(cols=2, rows=1, sharey=True)

    plt.subplot(1, 2, 1)
    plot_mesh(u, v, z, labels=labels, title="MLP", embedded=True)
    plot_points_in_mesh(plt, X_sampled, y_sampled, weights, colors, selected_x)
    plt.legend()

    plt.subplot(1, 2, 2)
    u2, v2, z2 = get_mesh(
        explainer, dataset, points_per_feature=points_per_feature)
    plot_mesh(u2, v2, z2, labels=labels, title="Decision Tree", embedded=True)
    plot_points_in_mesh(plt, X_sampled, y_sampled, weights, colors, selected_x)
    plt.legend()

    plt.show()

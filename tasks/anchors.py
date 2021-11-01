import sys
import os  # noqa
sys.path.insert(0, ".")  # noqa

from utils.styled_plot import plt
from utils.dataset import Dataset
from classifiers.mlp import MLP
import numpy as np

import re
from anchor import anchor_tabular
from tests.config import WORKING_DIR
module = __import__(f"{WORKING_DIR}.custom_lime",
                    fromlist=['get_mesh', 'plot_mesh', 'sample_points', 'plot_points_in_mesh'])


def get_explanation(explainer, model, dataset, selected_x, threshold=0.95):
    """
    Uses anchor-exp to get an explanation/anchor. Human-friendly text is converted as bounds.

    Parameters:
        explainer (anchor.anchor_tabular.AnchorTabularExplainer)
        model: Classifier which can call a predict method.
        dataset (utils.Dataset): Dataset with access to configspace and input labels.
        selected_x (np.ndarray): Single point with shape (2,).
        threshold (float): Desired confidence.

    Returns:
        precision (float): Precision value rounded to two decimal places.
        coverage (float): Coverage value rounded to two decimal places.
        bounds (dict): Uses labels/feature names as keys and maps to (lower, upper) bounds.
    """

    return None


def plot_anchor(plt, labels, precision, coverage, bounds, color="r"):
    """
    Adds the anchor (4 lines) and the selected point to the plot.
    Also, adds precision and coverage inside the anchor.

    Parameters:
        plt (matplotlib.pyplot or utils.styled_plot.plt): Plot with color mesh.
        labels (list): List of labels/feature names.
        selected_x (np.ndarray): Single point with shape (2,).
        precision (float): Precision value rounded to two decimal places.
        coverage (float): Coverage value rounded to two decimal places.
        bounds (dict): Uses labels/feature names as keys and maps to (lower, upper) bounds.
        color (str): Color of the anchor and point.
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

    classes = dataset.get_classes()
    labels = dataset.get_input_labels()
    points_per_feature = 50
    n_points = 500
    colors = {
        0: "purple",
        1: "green",
        2: "orange",
    }
    points = {
        (0.2, 0.7): "purple",
        (0.54, 0.74): "green",
        (0.95, 0.30): "orange",
        (0.18, 0.12): "green",
    }

    X_sampled, y_sampled = module.sample_points(model, dataset, n_points)

    np.random.seed(0)
    explainer = anchor_tabular.AnchorTabularExplainer(
        classes, labels, X_sampled)

    print("Run `get_mesh` ...")
    u, v, z = module.get_mesh(
        model, dataset, points_per_feature=points_per_feature)

    print("Run `plot_mesh` ...")
    plt = module.plot_mesh(u, v, z, labels=labels, title="MLP")
    module.plot_points_in_mesh(
        plt, X_sampled, y_sampled, colors=colors, size=2)

    print("Explaining points might take a while ...")
    for point, color in points.items():
        print(f"Explain point {point} ...")

        point = np.array(point)
        precision, coverage, bounds = get_explanation(
            explainer, model, dataset, point)
        module.plot_points_in_mesh(plt, selected_x=point)
        plot_anchor(plt, labels, precision, coverage, bounds, color=color)

    plt.show()

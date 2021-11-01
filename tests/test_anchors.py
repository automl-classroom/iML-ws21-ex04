import sys
import os  # noqa
sys.path.insert(0, ".")  # noqa

from classifiers.mlp import MLP
from tests.config import WORKING_DIR
from utils.dataset import Dataset
import numpy as np
import matplotlib
from anchor import anchor_tabular
import warnings


warnings.filterwarnings("ignore")
matplotlib.use('Agg')


module = __import__(f"{WORKING_DIR}.anchors", fromlist=[
    'get_explanation', 'plot_anchor'])

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


classes = dataset.get_classes()
labels = dataset.get_input_labels()

np.random.seed(0)
explainer = anchor_tabular.AnchorTabularExplainer(classes, labels, X_test)


def test_get_explanation():
    prec, cov, bounds = module.get_explanation(
        explainer,
        model,
        dataset,
        np.array([0.5, 0.5]),
        threshold=0.7,
    )

    assert prec == 0.91
    assert cov == 0.51

    assert 'perimeter' in bounds
    assert 'asymmetry' in bounds
    assert bounds["perimeter"] == (0.44, 1.0)
    assert bounds["asymmetry"] == (0.0, 1.0)

    prec, cov, bounds = module.get_explanation(
        explainer,
        model,
        unnormalized_dataset,
        np.array([0.5, 0.5]),
        threshold=0.7,
    )

    assert bounds["asymmetry"][1] == 8.315


def test_plot_anchor():
    prec = 0.7
    cov = 0.3
    bounds = {
        "perimeter": (0.2, 0.3),
        "asymmetry": (0.7, 0.8)
    }

    from utils.styled_plot import plt
    plt.figure()

    module.plot_anchor(
        plt,
        labels,
        prec,
        cov,
        bounds,
        color="green"
    )

    module.plot_anchor(
        plt,
        labels,
        prec,
        cov,
        bounds,
        color="red"
    )

    ax = plt.gca()

    count = 0
    for child in ax.get_children():
        if isinstance(child, matplotlib.text.Text):
            if child.get_text() != '':
                count += 1

    assert count == 2

    combinations = []
    for line in ax.lines:
        for x, y in zip(line.get_xdata(), line.get_ydata()):

            combinations.append([x, y])

    required_combinations = [
        [0.2, 0.7],
        [0.3, 0.7],
        [0.2, 0.8],
        [0.3, 0.8]
    ]

    for rc in required_combinations:
        assert rc in combinations


if __name__ == "__main__":
    test_get_explanation()
    test_plot_anchor()

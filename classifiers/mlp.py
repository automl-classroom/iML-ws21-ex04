from utils.dataset import PyTorchDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch import nn
import torch
import numpy as np
import warnings
import logging
import sys
sys.path.insert(0, '..')

logger = logging.getLogger('lightning')
logger.setLevel(logging.ERROR)

warnings.filterwarnings("ignore")

pl.utilities.distributed.log.setLevel(logging.ERROR)
pl.utilities.seed.log.setLevel(logging.ERROR)


def get_sequential_model(num_hidden_layers,
                         num_hidden_units,
                         input_units,
                         output_units,
                         dropout_rate=0.2):
    """
    Returns a sequential model with 2+num_hidden_layers linear layers.
    All linear layers (except the last one) are followed by a ReLU function.

    Parameters:
        num_hidden_layers (int): The number of hidden layers.
        num_hidden_units (int): The number of features from the hidden
            linear layers.
        input_units (int): The number of input units.
            Should be number of features.
        output_units (int): The number of output units. In case of regression task,
            it should be one.

    Returns:
        model (nn.Sequential): Neural network as sequential model.
    """

    layers = [
        nn.Linear(input_units, num_hidden_units),
        nn.BatchNorm1d(num_hidden_units),
        nn.ReLU(),
        nn.Dropout(p=dropout_rate)
    ]

    for _ in range(num_hidden_layers):
        layers += [
            nn.Linear(num_hidden_units, num_hidden_units),
            nn.BatchNorm1d(num_hidden_units),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        ]

    layers += [
        nn.Linear(num_hidden_units, output_units)
    ]

    return nn.Sequential(*layers)


class MLP(pl.LightningModule):
    """
    Multi Layer Perceptron wrapper for pytorch lightning.
    The model from `get_sequential_model` is used as classifier.
    """

    def __init__(self,
                 num_hidden_layers,
                 num_hidden_units,
                 input_units,
                 output_units,
                 lr=1e-3,
                 verbose=False):
        """
        Parameters:
            lr (float): Learning rate.
        """

        super().__init__()

        pl.seed_everything(0)
        self.model = get_sequential_model(
            num_hidden_layers,
            num_hidden_units,
            input_units,
            output_units)

        self.verbose = verbose
        self.lr = lr
        self.classification = output_units > 1

        if self.classification:
            task = "Binary"
            if output_units > 2:
                task = "Multi"

            if verbose:
                print(f"--- {task} Classification Task ---")
            self.loss_fn = nn.CrossEntropyLoss()
        else:

            if verbose:
                print("--- Regression Task ---")
            self.loss_fn = nn.MSELoss()

    def forward(self, batch):
        X, _ = batch
        y_pred = self.model(X)

        if self.classification:
            y_pred_softmax = torch.log_softmax(y_pred, dim=1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
            return y_pred_tags
        else:
            return y_pred

    def multi_acc(self, y_pred, y_test):
        y_pred_softmax = torch.log_softmax(y_pred, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

        correct_pred = (y_pred_tags == y_test).float()
        acc = correct_pred.sum() / len(correct_pred)

        return acc

    def validation_step(self, batch, _):
        """
        Receives the validation data and calculates mean square
        error on them.

        Parameters:
            batch: Tuple of validation data.

        Returns: 
            metrics (dict): Dict with mse.
        """

        X, y = batch
        y_hat = self.model(X)

        results = {}
        if self.classification:
            results["accuracy"] = self.multi_acc(y_hat, y)
            y = y.long()

        results["loss"] = self.loss_fn(y_hat, y)

        return results

    def validation_epoch_end(self, outputs):
        """
        Collects the outputs from `validation_step` and sets
        the average for mse.

        Parameters:
            outputs: List of dicts from `validation_step`.
        """

        self.val_loss = float(
            np.mean(torch.stack([o['loss'] for o in outputs]).numpy().flatten()))

        if self.classification:
            self.val_accuracy = float(
                np.mean(torch.stack([o['accuracy'] for o in outputs]).numpy().flatten()))
            if self.verbose:
                print(f"{self.current_epoch}: {self.val_accuracy}")
        else:
            if self.verbose:
                print(f"{self.current_epoch}: {self.val_loss}")

    def training_step(self, batch, _):
        """
        Receives the training data and calculates
        mean square error as loss, which is used to train
        the classifier.

        Parameters:
            batch: Tuple of training data.

        Returns: 
            loss (Tensor): Loss of current step. 

        """

        X, y = batch
        y_hat = self.model(X)

        if self.classification:
            y = y.long()

        return self.loss_fn(y_hat, y)

    def configure_optimizers(self):
        """
        Configures Adam as optimizer.

        Returns:
            optimizer (torch.optim): Optimizer used internally from
                pytorch lightning.
        """

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def fit(self, X_train, y_train, X_val=None, y_val=None, num_epochs=50, batch_size=8):
        print("Fitting MLP ...")
        pl.seed_everything(0)

        trainer = pl.Trainer(
            num_sanity_val_steps=0,  # No validation sanity
            max_epochs=num_epochs,  # We only train one epoch
            progress_bar_refresh_rate=0,  # No progress bar
            weights_summary=None  # No model summary
        )

        # Define training loader
        # `train_loader` is a lambda function, which takes batch_size as input
        train_loader = DataLoader(
            PyTorchDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0)

        if X_val is not None and y_val is not None:
            # Define validation loader
            val_loader = DataLoader(
                PyTorchDataset(X_val, y_val),
                batch_size=1,
                num_workers=0)

            # Train model
            trainer.fit(self, train_loader, val_loader)
        else:
            trainer.fit(self, train_loader)

    def predict(self, X):
        trainer = pl.Trainer(
            num_sanity_val_steps=0,  # No validation sanity
            progress_bar_refresh_rate=0,  # No progress bar
            weights_summary=None  # No model summary
        )

        loader = DataLoader(
            PyTorchDataset(X),
            batch_size=1,
            num_workers=0)

        results = trainer.predict(self, dataloaders=loader)
        return np.array([r.numpy()[0] for r in results]).flatten()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))

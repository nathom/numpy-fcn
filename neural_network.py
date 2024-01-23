import numpy as np

from activation import Activation
from layer import Layer


class NeuralNetwork:
    def __init__(self, config):
        self.y: np.ndarray  # Last computed output
        self.learning_rate: float = config["learning_rate"]

        num_layers = len(config["layer_specs"]) - 1
        specs = config["layer_specs"]
        activation = config["activation"]
        self.layers = [
            Layer(specs[i], specs[i + 1], Activation(activation))
            for i in range(num_layers - 1)
        ]
        self.layers.append(
            Layer(specs[num_layers - 1], specs[num_layers], Activation("output"))
        )

    def forward(self, x_batch):
        output = x_batch
        for layer in self.layers:
            output = self.append_bias(output)
            output = layer.forward(output)

        self.y = output

    @staticmethod
    def append_bias(x):
        return np.column_stack((x, np.ones((x.shape[0],))))

    def current_loss(self, targets: np.ndarray) -> np.ndarray:
        epsilon = 1e-15
        # avoid log(0.0)
        outputs = np.clip(self.y, epsilon, 1 - epsilon)
        return -np.sum(targets * np.log(outputs), axis=1)

    def output_loss(self, outputs, targets):
        return targets - outputs

    def num_correct(self, targets: np.ndarray):
        assert self.y is not None
        correct_vec = np.argmax(self.y, axis=1) == np.argmax(targets, axis=1)
        return correct_vec.sum()

    def get_failed_indices(
        self, targets: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (y_hat, y) for all failed images and inds (boolean array) for their indices."""
        assert self.y is not None
        y_hats = np.argmax(self.y, axis=1)
        ys = np.argmax(targets, axis=1)
        inds = y_hats != ys
        return y_hats[inds], ys[inds], inds

    def backward(self, l1: float, l2: float, gamma: float, targets: np.ndarray):
        delta = self.output_loss(self.y, targets)
        for layer in reversed(self.layers):
            delta = layer.backward(delta, self.learning_rate, gamma, l1, l2)

    def update_weights(self):
        for layer in self.layers:
            layer.update_weights()

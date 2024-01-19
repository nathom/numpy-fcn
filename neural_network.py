import numpy as np

from activation import Activation
from layer import Layer


class NeuralNetwork:
    """
    Create a Neural Network specified by the network configuration mentioned in the config yaml file.
    """

    __slots__ = ["layers", "x", "y", "learning_rate"]

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers: list[Layer] = []  # Store all layers in this list.
        num_layers = len(config["layer_specs"]) - 1  # Set num layers here
        self.x = None  # Save the input to forward in this
        self.y = None  # For saving the output vector of the model
        self.learning_rate: float = config["learning_rate"]

        # Add layers specified by layer_specs.
        for i in range(num_layers):
            if i < num_layers - 1:
                self.layers.append(
                    Layer(
                        config["layer_specs"][i],
                        config["layer_specs"][i + 1],
                        Activation(config["activation"]),
                    )
                )
            elif i == num_layers - 1:
                self.layers.append(
                    Layer(
                        config["layer_specs"][i],
                        config["layer_specs"][i + 1],
                        Activation("output"),
                    )
                )

    def forward(self, x_batch):
        output = x_batch
        for layer in self.layers:
            output = np.column_stack((output, np.ones((output.shape[0],))))
            output = layer.forward_batch(output)

        self.y = output

    def current_loss(self, targets: np.ndarray) -> np.ndarray:
        """Return the loss for the current y."""
        epsilon = 1e-15
        # avoid log(0.0)
        assert self.y is not None
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
        """Return (y_hat, y) for all failed images"""
        assert self.y is not None
        y_hats = np.argmax(self.y, axis=1)
        ys = np.argmax(targets, axis=1)
        i = y_hats != ys
        return y_hats[i], ys[i], i

    def backward(self, l1: float, l2: float, gamma, targets):
        delta = self.output_loss(self.y, targets)
        for layer in reversed(self.layers):
            delta = layer.backward_batch(delta, self.learning_rate, gamma, l1, l2)

    def update_weights(self):
        for layer in self.layers:
            layer.update_weights()

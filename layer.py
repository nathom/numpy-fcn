import numpy as np

from activation import Activation


class Layer:
    """
    This class implements Fully Connected layers for your neural network.
    """

    __slots__ = ["w", "x", "a", "activation", "dw"]

    def __init__(self, in_units, out_units, activation):
        """
        Define the architecture and create placeholders.
        """

        # Randomly initialize weights
        self.w: np.ndarray = 0.01 * np.random.random((in_units + 1, out_units))

        self.x: np.ndarray | None = None  # Save the input to forward in this
        self.a: np.ndarray | None = None  # output without activation
        self.activation: Activation = activation

        self.dw: np.ndarray = np.zeros_like(self.w)

    def forward_batch(self, x_batch: np.ndarray) -> np.ndarray:
        # x_batch (B,785)
        # w       (785,10)
        # return  (B,10)
        self.x = x_batch
        self.a = x_batch @ self.w
        z = self.activation(self.a)
        return z

    def backward_batch(
        self,
        next_delta: np.ndarray,
        learning_rate,
        momentum_gamma,
        l1: float,
        l2: float,
    ):
        assert self.a is not None
        assert self.x is not None

        if self.activation.activation_type == "output":
            # output delta passed in by caller
            # see NeuralNetwork.backward()
            this_delta = next_delta
        else:
            this_delta = self.activation.backward(self.a) * next_delta

        gradient = self.x.T @ this_delta - l1 - l2 * self.w
        if momentum_gamma > 0.0:
            self.dw *= momentum_gamma
        self.dw += learning_rate * gradient
        return this_delta @ self.w[:-1, :].T

    def get_gradient(self, next_delta):
        assert self.x is not None
        assert self.a is not None

        if self.activation.activation_type == "output":
            # output delta passed in by caller
            # see NeuralNetwork.backward()
            this_delta = next_delta
        else:
            this_delta = self.activation.backward(self.a) * next_delta
        return self.x.T @ this_delta

    def update_weights(self):
        self.w += self.dw

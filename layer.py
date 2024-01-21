import numpy as np

from activation import Activation


class Layer:
    def __init__(self, in_units, out_units, activation):
        self.x: np.ndarray
        self.a: np.ndarray
        self.activation: Activation = activation

        self.w: np.ndarray = 0.01 * np.random.random((in_units + 1, out_units))
        self.dw: np.ndarray = np.zeros_like(self.w)

    def forward(self, x_batch: np.ndarray) -> np.ndarray:
        self.x = x_batch
        self.a = x_batch @ self.w
        return self.activation.forward(self.a)

    def backward(
        self,
        next_delta: np.ndarray,
        learning_rate,
        momentum_gamma,
        l1: float,
        l2: float,
    ):
        if self.activation.activation_type == "output":
            # output delta passed in by caller
            # see NeuralNetwork.backward()
            this_delta = next_delta
        else:
            this_delta = self.activation.backward(self.a) * next_delta

        weight_decay = l1 * np.sign(self.w) + 2.0 * l2 * self.w
        gradient = self.x.T @ this_delta - weight_decay
        self.dw *= momentum_gamma
        self.dw += learning_rate * gradient
        return this_delta @ self.w[:-1, :].T

    def get_gradient(self, next_delta):
        if self.activation.activation_type == "output":
            # output delta passed in by caller
            # see NeuralNetwork.backward()
            this_delta = next_delta
        else:
            this_delta = self.activation.backward(self.a) * next_delta
        return self.x.T @ this_delta

    def update_weights(self):
        self.w += self.dw

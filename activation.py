import numpy as np


class Activation:
    def __init__(self, activation_type: str = "sigmoid"):
        activation_type = activation_type.lower()
        if activation_type not in ("sigmoid", "tanh", "relu", "output"):
            raise NotImplementedError(f"{activation_type} is not implemented.")

        self.activation_type: str = activation_type

    def forward(self, z) -> np.ndarray:
        if self.activation_type == "sigmoid":
            return self.sigmoid(z)

        elif self.activation_type == "tanh":
            return self.tanh(z)

        elif self.activation_type == "relu":
            return self.relu(z)

        elif self.activation_type == "output":
            return self.output(z)

        raise Exception

    def backward(self, z: np.ndarray) -> np.ndarray:
        if self.activation_type == "sigmoid":
            return self.grad_sigmoid(z)

        elif self.activation_type == "tanh":
            return self.grad_tanh(z)

        elif self.activation_type == "relu":
            return self.grad_relu(z)

        raise Exception

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def relu(self, x):
        return x * (x > 0.0)

    def output(self, x: np.ndarray):
        exp_x: np.ndarray = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def grad_sigmoid(self, x):
        s = self.sigmoid(x)
        return s * (-s + 1.0)

    def grad_tanh(self, x):
        return 1 - self.tanh(x) ** 2

    def grad_relu(self, x):
        return (x > 0.0) * 1.0

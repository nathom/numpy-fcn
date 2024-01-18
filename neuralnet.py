import numpy as np


class Activation:
    """
    The class implements different types of activation functions for
    your neural network layers.
    """

    def __init__(self, activation_type="sigmoid"):
        """
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU", "output"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type: str = activation_type

    def __call__(self, z):
        """
        This method allows your instances to be callable.
        """
        return self.forward(z)

    def forward(self, z) -> np.ndarray:
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(z)

        elif self.activation_type == "tanh":
            return self.tanh(z)

        elif self.activation_type == "ReLU":
            return self.ReLU(z)

        elif self.activation_type == "output":
            return self.output(z)

        raise Exception

    def backward(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            return self.grad_sigmoid(z)

        elif self.activation_type == "tanh":
            return self.grad_tanh(z)

        elif self.activation_type == "ReLU":
            return self.grad_ReLU(z)

        elif self.activation_type == "output":
            return self.grad_output(z)

        raise Exception

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(x))

    def tanh(self, x):
        return np.tanh(x)

    def ReLU(self, x):
        return x * (x > 0.0)

    def output(self, x: np.ndarray):
        """
        Softmax.

        Subtract maximum value for numerical stability
        """
        exp_x: np.ndarray = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def grad_sigmoid(self, x):
        """
        Compute the gradient for sigmoid here.
        """
        return self.sigmoid(x) * (1.0 - self.sigmoid(x))

    def grad_tanh(self, x):
        """
        Compute the gradient for tanh here.
        """
        return 1 - self.tanh(x) ** 2

    def grad_ReLU(self, x):
        """
        Compute the gradient for ReLU here.
        """
        return (x > 0.0) * 1

    def grad_output(self, x):
        """
        Deliberately returning 1 for output layer case since we don't multiply by any activation for final layer's delta. Feel free to use/disregard it
        """
        return x


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

    def forward(self, x) -> np.ndarray:
        """
        Compute the forward pass (activation of the weighted input) through the layer here and return it.
        """

        self.x = x  # (785)
        self.a = x @ self.w  # (785) * (785,10) -> (10,)
        z = self.activation(self.a)  #  (N,10)

        return z

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

    def forward_batch(self, x_batch):
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
        corr = np.argmax(self.y, axis=1) == np.argmax(targets, axis=1)
        return corr.sum()

    def backward_batch(self, l1: float, l2: float, gamma, targets):
        delta = self.output_loss(self.y, targets)
        for layer in reversed(self.layers):
            delta = layer.backward_batch(delta, self.learning_rate, gamma, l1, l2)

    def update_weights(self):
        for layer in self.layers:
            layer.update_weights()

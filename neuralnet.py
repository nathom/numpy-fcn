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

        # Placeholder for input. This can be used for computing gradients.
        self.x: np.ndarray | None = None

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
        exp_x: np.ndarray = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=0, keepdims=True)

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

    def __init__(self, in_units, out_units, activation):
        """
        Define the architecture and create placeholders.
        """

        # Randomly initialize weights
        self.w: np.ndarray = 0.01 * np.random.random((in_units + 1, out_units))

        self.x: np.ndarray | None = None  # Save the input to forward in this
        self.a: np.ndarray | None = None  # output without activation
        self.z: np.ndarray | None = None  # Output After Activation
        self.gradient: np.ndarray | None = None
        self.activation: Activation = activation

        self.dw: np.ndarray = np.zeros_like(self.w)
        self.velocity: np.ndarray = np.zeros_like(self.w)

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x) -> np.ndarray:
        """
        Compute the forward pass (activation of the weighted input) through the layer here and return it.
        """

        self.x = x  # (N,785)
        self.a = x @ self.w  # (N,785) * (785,10) -> (N,10)
        self.z = self.activation(self.a)  #  (N,10)
        return self.z

    def backward(
        self,
        next_delta: np.ndarray,  # (N,)
        learning_rate: float,
        momentum_gamma: float,
        regularization: float,
    ) -> np.ndarray:
        assert self.a is not None
        assert self.x is not None

        if self.activation.activation_type == "output":
            # output delta passed in by caller
            # see NeuralNetwork.backward()
            this_delta = next_delta
        else:
            this_delta = self.activation.backward(self.a) * next_delta

        # Accumulate gradient in mini batch
        # gradient = self.x.reshape((-1, 1)) @ this_delta.reshape((1, -1))
        self.gradient = self.x[:, np.newaxis] * this_delta
        if momentum_gamma > 0.0:
            self.dw *= momentum_gamma
        self.dw += learning_rate * self.gradient

        # Don't propagate the bias weight
        return self.w[:-1, :] @ this_delta

    def update_weights(self):
        self.w += self.dw
        self.dw = np.zeros_like(self.w)


class NeuralNetwork:
    """
    Create a Neural Network specified by the network configuration mentioned in the config yaml file.
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers: list[Layer] = []  # Store all layers in this list.
        self.num_layers = len(config["layer_specs"]) - 1  # Set num layers here
        self.x = None  # Save the input to forward in this
        self.y = None  # For saving the output vector of the model
        self.targets = None  # For saving the targets
        self.learning_rate: float = config["learning_rate"]

        # Add layers specified by layer_specs.
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                self.layers.append(
                    Layer(
                        config["layer_specs"][i],
                        config["layer_specs"][i + 1],
                        Activation(config["activation"]),
                    )
                )
            elif i == self.num_layers - 1:
                self.layers.append(
                    Layer(
                        config["layer_specs"][i],
                        config["layer_specs"][i + 1],
                        Activation("output"),
                    )
                )

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None) -> float | None:
        """
        TODO: Compute forward pass through all the layers in the network and return the loss.
        If targets are provided, return loss and accuracy/number of correct predictions as well.
        """
        output = x
        for layer in self.layers:
            output = layer.forward(np.append(output, 1))

        self.y = output
        if targets is not None:
            self.targets = targets
            return float(self.loss(output, targets))
        return None

    def loss(self, outputs, targets):
        """
        Compute the categorical cross-entropy loss and return it.
        """
        epsilon = 1e-15
        # avoid log(0.0)
        outputs = np.clip(outputs, epsilon, 1 - epsilon)
        return -np.sum(targets * np.log(outputs))

    def output_loss(self, outputs, targets):
        return targets - outputs

    def backward(self, gamma: float, targets: np.ndarray):
        """
        TODO: Implement backpropagation here by calling backward method of Layers class.
        Call backward methods of individual layers.
        """
        delta = self.output_loss(self.y, targets)  # (10,)
        for layer in reversed(self.layers):
            delta = layer.backward(delta, self.learning_rate, gamma, 0.0)

    def update_weights(self):
        for layer in self.layers:
            layer.update_weights()

    def new_batch(self):
        """Set the accumulated gradient on all layers to 0."""
        for layer in self.layers:
            layer.dw = np.zeros_like(layer.w)

import copy

import numpy as np

from neural_network import NeuralNetwork


def check_grad(model: NeuralNetwork, x_train, y_train):
    """
    Checks if gradients computed numerically are within O(epsilon**2)

    args:
        model
        x_train: Small subset of the original train dataset
        y_train: Corresponding target labels of x_train

    Prints gradient difference of values calculated via numerical approximation and backprop implementation
    """
    epsilon = 1e-2

    indices = [(0, 0, 0), (0, 1, 1), (0, -1, -1), (1, 0, 0), (1, 1, 1), (1, -1, -1)]
    actual_grads = []
    approx_grads = []
    diffs = []

    for layer, i, j in indices:
        model_copy = copy.deepcopy(model)

        # Obtain approximation
        model_copy.layers[layer].w[i][j] -= epsilon  # w = w - eps
        model_copy.forward(x_train[0].reshape((1, -1)))
        loss_1 = model_copy.current_loss(y_train[0])

        model_copy.layers[layer].w[i][j] += 2 * epsilon  # w = w + eps
        model_copy.forward(x_train[0].reshape((1, -1)))
        loss_2 = model_copy.current_loss(y_train[0])

        grad_approx = (loss_2 - loss_1) / (2 * epsilon)

        model.forward(x_train[0].reshape((1, -1)))

        # Obtain actual gradients
        delta = model.output_loss(model.y, y_train[0])  # (10,)

        delta2 = model.layers[1].backward(
            delta, model.learning_rate, momentum_gamma=0, l1=0, l2=0
        )
        grad_actual = None
        if layer == 1:
            grad_actual = -model.layers[1].get_gradient(delta)[i][j]

        delta = model.layers[0].backward(
            delta2, model.learning_rate, momentum_gamma=0, l1=0, l2=0
        )
        if layer == 0:
            grad_actual = -model.layers[0].get_gradient(delta2)[i][j]

        assert grad_actual is not None
        approx_grads.append(grad_approx)
        actual_grads.append(grad_actual)
        diffs.append(np.abs(grad_approx - grad_actual))

    for i in range(len(actual_grads)):
        print(
            "Actual Gradient: {: 8.12f}, Approximation Gradient {: 8.12f}, Absolute Difference {: 8.15f}".format(
                float(actual_grads[i]), float(approx_grads[i]), float(diffs[i])
            )
        )


def check_gradient(x_train, y_train, config):
    subsetSize = 10  # Feel free to change this
    sample_idx = np.random.randint(0, len(x_train), subsetSize)
    x_train_sample, y_train_sample = x_train[sample_idx], y_train[sample_idx]

    model = NeuralNetwork(config)
    check_grad(model, x_train_sample, y_train_sample)

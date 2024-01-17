import numpy as np
import copy

from neuralnet import NeuralNetwork


def check_grad(model, x_train, y_train):
    """
    TODO
        Checks if gradients computed numerically are within O(epsilon**2)

        args:
            model
            x_train: Small subset of the original train dataset
            y_train: Corresponding target labels of x_train

        Prints gradient difference of values calculated via numerical approximation and backprop implementation
    """
    epsilon = 1e-2

    indices = [(0, 0, 0), (0, 1, 1), (0, -1, -1), (1, 0, 0), (1, 1, 1), (1, -1, -1)]
    grads = []
    approxs = []
    diffs = []

    for layer, i, j in indices:
        model_copy = copy.deepcopy(model)

        # Obtain approximation
        model_copy.layers[layer].w[i][j] -= epsilon
        loss_1 = model_copy.forward(x_train[0], y_train[0])

        model_copy.layers[layer].w[i][j] += 2 * epsilon
        loss_2 = model_copy.forward(x_train[0], y_train[0])

        approx = (loss_2 - loss_1) / (2 * epsilon)

        model.forward(x_train[0], y_train[0])

        # Obtain actual gradients
        delta = model.output_loss(model.y, model.targets)  # (10,)

        delta = model.layers[1].backward(delta, model.learning_rate,
                                         momentum_gamma=0, l1=0, l2=0)
        if layer == 1:
            grad = - model.layers[1].gradient[i][j]

        delta = model.layers[0].backward(delta, model.learning_rate,
                                         momentum_gamma=0, l1=0, l2=0)
        if layer == 0:
            grad = - model.layers[0].gradient[i][j]

        approxs.append(approx)
        grads.append(grad)
        diffs.append(np.abs(approx - grad))

    for i in range(len(grads)):
        print("Actual Gradient: {: 8.12f}, Approximation Gradient {: 8.12f}, Absolute Difference {: 8.15f}".format(grads[i], approxs[i], diffs[i]))

    return 1
    raise NotImplementedError("check_grad not implemented.")


def check_gradient(x_train, y_train, config):
    subsetSize = 10  # Feel free to change this
    sample_idx = np.random.randint(0, len(x_train), subsetSize)
    x_train_sample, y_train_sample = x_train[sample_idx], y_train[sample_idx]

    model = NeuralNetwork(config)
    check_grad(model, x_train_sample, y_train_sample)

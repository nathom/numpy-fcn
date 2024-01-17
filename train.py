import copy

import numpy as np
from tqdm import tqdm

from neuralnet import NeuralNetwork


def model_train(
    model: NeuralNetwork, x_train, y_train, x_valid, y_valid, config
) -> tuple[NeuralNetwork, list[float], list[float], list[float], list[float]]:
    """
    TODO: Train your model here.
    Learns the weights (parameters) for our model
    Implements mini-batch SGD to train the model.
    Implements Early Stopping.
    Uses config to set parameters for training like learning rate, momentum, etc.

    args:
        model - an object of the NeuralNetwork class
        x_train - the train set examples
        y_train - the test set targets/labels
        x_valid - the validation set examples
        y_valid - the validation set targets/labels

    returns:
        the trained model
    """

    # Read in the esssential configs
    # batch_size = config["batch_size"]

    # x_train = append_bias(x_train)
    # x_valid = append_bias(x_valid)

    # average loss values logged every epoch
    train_epoch_losses: list[float] = []
    train_epoch_accuracy: list[float] = []
    val_epoch_losses: list[float] = []
    val_epoch_accuracy: list[float] = []

    # initialize with untrained performance 
    init_ta, init_tl = model_test(model, x_train, y_train)
    init_va, init_vl = model_test(model, x_valid, y_valid)

    train_epoch_losses.append(init_tl)
    val_epoch_losses.append(init_vl)
    train_epoch_accuracy.append(init_ta)
    val_epoch_accuracy.append(init_va)

    batch_size = config["batch_size"]
    epochs = config["epochs"]
    uses_momentum = config["momentum"]
    if uses_momentum:
        gamma = config["momentum_gamma"]
    else:
        gamma = 0.0

    uses_early_stop: bool = config["early_stop"]
    patience_limit: int = config["early_stop_epoch"] if uses_early_stop else 99999999999

    current_patience = 0
    current_best_loss = float("inf")
    current_best_model: NeuralNetwork | None = None

    # OVERRIDES
    # epochs = 1
    # batch_size = 128
    # patience_limit = 1
    print(f"Running with {epochs = }, {batch_size = }, {patience_limit = }")
    with tqdm(total=epochs, unit="epoch") as bar:
        for _ in range(epochs):
            train_loss = 0.0  # for loss
            correct = 0  # for accuracy
            n = 0  # total number of iterations

            for i in range(0, len(x_train), batch_size):
                x_train_batch = x_train[i : i + batch_size]
                y_train_batch = y_train[i : i + batch_size]

                for i, (x, y) in enumerate(zip(x_train_batch, y_train_batch)):
                    loss = model.forward(x, y)
                    assert loss is not None

                    train_loss += float(loss)
                    assert model.y is not None
                    if np.argmax(model.y) == np.argmax(y):
                        correct += 1
                    n += 1

                    # If we're on the last iteration of the batch, update weights
                    # otherwise only update the gradient accumulator
                    # See SGD (Algorithm 1) on homework
                    # u = i == x_train_batch.shape[0] - 1
                    model.backward(config["L1_penalty"], config["L2_penalty"], gamma, y)
                model.update_weights()

            epoch_loss = train_loss / n
            train_epoch_losses.append(epoch_loss)  # loss at end of epoch
            epoch_acc = correct / n
            train_epoch_accuracy.append(epoch_acc)  # accuracy at end of epoch

            # Evaluate on validation set
            val_loss = 0.0
            correct = 0
            n = x_valid.shape[0]
            for x, y in zip(x_valid, y_valid):
                loss = model.forward(x, y)
                assert loss is not None
                val_loss += loss
                assert model.y is not None
                if np.argmax(model.y) == np.argmax(y):
                    correct += 1

            val_loss_avg = val_loss / n

            val_epoch_losses.append(val_loss_avg)
            val_acc = correct / n
            val_epoch_accuracy.append(val_acc)

            if val_loss_avg < current_best_loss:
                current_best_loss = val_loss_avg
                if current_best_model is not None:
                    del current_best_model
                current_best_model = copy.deepcopy(model)
                current_patience = 0
            else:
                current_patience += 1

            if current_patience >= patience_limit:
                print(
                    f"Early stopping. Bad performance for more than {patience_limit} consecutive epochs."
                )
                break

            bar.desc = (
                f"T: {epoch_acc*100:.2f}%, V: {val_acc*100:.2f}%, P: {current_patience}"
            )
            bar.update(1)

    assert current_best_model is not None
    return (
        current_best_model,
        train_epoch_losses,
        train_epoch_accuracy,
        val_epoch_losses,
        val_epoch_accuracy,
    )


def model_train_fast(
    model: NeuralNetwork, x_train, y_train, x_valid, y_valid, config
) -> tuple[NeuralNetwork, list[float], list[float]]:
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    uses_momentum = config["momentum"]
    if uses_momentum:
        gamma = config["momentum_gamma"]
    else:
        gamma = 0.0

    uses_early_stop: bool = config["early_stop"]
    patience_limit: int = config["early_stop_epoch"] if uses_early_stop else 99999999999
    val_epoch_losses: list[float] = []
    val_epoch_accuracy: list[float] = []

    current_patience = 0
    current_best_loss = float("inf")
    current_best_model: NeuralNetwork | None = None

    # OVERRIDES
    # epochs = 1
    # batch_size = 128
    # patience_limit = 1
    print(f"Running with {epochs = }, {batch_size = }, {patience_limit = }")
    with tqdm(total=epochs, unit="epoch") as bar:
        for _ in range(epochs):
            for i in range(0, len(x_train), batch_size):
                x_train_batch = x_train[i : i + batch_size]
                y_train_batch = y_train[i : i + batch_size]

                for x, y in zip(x_train_batch, y_train_batch):
                    model.forward(x)
                    model.backward(gamma, y)

                model.update_weights()

            # Evaluate on validation set
            val_loss = 0.0
            correct = 0
            n = x_valid.shape[0]
            for x, y in zip(x_valid, y_valid):
                loss = model.forward(x, y)
                assert loss is not None
                val_loss += loss
                assert model.y is not None
                if np.argmax(model.y) == np.argmax(y):
                    correct += 1

            val_loss_avg = val_loss / n

            val_epoch_losses.append(val_loss_avg)
            val_acc = correct / n
            val_epoch_accuracy.append(val_acc)

            if val_loss_avg < current_best_loss:
                current_best_loss = val_loss_avg
                if current_best_model is not None:
                    del current_best_model
                current_best_model = copy.deepcopy(model)
                current_patience = 0
            else:
                current_patience += 1

            if current_patience >= patience_limit:
                print(
                    f"Early stopping. Bad performance for more than {patience_limit} consecutive epochs."
                )
                break

            bar.desc = f"V: {val_acc*100:.2f}%, P: {current_patience}"
            bar.update(1)

    assert current_best_model is not None
    return current_best_model, val_epoch_losses, val_epoch_losses


# This is the test method
def model_test(model: NeuralNetwork, X_test, y_test):
    """
    TODO
    Calculates and returns the accuracy & loss on the test set.

    args:
        model - the trained model, an object of the NeuralNetwork class
        X_test - the test set examples
        y_test - the test set targets/labels

    returns:
        test accuracy
        test loss
    """

    N = X_test.shape[0]
    loss = 0.0
    correct = 0
    for x, y in zip(X_test, y_test):
        _loss = model.forward(x, y)
        assert _loss is not None
        assert model.y is not None
        if np.argmax(model.y) == np.argmax(y):
            correct += 1
        loss += _loss

    return correct / N, loss / N

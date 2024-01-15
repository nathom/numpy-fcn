import numpy as np
from tqdm import trange

from neuralnet import NeuralNetwork
from util import append_bias


def model_train(model: NeuralNetwork, x_train, y_train, x_valid, y_valid, config):
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

    x_train = append_bias(x_train)
    x_valid = append_bias(x_valid)

    # average loss values logged every epoch
    train_epoch_losses: list[float] = []
    val_epoch_losses: list[float] = []
    train_epoch_accuracy: list[float] = []
    val_epoch_accuracy: list[float] = []

    batch_size = config["batch_size"]
    epochs = config["epochs"]
    epochs = 10
    print(f"Running with {epochs = }, {batch_size = }")
    for _ in trange(epochs):
        train_loss = 0.0  # for loss
        correct = 0  # for accuracy
        n = 0  # total number of iterations

        # Train model
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
                u = i == x_train_batch.shape[0] - 1
                model.backward(update_weights=u)

            model.new_batch()  # reset gradient accumulator

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
            val_loss += float(loss)
            assert model.y is not None
            if np.argmax(model.y) == np.argmax(y):
                correct += 1

        val_loss_avg = val_loss / n
        val_epoch_losses.append(val_loss_avg)
        val_acc = correct / n
        val_epoch_accuracy.append(val_acc)

        print(f"Train: {epoch_acc*100:.2f}%, Validate: {val_acc*100:.2f}%")

    return (
        model,
        train_epoch_losses,
        train_epoch_accuracy,
        val_epoch_losses,
        val_epoch_accuracy,
    )


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


def digit_show(x, y):
    print(f"Correct: {y}")
    import matplotlib.pyplot as plt

    plt.imshow(x.reshape(28, 28), cmap="gray")
    plt.axis("off")  # Turn off axis labels
    plt.show()

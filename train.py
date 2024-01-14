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
    batch_size = 128
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
                if (model.y == y).all():
                    correct += 1
                n += 1

                # If we're on the last iteration of the batch, update weights
                # otherwise just update the gradient accumulator
                # See SGD (Algorithm 1) on homework
                u = i == x_train_batch.shape[0] - 1
                model.backward(update_weights=u)

            model.new_batch()  # reset gradient accumulator

        train_epoch_losses.append(train_loss / n)  # loss at end of epoch
        train_epoch_accuracy.append(correct / n)  # accuracy at end of epoch

        # Evaluate on validation set
        val_loss = 0.0
        correct = 0
        n = x_valid.shape[0]
        for x, y in zip(x_valid, y_valid):
            loss = model.forward(x, y)
            assert loss is not None
            val_loss += float(loss)
            if (model.y == y).all():
                correct += 1
        val_epoch_losses.append(val_loss / n)
        val_epoch_accuracy.append(correct / n)

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

    losses = []
    correct = 0
    for x, y in zip(X_test, y_test):
        loss = model.forward(x, y)
        if (model.y == y).all():
            correct += 1
        losses.append(loss)
    return correct / len(y_test), losses

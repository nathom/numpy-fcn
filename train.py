import json

from tqdm import trange

from neuralnet import NeuralNetwork


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
    losses = []
    batch_size = 1
    for _ in range(config["epochs"]):
        for i in trange(0, len(x_train), batch_size):
            x_train_batch = x_train[i : i + batch_size]
            y_train_batch = y_train[i : i + batch_size]

            for x, y in zip(x_train_batch, y_train_batch):
                loss = model.forward(x, y)
                assert loss is not None
                losses.append(loss)

            model.backward()

    with open("loss.json", "w") as f:
        json.dump(losses, f)

    return model


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
        if model.y == y:
            correct += 1
        losses.append(loss)
    return correct / len(y_test), losses

import copy

from tqdm import tqdm

from neural_network import NeuralNetwork


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

    layers = config["layer_specs"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    uses_momentum = config["momentum"]
    l1 = config["L1_penalty"]
    l2 = config["L2_penalty"]

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
    print(
        f"Running with {layers = }, {epochs = }, {batch_size = }, {patience_limit = }, {gamma = }, {l1 = }, {l2 = }"
    )
    with tqdm(total=epochs, unit="epoch") as bar:
        for _ in range(epochs):
            train_loss = 0.0  # for loss
            correct = 0  # for accuracy
            n = 0  # total number of iterations

            for i in range(0, len(x_train), batch_size):
                x_train_batch = x_train[i : i + batch_size]
                y_train_batch = y_train[i : i + batch_size]

                model.forward(x_train_batch)
                losses = model.current_loss(y_train_batch)
                train_loss += losses.sum()
                correct += model.num_correct(y_train_batch)
                model.backward(gamma=gamma, targets=y_train_batch, l1=l1, l2=l2)
                model.update_weights()

            n = len(x_train)
            epoch_loss = train_loss / n
            train_epoch_losses.append(epoch_loss)  # loss at end of epoch
            epoch_acc = correct / n
            train_epoch_accuracy.append(epoch_acc)  # accuracy at end of epoch

            model.forward(x_valid)
            val_loss = model.current_loss(y_valid)
            correct = model.num_correct(y_valid)

            n = len(x_valid)
            val_loss_avg = val_loss.sum() / n
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
    model.forward(X_test)
    loss = model.current_loss(y_test).sum()
    correct = model.num_correct(y_test)
    return correct / N, loss / N

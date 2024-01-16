import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from numba import njit

import constants
from constants import save_location


def load_config(path):
    """
    Loads the config yaml from the specified path

    args:
        path - Complete path of the config yaml file to be loaded
    returns:
        yaml - yaml object containing the config file
    """
    return yaml.load(open(path, "r"), Loader=yaml.SafeLoader)


@njit
def normalize_data(inp):
    """
    TODO
    Normalizes image pixels here to have 0 mean and unit variance.

    args:
        inp : N X d 2D array where N is the number of examples and d is the number of dimensions

    returns:
        normalized inp: N X d 2D array

    """
    mean_val = np.mean(inp)
    std_dev = np.std(inp)

    normalized_data = (inp - mean_val) / std_dev

    return normalized_data


@njit
def one_hot_encoding(labels, num_classes=10):
    """
    TODO
    Encodes labels using one hot encoding.

    args:
        labels : N dimensional 1D array where N is the number of examples
        num_classes: Number of distinct labels that we have (10 for MNIST)

    returns:
        oneHot : N X num_classes 2D array
    """
    N = len(labels)
    labels = labels.astype(np.int32).flatten()
    ret = np.zeros((N, num_classes))
    # wow im so clever
    ret[range(N), labels] = 1.0
    return ret


def append_bias(X):
    """
    TODO
    Appends bias to the input
    args:
        X (N X d 2D Array)
    returns:
        X_bias (N X (d+1)) 2D Array
    """
    # return np.column_stack((X, np.ones((X.shape[0], 1))))
    return np.append(X, 1)


def plot(trainEpochLoss, trainEpochAccuracy, valEpochLoss, valEpochAccuracy, earlyStop):
    """
    Helper function for creating the plots
    """
    if not os.path.exists(constants.save_location):
        os.makedirs(constants.save_location)

    fig1, ax1 = plt.subplots()
    epochs = np.arange(1, len(trainEpochLoss) + 1, 1)
    ax1.plot(epochs, trainEpochLoss, "r", label="Training Loss")
    ax1.plot(epochs, valEpochLoss, "g", label="Validation Loss")
    if earlyStop is not None:
        plt.scatter(
            epochs[earlyStop],
            valEpochLoss[earlyStop],
            marker="x",
            c="g",
            s=400,
            label="Early Stop Epoch",
        )
    plt.xticks(ticks=np.arange(min(epochs), max(epochs) + 1, 10), fontsize=12)
    plt.yticks(fontsize=12)
    ax1.set_title("Loss Plots", fontsize=12.0)
    ax1.set_xlabel("Epochs", fontsize=12.0)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=12.0)
    ax1.legend(loc="upper right", fontsize=12.0)
    plt.savefig(constants.save_location + "loss.eps")
    plt.show()

    fig2, ax2 = plt.subplots()
    ax2.plot(epochs, trainEpochAccuracy, "r", label="Training Accuracy")
    ax2.plot(epochs, valEpochAccuracy, "g", label="Validation Accuracy")
    if earlyStop is not None:
        plt.scatter(
            epochs[earlyStop],
            valEpochAccuracy[earlyStop],
            marker="x",
            c="g",
            s=400,
            label="Early Stop Epoch",
        )
    plt.xticks(ticks=np.arange(min(epochs), max(epochs) + 1, 10), fontsize=12)
    plt.yticks(fontsize=12)
    ax2.set_title("Accuracy Plots", fontsize=12.0)
    ax2.set_xlabel("Epochs", fontsize=12.0)
    ax2.set_ylabel("Accuracy", fontsize=12.0)
    ax2.legend(loc="lower right", fontsize=12.0)
    plt.savefig(constants.save_location + "accuarcy.eps")
    plt.ylim(0, 1)
    plt.show()

    # Saving the losses and accuracies for further offline use
    pd.DataFrame(trainEpochLoss).to_csv(constants.save_location + "trainEpochLoss.csv")
    pd.DataFrame(valEpochLoss).to_csv(constants.save_location + "valEpochLoss.csv")
    pd.DataFrame(trainEpochAccuracy).to_csv(
        constants.save_location + "trainEpochAccuracy.csv"
    )
    pd.DataFrame(valEpochAccuracy).to_csv(
        constants.save_location + "valEpochAccuracy.csv"
    )


@njit
def train_validation_split(
    x_train, y_train, random_seed=42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    :return: x_train, y_train, x_validate, y_validate
    """
    # raise NotImplementedError("createTrainValSplit not implemented")
    # Assuming 'data' is your dataset and 'labels' are corresponding labels/targets
    # Specify the test_size to set the proportion of the dataset to include in the test split
    # Random_state ensures reproducibility, use a specific number or set to None for randomness

    np.random.seed(random_seed)
    N, w = x_train.shape
    train_size = int(N * 0.8)
    data = np.column_stack((x_train, y_train))
    assert data.shape == (N, w + 1)
    np.random.shuffle(data)
    return (
        data[:train_size, :w],
        data[:train_size, w:],
        data[train_size:, :w],
        data[train_size:, w:],
    )


def get_mnist():
    """
    This function code was taken from public repository: https://github.com/fgnt/mnist
    """
    # The code to download the mnist data original came from
    # https://cntk.ai/pythondocs/CNTK_103A_MNIST_DataLoader.html

    import gzip
    import os
    import struct
    from urllib.request import urlretrieve

    import numpy as np

    def load_data(src, num_samples):
        gzfname, h = urlretrieve(src, "./delete.me")
        try:
            with gzip.open(gzfname) as gz:
                n = struct.unpack("I", gz.read(4))
                # Read magic number.
                if n[0] != 0x3080000:
                    raise Exception("Invalid file: unexpected magic number.")
                # Read number of entries.
                n = struct.unpack(">I", gz.read(4))[0]
                if n != num_samples:
                    raise Exception(
                        "Invalid file: expected {0} entries.".format(num_samples)
                    )
                crow = struct.unpack(">I", gz.read(4))[0]
                ccol = struct.unpack(">I", gz.read(4))[0]
                if crow != 28 or ccol != 28:
                    raise Exception("Invalid file: expected 28 rows/cols per image.")
                # Read data.
                res = np.frombuffer(gz.read(num_samples * crow * ccol), dtype=np.uint8)
        finally:
            os.remove(gzfname)
        return res.reshape((num_samples, crow, ccol)) / 256

    def load_labels(src, num_samples):
        gzfname, h = urlretrieve(src, "./delete.me")
        try:
            with gzip.open(gzfname) as gz:
                n = struct.unpack("I", gz.read(4))
                # Read magic number.
                if n[0] != 0x1080000:
                    raise Exception("Invalid file: unexpected magic number.")
                # Read number of entries.
                n = struct.unpack(">I", gz.read(4))
                if n[0] != num_samples:
                    raise Exception(
                        "Invalid file: expected {0} rows.".format(num_samples)
                    )
                # Read labels.
                res = np.frombuffer(gz.read(num_samples), dtype=np.uint8)
        finally:
            os.remove(gzfname)
        return res.reshape((num_samples))

    def try_download(data_source, label_source, num_samples):
        data = load_data(data_source, num_samples)
        labels = load_labels(label_source, num_samples)
        return data, labels

    # Not sure why, but yann lecun's website does no longer support
    # simple downloader. (e.g. urlretrieve and wget fail, while curl work)
    # Since not everyone has linux, use a mirror from uni server.
    #     server = 'http://yann.lecun.com/exdb/mnist'
    server = "https://raw.githubusercontent.com/fgnt/mnist/master"

    # URLs for the train image and label data
    url_train_image = f"{server}/train-images-idx3-ubyte.gz"
    url_train_labels = f"{server}/train-labels-idx1-ubyte.gz"
    num_train_samples = 60000

    train_features, train_labels = try_download(
        url_train_image, url_train_labels, num_train_samples
    )

    # URLs for the test image and label data
    url_test_image = f"{server}/t10k-images-idx3-ubyte.gz"
    url_test_labels = f"{server}/t10k-labels-idx1-ubyte.gz"
    num_test_samples = 10000

    test_features, test_labels = try_download(
        url_test_image, url_test_labels, num_test_samples
    )

    return train_features, train_labels, test_features, test_labels


def load_data(path):
    """
    Loads, splits our dataset - MNIST into train, val and test sets and normalizes them

    args:
        path: Path to MNIST dataset
    returns:
        train_normalized_images, train_one_hot_labels, val_normalized_images, val_one_hot_labels,  test_normalized_images, test_one_hot_labels

    """

    # Make data directory
    if not os.path.exists(path):
        os.makedirs(path)

    raw_fn = "raw.pkl"
    raw_path = os.path.join(path, raw_fn)
    proc_fn = "processed.pkl"
    proc_path = os.path.join(path, proc_fn)
    # Check if raw data is cached, otherwise fetch and cache
    if not os.path.exists(raw_path):
        print("Fetching MNIST data...")
        train_features, train_labels, test_features, test_labels = get_mnist()
        # Save data using pickle
        with open(raw_path, "wb") as f:
            pickle.dump([train_features, train_labels, test_features, test_labels], f)
        print(f"Done. All raw data can be found in {path}")

    # Load raw data from pickle file
    with open(raw_path, "rb") as f:
        train_images, train_labels, test_images, test_labels = pickle.load(f)

    # Check if processed data is cached, otherwise process and cache
    if not os.path.exists(proc_path):
        print("Processing data")
        # Reformat the images and labels
        train_images, test_images = (
            train_images.reshape(train_images.shape[0], -1),
            test_images.reshape(test_images.shape[0], -1),
        )
        train_labels, test_labels = (
            np.expand_dims(train_labels, axis=1),
            np.expand_dims(test_labels, axis=1),
        )

        # Create 80-20 train-validation split
        train_images, train_labels, val_images, val_labels = train_validation_split(
            train_images, train_labels
        )

        # Preprocess data
        train_normalized_images = normalize_data(train_images)  # very expensive
        print(train_labels[0])
        train_one_hot_labels = one_hot_encoding(train_labels, num_classes=10)  # (n, 10)
        print(train_one_hot_labels[0])

        val_normalized_images = normalize_data(val_images)
        val_one_hot_labels = one_hot_encoding(val_labels, num_classes=10)  # (n, 10)

        test_normalized_images = normalize_data(test_images)
        test_one_hot_labels = one_hot_encoding(test_labels, num_classes=10)  # (n, 10)

        with open(os.path.join(path, "processed.pkl"), "wb") as g:
            pickle.dump(
                [
                    train_normalized_images,
                    train_one_hot_labels,
                    val_normalized_images,
                    val_one_hot_labels,
                    test_normalized_images,
                    test_one_hot_labels,
                ],
                g,
            )
        print(f"Done. All processed data can be found in {path}")

    # Load processed data from pickle file
    with open(proc_path, "rb") as f:
        (
            train_normalized_images,
            train_one_hot_labels,
            val_normalized_images,
            val_one_hot_labels,
            test_normalized_images,
            test_one_hot_labels,
        ) = pickle.load(f)

    return (
        train_normalized_images,
        train_one_hot_labels,
        val_normalized_images,
        val_one_hot_labels,
        test_normalized_images,
        test_one_hot_labels,
    )


def save_loss_accuracy(test_loss, test_acc, val_loss, val_acc):
    fn = os.path.join(save_location, "loss_acc_over_epochs.json")
    info = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }
    with open(fn, "w") as f:
        json.dump(info, f)

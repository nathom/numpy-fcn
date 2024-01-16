import argparse
import os
import pickle

import util
from constants import (
    config_dir,
    dataset_dir,
    models_dir,
)
from neuralnet import NeuralNetwork
from train import model_test, model_train, model_train_fast


# TODO
def main(args):
    # Read the required config
    # Create different config files for different experiments

    if args.config is None:
        raise Exception("Specify a config name to use")

    if not args.config.endswith("yaml"):
        args.config += ".yaml"

    config_fn = args.config

    # Load the data
    x_train, y_train, x_valid, y_valid, x_test, y_test = util.load_data(
        path=dataset_dir
    )

    # Load the configuration from the corresponding yaml file. Specify the file path and name
    assert config_dir is not None
    config = util.load_config(os.path.join(config_dir, config_fn))

    # Create a Neural Network object which will be our model
    model = NeuralNetwork(config)

    if args.load and args.save:
        raise Exception("Load and save flags cannot both be toggled.")

    if args.load:
        path = os.path.join(models_dir, args.load + ".pkl")
        if os.path.exists(path):
            print(f"Loading cached model from {path}")
            with open(path, "rb") as f:
                model, tl, ta, vl, va = pickle.load(f)
        else:
            raise Exception("File saved_model.pkl does not exist.")
    else:
        if args.save:
            path = os.path.join(models_dir, args.save + ".pkl")
            if os.path.exists(path):
                print(f"WARNING: {path} already exists. Overwriting.")
        else:
            path = None

        if args.fast:
            print(
                "WARNING: Training in fast mode. Training loss and accuracy not recorded."
            )
            model, vl, va = model_train_fast(
                model, x_train, y_train, x_valid, y_valid, config
            )
            tl, ta = [], []
        else:
            model, tl, ta, vl, va = model_train(
                model, x_train, y_train, x_valid, y_valid, config
            )

        if args.save:
            assert path is not None
            # Save cached model
            with open(path, "wb") as file:
                pickle.dump([model, tl, ta, vl, va], file)
                print(f"Trained model saved at {path}")

    # Model is loaded

    util.save_loss_accuracy(tl, ta, vl, va)

    if args.plot:
        # util.plot(tl, ta, vl, va, len(tl) - 1)
        util.plot(tl, ta, vl, va, None)

    # test the model
    test_acc, test_loss = model_test(model, x_test, y_test)

    # Print test accuracy and test loss
    print("Test Accuracy:", test_acc, " Test Loss:", test_loss)


if __name__ == "__main__":
    # Parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        # default="test_momentum",
        help="Specify the config that you want to run. Dont have to include .yaml extension",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the results",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Faster training, but no plot data.",
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Saves trained model as saved_model.pkl",
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Attempt to load model if saved_model.pkl exists.",
    )
    args = parser.parse_args()
    main(args)

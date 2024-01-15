import argparse
import os
import pickle

import gradient
import util
from constants import (
    config_dir,
    dataset_dir,
    models_dir,
)
from neuralnet import NeuralNetwork
from train import model_test, model_train

# install(show_locals=True)


# TODO
def main(args):
    # Read the required config
    # Create different config files for different experiments
    if args.experiment == "test_softmax":  # Rubric #4: Softmax Regression
        config_fn = "config_4.yaml"
    elif (
        args.experiment == "test_gradients"
    ):  # Rubric #5: Numerical Approximation of Gradients
        config_fn = "config_5.yaml"
    elif args.experiment == "test_momentum":  # Rubric #6: Momentum Experiments
        config_fn = "config_6.yaml"
    elif (
        args.experiment == "test_regularization"
    ):  # Rubric #7: Regularization Experiments
        raise NotImplementedError
    elif args.experiment == "test_activation":  # Rubric #8: Activation Experiments
        raise NotImplementedError
    elif args.experiment == "multilayer":  # Rubric #8: Activation Experiments
        config_fn = "multilayer.yaml"
    else:
        raise NotImplementedError

    # Load the data
    x_train, y_train, x_valid, y_valid, x_test, y_test = util.load_data(
        path=dataset_dir
    )

    # Load the configuration from the corresponding yaml file. Specify the file path and name
    assert config_fn is not None
    assert config_dir is not None
    config = util.load_config(os.path.join(config_dir, config_fn))

    if args.experiment == "test_gradients":
        gradient.check_gradient(x_train, y_train, config)
        return 1

    # Create a Neural Network object which will be our model
    model = NeuralNetwork(config)

    if args.load is not None and args.save is not None:
        raise Exception("Cannot load and save model")

    if args.load:
        path = os.path.join(models_dir, args.load + ".pkl")
        if not os.path.exists(path):
            raise Exception(f"File {path} does not exist.")

        print(f"Loading cached model from {path}")
        with open(path, "rb") as f:
            model, tl, ta, vl, va = pickle.load(f)
    else:
        path = os.path.join(models_dir, args.save + ".pkl")
        if os.path.exists(path):
            print(f"WARNING: {path} already exists. Overwriting.")

        model, tl, ta, vl, va = model_train(
            model, x_train, y_train, x_valid, y_valid, config
        )

        # Save cached model
        with open(path, "wb") as file:
            pickle.dump([model, tl, ta, vl, va], file)
            print(f"Trained model saved at {path}")

    util.save_loss_accuracy(tl, ta, vl, va)

    if args.plot:
        util.plot(tl, ta, vl, va, None)

    # test the model
    test_acc, test_loss = model_test(model, x_test, y_test)

    # Print test accuracy and test loss
    print("Test Accuracy:", test_acc, " Test Loss:", test_loss)


if __name__ == "__main__":
    # Parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        default="test_momentum",
        help="Specify the experiment that you want to run",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the results",
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Save model with name.",
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Load model with name.",
    )
    args = parser.parse_args()
    main(args)

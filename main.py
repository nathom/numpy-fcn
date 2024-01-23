import argparse
import os
import pickle

import numpy as np

import gradient
import util
from constants import (
    config_dir,
    dataset_dir,
    models_dir,
)
from neural_network import NeuralNetwork
from slideshow import show_slideshow
from train import model_test, model_train


def main(args):
    if args.config is None:
        raise Exception("Specify a config name to use")

    if not args.config.endswith("yaml"):
        args.config += ".yaml"

    config_fn = args.config

    x_train, y_train, x_valid, y_valid, x_test, y_test = util.load_data(
        path=dataset_dir
    )

    config = util.load_config(os.path.join(config_dir, config_fn))

    model = NeuralNetwork(config)

    if args.load and args.save:
        raise Exception("Load and save flags cannot both be toggled.")

    if args.grad:
        gradient.check_gradient(x_train, y_train, config)
        return 1

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

        model, tl, ta, vl, va = model_train(
            model, x_train, y_train, x_valid, y_valid, config
        )

        if args.save:
            for layer in model.layers:
                del layer.a
                del layer.x
                del layer.dw

            assert path is not None
            # Save cached model
            with open(path, "wb") as file:
                pickle.dump([model, tl, ta, vl, va], file)
                print(f"Trained model saved at {path}")

    if args.plot:
        if config["early_stop"]:
            es = len(tl) - 1
        else:
            es = None
        util.plot(tl, ta, vl, va, es)

    test_acc, test_loss = model_test(model, x_test, y_test)

    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test loss: {test_loss:.2f}")

    if args.show_fails:
        model.forward(x_test)
        y_hats, ys, inds = model.get_failed_indices(y_test)
        titles = [f"Guess: {y_hat} Correct: {y}" for y_hat, y in zip(y_hats, ys)]
        imgs = x_test[inds]
        show_slideshow(imgs, titles)
        util.tile_images(imgs[:42], titles[:42])

    if args.compare:
        path = os.path.join(models_dir, args.compare + ".pkl")
        if os.path.exists(path):
            print(f"Loading cached model from {path}")
            with open(path, "rb") as f:
                model2, tl, ta, vl, va = pickle.load(f)
        else:
            raise Exception("File saved_model.pkl does not exist.")

        model.forward(x_test)
        model2.forward(x_test)
        _, _, inds = model.get_failed_indices(y_test)
        _, _, inds2 = model2.get_failed_indices(y_test)

        imgs = []
        titles = []
        for i, (i1, i2) in enumerate(zip(inds, inds2)):
            if not i1 and not i2:
                continue
            g1 = np.argmax(model.y[i])
            g2 = np.argmax(model2.y[i])
            c = np.argmax(y_test[i])
            imgs.append(x_test[i])
            if g1 == c:
                winner = args.load
            elif g2 == c:
                winner = args.compare
            else:
                winner = "NOBODY"
            titles.append(
                f"Winner: {winner}\n{args.load} guessed {g1}. {args.compare} guessed {g2}. Correct is {c}"
            )

        show_slideshow(imgs, titles)

    if args.tile:
        model.forward(x_test)
        y_hats, ys, inds = model.get_failed_indices(y_test)
        loss = model.current_loss(y_test)
        sort_loss_inds = np.argsort(loss)[-42:]
        imgs = []
        titles = []
        for i in sort_loss_inds:
            imgs.append(x_test[i])
            g = np.argmax(model.y[i])
            c = np.argmax(y_test[i])
            titles.append(f"Guess: {g} Correct: {c}")

        util.tile_images(imgs, titles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Specify the config that you want to run. Dont have to include .yaml extension",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the results",
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

    parser.add_argument(
        "--grad",
        action="store_true",
        help="Runs numerical approximation test.",
    )
    parser.add_argument(
        "--show-fails",
        action="store_true",
        help="Show slideshow of failed digits.",
    )
    parser.add_argument(
        "--compare",
        type=str,
        help="Compare the performance of this model with the one loaded in a slideshow.",
    )
    parser.add_argument(
        "--tile",
        action="store_true",
        help="Tile the images that were confusing.",
    )
    args = parser.parse_args()
    main(args)

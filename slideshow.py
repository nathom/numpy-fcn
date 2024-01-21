import matplotlib.pyplot as plt
import numpy as np

images: np.ndarray
img_titles: list[str]


def show_slideshow(imgs: np.ndarray | list[np.ndarray], titles: list[str]):
    i = 0

    def fig1(fig):
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.set_title(f"[{i}/{len(titles)-1}] " + titles[i])
        ax.imshow(imgs[i].reshape(28, 28), cmap="gray")

    def onclick1(event, fig):
        nonlocal i
        fig.clear()
        if event.button == 1:
            i += 1
        elif event.button == 3:
            i -= 1
        i %= len(imgs)
        fig1(fig)
        plt.draw()

    fig = plt.figure()
    fig1(fig)
    fig.canvas.mpl_connect("button_press_event", lambda event: onclick1(event, fig))

    plt.show()

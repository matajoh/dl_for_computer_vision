import math

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from datasets import (load_emnist, load_mnist,
                      load_mnist_patches, load_emnist_patches)


def analyze(patches, gt: np.ndarray, num_samples=10000):
    tokens = patches["tokens"]

    order = np.arange(len(patches["val"]))[:num_samples]
    np.random.shuffle(order)
    sum_psnr = 0
    for i, o in tqdm(enumerate(order), total=len(order)):
        original = gt[o]
        pixels = np.zeros((28, 28), np.uint8)
        x = patches["val"][o]
        for j, code in enumerate(x):
            r = (j // 7) * 4
            c = (j % 7) * 4
            pixels[r:r+4, c:c+4] = tokens[code].reshape(4, 4)

        pixels_f = pixels.astype(np.float32) / 255
        original_f = original.astype(np.float32) / 255
        mse = np.square(pixels_f - original_f).mean()
        psnr = -10 * math.log10(mse)
        sum_psnr += psnr

    print(f"Average PSNR: {sum_psnr / len(order)}")


def figures(patches, gt: np.ndarray):
    tokens = patches["tokens"]

    rows = int(math.sqrt(len(tokens)) / math.sqrt(2))
    cols = len(tokens) // rows
    if rows * cols < len(tokens):
        rows += 1

    plt.rc("font", size=15)
    fig = plt.figure(figsize=(cols, rows))
    for i, t in enumerate(tokens):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(t.reshape(4, 4), cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    for i in range(8):
        original = gt[i]
        pixels = np.zeros((28, 28), np.uint8)
        x = patches["val"][i]
        for j, code in enumerate(x):
            r = (j // 7) * 4
            c = (j % 7) * 4
            pixels[r:r+4, c:c+4] = tokens[code].reshape(4, 4)

        compare = np.zeros((28, 56), np.uint8)
        compare[:, :28] = original
        compare[:, 28:] = pixels
        ax = plt.subplot(2, 4, i + 1)
        ax.imshow(compare, cmap="gray", interpolation="nearest")
        ax.set_xticks([14, 42], ["Original", "Quantised"])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def mnist_patches(K=64):
    """Load MNIST patch dictionary."""
    patches = load_mnist_patches(K)
    mnist = load_mnist()
    gt = mnist["data"][60000:].reshape(-1, 28, 28)
    return patches, gt


def emnist_patches(K=128):
    """Load EMNIST patch dictionary."""
    patches = load_emnist_patches(K)
    emnist = load_emnist()
    gt = emnist["test_images"].reshape(-1, 28, 28)
    return patches, gt


if __name__ == "__main__":
    # To regenerate patch dictionaries from scratch, delete the files
    # from data/ first. Note: regenerating will invalidate any
    # transformer/RNN models trained with the old dictionaries.

    # Computation
    mnist_p, mnist_gt = mnist_patches()
    emnist_p, emnist_gt = emnist_patches()

    # Visualization
    analyze(mnist_p, mnist_gt)
    figures(mnist_p, mnist_gt)
    analyze(emnist_p, emnist_gt)
    figures(emnist_p, emnist_gt)

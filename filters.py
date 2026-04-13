import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import MulticlassDataset


blur_kernel = np.array([[1, 4, 7, 4, 1],
                        [4, 16, 26, 16, 4],
                        [7, 26, 41, 26, 7],
                        [4, 16, 26, 16, 4],
                        [1, 4, 7, 4, 1]], dtype='float32')
blur_kernel /= 273

horizontal_edge_kernel = np.array([[1, 4, 0, -4, 1],
                                   [4, 16, 0, -16, -4],
                                   [7, 26, 0, -26, -7],
                                   [4, 16, 0, -16, -4],
                                   [1, 4, 0, -4, -1]], dtype='float32')
horizontal_edge_kernel /= 166

horizontal_bar_kernel = np.array([[0, 0, 0, 0, 0],
                                 [-2, -8, -13, -8, -2],
                                 [4, 16, 26, 16, 4],
                                 [-2, -8, -13, -8, -2],
                                 [0, 0, 0, 0, 0]], dtype='float32')
horizontal_bar_kernel /= 132

blob_kernel = np.array([[0, 1, 1, 2, 2, 2, 1, 1, 0],
                       [1, 2, 4, 5, 5, 5, 4, 2, 1],
                       [1, 4, 5, 3, 0, 3, 5, 4, 1],
                       [2, 5, 3, -12, -24, -12, 3, 5, 2],
                       [2, 5, 0, -24, -40, -24, 0, 5, 2],
                       [2, 5, 3, -12, -24, -12, 3, 5, 2],
                       [1, 4, 5, 3, 0, 3, 5, 4, 1],
                       [1, 2, 4, 5, 5, 5, 4, 2, 1],
                       [0, 1, 1, 2, 2, 2, 1, 1, 0]], dtype='float32')
blob_kernel /= np.sum(np.abs(blob_kernel))


def show_kernel(ax, kernel, label):
    ax.imshow(kernel, interpolation="nearest", cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(label, fontsize="xx-large")


def show_kernels():
    fig = plt.figure(figsize=(10, 2))
    ax = fig.add_subplot(1, 4, 1)
    show_kernel(ax, blur_kernel, "Blur")

    ax = fig.add_subplot(1, 4, 2)
    show_kernel(ax, horizontal_edge_kernel, "Edge")

    ax = fig.add_subplot(1, 4, 3)
    show_kernel(ax, horizontal_bar_kernel, "Bar")

    ax = fig.add_subplot(1, 4, 4)
    show_kernel(ax, blob_kernel, "Blob")

    plt.tight_layout()
    plt.show()


def convolve(image, kernel, pad=0, stride=1):
    """
    Convolve an image with a kernel.

    Arguments:

    image -- a 2D ndarray containing grayscale pixel values
    kernel -- a 2D ndarray containing filter values

    Keyword arguments:

    pad -- Amount of space to add on the edges of the image
    stride -- Frequency with which to apply the filter
    """
    rows, columns = image.shape
    output_rows = rows // stride
    output_columns = columns // stride
    result = np.zeros((output_rows, output_columns))
    if pad > 0:
        # create a new image with zero padding around the edges
        image = np.pad(image, pad, 'constant')

    kernel_size = kernel.size
    kernel_length = kernel.shape[0]
    half_kernel = kernel_length // 2
    kernel_flat = kernel.reshape(kernel_size, 1)  # flattening out the 2D kernel will aid in computation later
    offset = abs(half_kernel - pad)  # This ensures we are never sampling outside of the image
    for r in range(offset, rows - offset, stride):
        for c in range(offset, columns - offset, stride):
            rr = r - half_kernel + pad
            cc = c - half_kernel + pad
            patch = image[rr:rr + kernel_length, cc:cc + kernel_length]
            # flatten out the patch and then just do a dot product
            val = np.dot(patch.reshape(1, kernel_size), kernel_flat)
            result[r//stride, c//stride] = val[0, 0]

    return result


def show_convolution(kernel, stride=1):
    """Displays the effect of convolving with the given kernel."""
    fig = plt.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(3, 3, height_ratios=[3, 1, 3])
    dataset = MulticlassDataset.mnist()
    examples = dataset.train.values[:3]
    for i, image in enumerate(examples):
        image = image.reshape(28, 28)
        padding = kernel.shape[0] // 2
        conv = convolve(image, kernel, padding, stride)
        ax = fig.add_subplot(gs[i])
        plt.imshow(image, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        ax = fig.add_subplot(gs[i + 3])
        plt.imshow(kernel, cmap='gray', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        ax = fig.add_subplot(gs[i + 6])
        plt.imshow(conv, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()


def show_convolutions():
    show_convolution(blur_kernel)
    show_convolution(horizontal_edge_kernel)
    show_convolution(horizontal_bar_kernel)
    show_convolution(blob_kernel)
    plt.show()


from typing import NamedTuple


ConvFrame = NamedTuple("ConvFrame", [("loc", np.ndarray), ("patch", np.ndarray), ("output", np.ndarray)])


def conv2d_frames():
    """Precompute all frames for an interactive 2D convolution animation."""
    dataset = MulticlassDataset.mnist()
    input_image = dataset.train.values[0].reshape(28, 28)
    output_image = np.zeros_like(input_image)
    input_image = np.pad(input_image, [[2, 2], [2, 2]], mode='constant')
    np.random.seed(42)
    filter_image = np.random.normal(size=(5, 5)).astype(np.float32)
    filter_image /= np.linalg.norm(filter_image, ord=2)

    frames = []
    for row in range(28):
        for col in range(28):
            patch = input_image[row:row+5, col:col+5].copy()
            output_image[row, col] = np.sum(patch * filter_image)
            frames.append(ConvFrame(np.array([col, row]), patch, output_image.copy()))

    return input_image, filter_image, frames


def plot_conv2d_frame(fig, input_image, filter_image, frames, index):
    """Plot a single frame of the convolution animation."""
    import matplotlib.patches as mpatches

    frame = frames[index]
    vmin = np.min(frames[-1].output)
    vmax = np.max(frames[-1].output)
    grid = plt.GridSpec(2, 3, width_ratios=[6, 2, 6])
    input_ax = fig.add_subplot(grid[:, 0])
    input_ax.imshow(input_image[2:-2, 2:-2], interpolation='nearest', cmap='gray')
    input_rect = mpatches.Rectangle(frame.loc - 2, 5, 5, edgecolor='r', fill=False)
    input_ax.add_patch(input_rect)
    input_ax.set_xticks([])
    input_ax.set_yticks([])
    input_ax.set_title("input")
    filter_ax = fig.add_subplot(grid[0, 1])
    filter_ax.imshow(filter_image, interpolation='nearest', cmap='gray')
    filter_ax.set_xticks([])
    filter_ax.set_yticks([])
    filter_ax.set_title("filter")
    patch_ax = fig.add_subplot(grid[1, 1])
    patch_ax.imshow(frame.patch, interpolation='nearest', cmap='gray')
    patch_ax.set_xticks([])
    patch_ax.set_yticks([])
    patch_ax.set_title("patch")
    output_ax = fig.add_subplot(grid[:, 2])
    output_ax.imshow(frame.output, interpolation='nearest', cmap='gray', vmin=vmin, vmax=vmax)
    output_rect = mpatches.Rectangle(frame.loc - 0.5, 1, 1, edgecolor='r', fill=False)
    output_ax.add_patch(output_rect)
    output_ax.set_xticks([])
    output_ax.set_yticks([])
    output_ax.set_title("output")
    plt.tight_layout()


alien = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                  [0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                  [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                  [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype='float32')


def display_image(ax, x, label, vrange=None):
    assert len(x.shape) == 2, str(x.shape)

    if vrange is None:
        vmin = np.min(x)
        vmax = np.max(x)
    else:
        vmin, vmax = vrange

    ax.imshow(x, interpolation='nearest', cmap='gray', extent=[0, x.shape[1], x.shape[0], 0], vmin=vmin, vmax=vmax)
    ax.grid(True)
    ax.set_xticks(np.arange(x.shape[1]))
    ax.xaxis.set_ticklabels([])
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(x.shape[0]))
    ax.yaxis.set_ticklabels([])
    ax.set_xlabel(label)


def train_transposed_convolution():
    image = torch.from_numpy(2 * alien - 1).unsqueeze(0).unsqueeze(0)
    conv = nn.Conv2d(1, 2, kernel_size=3, padding=1, stride=2, bias=False, padding_mode="reflect")
    tconv = nn.ConvTranspose2d(2, 1, kernel_size=3, padding=1, stride=2, bias=False)
    module = nn.Sequential(conv, tconv, nn.Tanh())

    optim = torch.optim.Adam(module.parameters(), lr=0.01)

    for i in range(5000):
        optim.zero_grad()
        o = module(image)
        loss = F.mse_loss(o, image)
        loss.backward()
        optim.step()

        if i % 100 == 0:
            print(i, loss.item())

    c = conv.weight[:, 0].detach().numpy()
    t = tconv.weight[:, 0].detach().numpy()
    z = conv(image).detach().squeeze().numpy()
    result = module(image).detach().squeeze().numpy()
    return {"c": c, "t": t, "z": z, "result": result}


def show_transposed_convolution(data):
    c = data["c"]
    t = data["t"]
    z = data["z"]
    result = data["result"]

    plt.rc('font', size=15)
    fig = plt.figure(figsize=(10, 4))
    gs = fig.add_gridspec(2, 3, width_ratios=[13, 3, 7])

    display_image(fig.add_subplot(gs[:, 0]), alien, "Input")
    display_image(fig.add_subplot(gs[0, 1]), c[0], "C0", (c.min(), c.max()))
    display_image(fig.add_subplot(gs[1, 1]), c[1], "C1", (c.min(), c.max()))
    display_image(fig.add_subplot(gs[0, 2]), z[0], "Z0", (z.min(), z.max()))
    display_image(fig.add_subplot(gs[1, 2]), z[1], "Z1", (z.min(), z.max()))
    fig.tight_layout()

    fig = plt.figure(figsize=(10, 4))
    gs = fig.add_gridspec(2, 3, width_ratios=[7, 3, 13])

    display_image(fig.add_subplot(gs[0, 0]), z[0], "Z0", (z.min(), z.max()))
    display_image(fig.add_subplot(gs[1, 0]), z[1], "Z1", (z.min(), z.max()))
    display_image(fig.add_subplot(gs[0, 1]), t[0], "T0", (t.min(), t.max()))
    display_image(fig.add_subplot(gs[1, 1]), t[1], "T1", (t.min(), t.max()))
    display_image(fig.add_subplot(gs[:, 2]), result, "Output")
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    # Computation
    tconv_data = train_transposed_convolution()

    # Visualization
    show_kernels()
    show_convolutions()
    show_transposed_convolution(tconv_data)

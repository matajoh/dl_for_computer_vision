import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def pool(x, size, stride, reduce):
    rows, cols = x.shape
    pad = size // 2
    output_rows = (rows + 2*pad) // stride
    output_cols = (cols + 2*pad) // stride

    x = np.pad(x, pad, 'constant')

    o = np.zeros((output_rows, output_cols), dtype='float32')

    for r in range(0, rows, stride):
        for c in range(0, cols, stride):
            o[r//stride, c//stride] = reduce(x[r:r+size, c:c+size])

    return o


def add_input_references(ax):
    ax.plot([0.5, 1.5, 0.5, 1.5], [0.5, 0.5, 1.5, 1.5], 'b+', ms=12, mew=4)
    ax.plot([3.5, 4.5, 5.5, 3.5, 4.5, 5.5, 3.5, 4.5, 5.5], [
            5.5, 5.5, 5.5, 6.5, 6.5, 6.5, 7.5, 7.5, 7.5], 'r.', ms=12, mew=4)
    ax.plot([9.5, 10.5, 11.5, 9.5, 10.5, 11.5, 9.5, 10.5, 11.5], [
            3.5, 3.5, 3.5, 4.5, 4.5, 4.5, 5.5, 5.5, 5.5], 'g*', ms=12, mew=4, mec='g')


def add_output_references(ax):
    ax.plot(0.5, 0.5, 'b+', ms=12, mew=4)
    ax.plot(2.5, 3.5, 'r.', ms=12, mew=4)
    ax.plot(5.5, 2.5, 'g*', ms=12, mew=4, mec='g')


def display_image(ax, x, label):
    ax.imshow(1 - x, interpolation='nearest', cmap='gray', extent=[0, x.shape[1], x.shape[0], 0])
    ax.grid(True)
    ax.set_xticks(np.arange(x.shape[1]))
    ax.xaxis.set_ticklabels([])
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(x.shape[0]))
    ax.yaxis.set_ticklabels([])
    ax.set_xlabel(label, fontsize="xx-large")
    if label == 'i':
        add_input_references(ax)
    else:
        add_output_references(ax)


def display_images(i, o):
    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
    display_image(fig.add_subplot(gs[0]), i, "i")
    display_image(fig.add_subplot(gs[1]), o, "o")
    plt.tight_layout()


i = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
              [0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0],
              [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
              [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
              [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
              [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype='float32')


def avg_pool():
    o = pool(i, 3, 2, np.average)
    display_images(i, o)
    plt.show()


def max_pool():
    o = pool(i, 3, 2, np.max)
    display_images(i, o)
    plt.show()


def main():
    avg_pool()
    max_pool()


if __name__ == "__main__":
    main()

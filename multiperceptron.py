from typing import List, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from datasets import Data, MulticlassDataset
from activations import tanh, dtanh


def softmax(x):
    x_max = x.max(axis=1, keepdims=True)
    x_exp = np.exp(x - x_max)  # this avoids numerical issues
    x_exp_sum = x_exp.sum(axis=1, keepdims=True)
    return x_exp / x_exp_sum


def linear(x, W, b):
    return np.dot(x, W.T) + b


def cross_entropy_loss(p, t):
    p_of_t = p[np.arange(len(t)), t]
    log_prob = np.log(p_of_t + 1e-10)
    return -log_prob.mean()


def dcross_entropy_loss(p, t):
    labels = np.zeros_like(p)
    labels[np.arange(len(t)), t] = 1
    return (p - labels) / len(t)


def dlinear(x, W, b, dy):
    dx = np.dot(dy, W)
    dW = np.dot(dy.T, x)
    db = dy.sum(axis=0)
    return dx, dW, db


def plot_decision_boundaries(ax: plt.Axes, data: Data, W: np.ndarray, b: np.ndarray, batch: np.ndarray):
    """
    Plots the provided points along with class-wise decision boundaries between them defined by the provided weights

    Arguments:

    data -- ndarray (num_examples, num_dims) of data points
    labels -- ndarray (num_examples, 1) with class labels for all points
    W -- the weights of the classifier
    b -- the biases of the classifier
    """

    colors = ['b', 'r', 'g']
    markers = ['+', '.', '*']

    if batch is not None:
        data.plot(ax, ['gray', 'gray', 'gray'], markers)
        data.subset(batch).plot(ax, colors, markers)
    else:
        data.plot(ax, colors, markers)

    for i in range(data.num_classes):
        weights = W[i]
        bias = b[i]
        if np.isclose(weights[1], 0):
            if np.isclose(weights[0], 0):
                x = y = np.array(data.limits, dtype='float32')
            else:
                y = np.array(data.limits, dtype='float32')
                x = -(weights[1] * y + bias)/weights[0]
        else:
            x = np.array(data.limits, dtype='float32')
            y = -(weights[0] * x + bias)/weights[1]

        ax.plot(x, y, colors[i], linewidth=2.0)


class Snapshot(NamedTuple("Snapshot", [("num_instances", int), ("accuracy", float), ("loss", float),
                                       ("weights", np.ndarray), ("biases", np.ndarray), ("batch", np.ndarray)])):
    def plot(self, ax: plt.Axes, data: MulticlassDataset, include_acc=True, show_batch=False):
        if show_batch:
            plot_decision_boundaries(ax, data.train, self.weights, self.biases, self.batch)
        else:
            plot_decision_boundaries(ax, data.val, self.weights, self.biases, None)

        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        font = {'family': 'sans-serif',
                'color':  'black',
                'weight': 'normal',
                'size': 15,
                }

        if include_acc:
            ax.text(-4, 3.7, "Acc: {:.2f}".format(self.accuracy), fontdict=font)


def forward(x: np.ndarray, W: np.ndarray, b: np.ndarray):
    h = linear(x, W, b)
    s = tanh(h)
    return softmax(s)


def train(data: MulticlassDataset, learning_rate=0.1, num_epochs=1, batch_size=10, report_frequency=0) -> List[Snapshot]:
    """
    Trains a perceptron

    Arguments:

    data -- the dataset, in the form of a dictionary of ['train': (data, labels), 'val': (data, labels)]

    Keyword arguments:

    learning_rate -- Rate at which gradients are applied
    num_epochs -- Number of times to show all the data to the model
    batch_size -- The size of the batches to show the model

    Returns:

    List of snapshots in the form [(instances, weights_0, val_accuracy_0), (instances, weights_1, val_accuracy_1), ...]    
    """

    train_data, train_labels = data.train
    val_data, val_labels = data.val
    num_train = train_data.shape[0]
    num_dims = train_data.shape[1]
    num_classes = np.max(train_labels) + 1
    W = np.zeros((num_classes, num_dims), dtype='float32')
    b = np.zeros((num_classes,), dtype='float32')

    np.set_printoptions(precision=2, suppress=True)

    snapshots: List[Snapshot] = []

    def evaluate_model():
        p = forward(val_data, W, b)
        loss = cross_entropy_loss(p, val_labels)

        actual = np.argmax(p, axis=1)
        accuracy = (actual == val_labels).sum() / val_data.shape[0]
        return accuracy, loss

    num_instances = 0
    last_accuracy = -1.0
    last_instances = 0

    def do_report(accuracy, num_instances):
        if report_frequency == 0:
            accuracy_freq = .01
            instances_freq = num_train / 5

            accuracy_diff = accuracy - last_accuracy
            instances_diff = num_instances - last_instances
            return accuracy_diff >= accuracy_freq or instances_diff >= instances_freq
        
        if num_instances - last_instances >= report_frequency:
            return True       

    accuracy, loss = evaluate_model()
    snapshots.append(Snapshot(num_instances, accuracy, loss, np.copy(W), np.copy(b), np.random.choice(num_train, batch_size)))

    for epoch in range(num_epochs):
        perm = np.random.permutation(num_train)

        batches = tqdm(range(0, num_train, batch_size), desc="Epoch {}".format(epoch)) 
        for i in batches:
            accuracy, loss = evaluate_model()
            batches.set_postfix({"accuracy": "{:.3f}".format(accuracy), "loss": "{:.3f}".format(loss)})
            x = np.asarray(train_data[perm[i:i + batch_size]])
            if do_report(accuracy, num_instances):
                snapshots.append(Snapshot(num_instances, accuracy, loss, np.copy(W), np.copy(b), np.copy(perm[i:i + batch_size])))
                last_accuracy = accuracy
                last_instances = num_instances

            t = np.asarray(train_labels[perm[i:i + batch_size]])
            h = linear(x, W, b)
            s = tanh(h)
            p = forward(x, W, b)
            loss = cross_entropy_loss(p, t)                

            dloss = dcross_entropy_loss(p, t)
            ds = dtanh(s, dloss)
            _, dW, db = dlinear(x, W, b, ds)

            W -= learning_rate * dW
            b -= learning_rate * db
            num_instances += x.shape[0]

    accuracy, loss = evaluate_model()
    snapshots.append(Snapshot(num_instances, accuracy, loss, W, b, None))
    return snapshots


def three_class():
    data = MulticlassDataset.generate_three_class(200, 100)
    snapshots = train(data, 0.1, 2, 10)

    plt.rc('font', size=15)
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(2, 2, 1)
    data.plot(ax)
    ax.set_title("Dataset")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = plt.subplot(2, 2, 2)
    snapshots[0].plot(ax, data)
    ax.set_title("Initial")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = plt.subplot(2, 2, 3)
    snapshots[2].plot(ax, data)
    ax.set_title(str(snapshots[2].num_instances))
    ax.set_xticks([])
    ax.set_yticks([])

    ax = plt.subplot(2, 2, 4)
    snapshots[-1].plot(ax, data)
    ax.set_title(str(snapshots[-1].num_instances))
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def uncertainty():
    data = MulticlassDataset.generate_two_class(200, 100)
    snapshots = train(data, 0.1, 2, 10)

    p = forward(data.train.values, snapshots[-1].weights, snapshots[-1].biases)
    start = p[:, 0].argmax()
    end = p[:, 1].argmax()

    stops = [0, 0.45, 0.55, 1]
    start = np.array([0, 3.5], np.float32)
    end = np.array([2, -3.5], np.float32)
    dir = end - start

    plt.rc('font', size=15)
    plt.figure(figsize=(8, 8))

    for i, stop in enumerate(stops):
        ax = plt.subplot(2, 2, i+1)
        point = start + stop * dir
        p = forward(point.reshape(1, 2), snapshots[-1].weights, snapshots[-1].biases)[0]
        print(p)
        plot_decision_boundaries(ax, data.train, snapshots[-1].weights, snapshots[-1].biases, None)
        ax.plot(point[0], point[1], 's', c="magenta", ms='15', mew='2.0')
        bins = [-4, -3]
        ax.bar(bins, p, color=['b', 'r'], width=1)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_title(f"p = [{p[0]:.2f}, {p[1]:.2f}]")     
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def plot_mnist(fig: plt.Figure, data: MulticlassDataset, snapshots: List[Snapshot], step: int, max_steps: int):
    plt.rc('font', size=15)
    gs = fig.add_gridspec(1, 3, width_ratios=[2, 1, 2])
    num_instances, accuracy, loss, W, b, batch = snapshots[step]
    
    tiled_batch = np.zeros((28*5, 28*4), dtype='float32')
    for i in range(20):
        r = i % 5
        c = i // 5
        tiled_batch[r*28:(r+1)*28, c*28:(c+1)*28] = data.train.values[batch[i]].reshape(28, 28)

    tiled_weights = np.zeros((28*5, 28*2), dtype='float32')
    for i in range(10):
        r = i % 5
        c = i // 5
        tiled_weights[r*28:(r+1)*28, c*28:(c+1)*28] = W[i].reshape(28, 28)

    a0 = fig.add_subplot(gs[0, 0])
    a0.set_title("Input")
    a0.imshow(tiled_batch, cmap='gray', interpolation='nearest')
    a0.set_xticks([])
    a0.set_yticks([])

    a1 = fig.add_subplot(gs[0, 1])
    a1.set_title("Weights")
    a1.imshow(tiled_weights, interpolation='nearest')
    a1.set_xticks([])
    a1.set_yticks([])

    x = [snapshots[i][0] for i in range(max_steps)]
    y1 = [snapshots[i][1] for i in range(max_steps)]
    y2 = [snapshots[i][2] for i in range(max_steps)]

    a2 = fig.add_subplot(gs[0, 2])
    a2.set_title("Accuracy / Loss")
    acc_line = a2.plot(x, y1, 'b-')
    a2.set_xticks([])
    a2.set_ylim(0, 1.1)
    a2.tick_params('y', colors='b')
    a2.plot(num_instances, accuracy, 'bo', ms="15.0")

    a3 = a2.twinx()

    loss_line = a3.plot(x, y2, 'g--')
    a3.tick_params('y', colors='g')
    a3.plot(num_instances, loss, 'go', ms="15.0")

    a2.legend(acc_line + loss_line, ["Accuracy", "Loss"], loc="center right")

    plt.tight_layout()


def mnist():
    data = MulticlassDataset.mnist()
    snapshots = train(data, 0.1, 2, 100)
    num_snaps = len(snapshots)
    fig = plt.figure(figsize=(12, 4))
    plot_mnist(fig, data, snapshots, num_snaps - 2, num_snaps)
    plt.show()


def xor_bullseye():
    xor = MulticlassDataset.generate_xor(200, 100)
    bullseye = MulticlassDataset.generate_bullseye(200, 100)
    xor_snapshots = train(xor, 0.05, 5, 10, 10)
    bullseye_snapshots = train(bullseye, 0.05, 5, 10, 10)

    def plot(fig: plt.Figure, i: int):
        ax = fig.add_subplot(1, 2, 1)
        xor_snapshots[i].plot(ax, xor, False, True)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("XOR")

        ax = fig.add_subplot(1, 2, 2)
        bullseye_snapshots[i].plot(ax, bullseye, False, True)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Bullseye")

    plt.rc('font', size=15)
    fig = plt.figure(figsize=(8, 4))
    plot(fig, len(xor_snapshots) - 1)
    plt.show()

def xor_bullseye_features():
    xor = MulticlassDataset.generate_xor(200, 100)
    bullseye = MulticlassDataset.generate_bullseye(200, 100)

    def xor_feature(x: np.ndarray):
        return np.stack([x[:, 0], x.prod(1) / np.abs(x[:, 0])], axis=1)

    def bullseye_feature(x: np.ndarray):
        r = np.sqrt((x ** 2).sum(1))
        return np.stack([x[:, 0], r], axis=1)

    xor = xor.apply_feature(xor_feature)
    bullseye = bullseye.apply_feature(bullseye_feature)

    xor_snapshots = train(xor, 0.05, 5, 10, 10)
    bullseye_snapshots = train(bullseye, 0.05, 5, 10, 10)

    def plot(fig: plt.Figure, i: int):
        ax = fig.add_subplot(1, 2, 1)
        xor_snapshots[i].plot(ax, xor, False, True)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("XOR")

        ax = fig.add_subplot(1, 2, 2)
        bullseye_snapshots[i].plot(ax, bullseye, False, True)
        ax.set_xlim(-10, 10)
        ax.set_ylim(0, 10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Bullseye")

    plt.rc('font', size=15)
    fig = plt.figure(figsize=(8, 4))
    plot(fig, len(xor_snapshots) - 1)
    plt.show()


def main():
    three_class()
    mnist()
    uncertainty()
    xor_bullseye()
    xor_bullseye_features()


if __name__ == "__main__":
    main()

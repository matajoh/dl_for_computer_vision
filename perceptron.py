from typing import List, NamedTuple, Union

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from datasets import BinaryDataset


Snapshot = NamedTuple("Snapshot", [("step", int),
                                   ("weights", np.ndarray),
                                   ("accuracy", float),
                                   ("examples", Union[None, List[np.ndarray]])])


def plot_2d_decision_boundary(dataset: BinaryDataset, snapshot: Snapshot, ax: plt.Axes, title: str):
    """
    Plots the provided points along with a decision boundary between them defined by the provided weights

    Arguments:

    positive_examples -- ndarray (num_examples, num_dims) of data points from the positive class, drawn as blue circles
    negative_examples -- ndarray (num_examples, num_dims) of data points from the negative class, drawn as red circles
    weights -- [A,B,C] weights for the line equation, used to draw the decision boundary in green    
    """
    min_max = dataset.min_max()

    if np.isclose(snapshot.weights[1], 0):
        if np.isclose(snapshot.weights[0], 0):
            x = y = None
        else:
            y = np.array(min_max, dtype='float32')
            x = -(snapshot.weights[1] * y + snapshot.weights[2])/snapshot.weights[0]
    else:
        x = np.array(min_max, dtype='float32')
        y = -(snapshot.weights[0] * x + snapshot.weights[2])/snapshot.weights[1]

    dataset.plot(ax)
    if x is not None:
        ax.plot(x, y, 'g', linewidth=2.0)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

    if snapshot.examples is not None:
        for example in snapshot.examples:
            ax.plot(example[0], example[1], 'o', color="magenta", ms='14.0', mew='3.0', mfc='none')


def train(dataset: BinaryDataset, num_iterations: int) -> List[Snapshot]:
    """
    Trains a perceptron

    Arguments:

    positive_examples -- ndarray (num_examples, num_dims) of data points from the positive class
    negative_examples -- ndarray (num_examples, num_dims) of data points from the negative class

    Keyword arguments:

    num_iterations -- Number of iterations to run the algorithm.

    Returns:

    List of snapshots in the form [(step, weights_0, confusion_matrix_0), (weights_1, confusion_matrix_1), ...]
    """
    num_dims = dataset.num_dims
    weights = np.zeros((num_dims, 1))  # initialise the weights

    pos_count = dataset.positive.shape[0]
    neg_count = dataset.negative.shape[0]

    np.set_printoptions(precision=2, suppress=True)

    snapshots: List[Snapshot] = []

    def evaluate_model():
        pos_out = np.dot(dataset.positive, weights)
        neg_out = np.dot(dataset.negative, weights)
        test = np.concatenate((dataset.positive, pos_out), axis=1)
        test = np.sort(test, axis=0)

        pos_correct = (pos_out >= 0).sum()
        neg_correct = (neg_out < 0).sum()

        return (pos_correct + neg_correct) / (pos_count + neg_count)

    acc = evaluate_model()
    snapshots.append(Snapshot(0, np.copy(weights), acc, []))
    num_changes = 0
    i = 0

    for i in tqdm(range(num_iterations), desc="Training perceptron"):
        # select a positive and a negative example
        pos = dataset.positive[i % pos_count]
        neg = dataset.negative[i % neg_count]

        # present the positive example
        pos_out = np.dot(pos, weights)

        if pos_out < 0:
            # if there was a mistake, update the weights
            weights = weights + pos.reshape(weights.shape)
            num_changes += 1
            acc = evaluate_model()
            snapshots.append(Snapshot(num_changes, np.copy(weights), acc, [pos]))
        
        # present the negative example
        neg_out = np.dot(neg, weights)
        if neg_out >= 0:
            # if there was a mistake, update the weights
            weights = weights - neg.reshape(weights.shape)
            num_changes += 1
            acc = evaluate_model()
            snapshots.append(Snapshot(num_changes, np.copy(weights), acc, [neg]))

    snapshots.append(Snapshot(num_changes, weights, acc, []))

    return snapshots


def plot_mnist(fig: plt.Figure, snapshot: Snapshot, snapshots: List[Snapshot]):
    plt.rc('font', size=15)
    i, weights, accuracy, examples = snapshot
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(examples[0].reshape(28, 28), cmap='gray', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Input")

    ax = fig.add_subplot(1, 3, 2)
    plt.imshow(weights.reshape(28, 28), interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    min_val = np.min(weights)
    max_val = np.max(weights)
    ax.set_title("Weights (range: {:.2f})".format(max_val - min_val))

    ax = fig.add_subplot(1, 3, 3)
    ax.set_title("Accuracy")
    plt.bar([0, 1], [1-accuracy, accuracy])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Mistakes", "Correct"])
    ax.set_ylim(0, 1.1)           

    plt.tight_layout()


def binary():
    dataset = BinaryDataset.generate_from_normals(200)
    snapshots = train(dataset, 200)
    
    plt.rc('font', size=15)
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(2, 2, 1)
    plot_2d_decision_boundary(dataset, snapshots[0], ax, "Initial")
    ax = plt.subplot(2, 2, 2)
    plot_2d_decision_boundary(dataset, snapshots[1], ax, "1")
    ax = plt.subplot(2, 2, 3)
    plot_2d_decision_boundary(dataset, snapshots[2], ax, "2")
    ax = plt.subplot(2, 2, 4)
    plot_2d_decision_boundary(dataset, snapshots[-1], ax, "Final")
    plt.tight_layout()
    plt.show()


def mnist(pos: int, neg: int):
    dataset = BinaryDataset.mnist(pos, neg)
    snapshots = train(dataset, 1000)
    fig = plt.figure(figsize=(12, 4))
    plot_mnist(fig, snapshots[len(snapshots)-2], snapshots)
    plt.show()


def main():
    binary()
    mnist(1, 0)

if __name__ == "__main__":
    main()

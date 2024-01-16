import copy
import os
from typing import List, NamedTuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from datasets import MulticlassDataset


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MultiLayerPerceptron, self).__init__()
        self.linear0 = nn.Linear(input_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, output_dim)

    def __call__(self, x):
        h = self.linear0(x)
        s = h.tanh()
        y = self.linear1(s)

        return y

    def snapshot(self):
        return copy.deepcopy(self.state_dict())


Snapshot = NamedTuple("Snapshot", [("step", int), ("accuracy", float), ("loss", float),
                                   ("state_dict", dict)])


def train(dataset: MulticlassDataset, net: MultiLayerPerceptron,
                                criterion, optimizer, num_epochs=1, batch_size=100, report_frequency=0):
    train_data, train_labels = dataset.train
    val_data, val_labels = dataset.val

    train_data = torch.from_numpy(train_data)
    train_labels = torch.from_numpy(train_labels)
    val_data = torch.from_numpy(val_data)
    val_labels = torch.from_numpy(val_labels)

    num_train = train_data.shape[0]

    snapshots = []

    np.set_printoptions(precision=2, suppress=True)
    num_instances = 0
    last_accuracy = -1.0
    last_instances = 0

    def compute_accuracy(output, expected_labels):
        _, actual_labels = torch.max(output, -1)
        num_correct = (actual_labels == expected_labels).sum().item()
        total = len(actual_labels)
        return num_correct / total

    def evaluate_model():
        net.eval()
        x = val_data
        t = val_labels
        y = net(x)
        loss = criterion(y, t).item()
        accuracy = compute_accuracy(y, t)
        net.train()
        return accuracy, loss

    def do_report(accuracy, num_instances):
        if report_frequency == 0:
            accuracy_freq = .01
            instances_freq = num_train / 5

            accuracy_diff = accuracy - last_accuracy
            instances_diff = num_instances - last_instances
            return accuracy_diff >= accuracy_freq or instances_diff >= instances_freq

        return num_instances - last_instances >= report_frequency

    epochs = []

    for epoch in range(num_epochs):
        perm = np.random.permutation(num_train)

        sum_loss = 0
        sum_accuracy = 0
        batches = tqdm(range(0, num_train, batch_size), desc="Epoch {}".format(epoch))
        for i in batches:
            accuracy, loss = evaluate_model()
            batches.set_postfix({"accuracy": "{:.3f}".format(accuracy), "loss": "{:.3f}".format(loss)})
            if do_report(accuracy, num_instances):
                snapshots.append(Snapshot(num_instances, accuracy, loss, net.snapshot()))
                last_accuracy = accuracy
                last_instances = num_instances

            x = train_data[perm[i:i + batch_size]]
            t = train_labels[perm[i:i + batch_size]]

            optimizer.zero_grad()
            y = net(x)
            loss = criterion(y, t)
            accuracy = compute_accuracy(y, t)
            loss.backward()
            optimizer.step()

            num_instances += x.shape[0]
            sum_loss += loss.item() * len(t)
            sum_accuracy += accuracy * len(t)

        accuracy = sum_accuracy / num_train
        loss = sum_loss / num_train
        val_accuracy, val_loss = evaluate_model()
        epochs.append((num_train*(epoch+1), accuracy, loss, val_accuracy, val_loss, net.snapshot()))

    accuracy, loss = evaluate_model()
    snapshots.append(Snapshot(num_instances, accuracy, loss, net.snapshot()))
    return {
        "snapshots": snapshots,
        "epochs": epochs
    }


ExploreSnapshot = NamedTuple("ExploreSnapshot", [("step", int), ("p0", float), ("p1", float)])


def plot_xor(fig: plt.Figure, snapshots: List[Union[Snapshot, ExploreSnapshot]], step: int, net: MultiLayerPerceptron,
             dataset: MulticlassDataset, examples: List[np.ndarray] = None,
             bounds=10, azimuth=None, planes=False):
    ax = fig.add_subplot(1, 3, 1)
    ax.set_title("Input")
    ax.set_xticks([])
    ax.set_yticks([])
    snapshot = snapshots[step]

    scale = (2 * bounds) / 100
    space = np.zeros((100, 100, 3), dtype='uint8')
    for row in range(100):
        batch = torch.zeros(100, 2, dtype=torch.float32)
        for col in range(100):
            batch[col, 0] = col-50
            batch[col, 1] = 50 - row

        batch *= scale
        output = net(batch).detach().numpy()
        labels = np.argmax(output, axis=1)
        for col in range(100):
            space[row, col] = [255, 182, 193] if labels[col] else [173, 216, 230]

    data, labels = dataset.train
    color = ['b', 'r']
    marker = ['+', '.']
    ax.imshow(space, interpolation='nearest', extent=(-bounds, bounds, -bounds, bounds))
    ax.autoscale(False)
    for i in range(2):
        points = data[np.where(labels == i)]
        ax.plot(points[:, 0], points[:, 1], color[i] + marker[i])

    if examples is not None:
        x, t = examples[step]
        ax.plot(x[:, 0], x[:, 1], color="magenta", marker="s", ms="15.0")

    ax1 = fig.add_subplot(1, 3, 2, projection='3d')
    if azimuth is not None:
        ax1.view_init(azim=azimuth)

    ax1.set_title("Latent Space")
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_zlim(-1, 1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    latent = net.linear0(torch.from_numpy(data)).tanh().detach().numpy()
    for i in range(2):
        points = latent[np.where(labels == i)]
        ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=color[i], marker=marker[i])
        if planes:
            xx, yy = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
            a, b, c = net.linear1.weight.detach().numpy()[i]
            d = net.linear1.bias.detach().numpy()[i]
            z = (-a * xx - b * yy - d) / c
            ax1.plot_surface(xx, yy, z, color=color[i], alpha=0.5)

    if examples is not None:
        x, t = examples[step]
        latent = net.linear0(torch.from_numpy(x)).tanh().detach().numpy()
        ax1.scatter(latent[:, 0], latent[:, 1], latent[:, 2], c="magenta", marker='s', sizes=[200])

    ax2 = fig.add_subplot(1, 3, 3)

    x = [snapshots[i][0] for i in range(len(snapshots))]
    y1 = [snapshots[i][1] for i in range(len(snapshots))]
    y2 = [snapshots[i][2] for i in range(len(snapshots))]

    if isinstance(snapshots[i], Snapshot):
        ax2.set_title("Loss/Accuracy")  
        acc_line = ax2.plot(x, y1, 'b-')
        ax2.plot(snapshot[0], snapshot[1], 'bo')
        ax2.set_xticks([])
        ax2.set_ylim(0, 1.1)
        ax2.tick_params('y', colors='b')
        ax3 = ax2.twinx()
        ax3.tick_params('y', colors='g')
        loss_line = ax3.plot(x, y2, 'g--')
        ax3.plot(snapshot[0], snapshot[2], 'go')
        ax3.legend(acc_line + loss_line, ["Accuracy", "Loss"], loc="center right")
    else:
        ax2.set_title("Prediction")
        ax2.plot(x, y1, 'b-')
        ax2.plot(x, y2, 'r-')
        ax2.plot(snapshot[0], snapshot[1], 'b+', ms="15.0")
        ax2.plot(snapshot[0], snapshot[2], 'r.', ms="15.0")
        ax2.set_xticks([])
        ax2.set_ylim(0, 1)

    plt.tight_layout()


def xor(dataset: MulticlassDataset):
    net = MultiLayerPerceptron(2, 3, 2)
    if os.path.exists("xor.results"):
        results = torch.load("xor.results")
        print("xor.results loaded")
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.01)
        results = train(dataset, net, criterion, optimizer, num_epochs=5, batch_size=5, report_frequency=10)
        torch.save(results, "xor.results")

    snapshots = results['snapshots']
    net.load_state_dict(snapshots[1].state_dict)

    plt.rc('font', size=15)
    fig = plt.figure(figsize=(12, 4))

    net.load_state_dict(snapshots[-1].state_dict)
    plot_xor(fig, snapshots, len(snapshots)-1, net, dataset)
    plt.show()


def explore(x: np.ndarray, t: np.ndarray, model: nn.Module, criterion: nn.Module):
    x = torch.from_numpy(x)
    x.requires_grad = True
    t = torch.from_numpy(t)
    optimizer = optim.SGD([x], lr=0.1)
    model.eval()
    switch = None
    snapshots = []
    examples = []
    for i in range(100):
        if switch and i > 2 * switch:
            break

        optimizer.zero_grad()
        y = model(x)
        p = y.softmax(1)
        p0 = p[0, 0].item()
        p1 = p[0, 1].item()
        if p1 > p0 and switch is None:
            switch = i

        loss = criterion(y, t)
        loss.backward()
        x.data -= x.grad.data
        x.grad.data.zero_()
        snapshots.append(ExploreSnapshot(i, p0, p1))
        examples.append((x.data.numpy().copy(), t.item()))

    return snapshots, examples


def xor_explore(dataset: MulticlassDataset):
    results = torch.load("xor.results")
    print("xor.results loaded")
    snapshots = results['snapshots']
    net = MultiLayerPerceptron(2, 3, 2)
    net.load_state_dict(snapshots[-1].state_dict)
    criterion = nn.CrossEntropyLoss()

    x = dataset.train.values[:1].copy()
    t = 1 - dataset.train.labels[:1]
    snapshots, examples = explore(x, t, net, criterion)

    plt.rc("font", size=15)
    fig = plt.figure(figsize=(12, 4))
    plot_xor(fig, snapshots, len(snapshots) - 1, net, dataset, examples, 13)
    plt.show()


def bullseye(dataset: MulticlassDataset):
    net = MultiLayerPerceptron(2, 3, 2)
    if os.path.exists("bullseye.results"):
        results = torch.load("bullseye.results")
        print("bullseye.results loaded")
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.01)
        results = train(dataset, net, criterion, optimizer, num_epochs=15, batch_size=5, report_frequency=10)
        torch.save(results, "bullseye.results")

    snapshots = results['snapshots']

    def plot(fig: plt.Figure, step: int):
        net.load_state_dict(snapshots[step].state_dict)
        azimuth = 360 * step / len(snapshots)
        plot_xor(fig, snapshots, step, net, dataset, azimuth=azimuth, planes=step > len(snapshots) // 2)

    fig = plt.figure(figsize=(12, 4))
    plot(fig, len(snapshots) - 1)
    plt.show()


def plot_mnist(fig: plt.Figure, snapshots: List[Snapshot], step: int, net: MultiLayerPerceptron):
    gs = fig.add_gridspec(1, 3, width_ratios=[2, 1, 2])
    num_instances, accuracy, loss, state_dict = snapshots[step]
    net.load_state_dict(state_dict)

    W = net.linear0.weight.detach().numpy()
    tiled = np.zeros((28*8, 28*8), dtype='float32')
    for i in range(64):
        r = i % 8
        c = i // 8
        tiled[r*28:(r+1)*28, c*28:(c+1)*28] = W[i].reshape(28, 28)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(tiled, interpolation='nearest')
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_title("W0")

    W = net.linear1.weight.detach().numpy()
    tiled = np.zeros((8*5, 8*2), dtype="float32")
    for i in range(10):
        r = i % 5
        c = i // 5
        tiled[r*8:(r+1)*8, c*8:(c+1)*8] = W[i].reshape(8, 8)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(tiled, interpolation='nearest')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("W1")

    x = [snapshots[i][0] for i in range(len(snapshots))]
    y1 = [snapshots[i][1] for i in range(len(snapshots))]
    y2 = [snapshots[i][2] for i in range(len(snapshots))]

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_title("Accuracy / Loss")
    ax2.set_ylim(0, 1.1)
    ax2.set_xticks([])
    ax2.tick_params('y', colors='b')
    acc_line = ax2.plot(x, y1, 'b-')
    ax2.plot(num_instances, accuracy, 'bo', ms="15.0")

    ax3 = ax2.twinx()
    ax3.set_ylim(0, max(y2) * 1.1)
    ax3.tick_params('y', colors='g')
    loss_line = ax3.plot(x, y2, 'g--')
    ax3.plot(num_instances, loss, 'go', ms="15.0")

    ax2.legend(acc_line + loss_line, ["Accuracy", "Loss"], loc="center right")

    plt.tight_layout()


def mnist_figure():
    dataset = MulticlassDataset.mnist()
    net = MultiLayerPerceptron(784, 64, 10)

    if os.path.exists("mnist.results"):
        results = torch.load("mnist.results")
        print("mnist.results loaded")
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.1)
        results = train(dataset, net, criterion, optimizer, num_epochs=5)
        torch.save(results, "mnist.results")

    snapshots = results["snapshots"]
    plt.rc('font', size=15)
    fig = plt.figure(figsize=(12, 4))
    plot_mnist(fig, snapshots, len(snapshots)-1, net)
    plt.show()


AttackSnapshot = NamedTuple("AttackSnapshot", [("step", int), ("p0", float), ("p1", float)])


def plot_mnist_explore(fig: plt.Figure, snapshots: List[AttackSnapshot], examples: List[np.ndarray], labels: List[str], step: int):
    x, dx = examples[step]

    ax0 = fig.add_subplot(1, 3, 1)
    ax0.imshow(x.reshape(28, 28), cmap="gray", vmin=0, vmax=1, interpolation='nearest')
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_title("Altered Image")

    ax1 = fig.add_subplot(1, 3, 2)
    ax1.imshow(dx.reshape(28, 28), vmin=-0.3, vmax=0.3, interpolation='nearest')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("Difference")

    num_instances, p0, p1 = snapshots[step]

    ax2 = fig.add_subplot(1, 3, 3)
    ax2.set_title("Prediction")
    x = [snapshots[i].step for i in range(len(snapshots))]
    y1 = [snapshots[i].p0 for i in range(len(snapshots))]
    y2 = [snapshots[i].p1 for i in range(len(snapshots))]

    ax2.set_xticks([])
    ax2.set_ylim(0, 1.0)
    ax2.plot(x, y1, 'b-', label=labels[0])
    ax2.plot(num_instances, p0, 'bo')
    ax2.plot(x, y2, 'g--', label=labels[1])
    ax2.plot(num_instances, p1, 'go')
    ax2.legend(loc="center right")

    plt.tight_layout()


def attack(x: np.ndarray, t0: np.ndarray, t1: np.ndarray, model: nn.Module, criterion: nn.Module):
    x0 = torch.from_numpy(x.copy())
    x = torch.from_numpy(x)
    x.requires_grad = True
    t = torch.LongTensor([t1])
    p = model(x0).softmax(1)
    target = p[0, t0].item()
    model.eval()
    switch = None
    snapshots = []
    examples = []
    optimizer = optim.SGD([x], 0.01)

    y = model(x)
    loss = criterion(x0, x, y, t)
    snapshots.append(AttackSnapshot(0, p[0, t0].item(), p[0, t1].item()))
    examples.append((x.data.numpy().copy(), np.zeros((28, 28), np.float32)))

    for i in range(1000):
        optimizer.zero_grad()
        y = model(x)
        p = y.softmax(1)
        p0 = p[0, t0].item()
        p1 = p[0, t1].item()

        if p1 > target:
            break

        if p1 > p0 and switch is None:
            print("switch at", i)
            switch = i

        loss = criterion(x0, x, y, t)
        loss.backward()
        optimizer.step()
        x.data = x.data.clamp(0, 1)
        snapshots.append(AttackSnapshot(i, p0, p1))
        examples.append((x.data.numpy().copy(), (x - x0).data.numpy().copy()))

    return snapshots, examples, [str(t0), str(t1)]


def mnist_attack():
    dataset = MulticlassDataset.mnist()
    results = torch.load("mnist.results")
    print("mnist.results loaded")
    snapshots = results['snapshots']
    net = MultiLayerPerceptron(784, 64, 10)
    net.load_state_dict(snapshots[-1].state_dict)

    def criterion(x0, x, y, t):
        return F.cross_entropy(y, t) + 0.01 * F.mse_loss(x, x0, reduction='sum')

    print(dataset.train.labels[:10])

    x = dataset.train.values[2:3].copy()
    t0 = dataset.train.labels[2:3].copy().item()
    t1 = dataset.train.labels[4:5].copy().item()
    
    snapshots, examples, labels = attack(x, t0, t1, net, criterion)

    plt.rc('font', size=15)
    fig = plt.figure(figsize=(12, 4))
    plot_mnist_explore(fig, snapshots, examples, labels, len(snapshots) - 1)
    plt.show()


def gradients(activation: str):
    num_layers = 20
    layers = []
    for i in range(num_layers):
        layers.append(torch.rand(2, 2, dtype=torch.float32, requires_grad=True))

    x = torch.rand(3, 2, dtype=torch.float32, requires_grad=True)
    outputs = [x]
    for i in range(num_layers):
        h = torch.mm(outputs[i], torch.transpose(layers[i], 0, 1))
        if activation == "sigmoid":
            s = h.sigmoid()
        elif activation == "relu":
            s = h.relu()
        else:
            raise RuntimeError("Unsupported activation")

        outputs.append(s)

    y = outputs[-1]
    grad = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.float32)

    grad_mag = []
    y.backward(grad)
    for i in range(num_layers):
        value = np.linalg.norm(layers[i].grad.detach().numpy())
        grad_mag.append(value)

    plt.rc('font', size=15)
    plt.figure(figsize=(9, 3))
    plt.plot(range(1, num_layers+1), grad_mag, c="black", linestyle='dashed')
    plt.xlim(0, num_layers)
    plt.xlabel("Layer")
    xticks = list(range(0, num_layers + 1, 2))
    plt.xticks(xticks)
    plt.grid(visible=True)
    plt.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.ylabel("Gradient Magnitude")
    if activation == "sigmoid":
        plt.yscale("log")

    plt.tight_layout()
    plt.show()


def main():
    np.random.seed(20080524)
    torch.manual_seed(20080524)

    dataset = MulticlassDataset.generate_xor(200, 100)
    #xor(dataset)
    #xor_explore(dataset)

    dataset = MulticlassDataset.generate_bullseye(200, 100)
    #bullseye(dataset)

    #mnist_figure()
    mnist_attack()

    gradients("sigmoid")
    gradients("relu")


if __name__ == "__main__":
    main()

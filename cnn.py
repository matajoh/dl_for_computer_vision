import os
from typing import List, NamedTuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from tqdm import tqdm

from datasets import load_coco, MulticlassDataset


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.c1 = nn.Conv2d(1, 9, 5)
        self.f3 = nn.Linear(5184, 10)

        self.outputs = [0] * 4
        self.num_layers = 4

    def __call__(self, x: torch.Tensor):
        self.outputs[0] = output = x
        self.outputs[1] = output = self.c1(output)
        self.outputs[2] = output = output.tanh()
        self.outputs[3] = output = self.f3(output.flatten(1))

        return output

    def get_weights(self, layer):
        if layer == 1:
            return self.c1.weight.cpu().detach().numpy()
        elif layer == 3:
            return self.f3.weight.cpu().detach().numpy()
        else:
            raise ValueError("Layer does not have weights: {}".format(layer))


class PoolingCNN(nn.Module):
    def __init__(self):
        super(PoolingCNN, self).__init__()
        self.c1 = nn.Conv2d(1, 9, 5)
        self.f3 = nn.Linear(1296, 10)

        self.num_layers = 4
        self.outputs = [0] * self.num_layers

    def __call__(self, x):
        self.outputs[0] = output = x
        self.outputs[1] = output = self.c1(output)
        self.outputs[2] = output = F.max_pool2d(output, 2, stride=2).relu()
        self.outputs[3] = output = self.f3(output.flatten(1))

        return output

    def get_weights(self, layer):
        if layer == 1:
            return self.c1.weight.cpu().detach().numpy()
        elif layer == 3:
            return self.f3.weight.cpu().detach().numpy()
        else:
            raise ValueError("Layer does not have weights: {}".format(layer))


class CifarCNN(nn.Module):
    def __init__(self):
        super(CifarCNN, self).__init__()
        self.c1 = nn.Conv2d(3, 32, 5, stride=1, padding=2)
        self.c3 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.c5 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.f7 = nn.Linear(1024, 64)
        self.d8 = nn.Dropout(0.5)
        self.f9 = nn.Linear(64, 10)

        self.num_layers = 10
        self.outputs = [0] * self.num_layers

    def __call__(self, x):
        self.outputs[0] = x
        self.outputs[1] = h = self.c1(x)
        self.outputs[2] = h = F.max_pool2d(h, 3, stride=2).relu()
        self.outputs[3] = h = self.c3(h).relu()
        self.outputs[4] = h = F.avg_pool2d(h, 3, stride=2, padding=1)
        self.outputs[5] = h = self.c5(h).relu()
        self.outputs[6] = h = F.avg_pool2d(h, 3, stride=2, padding=1)
        self.outputs[7] = h = self.f7(h.flatten(1))
        self.outputs[8] = h = self.d8(h).relu()
        self.outputs[9] = h = self.f9(h)
        return h

    def get_weights(self, layer):
        if layer == 1:
            return self.c1.weight.cpu().detach().numpy()
        elif layer == 3:
            return self.c3.weight.cpu().detach().numpy()
        elif layer == 5:
            return self.c5.weight.cpu().detach().numpy()
        elif layer == 7:
            return self.f7.weight.cpu().detach().numpy()
        elif layer == 9:
            return self.f9.weight.cpu().detach().numpy()
        else:
            raise ValueError("Layer does not have weights: {}".format(layer))


class FCNSimple(nn.Module):
    def __init__(self):
        super(FCNSimple, self).__init__()
        self.c1 = nn.Conv2d(3, 32, 5, stride=1, padding=2)
        self.c3 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.c5 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.c7 = nn.Conv2d(64, 128, 1)
        self.d8 = nn.Dropout(0.5)
        self.c9 = nn.Conv2d(128, 10, 1)

        self.num_layers = 11
        self.outputs = [0] * self.num_layers

    def __call__(self, x):
        self.outputs[0] = x
        self.outputs[1] = h = self.c1(x)
        self.outputs[2] = h = F.max_pool2d(h, 3, stride=2).relu()
        self.outputs[3] = h = self.c3(h).relu()
        self.outputs[4] = h = F.avg_pool2d(h, 3, stride=2, padding=1)
        self.outputs[5] = h = self.c5(h).relu()
        self.outputs[6] = h = F.avg_pool2d(h, 3, stride=2, padding=1)
        self.outputs[7] = h = self.c7(h)
        self.outputs[8] = h = self.d8(h).relu()
        self.outputs[9] = h = self.c9(h)
        width = h.shape[-2]
        height = h.shape[-1]
        self.outputs[10] = h = F.avg_pool2d(h, (width, height)).flatten(1)
        return h

    def get_weights(self, layer):
        if layer == 1:
            return self.c1.weight.cpu().detach().numpy()
        elif layer == 3:
            return self.c3.weight.cpu().detach().numpy()
        elif layer == 5:
            return self.c5.weight.cpu().detach().numpy()
        elif layer == 7:
            return self.c7.weight.cpu().detach().numpy()
        elif layer == 9:
            return self.c9.weight.cpu().detach().numpy()
        else:
            raise ValueError("Layer does not have weights: {}".format(layer))


class NIN(nn.Module):
    def __init__(self, num_classes):
        super(NIN, self).__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(3, 192, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 160, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 96, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            nn.Dropout(inplace=True),

            nn.Conv2d(96, 192, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(3, stride=2, ceil_mode=True),
            nn.Dropout(inplace=True),

            nn.Conv2d(192, 192, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, self.num_classes, 1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(8, stride=1)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), self.num_classes)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                if m.bias is not None:
                    m.bias.data.zero_()


def compute_accuracy(output, expected_labels):
    _, actual_labels = torch.max(output, -1)
    num_correct = (actual_labels == expected_labels).sum().item()
    total = len(actual_labels)
    return num_correct / total


Snapshot = NamedTuple("Snapshot", [("step", int), ("accuracy", float), ("loss", float)])


def train_model(dataset: MulticlassDataset, net: nn.Module, criterion, batch_size=100, num_epochs=20, device="cpu"):
    """
    Trains a PyTorch model.

    Arguments:

    model -- The model to train

    Keyword arguments:

    batchsize -- The batchsize to use during training
    epoch -- The number of epochs to train
    """

    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    snapshots = []
    train_loader = torch.utils.data.DataLoader(dataset.train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset.val, batch_size=batch_size, shuffle=False)
    for epoch in range(1, num_epochs + 1):
        # training
        net.train()
        sum_accuracy = 0
        sum_loss = 0
        total = 0
        batches = tqdm(train_loader, desc="Epoch {}".format(epoch))
        for x, t in batches:
            x = x.to(device)
            t = t.to(device)

            optimizer.zero_grad()
            y = net(x)
            loss = criterion(y, t)
            accuracy = compute_accuracy(y, t)

            loss.backward()
            optimizer.step()

            sum_loss += loss.item() * len(t)
            sum_accuracy += accuracy * len(t)
            total += len(t)
            batches.set_postfix({"accuracy": "{:.3f}".format(sum_accuracy / total),
                                 "loss": "{:.3f}".format(sum_loss / total)})

        # evaluation
        net.eval()
        sum_accuracy = 0
        sum_loss = 0
        total = 0
        batches = tqdm(val_loader, desc="Evaluation")
        for x, t in batches:
            x = x.to(device)
            t = t.to(device)
            y = net(x)
            loss = criterion(y, t)
            accuracy = compute_accuracy(y, t)

            sum_loss += loss.item() * len(t)
            sum_accuracy += accuracy * len(t)
            total += len(t)
            batches.set_postfix({"accuracy": "{:.3f}".format(sum_accuracy / total),
                                 "loss": "{:.3f}".format(sum_loss / total)})

        snapshots.append(Snapshot(epoch, sum_accuracy / total, sum_loss / total))

    return snapshots


def evaluate_model(dataset: MulticlassDataset, net: nn.Module, snapshots: List[Snapshot], device="cpu"):
    label_names = dataset.label_names
    num_examples = 6
    examples, _ = dataset.val[0:num_examples]
    columns = num_examples // 2

    net.to(device)
    net.eval()
    x = examples.to(device)
    output = F.softmax(net(x), dim=1).cpu().detach().numpy()
    top_labels = np.argpartition(output, -2)[:, -2:]

    plt.rc('font', size=15)
    fig = plt.figure(figsize=(10, 4))
    gs0 = gridspec.GridSpec(1, 2, wspace=.3)
    gs00 = gridspec.GridSpecFromSubplotSpec(2, columns, gs0[0])

    for i in range(num_examples):
        row = i // columns
        col = i % columns
        ax = fig.add_subplot(gs00[row, col])
        if examples[i].shape[0] == 3:
            reorder = dataset.unnormalize(examples[i].detach().cpu().numpy())
            reorder = np.swapaxes(reorder, 0, 1)
            reorder = np.swapaxes(reorder, 1, 2)
            ax.imshow(reorder, interpolation='nearest')
        else:
            ax.imshow(examples[i].reshape(examples[i].shape[1:]), cmap='gray', interpolation='nearest')

        certainty1, certainty0 = output[i, top_labels[i]]
        text1 = label_names[top_labels[i, 0]]
        text0 = label_names[top_labels[i, 1]]
        ax.set_xlabel("{} ({})\n{} ({})".format(text0, round(float(certainty0), 2), text1, round(float(certainty1), 2)))
        ax.set_xticks([])
        ax.set_yticks([])

    ax1 = fig.add_subplot(gs0[1])
    x = [snapshots[i][0] for i in range(len(snapshots))]
    y1 = [snapshots[i][1] for i in range(len(snapshots))]
    y2 = [snapshots[i][2] for i in range(len(snapshots))]

    acc_line = ax1.plot(x, y1, 'b-')
    ax1.set_xticks([])
    ax1.set_ylim(0, 1.1)
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()

    loss_line = ax2.plot(x, y2, 'g--')
    ax2.tick_params('y', colors='g')

    ax2.legend(acc_line + loss_line, ["Accuracy", "Loss"], loc="center right")


def mnist_simple():
    simple_net = SimpleCNN()
    dataset = MulticlassDataset.mnist(28).to_torch()
    path = "mnist_simple_cnn.results"

    if os.path.exists(path):
        results = torch.load(path)
        print(path, "loaded")
    else:
        criterion = nn.CrossEntropyLoss()
        snapshots = train_model(dataset, simple_net, criterion, num_epochs=5)
        results = {"snapshots": snapshots, "net": simple_net.state_dict()}
        torch.save(results, path)

    snapshots = results["snapshots"]
    simple_net.load_state_dict(results["net"])
    evaluate_model(dataset, simple_net, snapshots)
    plt.show()


def filters_grid(input, fig, cell):
    num_filters = input.shape[0]
    num_channels = input.shape[1]
    size = input.shape[2]

    rows = int(np.sqrt(num_filters))
    cols = num_filters // rows
    if num_filters > rows * cols:
        cols += 1

    cside = int(np.sqrt(num_channels))
    if num_channels > cside*cside:
        cside += 1

    grid = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=cell)

    for i in range(num_filters):
        ax = plt.Subplot(fig, grid[i])

        composite = np.zeros((cside*size, cside*size), dtype='float32')
        for j in range(num_channels):
            r = (j // cside)*size
            c = (j % cside)*size
            composite[r:r + size, c:c + size] = input[i, j]

        ax.imshow(composite, cmap='gray', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)


def images_grid(input, fig, cell):
    num_channels = input.shape[0]

    side = int(np.sqrt(num_channels))
    if num_channels > side*side:
        side += 1

    grid = gridspec.GridSpecFromSubplotSpec(side, side, subplot_spec=cell, wspace=0.1, hspace=0.1)

    vmin = input.min()
    vmax = input.max()
    for i in range(num_channels):
        ax = plt.Subplot(fig, grid[i])
        ax.imshow(input[i], interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)


def visualize_layer_output(net, layer, index, widths, height):
    outputs = net.outputs
    plt.rc('font', size=15)
    fig = plt.figure(figsize=(np.sum(widths), height))
    gs = gridspec.GridSpec(1, len(widths),
                           width_ratios=widths)

    input_image = outputs[layer - 1].cpu().detach().numpy()[index]
    if len(input_image.shape) == 3:
        ax = plt.Subplot(fig, gs[0, 0])
        ax.set_title("Input")
        images_grid(input_image, fig, gs[0, 0])
    else:
        ax = plt.Subplot(fig, gs[0, 0])
        ax.set_title("Input")
        if layer == net.num_layers - 2:
            input_image = input_image.reshape(8, 8)
        else:
            input_image = input_image.reshape(1, 10)
        ax.imshow(input_image, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)

    weights = net.get_weights(layer)
    if len(weights.shape) == 2:
        if len(input_image.shape) == 3:
            weights = weights.reshape(weights.shape[0], *input_image.shape)

    if len(weights.shape) == 4:
        ax = plt.Subplot(fig, gs[0, 1])
        ax.set_title("Weights")
        filters_grid(weights, fig, gs[0, 1])
    else:
        ax = plt.Subplot(fig, gs[0, 1])
        ax.set_title("Weights")
        ax.imshow(weights, cmap='gray', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)

    output_image = outputs[layer].cpu().detach().numpy()[index]
    if len(output_image.shape) == 3:
        ax = plt.Subplot(fig, gs[0, 2])
        ax.set_title("Output")
        images_grid(output_image, fig, gs[0, 2])
    else:
        ax = plt.Subplot(fig, gs[0, 2])
        ax.set_title("Output")
        if layer == net.num_layers - 2:
            output_image = output_image.reshape(8, 8)
        else:
            output_image = output_image.reshape(1, 10)

        ax.imshow(output_image, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)

    plt.tight_layout()


def mnist_simple_filters():
    simple_net = SimpleCNN()
    dataset = MulticlassDataset.mnist(28).to_torch()
    path = "mnist_simple_cnn.results"
    results = torch.load(path)
    print(path, "loaded")

    snapshots = results["snapshots"]
    simple_net.load_state_dict(results["net"])
    evaluate_model(dataset, simple_net, snapshots)
    visualize_layer_output(simple_net, 1, 3, [3, 4, 3], 4)
    plt.show()


def mnist_pool():
    pool_net = PoolingCNN()
    dataset = MulticlassDataset.mnist(28).to_torch()
    path = "mnist_pool_cnn.results"

    if os.path.exists(path):
        results = torch.load(path)
        print(path, "loaded")
    else:
        criterion = nn.CrossEntropyLoss()
        snapshots = train_model(dataset, pool_net, criterion, num_epochs=5)
        results = {"snapshots": snapshots, "net": pool_net.state_dict()}
        torch.save(results, path)

    snapshots = results["snapshots"]
    pool_net.load_state_dict(results["net"])
    evaluate_model(dataset, pool_net, snapshots)
    plt.show()


def mnist_pool_filters():
    pool_net = PoolingCNN()
    dataset = MulticlassDataset.mnist(28).to_torch()
    path = "mnist_pool_cnn.results"
    results = torch.load(path)
    print(path, "loaded")

    snapshots = results["snapshots"]
    pool_net.load_state_dict(results["net"])
    evaluate_model(dataset, pool_net, snapshots)
    visualize_layer_output(pool_net, 1, 3, [3, 4, 3], 4)
    plt.show()
    visualize_layer_output(pool_net, 3, 3, [3, 4, 3], 4)
    plt.show()


def cifar():
    dataset = MulticlassDataset.cifar().normalize().to_torch()
    cifar10_net = CifarCNN()
    path = "cifar10.results"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if os.path.exists(path):
        results = torch.load(path)
        print(path, "loaded")
    else:
        print("Running on", device)
        criterion = nn.CrossEntropyLoss()
        snapshots = train_model(dataset, cifar10_net, criterion, device=device, num_epochs=30)
        results = {"snapshots": snapshots, "net": cifar10_net.state_dict()}
        torch.save(results, path)

    cifar10_net.load_state_dict(results["net"])
    snapshots = results["snapshots"]
    evaluate_model(dataset, cifar10_net, snapshots, device=device)
    plt.show()


def cifar_filters():
    dataset = MulticlassDataset.cifar().normalize()
    cifar10_net = CifarCNN()
    path = "cifar10.results"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = torch.load(path)
    index = 0

    image = dataset.unnormalize(dataset.val.values[index])
    plt.figure(figsize=(4, 4))
    image = np.swapaxes(image, 0, 1)
    image = np.swapaxes(image, 1, 2)
    plt.imshow(image, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    dataset = dataset.to_torch()
    cifar10_net.load_state_dict(results["net"])
    snapshots = results["snapshots"]
    evaluate_model(dataset, cifar10_net, snapshots, device=device)
    visualize_layer_output(cifar10_net, 1, index, [3, 4, 3], 4)
    visualize_layer_output(cifar10_net, 3, index, [3, 4, 3], 4)
    visualize_layer_output(cifar10_net, 5, index, [3, 4, 3], 4)
    plt.show()


def fcn_simple():
    dataset = MulticlassDataset.cifar().normalize().to_torch()
    net = FCNSimple()
    path = "fcn_simple.results"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if os.path.exists(path):
        results = torch.load(path)
        print(path, "loaded")
    else:
        print("Running on", device)
        criterion = nn.CrossEntropyLoss()
        snapshots = train_model(dataset, net, criterion, num_epochs=50, device=device)
        results = {"snapshots": snapshots, "net": net.state_dict()}
        torch.save(results, path)

    net.load_state_dict(results["net"])
    snapshots = results["snapshots"]
    evaluate_model(dataset, net, snapshots, device=device)
    plt.show()


def plot_outputs(outputs, index):
    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(1, 3)

    for i, output in enumerate(outputs[:3]):
        output_image = output.cpu().detach().numpy()[index]
        if len(output_image.shape) == 3:
            images_grid(output_image, fig, gs[0, i])
        else:
            ax = plt.Subplot(fig, gs[0, 2])
            output_image = output_image.reshape(1, 10)

            ax.imshow(output_image, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)

    plt.tight_layout()


def fcn_simple_outputs():
    dataset = MulticlassDataset.cifar().normalize()
    net = FCNSimple()
    path = "fcn_simple.results"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = torch.load(path)
    print(path, "loaded")

    index = 4
    image = dataset.unnormalize(dataset.val.values[index])
    plt.figure(figsize=(4, 4))
    image = np.swapaxes(image, 0, 1)
    image = np.swapaxes(image, 1, 2)
    plt.imshow(image, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()

    dataset = dataset.to_torch()
    net.load_state_dict(results["net"])
    snapshots = results["snapshots"]
    evaluate_model(dataset, net, snapshots, device=device)

    outputs = net.outputs

    plot_outputs([outputs[0], outputs[2], outputs[4]], index)
    plot_outputs([outputs[6], outputs[9], outputs[10]], index)


def image_data_sample():
    data = np.load("coco_minival.npz")
    images = data["images"]

    plt.rc('font', size=15)
    fig = plt.figure(figsize=(12, 4))
    image = images[1]
    sample = np.random.uniform(0, 1, size=image.shape)
    scramble = image.copy().reshape(-1)
    np.random.shuffle(scramble)
    scramble = scramble.reshape(image.shape)

    ax = fig.add_subplot(131)
    ax.imshow(sample, interpolation='nearest', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Sample")

    ax = fig.add_subplot(132)
    ax.imshow(image, interpolation='nearest', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Structured")

    ax = fig.add_subplot(133)
    ax.imshow(scramble, interpolation='nearest', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Unstructured")

    plt.tight_layout()
    plt.show()


Segmentation = NamedTuple("Segmentation", [("img", np.ndarray), ("seg", np.ndarray), ("gt", np.ndarray)])


def segment(dataset, num_images: int, batch_size: int) -> List[Segmentation]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    images = dataset["images"]
    cat_names = dataset["cat_names"].tolist()
    cat_ids = dataset["cat_ids"].tolist()
    cat_colors = dataset["cat_colors"]

    weights = FCN_ResNet50_Weights.DEFAULT
    model = fcn_resnet50(weights=weights)
    model.to(device)
    model.eval()

    preprocess = weights.transforms(antialias=True)
    class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
    coco_to_voc = {
        "airplane": "aeroplane",
        "dining table": "diningtable",
        "motorcycle": "motorbike",
        "potted plant": "pottedplant",
        "couch": "sofa",
        "tv": "tvmonitor"
    }

    result = []
    start = 0
    progress = tqdm(total=num_images)
    while len(result) < num_images and start < len(images):
        end = min(start + batch_size, len(images))
        batch = torch.from_numpy(dataset["images"][start:end]).permute(0, 3, 1, 2).float() / 255.0
        batch = preprocess(batch).to(device)

        prediction = model(batch)["out"]
        segs = prediction.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)

        for i in range(start, end):
            gt = dataset["annotations"][i]
            gt_labels = np.unique(gt)
            gt_palette = np.zeros((256, 3), np.uint8)
            seg_palette = np.zeros_like(gt_palette)
            valid = False
            for label in gt_labels:
                if label == 0:
                    continue

                cat_id = cat_ids.index(label)
                cat_name = cat_names[cat_id]
                if cat_name in coco_to_voc:
                    cat_name = coco_to_voc[cat_name]

                if cat_name not in class_to_idx:
                    continue

                valid = True
                gt_palette[label] = cat_colors[cat_id]
                seg_palette[class_to_idx[cat_name]] = cat_colors[cat_id]

            if valid:
                gt = Image.fromarray(gt, "P")
                gt.putpalette(gt_palette.reshape(-1).tolist())

                seg = Image.fromarray(segs[i - start], "P")
                seg.putpalette(seg_palette.reshape(-1).tolist())
                result.append(Segmentation(dataset["images"][i], seg, gt))
                progress.update(1)
                if len(result) == num_images:
                    break

        start = end

    progress.close()
    return result


def plot_coco_example(dataset, index):
    images = dataset["images"]
    annotations = dataset["annotations"]
    image_ids = dataset["image_ids"]
    image_objects = dataset["image_objects"]
    bboxes = dataset["bboxes"]
    cat_colors = dataset["cat_colors"] / 255.0
    palette = dataset["cat_palette"].tolist()

    img = images[index]
    ann = annotations[index]
    img_mask = image_objects[:, 0] == image_ids[index]
    img_objs = image_objects[img_mask]
    img_bboxes = bboxes[img_mask]

    plt.rc('font', size=15)
    plt.figure(figsize=(12, 4))
    ax = plt.subplot(1, 3, 1)
    ax.imshow(img)
    ax.set_title("Image")
    ax.axis('off')

    ax = plt.subplot(1, 3, 2)
    ax.set_title("Segmentation")

    ann = Image.fromarray(ann, "P")
    ann_ids = np.unique(ann)
    for ann_id in ann_ids:
        if ann_id == 0:
            continue

    ann.putpalette(palette)
    ax.imshow(ann)
    ax.axis('off')

    ax = plt.subplot(1, 3, 3)
    ax.set_title("Objects")
    ax.imshow(img)
    ax.axis('off')
    for obj, bbox in zip(img_objs, img_bboxes):
        ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                   color=cat_colors[obj[1]],
                                   fill=True, alpha=0.4))
        ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                   fill=False, edgecolor=cat_colors[obj[1]], linewidth=3))

    plt.tight_layout()


def fcn_coco():
    dataset = load_coco("minival")
    segs = segment(dataset, 1, 1)
    seg = segs[0]

    plt.rc('font', size=15)
    plt.figure(figsize=(12, 4))
    ax = plt.subplot(1, 3, 1)
    ax.imshow(seg.img)
    ax.axis('off')
    ax.set_title("Image")

    ax = plt.subplot(1, 3, 2)
    ax.imshow(seg.seg)
    ax.axis('off')
    ax.set_title("Segmentation")

    ax = plt.subplot(1, 3, 3)
    ax.imshow(seg.gt)
    ax.axis('off')
    ax.set_title("Ground truth")

    plt.tight_layout()
    plt.show()


def main():
    mnist_simple()
    mnist_simple_filters()
    mnist_pool()
    mnist_pool_filters()
    cifar()
    cifar_filters()
    fcn_simple()
    fcn_simple_outputs()
    image_data_sample()
    fcn_coco()


if __name__ == "__main__":
    main()

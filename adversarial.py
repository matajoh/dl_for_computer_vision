import os
import pickle
import sys
from typing import List, NamedTuple

import matplotlib.pyplot as plt
from mlp import AttackSnapshot, Snapshot  # noqa: F401 — needed for torch.load unpickling
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights

from datasets import MulticlassDataset, DATA_DIR

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

Snap = NamedTuple("Snap", [("step", int), ("p0", float), ("p1", float)])


def _load_results(path):
    _current_module = sys.modules[__name__]

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "__main__":
                obj = getattr(_current_module, name, None)
                if obj is not None:
                    return obj
            return super().find_class(module, name)

    _pickle_module = type(sys)('_pickle_compat')
    _pickle_module.Unpickler = _Unpickler
    _pickle_module.load = pickle.load
    _pickle_module.UnpicklingError = pickle.UnpicklingError
    return torch.load(path, weights_only=False, pickle_module=_pickle_module)


def to_np_image(x: torch.Tensor, natural_image=False) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    x = x.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    if natural_image:
        x = x * std + mean
        x = np.clip(x, 0, 1)
        return (x * 255).astype(np.uint8)

    return np.abs(x).sum(-1)


def plot_explore(fig: plt.Figure, snapshots, examples: list, labels: List[str],
                 step: int, row: int = 0, nrows: int = 1,
                 vmax: float = None, cmap="viridis", reshape=None):
    x, dx = examples[step]

    ax0 = fig.add_subplot(nrows, 3, row * 3 + 1)
    if reshape:
        x = x.reshape(reshape)
        dx = dx.reshape(reshape)
    ax0.imshow(x, cmap="gray" if len(x.shape) == 2 else None,
               interpolation='nearest', vmin=0, vmax=1 if len(x.shape) == 2 else None)
    ax0.set_xticks([])
    ax0.set_yticks([])
    if row == 0:
        ax0.set_title("Altered Image")

    ax1 = fig.add_subplot(nrows, 3, row * 3 + 2)
    kw = {"interpolation": "nearest"}
    if vmax is not None:
        if len(dx.shape) == 2 and dx.min() < 0:
            kw["vmin"] = -vmax
            kw["vmax"] = vmax
        else:
            kw["vmin"] = 0
            kw["vmax"] = vmax
    ax1.imshow(dx, cmap=cmap, **kw)
    ax1.set_xticks([])
    ax1.set_yticks([])
    if row == 0:
        ax1.set_title("Difference")

    num_instances, p0, p1 = snapshots[step]

    ax2 = fig.add_subplot(nrows, 3, row * 3 + 3)
    sx = [snapshots[i][0] for i in range(len(snapshots))]
    y1 = [snapshots[i][1] for i in range(len(snapshots))]
    y2 = [snapshots[i][2] for i in range(len(snapshots))]

    ax2.plot(sx, y1, 'b-', label=labels[0])
    ax2.plot(sx, y2, 'g--', label=labels[1])
    ax2.set_ylim(0, 1)
    ax2.plot(num_instances, p0, 'bo', ms=12)
    ax2.plot(num_instances, p1, 'go', ms=12)
    ax2.set_xticks([])
    if row == 0:
        ax2.set_title("Prediction")
        ax2.legend(loc="center right")


MNIST_SWITCH = 27
MNIST_STEPS = 54  # equal steps either side of crossover


def generate_mnist_attack():
    from mlp import MultiLayerPerceptron

    dataset = MulticlassDataset.mnist()
    mlp_path = os.path.join(RESULTS_DIR, "mnist.results")
    if not os.path.exists(mlp_path):
        raise FileNotFoundError(f"Train MLP first by running mlp.py: {mlp_path}")

    mlp_results = _load_results(mlp_path)
    net = MultiLayerPerceptron(784, 64, 10)
    net.load_state_dict(mlp_results["snapshots"][-1].state_dict)
    net.eval()

    def criterion(x0, x, y, t):
        return F.cross_entropy(y, t) + 0.01 * F.mse_loss(x, x0, reduction='sum')

    x = torch.from_numpy(dataset.train.values[2:3].copy())
    t0 = dataset.train.labels[2:3].copy().item()
    t1 = dataset.train.labels[4:5].copy().item()
    t = torch.from_numpy(dataset.train.labels[4:5].copy())
    x0 = x.clone().detach()
    x.requires_grad = True

    p = net(x0).softmax(1)
    optimizer = optim.SGD([x], lr=0.01)
    snapshots = []
    examples = []

    snapshots.append(AttackSnapshot(0, p[0, t0].item(), p[0, t1].item()))
    examples.append((x.data.numpy().copy(), np.zeros((28, 28), np.float32)))

    for i in range(MNIST_STEPS):
        optimizer.zero_grad()
        y = net(x)
        p = y.softmax(1)
        p0 = p[0, t0].item()
        p1 = p[0, t1].item()

        loss = criterion(x0, x, y, t)
        loss.backward()
        optimizer.step()
        x.data = x.data.clamp(0, 1)
        snapshots.append(AttackSnapshot(i, p0, p1))
        examples.append((x.data.numpy().copy(), (x - x0).data.numpy().copy()))

    labels = [str(t0), str(t1)]

    path = os.path.join(RESULTS_DIR, "mnist_attack.results")
    results = {"snapshots": snapshots, "examples": examples, "labels": labels}
    torch.save(results, path)
    print(f"Saved {path}")
    return results


def mnist_attack():
    path = os.path.join(RESULTS_DIR, "mnist_attack.results")
    if os.path.exists(path):
        results = _load_results(path)
        print(f"{path} loaded")
    else:
        results = generate_mnist_attack()

    snapshots = results["snapshots"]
    examples = results["examples"]
    labels = results["labels"]

    plt.rc('font', size=15)
    fig = plt.figure(figsize=(12, 12))
    plot_explore(fig, snapshots, examples, labels, step=0, row=0, nrows=3, vmax=0.3, reshape=(28, 28))
    plot_explore(fig, snapshots, examples, labels, step=MNIST_SWITCH, row=1, nrows=3, vmax=0.3, reshape=(28, 28))
    plot_explore(fig, snapshots, examples, labels, step=len(snapshots) - 1, row=2, nrows=3, vmax=0.3, reshape=(28, 28))
    plt.tight_layout()
    plt.show()


PANDA_SWITCH = 24
PANDA_STEPS = 48  # equal steps either side of crossover


def generate_panda_attack(target_label="soccer ball"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = read_image(os.path.join(DATA_DIR, "panda.jpg"))

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).to(device)
    model.eval()

    preprocess = weights.transforms()
    batch = preprocess(img).unsqueeze(0).to(device)

    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")

    def criterion(x0, x, y, t):
        return F.cross_entropy(y, t) + 4e-1 * F.mse_loss(x, x0, reduction='sum')

    x = batch.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x])
    t0 = class_id
    t1 = weights.meta["categories"].index(target_label)
    print(f"Target: {category_name} ({t0}) -> {target_label} ({t1})")
    t = torch.tensor([t1], dtype=torch.long, device=device)
    x0 = x.clone().detach()
    x0_min = x0.min().item()
    x0_max = x0.max().item()

    snapshots = []
    examples = []

    snapshots.append(Snap(0, prediction[t0].item(), prediction[t1].item()))
    examples.append((to_np_image(x, True), np.zeros((224, 224, 3), np.float32)))

    for i in range(PANDA_STEPS):
        optimizer.zero_grad()
        y = model(x)

        p = y.softmax(1)
        p0 = p[0, t0].item()
        p1 = p[0, t1].item()

        loss = criterion(x0, x, y, t)
        loss.backward()
        optimizer.step()
        x.data = torch.clamp(x.data, x0_min, x0_max)

        snapshots.append(Snap(i, p0, p1))
        examples.append((to_np_image(x, True), to_np_image(x - x0, False)))
        x.grad.data.zero_()

    max_diff = max([diff.max() for _, diff in examples])
    labels = [category_name, target_label]

    path = os.path.join(RESULTS_DIR, "panda_attack.results")
    results = {"snapshots": snapshots, "examples": examples, "labels": labels, "max_diff": max_diff}
    torch.save(results, path)
    print(f"Saved {path}")
    return results


def panda_attack():
    path = os.path.join(RESULTS_DIR, "panda_attack.results")
    if os.path.exists(path):
        results = _load_results(path)
        print(f"{path} loaded")
    else:
        results = generate_panda_attack()

    snapshots = results["snapshots"]
    examples = results["examples"]
    labels = results["labels"]
    max_diff = results["max_diff"]

    plt.rc('font', size=15)
    fig = plt.figure(figsize=(12, 12))
    plot_explore(fig, snapshots, examples, labels, step=0, row=0, nrows=3, vmax=max_diff)
    plot_explore(fig, snapshots, examples, labels, step=PANDA_SWITCH, row=1, nrows=3, vmax=max_diff)
    plot_explore(fig, snapshots, examples, labels, step=len(snapshots) - 1, row=2, nrows=3, vmax=max_diff)
    plt.tight_layout()
    plt.show()


def load_mnist_attack():
    path = os.path.join(RESULTS_DIR, "mnist_attack.results")
    return _load_results(path)


def load_panda_attack():
    path = os.path.join(RESULTS_DIR, "panda_attack.results")
    return _load_results(path)


if __name__ == "__main__":
    # Generation
    generate_mnist_attack()
    generate_panda_attack()

    # Visualization
    mnist_attack()
    panda_attack()

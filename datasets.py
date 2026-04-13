import json
import os
from typing import Any, Callable, List, NamedTuple, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def _data_path(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)


def download(url: str, path: str):
    print(f"Downloading {url} to {path}")
    with open(path, 'wb') as file:
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=path) as progress:
            response = requests.get(url, stream=True)
            for chunk in response.iter_content(chunk_size=512):
                if chunk:
                    file.write(chunk)
                    progress.update(len(chunk))


def load_mnist():
    url = "https://matajohdata.blob.core.windows.net/datasets/cued/mnist.npz"
    path = _data_path("mnist.npz")
    if not os.path.exists(path):
        download(url, path)

    return np.load(path)


def load_cifar():
    url = "https://matajohdata.blob.core.windows.net/datasets/cued/cifar10.npz"
    path = _data_path("cifar10.npz")
    if not os.path.exists(path):
        download(url, path)

    return np.load(path)


def _create_coco_demo():
    import glob
    coco_dir = os.path.join(DATA_DIR, "coco")
    jpg_paths = sorted(glob.glob(os.path.join(coco_dir, "*.jpg")))
    local_ids = [int(os.path.splitext(os.path.basename(p))[0]) for p in jpg_paths]

    full = load_coco("minival")
    full_ids = full["image_ids"]
    indices = []
    for lid in local_ids:
        idx = np.where(full_ids == lid)[0]
        if len(idx) == 1:
            indices.append(idx[0])

    indices = np.array(indices)
    subset_image_ids = full_ids[indices]

    obj_mask = np.isin(full["image_objects"][:, 0], subset_image_ids)

    path = _data_path("coco_demo.npz")
    np.savez(
        path,
        metadata=full["metadata"],
        images=full["images"][indices],
        image_licenses=full["image_licenses"][indices],
        annotations=full["annotations"][indices],
        bboxes=full["bboxes"][obj_mask],
        image_ids=subset_image_ids,
        image_objects=full["image_objects"][obj_mask],
        cat_names=full["cat_names"],
        cat_ids=full["cat_ids"],
        cat_colors=full["cat_colors"],
        cat_palette=full["cat_palette"],
        object_ids=full["object_ids"],
        object_names=full["object_names"],
    )
    return path


def load_coco(name: str):
    if name == "minitrain":
        url = "https://matajohdata.blob.core.windows.net/datasets/cued/coco_minitrain.npz"
    elif name == "demo":
        url = "https://matajohdata.blob.core.windows.net/datasets/cued/coco_demo.npz"
    else:
        url = "https://matajohdata.blob.core.windows.net/datasets/cued/coco_minival.npz"

    path = _data_path(f"coco_{name}.npz")
    if not os.path.exists(path):
        download(url, path)

    return np.load(path)


def load_lfw(name: str):
    if name == "train":
        url = "https://matajohdata.blob.core.windows.net/datasets/cued/lfw_train.npz"
    else:
        url = "https://matajohdata.blob.core.windows.net/datasets/cued/lfw_test.npz"

    path = _data_path(f"lfw_{name}.npz")
    if not os.path.exists(path):
        download(url, path)

    return np.load(path)


def load_emnist():
    url = "https://matajohdata.blob.core.windows.net/datasets/cued/emnist.npz"
    path = _data_path("emnist.npz")
    if not os.path.exists(path):
        download(url, path)

    return np.load(path)


def load_postcodes():
    url = "https://matajohdata.blob.core.windows.net/datasets/cued/postcodes_small.npz"
    path = _data_path("postcodes_small.npz")
    if not os.path.exists(path):
        download(url, path)

    return np.load(path)


def to_patches(pixels: np.ndarray) -> np.ndarray:
    patches = np.zeros((pixels.shape[0], 7*7, 16), np.float32)
    for i in range(7):
        r = i * 4
        for j in range(7):
            c = j * 4
            patches[:, i*7 + j] = pixels[:, r:r+4, c:c+4].reshape(-1, 16)

    return patches


def read_mnist_patches():
    mnist = load_mnist()
    pixels = mnist["data"]
    pixels = pixels.astype(np.float32) / 255
    pixels = pixels.reshape((-1, 28, 28))

    patches = to_patches(pixels)
    return patches[:60000], patches[60000:]


def read_emnist_patches():
    emnist = load_emnist()
    train_pixels = emnist["train_images"]
    train_pixels = train_pixels.astype(np.float32) / 255

    test_pixels = emnist["test_images"]
    test_pixels = test_pixels.astype(np.float32) / 255

    return to_patches(train_pixels), to_patches(test_pixels)


def cluster_patches(path: str, dataset, K=32, num_steps=30, N=200000):
    train, val = dataset
    D = train.shape[-1]

    patches = train.reshape(-1, D)
    index = np.arange(min(N, len(patches)))
    np.random.shuffle(index)
    patches = patches[index]
    N = len(patches)

    # Kickstart with random cluster assignment
    membership = np.arange(N) % K
    np.random.shuffle(membership)

    clusters = np.zeros((K, D), np.float64)
    counts = []
    for k in range(K):
        members = membership == k
        counts.append(members.sum())
        clusters[k] = patches[members].sum(0) / members.sum()

    counts = np.array(counts)

    with tqdm(range(num_steps)) as pb:
        for step in pb:
            temp = clusters.reshape(1, K, D) - patches.reshape(N, 1, D)
            sq_dist = np.square(temp).sum(2)
            new_membership = sq_dist.argmin(-1)

            changed = (membership != new_membership).sum()
            membership = new_membership
            for k in range(K):
                members = membership == k
                count = members.sum()
                if count > 0:
                    clusters[k] = patches[members].sum(0) / count
                elif step % 7 == 0:
                    clusters[k] = np.random.uniform(size=patches[0].shape)
                else:
                    clusters[k] = np.full_like(patches[0], 0.5)

            pb.set_postfix({"#d": changed})
            if changed == 0:
                break

    merged = []
    cluster_set = set()
    for k, c in enumerate(clusters):
        members = membership == k
        if members.any():
            discrete = tuple([int(p.item() * 255) for p in c])
            if discrete not in cluster_set:
                merged.append(c)
                cluster_set.add(discrete)

    print("merged", len(clusters), "to", len(merged), "based on appearance")
    clusters = np.array(merged, np.float32)

    def assign_to_cluster(values: np.ndarray) -> np.ndarray:
        codes = []
        for x in tqdm(values):
            x = x.reshape(-1, 1, D)
            temp = clusters.reshape(1, K, D) - x
            sq_dist = np.square(temp).sum(2)
            codes.append(sq_dist.argmin(-1))

        return np.array(codes, np.uint8)

    train_codes = assign_to_cluster(train)
    val_codes = assign_to_cluster(val)
    np.savez(path, dists=clusters,
             tokens=(clusters * 255).astype(np.uint8),
             train=train_codes, val=val_codes)


def load_mnist_patches(K=64, num_steps=100):
    path = _data_path(f"mnist_patch{K}.npz")
    if not os.path.exists(path):
        url = f"https://matajohdata.blob.core.windows.net/datasets/cued/mnist_patch{K}.npz"
        try:
            download(url, path)
        except Exception:
            print(f"Download failed, building mnist_patch{K} locally...")
            cluster_patches(path, read_mnist_patches(), K, num_steps)

    return np.load(path)


def load_emnist_patches(K=128, num_steps=100):
    path = _data_path(f"emnist_patch{K}.npz")
    if not os.path.exists(path):
        url = f"https://matajohdata.blob.core.windows.net/datasets/cued/emnist_patch{K}.npz"
        try:
            download(url, path)
        except Exception:
            print(f"Download failed, building emnist_patch{K} locally...")
            cluster_patches(path, read_emnist_patches(), K, num_steps)

    return np.load(path)


class BinaryDataset:
    def __init__(self, positive: np.ndarray, negative: np.ndarray):
        self.positive = positive
        self.negative = negative

    def generate_from_normals(num_points: int) -> "BinaryDataset":
        positive = np.ones((num_points, 3))
        negative = np.ones((num_points, 3))
        positive[:, :-1] = np.random.multivariate_normal([0, -2.5], [[0.5, 0], [0, 0.5]], num_points)
        negative[:, :-1] = np.random.multivariate_normal([-2, 2.5], [[1, 0], [0, 1]], num_points)
        return BinaryDataset(positive, negative)

    def generate_xor(num_instances: int) -> "BinaryDataset":
        positive = np.ones((200, 3))
        negative = np.ones((200, 3))
        positive[:100, :-1] = np.random.multivariate_normal([7, 7], [[1, 0], [0, 1]], 100)
        positive[100:, :-1] = np.random.multivariate_normal([-7, -7], [[1, 0], [0, 1]], 100)
        negative[:100, :-1] = np.random.multivariate_normal([7, -7], [[1, 0], [0, 1]], 100)
        negative[100:, :-1] = np.random.multivariate_normal([-7, 7], [[1, 0], [0, 1]], 100)
        np.random.shuffle(positive)
        np.random.shuffle(negative)
        return BinaryDataset(positive, negative)

    def mnist(positive_label: int, negative_label: int) -> "BinaryDataset":
        mnist = load_mnist()
        data = mnist['data'].astype(np.float32) / 255
        target = mnist['target'].astype(np.int32)

        positive_indices = target == positive_label
        negative_indices = target == negative_label

        positive_images = data[positive_indices]
        negative_images = data[negative_indices]

        return BinaryDataset(positive_images, negative_images)

    def generate_bullseye(num_instances: int) -> "BinaryDataset":
        positive = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], num_instances)

        points = []
        while len(points) < num_instances:
            samples = np.random.uniform(-10, 10, (5*num_instances, 2))
            for sample in samples:
                if 5 < np.linalg.norm(sample, 2) < 8:
                    points.append(sample)

        negative = np.array(points[:num_instances])

        return BinaryDataset(positive, negative)

    def generate_sin(num_instances: int) -> "BinaryDataset":
        pass

    def plot(self, ax: plt.Axes):
        min_max = self.min_max()
        ax.set_xlim(min_max[0], min_max[1])
        ax.set_ylim(min_max[0], min_max[1])
        ax.plot(self.positive[:, 0], self.positive[:, 1], 'b+')
        ax.plot(self.negative[:, 0], self.negative[:, 1], 'r.')

    def save(self, filename: str):
        plt.figure(figsize=(4, 4))
        self.plot(plt.subplot(111))
        plt.savefig(filename)

    def min_max(self) -> Tuple[int, int]:
        examples = np.stack([self.positive[:, :2], self.negative[:, :2]])
        min_val = np.amin(examples)
        max_val = np.amax(examples)
        return int(min_val) - 1, int(max_val) + 1

    @property
    def num_dims(self) -> int:
        return self.positive.shape[1]


class Data(NamedTuple("Data", [("values", np.ndarray), ("labels", np.ndarray)])):
    def plot(self, ax: plt.Axes, colors: Tuple[str, ...], markers: Tuple[str, ...]):
        ax.set_xlim(*self.limits)
        ax.set_ylim(*self.limits)

        for i in range(self.num_classes):
            points = self.values[np.where(self.labels == i)]
            ax.scatter(points[:, 0], points[:, 1], c=colors[i], marker=markers[i])

    @property
    def limits(self) -> Tuple[float, float]:
        min_val = np.amin(self.values).astype('int32') - 1
        max_val = np.amax(self.values).astype('int32') + 1
        return min_val, max_val

    def subset(self, idx: np.ndarray) -> "Data":
        values = self.values[idx].copy()
        labels = self.labels[idx].copy()
        return Data(values, labels)

    def values_for(self, label: int) -> np.ndarray:
        return self.values[np.where(self.labels == label)]

    def to_torch(self) -> TensorDataset:
        values = torch.from_numpy(self.values)
        labels = torch.from_numpy(self.labels)
        return TensorDataset(values, labels)

    def normalize(self, mean: np.ndarray, std: np.ndarray) -> "Data":
        values = (self.values - mean) / std
        return Data(values, self.labels)

    def apply_feature(self, feature: Callable[[np.ndarray], np.ndarray]) -> "Data":
        values = feature(self.values)
        return Data(values, self.labels)

    @property
    def num_classes(self) -> int:
        return self.labels.max() + 1

    def generate_two_class(num_instances):
        positive_points = np.random.multivariate_normal([1, 2], [[1.5, 0], [0, 1.5]], num_instances)
        negative_points = np.random.multivariate_normal([2, -2], [[0.5, 0], [0, 1]], num_instances)

        values = np.concatenate((positive_points, negative_points), 0)
        labels = np.array([(0 if i < num_instances else 1) for i in range(values.shape[0])], dtype='int32')

        return Data(values, labels)

    def generate_three_class(num_instances: int) -> "Data":
        class0 = np.random.multivariate_normal([1.5, -2], [[1, 0], [0, 1]], num_instances).astype('float32')
        class1 = np.random.multivariate_normal([1, 2], [[0.5, 0], [0, 0.5]], num_instances).astype('float32')
        class2 = np.random.multivariate_normal([-2, 0], [[1, 0], [0, 0.5]], num_instances).astype('float32')

        values = np.concatenate((class0, class1, class2), 0)
        labels = np.array([i // num_instances for i in range(values.shape[0])], dtype='int32')

        return Data(values, labels)

    def generate_xor(num_instances):
        num_instances = num_instances // 2
        quadrant0 = np.random.multivariate_normal([7, 7], [[1, 0], [0, 1]], num_instances)
        quadrant2 = np.random.multivariate_normal([-7, -7], [[1, 0], [0, 1]], num_instances)
        quadrant1 = np.random.multivariate_normal([7, -7], [[1, 0], [0, 1]], num_instances)
        quadrant3 = np.random.multivariate_normal([-7, 7], [[1, 0], [0, 1]], num_instances)

        values = np.concatenate([quadrant0, quadrant2, quadrant1, quadrant3], 0).astype(np.float32)
        labels = np.array([i // (2*num_instances) for i in range(values.shape[0])], dtype=np.int64)

        return Data(values, labels)

    def generate_bullseye(num_instances):
        class0 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], num_instances)

        points = []
        while len(points) < num_instances:
            samples = np.random.uniform(-10, 10, (5*num_instances, 2))
            for sample in samples:
                if 5 < np.linalg.norm(sample, 2) < 8:
                    points.append(sample)

        class1 = np.array(points[:num_instances])

        values = np.concatenate([class0, class1], axis=0).astype(np.float32)
        labels = np.array([0] * num_instances + [1] * num_instances).astype(np.int64)
        return Data(values, labels)


class MulticlassDataset:
    def __init__(self, train: Union[Data, TensorDataset], val: Union[Data, TensorDataset],
                 label_names: List[str] = None):
        if isinstance(train, TensorDataset):
            self.shape = train.tensors[0].shape[1:]
        else:
            self.shape = train.values.shape[1:]

        self.train = train
        self.val = val
        self.label_names = label_names
        self.mean = np.zeros((1,) + self.shape, np.float32)
        self.std = np.ones((1,) + self.shape, np.float32)

    def normalize(self) -> "MulticlassDataset":
        mean = self.train.values.mean(0, keepdims=True)
        std = self.train.values.std(0, keepdims=True)
        train = self.train.normalize(mean, std)
        val = self.val.normalize(mean, std)
        dataset = MulticlassDataset(train, val, self.label_names)
        dataset.mean = mean
        dataset.std = std
        return dataset

    def unnormalize(self, value: np.ndarray) -> np.ndarray:
        if len(value.shape) == 3:
            return (value.reshape((1,) + value.shape) * self.std + self.mean).reshape(value.shape)

        return value * self.std + self.mean

    def apply_feature(self, feature: Callable[[np.ndarray], np.ndarray]) -> "MulticlassDataset":
        train = self.train.apply_feature(feature)
        val = self.val.apply_feature(feature)
        return MulticlassDataset(train, val, self.label_names)

    def to_torch(self) -> "MulticlassDataset":
        dataset = MulticlassDataset(self.train.to_torch(), self.val.to_torch(), self.label_names)
        dataset.mean = self.mean
        dataset.std = self.std
        return dataset

    def plot(self, ax: plt.Axes):
        self.val.plot(ax, ('b', 'r', 'g'), ('+', '.', '*'))

    def save(self, filename: str):
        plt.figure(figsize=(4, 4))
        self.plot()
        plt.savefig(filename)

    def generate_two_class(num_train: int, num_val: int) -> "MulticlassDataset":
        train = Data.generate_two_class(num_train)
        val = Data.generate_two_class(num_val)
        return MulticlassDataset(train, val)

    def generate_three_class(num_train: int, num_val: int) -> "MulticlassDataset":
        train = Data.generate_three_class(num_train)
        val = Data.generate_three_class(num_val)
        return MulticlassDataset(train, val)

    def generate_xor(num_train: int, num_val: int) -> "MulticlassDataset":
        train = Data.generate_xor(num_train)
        val = Data.generate_xor(num_val)
        return MulticlassDataset(train, val)

    def generate_bullseye(num_train: int, num_val: int) -> "MulticlassDataset":
        train = Data.generate_bullseye(num_train)
        val = Data.generate_bullseye(num_val)
        return MulticlassDataset(train, val)

    def mnist(as_images=False) -> "MulticlassDataset":
        mnist = load_mnist()
        data = mnist['data'].astype(np.float32) / 255
        if as_images:
            data = data.reshape((-1, 1, 28, 28))
            if as_images != 28:
                padded = np.zeros((data.shape[0], 1, as_images, as_images), np.float32)
                x = (as_images - 28) // 2
                padded[:, :, x:x+28, x:x+28] = data
                data = padded

        target = mnist['target'].astype(np.int64)

        train = Data(data[:60000], target[:60000])
        val = Data(data[60000:], target[60000:])
        label_names = [str(i) for i in range(10)]
        return MulticlassDataset(train, val, label_names)

    def emnist() -> "MulticlassDataset":
        emnist = load_emnist()
        train_images = emnist["train_images"].astype(np.float32) / 255
        train_labels = emnist["train_labels"].astype(np.int64)
        val_images = emnist["test_images"].astype(np.float32) / 255
        val_labels = emnist["test_labels"].astype(np.int64)
        train_images = train_images.reshape(-1, 1, 28, 28)
        val_images = val_images.reshape(-1, 1, 28, 28)
        train = Data(train_images, train_labels)
        val = Data(val_images, val_labels)
        label_names = emnist["label_names"]
        return MulticlassDataset(train, val, label_names)

    def cifar(raw=False) -> "MulticlassDataset":
        cifar = load_cifar()

        if raw:
            cifar_data = cifar["data"]
            cifar_data = cifar_data.reshape(cifar_data.shape[0], 3, 32, 32)
            cifar_data = np.moveaxis(cifar_data, 1, 3)
            cifar_target = cifar["target"]
        else:
            cifar_data = cifar['data'].astype(np.float32)
            cifar_data = cifar_data.reshape(cifar_data.shape[0], 3, 32, 32)
            cifar_data /= 255
            cifar_target = cifar['target'].astype(np.int64)

        N_val = 10000
        val = Data(cifar_data[:N_val], cifar_target[:N_val])
        train = Data(cifar_data[N_val:], cifar_target[N_val:])

        label_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        return MulticlassDataset(train, val, label_names)

    def coco(raw=False) -> "MulticlassDataset":
        print("Loading coco...")
        train = load_coco("minitrain")
        val = load_coco("minival")

        def convert_images(raw_images: np.ndarray) -> np.ndarray:
            if raw:
                return raw_images

            images = raw_images.astype(np.float32) / 255.0
            return np.moveaxis(images, 3, 1)

        def convert_labels(raw_labels: np.ndarray, values: List[int]) -> np.ndarray:
            if raw:
                return raw_labels

            values = np.array(values, np.uint8)
            lookup = np.zeros(values.max() + 1, dtype=np.uint8)
            for i, value in enumerate(values):
                lookup[value] = i

            labels = raw_labels.reshape(-1)
            labels = lookup[labels]
            return labels.reshape(raw_labels.shape)

        cat_ids = [0] + train["cat_ids"].tolist()
        label_names = ["background"] + train["cat_names"].tolist()

        print("Converting images...")
        train_images = convert_images(train['images'])
        val_images = convert_images(val['images'])

        print("Convert labels...")
        train_labels = convert_labels(train["annotations"], cat_ids)
        val_labels = convert_labels(val["annotations"], cat_ids)

        print("Complete")
        train = Data(train_images, train_labels)
        val = Data(val_images, val_labels)
        return MulticlassDataset(train, val, label_names)


class MetricData(NamedTuple("MetricData", [("values", Union[np.ndarray, torch.Tensor]),
                                           ("positive", Union[np.ndarray, torch.Tensor]),
                                           ("negative", Union[np.ndarray, torch.Tensor]),
                                           ("ids", np.ndarray),
                                           ("labels", np.ndarray)])):
    def to_torch(self) -> "MetricData":
        return MetricData(torch.from_numpy(self.values),
                          torch.from_numpy(self.positive),
                          torch.from_numpy(self.negative),
                          self.ids, self.labels)


class MetricDataset:
    def __init__(self, train: MetricData, val: MetricData):
        self.train = train
        self.val = val

    def to_torch(self) -> "MetricDataset":
        return MetricDataset(torch.from_numpy(self.train), torch.from_numpy(self.val),
                             self.positive_pairs, self.negative_pairs)

    def lfw() -> "MetricDataset":
        train = load_lfw("train")
        val = load_lfw("test")

        def convert_images(raw_images: np.ndarray) -> np.ndarray:
            images = raw_images.astype(np.float32) / 255.0
            return np.moveaxis(images, 3, 1)

        print("Converting images...")
        train_images = convert_images(train['images'])
        val_images = convert_images(val['images'])
        print("Complete")

        train_positive = train["positive_pairs"]
        train_negative = train["negative_pairs"]
        train_ids = train["image_ids"]
        train_labels = np.array([s.decode("ascii") for s in train["names"]])
        val_positive = val["positive_pairs"]
        val_negative = val["negative_pairs"]
        val_ids = val["image_ids"]
        val_labels = np.array([s.decode("ascii") for s in val["names"]])

        train = MetricData(train_images, train_positive, train_negative, train_ids, train_labels)
        val = MetricData(val_images, val_positive, val_negative, val_ids, val_labels)
        return MetricDataset(train, val)

    def lfw_minitrain() -> "MetricDataset":
        path = _data_path("lfw_minitrain.npz")
        if not os.path.exists(path):
            url = "https://matajohdata.blob.core.windows.net/datasets/cued/lfw_minitrain.npz"
            try:
                download(url, path)
            except Exception:
                print("Download failed, building lfw_minitrain locally...")
                dataset = MetricDataset.lfw()
                counts = np.bincount(dataset.train.ids)
                id_by_counts = [[] for _ in range(counts.max())]
                for i in range(len(counts)):
                    id_by_counts[counts[i] - 1].append(i)

                train_ids = id_by_counts[-1] + id_by_counts[-2]

                idx = []
                for i in train_ids:
                    idx.append(np.where(dataset.train.ids == i)[0])

                idx = np.concatenate(idx)
                idx.sort()

                values = dataset.train.values[idx]
                ids = dataset.train.ids[idx]

                np.savez(path, values=values, ids=ids, labels=dataset.train.labels)

        data = np.load(path)
        train = MetricData(data["values"], np.empty(0), np.empty(0), data["ids"], data["labels"])
        return MetricDataset(train, None)


class SequenceDataset(NamedTuple("SequenceDataset", [("input_size", int),
                                                     ("num_classes", int),
                                                     ("max_length", int),
                                                     ("tokens", List[Any]),
                                                     ("train", Union[Data, TensorDataset]),
                                                     ("val", Union[Data, TensorDataset]),
                                                     ("label_names", list),
                                                     ("mean", np.ndarray),
                                                     ("std", np.ndarray)])):

    def to_torch(self) -> "SequenceDataset":
        return SequenceDataset(self.input_size,
                               self.num_classes,
                               self.max_length,
                               self.tokens,
                               self.train.to_torch(),
                               self.val.to_torch(),
                               self.label_names,
                               torch.from_numpy(self.mean) if self.mean is not None else None,
                               torch.from_numpy(self.std) if self.std is not None else None)

    def normalize(self) -> "SequenceDataset":
        mean = self.train.values.mean(0, keepdims=True)
        std = self.train.values.std(0, keepdims=True)
        train = self.train.normalize(mean, std)
        val = self.val.normalize(mean, std)
        return SequenceDataset(self.input_size, self.num_classes, self.max_length,
                               self.tokens, train, val, self.label_names,
                               mean, std)

    def unnormalize(self, value: np.ndarray) -> np.ndarray:
        if len(value.shape) == 2:
            return (value.reshape((1,) + value.shape) * self.std + self.mean).reshape(value.shape)

        return value * self.std + self.mean

    def start_dist(self) -> np.ndarray:
        if isinstance(self.train, Data):
            data = self.train.values
        else:
            data = self.train.tensors[0].numpy()

        counts = np.bincount(data[:, 0], minlength=self.input_size).astype(np.float32)
        counts = np.clip(counts, 1, None)
        return counts / counts.sum()

    def random_tokens(self, count: int, prefix=1) -> List[int]:
        if self.tokens:
            samples = np.random.choice(np.arange(self.input_size), count, p=self.start_dist())
            return [[s.item()] for s in samples]

        index = np.arange(len(self.train))
        np.random.shuffle(index)
        index = index[:count]
        samples, _ = self.train[index]
        tokens = samples[:, :prefix]
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()

        return tokens

    def to_token(self, logits: torch.Tensor, top_p=0.95, filter_value=-float('Inf'), temperature=1) -> torch.Tensor:

        if self.tokens is None:
            return logits.unsqueeze(0)

        logits = logits / temperature
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)

        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        sorted_logits[sorted_indices_to_remove] = filter_value
        logits = torch.gather(sorted_logits, -1, sorted_indices.argsort(-1))

        dist = F.softmax(logits, -1)

        samples = torch.multinomial(dist, 1)
        return samples

    @staticmethod
    def language(length: int, num_val: int) -> 'SequenceDataset':
        def word_to_tensor(word: str) -> np.ndarray:
            indices = [ord(c) - ord('a') for c in word]
            if not all(0 <= i < 26 for i in indices):
                raise ValueError(f"Invalid character in word: {word}")

            return np.array(indices, np.int64)

        with open(_data_path(f"language_{length}.json")) as f:
            dataset = json.load(f)

        japanese = np.stack([word_to_tensor(word) for word in dataset["japanese"]])
        english = np.stack([word_to_tensor(word) for word in dataset["english"]])
        french = np.stack([word_to_tensor(word) for word in dataset["french"]])

        order = np.arange(len(japanese))
        val_words = np.concatenate([japanese[order[:num_val]],
                                    english[order[:num_val]],
                                    french[order[:num_val]]])
        val_labels = np.concatenate([np.zeros(num_val, np.int64),
                                     np.ones(num_val, np.int64),
                                     np.full(num_val, 2, np.int64)])
        train_words = np.concatenate([japanese[order[num_val:]],
                                      english[order[num_val:]],
                                      french[order[num_val:]]])
        train_labels = np.concatenate([np.zeros(len(japanese) - num_val, np.int64),
                                       np.ones(len(english) - num_val, np.int64),
                                       np.full(len(french) - num_val, 2, np.int64)])
        train = Data(train_words, train_labels)
        val = Data(val_words, val_labels)
        input_size = 26
        tokens = [chr(c) for c in range(ord('a'), ord('z') + 1)]
        return SequenceDataset(input_size, 3, length, tokens,
                               train, val, ["japanese", "english", "french"],
                               None, None)

    @staticmethod
    def mnist(even_odd=False) -> 'SequenceDataset':
        mnist = load_mnist()
        patches = load_mnist_patches()

        targets = mnist["target"].astype(np.int64)

        if even_odd:
            targets = targets % 2

        train = Data(patches["train"].astype(np.int64), targets[:60000])
        val = Data(patches["val"].astype(np.int64), targets[60000:])
        if even_odd:
            label_names = ["even", "odd"]
        else:
            label_names = [str(i) for i in range(10)]

        input_size = train.values.max() + 1
        max_length = train.values.shape[1]
        tokens = patches["tokens"]
        return SequenceDataset(input_size, 2 if even_odd else 10, max_length,
                               tokens, train, val, label_names,
                               np.zeros((1, input_size), np.float32),
                               np.ones((1, input_size), np.float32))

    @staticmethod
    def emnist(tokens=False, alpha_num=False) -> 'SequenceDataset':
        emnist = load_emnist()

        train_labels = emnist["train_labels"].astype(np.int64)
        val_labels = emnist["test_labels"].astype(np.int64)

        if alpha_num:
            train_labels = np.where(train_labels < 10, 1, 0)
            val_labels = np.where(val_labels < 10, 1, 0)

        if tokens:
            patches = load_emnist_patches()
            train = Data(patches["train"].astype(np.int64), train_labels)
            val = Data(patches["val"].astype(np.int64), val_labels)
            tokens = patches["tokens"]
            input_size = train.values.max() + 1
        else:
            train_images = emnist["train_images"].astype(np.float32) / 255
            val_images = emnist["test_images"].astype(np.float32) / 255
            train = Data(to_patches(train_images), train_labels)
            val = Data(to_patches(val_images), val_labels)
            input_size = 16
            tokens = None

        if alpha_num:
            label_names = ["alpha", "num"]
        else:
            label_names = emnist["label_names"]

        num_classes = train_labels.max() + 1
        max_length = train.values.shape[1]
        return SequenceDataset(input_size, num_classes, max_length,
                               tokens, train, val, label_names,
                               np.zeros((1, input_size), np.float32),
                               np.ones((1, input_size), np.float32))

    @staticmethod
    def cifar() -> "SequenceDataset":
        cifar = load_cifar()
        cifar_data = cifar['data'].astype(np.float32)
        cifar_data = cifar_data.reshape(cifar_data.shape[0], 3, 32, 32)
        cifar_data /= 255
        cifar_data = np.moveaxis(cifar_data, 1, 3)
        cifar_target = cifar['target'].astype(np.int64)

        def to_patches(pixels: np.ndarray) -> np.ndarray:
            patches = np.zeros((pixels.shape[0], 8*8, 48), np.float32)
            for i in range(8):
                r = i * 4
                for j in range(8):
                    c = j * 4
                    patches[:, i*8 + j] = pixels[:, r:r+4, c:c+4].reshape(-1, 48)

            return patches

        N_val = 10000
        val_images, val_labels = cifar_data[:N_val], cifar_target[:N_val]
        train_images, train_labels = cifar_data[N_val:], cifar_target[N_val:]

        train_patches = to_patches(train_images)
        val_patches = to_patches(val_images)
        label_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        train = Data(train_patches, train_labels)
        val = Data(val_patches, val_labels)
        input_size = 48
        tokens = None
        return SequenceDataset(input_size, 10, 64, tokens, train, val, label_names,
                               np.zeros((1, input_size), np.float32),
                               np.ones((1, input_size), np.float32))


class PositionalDataset(NamedTuple("PositionalDataset", [("input_size", int), ("output_size", int),
                                                         ("train", Data), ("val", Data)])):
    @staticmethod
    def cat(size=512) -> 'PositionalDataset':
        image = Image.open(_data_path("cat.jpg"))

        pixels = np.array(image).astype(np.float32) / 255
        vals = np.linspace(0, 2, size // 2, endpoint=False, dtype=np.float32)
        train_uv = np.stack(np.meshgrid(vals, vals), axis=-1)
        train_color = pixels[::2, ::2, :]

        vals = np.linspace(0, 2, size, endpoint=False, dtype=np.float32)
        val_uv = np.stack(np.meshgrid(vals, vals), axis=-1)
        val_color = pixels

        return PositionalDataset(2, 3, Data(train_uv, train_color), Data(val_uv, val_color))

    @staticmethod
    def polynomial(num_samples: int, sample_rate: int) -> "PositionalDataset":
        x = np.linspace(-1, 1, num_samples * sample_rate, endpoint=False)
        y = -5*x**4 - 0.5*x**3 + 4.5*x**2 + 0.3*x - .15
        x = x.astype(np.float32).reshape(-1, 1)
        y = y.astype(np.float32).reshape(-1, 1)
        return PositionalDataset(1, 1, Data(x[::sample_rate], y[::sample_rate]), Data(x, y))

    @staticmethod
    def periodic(num_samples: int, sample_rate: int) -> "PositionalDataset":
        x = np.linspace(0, 2, num_samples * sample_rate, endpoint=False)
        y = 2 + np.sin(x * np.pi) + 0.5 * np.sin(2 * x * np.pi) - 0.2 * np.cos(5 * x * np.pi)
        x = x.astype(np.float32).reshape(-1, 1)
        y = y.astype(np.float32).reshape(-1, 1)
        return PositionalDataset(1, 1, Data(x[::sample_rate], y[::sample_rate]), Data(x, y))

    @staticmethod
    def example(num_samples: int, sample_rate: int) -> "PositionalDataset":
        x = np.linspace(0, 2, num_samples * sample_rate)
        y = 2*np.cos(x * np.pi) + 1/3 * np.sin(6 * x * np.pi) + 0.5 * np.cos(4 * x * np.pi)
        x = x.astype(np.float32).reshape(-1, 1)
        y = y.astype(np.float32).reshape(-1, 1)
        return PositionalDataset(1, 1, Data(x[::sample_rate], y[::sample_rate]), Data(x, y))


class TranslationDataset(NamedTuple("TranslationDataset", [("input_size", int),
                                                           ("output_size", int),
                                                           ("input_max_length", int),
                                                           ("output_max_length", int),
                                                           ("start_token", int),
                                                           ("end_token", int),
                                                           ("input_tokens", List[Any]),
                                                           ("output_tokens", List[Any]),
                                                           ("train", Data),
                                                           ("val", Data)])):
    def to_torch(self) -> "TranslationDataset":
        return TranslationDataset(self.input_size, self.output_size, self.input_max_length,
                                  self.output_max_length, self.start_token, self.end_token,
                                  self.input_tokens, self.output_tokens,
                                  self.train.to_torch(), self.val.to_torch())

    @staticmethod
    def postcodes():
        print("Loading postcodes...")
        postcodes = load_postcodes()

        print("Reading images and sequences...")
        train_images = postcodes["train_images"].astype(np.float32) / 255
        val_images = postcodes["val_images"].astype(np.float32) / 255
        train_sequences = postcodes["train_labels"]
        val_sequences = postcodes["val_labels"]

        rows, cols = train_images.shape[1:]
        rows = rows // 4
        cols = cols // 4

        def patchify(image: np.ndarray) -> np.ndarray:
            pbar = tqdm(total=rows * cols, desc="Creating patches")
            patches = np.zeros((image.shape[0], rows * cols, 16), np.float32)
            for i in range(rows):
                r = i * 4
                for j in range(cols):
                    c = j * 4
                    patches[:, i*cols + j] = image[:, r:r+4, c:c+4].reshape(-1, 16)
                    pbar.update(1)

            return patches

        tokens = " 0123456789abcdefghijklmnopqrstuvwxyz^$"

        def word_to_tensor(raw_sequences: np.ndarray) -> np.ndarray:
            sequences = np.zeros((len(raw_sequences), 8), np.int64)
            for s, sequence in tqdm(enumerate(raw_sequences), "Converting sequences", len(raw_sequences)):
                for i, c in enumerate(sequence):
                    if c == ' ':
                        continue

                    c = c.lower()
                    if c.isalpha():
                        sequences[s, i] = ord(c) - ord('a') + 11
                    else:
                        sequences[s, i] = ord(c) - ord('0') + 1

            return sequences

        train = Data(patchify(train_images), word_to_tensor(train_sequences))
        val = Data(patchify(val_images), word_to_tensor(val_sequences))
        start_token = len(tokens) - 2
        end_token = len(tokens) - 1
        return TranslationDataset(16, len(tokens), rows * cols, 8, start_token, end_token, None, tokens, train, val)

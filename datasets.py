import base64
import os
import pickle
from typing import Callable, List, NamedTuple, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm


def _create_onedrive_directdownload(onedrive_link: str):
    data_bytes64 = base64.b64encode(bytes(onedrive_link, "utf-8"))
    data_bytes64 = data_bytes64.decode("utf-8")
    data_bytes64 = data_bytes64.replace("/", "_").replace("+", "-").rstrip("=")
    return f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64}/root/content"


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
    url = _create_onedrive_directdownload("https://1drv.ms/u/s!AnWvK2b51nGql5MIRaiwi0M-0lnNIw")
    if not os.path.exists("mnist.pkl"):
        download(url, "mnist.pkl")

    with open("mnist.pkl", "rb") as mnist_pickle:
        return pickle.load(mnist_pickle, encoding='bytes')


def load_cifar():
    url = _create_onedrive_directdownload("https://1drv.ms/u/s!AnWvK2b51nGql5MJKmZLgbChcO7W3A")
    if not os.path.exists("cifar10.pkl"):
        download(url, "cifar10.pkl")

    with open("cifar10.pkl", "rb") as cifar_pickle:
        return pickle.load(cifar_pickle, encoding='bytes')


def load_coco(name: str):
    if name == "minitrain":
        url = _create_onedrive_directdownload("https://1drv.ms/u/s!AnWvK2b51nGql5MNTvePXNAC7FIaHQ")
    else:
        url = _create_onedrive_directdownload("https://1drv.ms/u/s!AnWvK2b51nGql5MMVa2fcxLmBtlvcA")

    path = f"coco_{name}.npz"
    if not os.path.exists(path):
        download(url, path)

    return np.load(path)


def load_lfw(name: str):
    if name == "train":
        url = _create_onedrive_directdownload("https://1drv.ms/u/s!AnWvK2b51nGql5MLX6uqC-ahrs_8gA")
    else:
        url = _create_onedrive_directdownload("https://1drv.ms/u/s!AnWvK2b51nGql5MKhMjA7RpH3ig_TA")

    path = f"lfw_{name}.npz"
    if not os.path.exists(path):
        download(url, path)

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
        mnist['data'] = mnist['data'].astype(np.float32)  # convert the uint8s to floats
        mnist['data'] /= 255  # scale to be from 0 to 1
        mnist['target'] = mnist['target'].astype(np.int32)  # convert the uint8s to int32s

        positive_indices = mnist["target"] == positive_label
        negative_indices = mnist["target"] == negative_label

        positive_images = mnist['data'][positive_indices]
        negative_images = mnist['data'][negative_indices]

        return BinaryDataset(positive_images, negative_images)

    def generate_rings(num_instances: int) -> "BinaryDataset":
        pass

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
        mnist['data'] = mnist['data'].astype(np.float32)  # convert the uint8s to floats
        mnist['data'] /= 255  # scale to be from 0 to 1
        if as_images:
            mnist['data'] = mnist['data'].reshape((-1, 1, 28, 28))
            if as_images != 28:
                data = np.zeros((mnist['data'].shape[0], 1, as_images, as_images), np.float32)
                x = (as_images - 28) // 2
                data[:, :, x:x+28, x:x+28] = mnist['data']
                mnist['data'] = data

        mnist['target'] = mnist['target'].astype(np.int64)  # convert the uint8s to int32s

        train = Data(mnist['data'][:60000], mnist['target'][:60000])
        val = Data(mnist['data'][60000:], mnist['target'][60000:])
        label_names = [str(i) for i in range(10)]
        return MulticlassDataset(train, val, label_names)

    def cifar(raw=False) -> "MulticlassDataset":
        cifar = load_cifar()

        if raw:
            cifar["data"] = cifar[b"data"]
            cifar['data'] = cifar['data'].reshape(cifar['data'].shape[0], 3, 32, 32)
            cifar['data'] = np.moveaxis(cifar['data'], 1, 3)
            cifar["target"] = cifar[b"target"]
        else:
            cifar['data'] = cifar[b'data'].astype(np.float32)
            cifar['data'] = cifar['data'].reshape(cifar['data'].shape[0], 3, 32, 32)
            cifar['data'] /= 255
            cifar['target'] = cifar[b'target'].astype(np.int64)

        N_val = 10000
        val = Data(cifar['data'][:N_val], cifar['target'][:N_val])
        train = Data(cifar['data'][N_val:], cifar['target'][N_val:])

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
        train_labels = train["names"]
        val_positive = val["positive_pairs"]
        val_negative = val["negative_pairs"]
        val_ids = val["image_ids"]
        val_labels = val["names"]

        train = MetricData(train_images, train_positive, train_negative, train_ids, train_labels)
        val = MetricData(val_images, val_positive, val_negative, val_ids, val_labels)
        return MetricDataset(train, val)

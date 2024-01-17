from typing import List, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

from datasets import load_coco


Prediction = NamedTuple("Prediction", [("id", int), ("score", float)])


def classify(images: np.ndarray, batch_size: int):
    num_images = len(images)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.to(device)
    model.eval()

    preprocess = weights.transforms(antialias=True)

    top5: List[List[Prediction]] = []
    for start in tqdm(range(0, num_images, batch_size)):
        end = min(start + batch_size, num_images)
        batch = torch.from_numpy(images[start:end]).permute(0, 3, 1, 2).float() / 255.0
        batch = preprocess(batch).to(device)

        out = model(batch).softmax(dim=1)
        val, idx = out.topk(5, dim=1)
        val = val.detach().cpu().numpy()
        idx = idx.detach().cpu().numpy()
        for i in range(end - start):
            top5.append([Prediction(idx[i, j], val[i, j]) for j in range(5)])

    return weights.meta["categories"], images, top5


def coco(num_images: int, batch_size: int):
    dataset = load_coco("minival")
    images = dataset["images"][:num_images]
    categories, images, top5 = classify(images, batch_size)
    image = 0
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(images[image])
    ax.axis('off')
    ax = fig.add_subplot(1, 2, 2)
    width = [top5[image][j].score for j in range(5)][::-1]
    names = [categories[top5[image][j].id].replace(" ", "\n") for j in range(5)][::-1]
    ax.barh(range(5), width, tick_label=names)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    coco(32, 8)

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO

from datasets import DATA_DIR

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEFAULT_MODEL = os.path.join(RESULTS_DIR, "yolo11n.pt")


def detect(image_path: str = None, model_name: str = None, conf: float = 0.25):
    if image_path is None:
        image_path = os.path.join(DATA_DIR, "train.jpg")
    if model_name is None:
        model_name = DEFAULT_MODEL

    model = YOLO(model_name)
    results = model(image_path, conf=conf)
    return results


def visualise(image_path: str = None, model_name: str = None, conf: float = 0.25):
    results = detect(image_path, model_name, conf)

    for result in results:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        img = np.array(Image.open(result.path))
        axes[0].imshow(img)
        axes[0].set_title("Original")
        axes[0].axis("off")

        annotated = result.plot()[..., ::-1]
        axes[1].imshow(annotated)
        axes[1].set_title(f"YOLO Detections (conf>{conf})")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()


def detect_coco_samples(num_samples: int = 4, model_name: str = None):
    coco_dir = os.path.join(DATA_DIR, "coco")
    import glob
    paths = sorted(glob.glob(os.path.join(coco_dir, "*.jpg")))[:num_samples]

    fig, axes = plt.subplots(2, len(paths), figsize=(4 * len(paths), 8))

    if model_name is None:
        model_name = DEFAULT_MODEL
    model = YOLO(model_name)

    for i, path in enumerate(paths):
        img = np.array(Image.open(path))
        axes[0, i].imshow(img)
        axes[0, i].set_title(os.path.basename(path))
        axes[0, i].axis("off")

        results = model(path, conf=0.25, verbose=False)
        annotated = results[0].plot()[..., ::-1]
        axes[1, i].imshow(annotated)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Detections")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualise()
    detect_coco_samples()

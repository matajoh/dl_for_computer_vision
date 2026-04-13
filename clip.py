import os

import matplotlib.pyplot as plt
import numpy as np
import open_clip
from PIL import Image
import torch
import torch.nn.functional as F

from datasets import DATA_DIR

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_model(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer


def zero_shot_classify(image_path: str, labels: list, model=None, preprocess=None, tokenizer=None):
    if model is None:
        model, preprocess, tokenizer = load_model()

    image = preprocess(Image.open(image_path)).unsqueeze(0)
    text = tokenizer(labels)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        similarity = (image_features @ text_features.T).squeeze(0)
        probs = F.softmax(similarity * 100, dim=-1)

    return {label: prob.item() for label, prob in zip(labels, probs)}


def demo(image_path=None):
    if image_path is None:
        image_path = os.path.join(DATA_DIR, "cat.jpg")

    model, preprocess, tokenizer = load_model()

    labels = ["a photo of a cat", "a photo of a dog", "a photo of a bird",
              "a photo of a car", "a photo of a house"]

    results = zero_shot_classify(image_path, labels, model, preprocess, tokenizer)

    plt.rc("font", size=15)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].imshow(Image.open(image_path))
    axes[0].axis("off")
    axes[0].set_title("Input Image")

    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    names = [r[0] for r in sorted_results]
    probs = [r[1] for r in sorted_results]
    colors = ["green" if i == 0 else "steelblue" for i in range(len(names))]
    axes[1].barh(range(len(names)), probs, color=colors)
    axes[1].set_yticks(range(len(names)), names)
    axes[1].set_xlabel("Probability")
    axes[1].set_title("CLIP Zero-Shot Classification")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.show()


def image_text_similarity(image_paths: list, texts: list):
    model, preprocess, tokenizer = load_model()

    images = torch.stack([preprocess(Image.open(p)) for p in image_paths])
    text_tokens = tokenizer(texts)

    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(text_tokens)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        similarity = image_features @ text_features.T

    similarity = similarity.cpu().numpy()

    plt.rc("font", size=12)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(similarity, cmap="viridis")

    ax.set_xticks(range(len(texts)), texts, rotation=45, ha="right")
    ax.set_yticks(range(len(image_paths)), [os.path.basename(p) for p in image_paths])
    ax.set_title("CLIP Image-Text Similarity")
    fig.colorbar(im, ax=ax)

    for i in range(len(image_paths)):
        for j in range(len(texts)):
            ax.text(j, i, f"{similarity[i, j]:.2f}", ha="center", va="center",
                    color="white" if similarity[i, j] < 0.5 else "black")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo()

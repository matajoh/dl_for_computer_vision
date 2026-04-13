import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import v2
from torchvision.io import decode_image

from datasets import DATA_DIR

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def dino_augmentations(image_path=None, seed=20080524):
    """Visualize DINO-style teacher and student augmentations."""
    if image_path is None:
        image_path = os.path.join(DATA_DIR, "cat.jpg")

    torch.manual_seed(seed)
    image = decode_image(image_path)

    teacher_transform = v2.Compose([
        v2.Resize(224),
        v2.CenterCrop(224),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])

    student_transform = v2.Compose([
        v2.Resize(256),
        v2.RandomCrop(size=(224, 224)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])

    original = v2.functional.resize(image, 224)
    original = v2.functional.center_crop(original, 224)
    teacher1 = teacher_transform(image)
    teacher2 = teacher_transform(image)
    student1 = student_transform(image)
    student2 = student_transform(image)
    student3 = student_transform(image)
    student4 = student_transform(image)

    plt.rc("font", size=12)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    def show(ax, img, title):
        ax.imshow(img.numpy().transpose(1, 2, 0))
        ax.set_title(title)
        ax.axis("off")

    show(axes[0, 0], original, "Original")
    show(axes[0, 1], teacher1, "Teacher crop 1")
    show(axes[0, 2], teacher2, "Teacher crop 2")
    axes[0, 3].axis("off")

    show(axes[1, 0], student1, "Student crop 1")
    show(axes[1, 1], student2, "Student crop 2")
    show(axes[1, 2], student3, "Student crop 3")
    show(axes[1, 3], student4, "Student crop 4")

    plt.suptitle("DINO: Teacher (global) and Student (local) Augmentations", fontsize=14)
    plt.tight_layout()
    plt.show()


def dino_masking(image_path=None, mask_ratio=0.125, seed=20080524):
    """Visualize random patch masking as in iBOT / DINOv2."""
    if image_path is None:
        image_path = os.path.join(DATA_DIR, "cat.jpg")

    torch.manual_seed(seed)
    image = decode_image(image_path)
    image = v2.functional.resize(image, 224)
    image = v2.functional.center_crop(image, 224)

    masked = image.clone()
    for r in range(16):
        rr = r * 14
        for c in range(16):
            cc = c * 14
            if torch.rand(1).item() < mask_ratio:
                masked[0, rr:rr+14, cc:cc+14] = 255
                masked[1, rr:rr+14, cc:cc+14] = 255
                masked[2, rr:rr+14, cc:cc+14] = 0

    plt.rc("font", size=14)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image.numpy().transpose(1, 2, 0))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(masked.numpy().transpose(1, 2, 0))
    axes[1].set_title(f"Masked ({mask_ratio:.0%} patches)")
    axes[1].axis("off")

    plt.suptitle("DINOv2 / iBOT Masking Strategy", fontsize=14)
    plt.tight_layout()
    plt.show()


def dino_features(image_path=None, compare_path=None):
    """Extract and visualize DINOv2 patch features using PCA."""
    if image_path is None:
        image_path = os.path.join(DATA_DIR, "dino_dog_0.jpg")
    if compare_path is None:
        compare_path = os.path.join(DATA_DIR, "dino_dog_1.jpg")

    warnings.filterwarnings("ignore", message="xFormers is")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval()

    transform = v2.Compose([
        v2.Resize(224),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_paths = [image_path, compare_path]
    images = [decode_image(p) for p in image_paths]
    batch = torch.stack([transform(img) for img in images])

    with torch.no_grad():
        features = model.forward_features(batch)
        patch_tokens = features["x_norm_patchtokens"]

    all_patches = patch_tokens.numpy().reshape(-1, patch_tokens.shape[-1])
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    # Step 1: first PCA to get component 0 for fg/bg separation
    pca_bg = PCA(n_components=3)
    pca_features = pca_bg.fit_transform(all_patches)

    # Use K-means (k=2) on the first component to find fg vs bg
    # (the threshold is image-dependent; simple zero doesn't always work)
    labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(
        pca_features[:, 0:1]
    )
    # The cluster with fewer patches is the foreground
    cluster_sizes = [np.sum(labels == k) for k in range(2)]
    fg_label = int(np.argmin(cluster_sizes))
    fg_mask = labels == fg_label

    # Step 2: re-fit 3-component PCA on foreground patches only
    pca = PCA(n_components=3)
    fg_features = pca.fit_transform(all_patches[fg_mask])
    # Normalize each component to [0, 1] via min-max scaling
    for c in range(3):
        lo, hi = fg_features[:, c].min(), fg_features[:, c].max()
        fg_features[:, c] = (fg_features[:, c] - lo) / (hi - lo + 1e-8)

    # Place fg colors, bg stays black
    pca_all = np.zeros((len(all_patches), 3), dtype=np.float32)
    pca_all[fg_mask] = fg_features

    n_patches = patch_tokens.shape[1]
    grid_size = int(np.sqrt(n_patches))

    plt.rc("font", size=14)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for row, (img, path) in enumerate(zip(images, image_paths)):
        original = v2.functional.resize(img, 224)
        original = v2.functional.center_crop(original, 224)
        axes[row, 0].imshow(original.numpy().transpose(1, 2, 0))
        axes[row, 0].set_title(os.path.basename(path))
        axes[row, 0].axis("off")

        pca_features = pca_all[row * n_patches:(row + 1) * n_patches]
        pca_image = pca_features.reshape(grid_size, grid_size, 3)
        axes[row, 1].imshow(pca_image, interpolation="nearest")
        axes[row, 1].set_title("DINOv2 Patch Features (PCA)")
        axes[row, 1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dino_augmentations()
    dino_masking()
    dino_features()

"""Download pre-computed results so notebooks can be used without training.

Note: some data and results files have dependencies. Regenerating them will
invalidate downstream results and require retraining:

  data/emnist_patch128.npz  -> transformer_emnist_gpt_128, rnn_emnist_patch_128,
                               transformer_emnist_128_finetune
  data/lfw_minitrain.npz    -> facenet
  transformer_language_128  -> transformer_language_128_finetune
  transformer_emnist_gpt_128 -> transformer_emnist_128_finetune
"""

import os
import sys

import requests
from tqdm import tqdm

BASE_URL = "https://matajohdata.blob.core.windows.net/models"

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FILES = [
    "bullseye.results",
    "cat_image_0.results",
    "cat_image_16.results",
    "cifar10.results",
    "facenet.results",
    "fcn_simple.results",
    "mnist.results",
    "mnist_attack.results",
    "mnist_pool_cnn.results",
    "mnist_simple_cnn.results",
    "panda_attack.results",
    "rnn_emnist_finetune_128.results",
    "rnn_emnist_patch_128.results",
    "rnn_language_6_512.results",
    "transformer_cifar_vit.results",
    "transformer_emnist_128_finetune.results",
    "transformer_emnist_gpt_128.results",
    "transformer_emnist_vit.results",
    "transformer_language_128.results",
    "transformer_language_128_finetune.results",
    "transformer_language_vit.results",
    "transformer_postcodes_128.results",
    "vae.results",
    "xor.results",
    "yolo11n.pt",
]


def download(url: str, path: str):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(path, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(path)) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def main():
    force = "--force" in sys.argv
    skipped = 0
    downloaded = 0

    for filename in FILES:
        path = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(path) and not force:
            skipped += 1
            continue

        url = f"{BASE_URL}/{filename}"
        print(f"Downloading {filename}...")
        try:
            download(url, path)
            downloaded += 1
        except requests.HTTPError as e:
            print(f"  Failed: {e}")

    print(f"\nDone. Downloaded: {downloaded}, skipped (already exist): {skipped}")
    if skipped and not force:
        print("Use --force to re-download existing files.")


if __name__ == "__main__":
    main()

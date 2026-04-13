# Deep Learning for Computer Vision

Companion repository for **4F12: Computer Vision** (Engineering Tripos Part IIB, University of Cambridge).

## Lectures

| # | Topic | Video |
|---|-------|-------|
| 1 | Multi-layer Perceptrons | [YouTube](https://youtu.be/LDcBRixd0jQ) |
| 2 | Convolutional Neural Nets | [YouTube](https://youtu.be/HfX_IOn5wjA) |

## Notebooks

Interactive lecture notes are provided as Jupyter Notebooks:

- **`lecture_notes_0.ipynb`** — Perceptrons, Classification, Backpropagation, Gradient Descent, Optimisation
- **`lecture_notes_1.ipynb`** — Convolution, Pooling, ResNet, Segmentation, VAE, Object Detection, Face Recognition, Adversarial Attacks
- **`lecture_notes_2.ipynb`** — RNNs, LSTM, Transformers, Vision Transformers, Generative Pretraining, CLIP, DINOv2

## Topics Covered

| Topic | Module(s) |
|-------|-----------|
| Perceptrons & MLPs | `perceptron.py`, `multiperceptron.py`, `mlp.py` |
| Activations | `activations.py` |
| Gradient Descent & Optimisation | `gradient_descent.py` |
| Convolution & Pooling | `filters.py`, `cnn.py`, `pooling.py` |
| ResNet | `resnet.py` |
| Salience Maps | `salience.py` |
| Datasets (MNIST, CIFAR-10, LFW, EMNIST, COCO) | `datasets.py` |
| Variational Autoencoders | `vae.py` |
| FaceNet & Triplet Loss | `facenet.py` |
| Object Detection (YOLO) | `yolo.py` |
| Adversarial Examples (FGSM) | `adversarial.py` |
| Recurrent Neural Networks | `rnn.py` |
| Transformers (ViT, GPT, Encoder-Decoder) | `transformers_model.py` |
| Patch Clustering (K-means Tokenisation) | `patch_clustering.py` |
| CLIP (Zero-shot Classification) | `clip.py` |
| DINO / DINOv2 (Self-supervised Vision) | `dino.py` |

## Getting Started

It is recommended to use a virtual environment:

```bash
python -m venv .env
source .env/bin/activate   # Linux/macOS
# .env\Scripts\activate    # Windows
```

Install dependencies:

```bash
python -m pip install pip --upgrade
pip install -r requirements.txt
```

Then follow the [instructions to install PyTorch](https://pytorch.org/get-started/locally/).

### Pre-computed Results

The notebooks rely on pre-computed training results (saved weights, embeddings, etc.). To download them:

```bash
python download_results.py
```

This downloads ~69 MB of results files into `results/`. Use `--force` to re-download existing files.

If you prefer to generate the results yourself, run the corresponding Python modules directly (e.g. `python facenet.py`), though this requires a GPU and considerably more time.

> **Warning — regeneration dependencies:** Some data and results files depend on each other. Regenerating them will invalidate downstream results, requiring those to be retrained as well.
>
> | If you regenerate… | …you must also regenerate |
> |---|---|
> | `data/emnist_patch128.npz` | `transformer_emnist_gpt_128.results`, `transformer_emnist_128_finetune.results`, `rnn_emnist_patch_128.results` |
> | `data/lfw_minitrain.npz` | `facenet.results` |
> | `transformer_language_128.results` | `transformer_language_128_finetune.results` |
> | `transformer_emnist_gpt_128.results` | `transformer_emnist_128_finetune.results` |

The scripts can be run standalone, but are primarily designed to be called from the Jupyter notebooks. [Learn more about Jupyter](https://jupyter.org/).

## Requirements

See [`requirements.txt`](requirements.txt). Key dependencies: PyTorch ≥ 2.0, torchvision, matplotlib, numpy, ultralytics (YOLO), open-clip-torch (CLIP).

## License

See [`LICENSE`](LICENSE).

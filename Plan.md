# Completion Plan for dl_for_computer_vision

This document analyses the gap between the 4F12 Deep Learning for Computer Vision course
lecture notes and this companion repository, and proposes a plan to address it.

## Table of Contents

1. [Current State](#1-current-state)
2. [Gap Analysis](#2-gap-analysis)
   - 2a. [CNN/DNN Gaps (lecture_notes_1 targets)](#2a-cnndnn-gaps-lecture_notes_1-targets)
   - 2b. [RNN & Transformer Gaps (lecture_notes_2 targets)](#2b-rnn--transformer-gaps-lecture_notes_2-targets)
3. [Import Plan](#3-import-plan)
4. [LaTeX → Markdown Mapping](#4-latex--markdown-mapping)
5. [Modernisation Plan](#5-modernisation-plan)

---

## 1. Current State

### Course Lecture Notes (LaTeX, 113 slides)

The course covers five broad areas across 113 slides:

| Section | Slides | Topics |
|---------|--------|--------|
| Foundations | 1–22 | Perceptrons, activation functions, gradient descent, backpropagation, cross-entropy, SGD, datasets, MLPs, latent space |
| CNN/DNN | 23–71 | MNIST, vanishing gradients, ReLU, momentum, convolution, CNNs, pooling, CIFAR-10, data augmentation, batch norm, ImageNet, AlexNet, ResNets, NiN, FCN, segmentation, COCO, VAE, transposed convolution, YOLO, salience, FaceNet, adversarial attacks |
| RNNs | 72–83 | Elman RNN, self-supervision, sequential datasets, EMNIST, patch tokenisation, Bernoulli mixtures, LSTM |
| Transformers | 84–103 | Positional encoding, SDPA, multi-head self-attention, ViT, layer norm, GELU, decoder transformers, patch clustering, sampling/beam search, encoder-decoder, vision-language models |
| SOTA | 104–112 | VGGT, CLIP, DINOv2, DINO/iBOT losses, responsible AI |

### Companion Repository (this repo)

| File | Covers |
|------|--------|
| `lecture_notes_0.ipynb` | Slides 1–22: Perceptrons → MLPs (complete) |
| `lecture_notes_1.ipynb` | Slides 23–59: MNIST → FCN/segmentation/COCO (mostly complete) |
| `perceptron.py` | Binary perceptron, 2D decision boundaries, MNIST binary |
| `multiperceptron.py` | Multi-class softmax classifier, XOR, bullseye |
| `activations.py` | Step, tanh, dtanh, ReLU |
| `gradient_descent.py` | 1D GD with/without momentum |
| `datasets.py` | MNIST, CIFAR-10, COCO, LFW loaders |
| `filters.py` | Hand-crafted convolution kernels, im2col, transposed convolution |
| `pooling.py` | Average and max pooling |
| `cnn.py` | SimpleCNN, PoolingCNN, CifarCNN, FCNSimple, NIN, training/eval |
| `mlp.py` | MLP with hidden layers for MNIST |
| `resnet.py` | Pre-trained ResNet-50 classification on COCO |
| `salience.py` | Grad-CAM visualisation for ResNet-50 |

### Course Scripts (figure-generation code, not yet in repo)

Scripts in the course repo that produce lecture figures but are not yet adapted:

| Script | Purpose |
|--------|---------|
| `rnn.py` | Elman RNN, beam search, BernoulliMixture, PatchDecoder |
| `transformers.py` | Positional MLP, MHSA, ViT/GPT training, positional encoding |
| `facenet.py` | FaceNet with triplet loss on LFW |
| `vae.py` | Variational autoencoder with ELBO loss |
| `unet.py` | U-Net for semantic segmentation |
| `dino.py` | DINO augmentation visualisation |
| `patch_clustering.py` | K-means patch dictionary, vector quantisation |
| `cross_entropy.py` | Cross-entropy distribution visualisation |
| `conv2d_animation.py` | Animated 2D convolution |
| `conv3d_animation.py` | Animated 3D convolution |
| `video.py` | Matplotlib → MP4 utility using scenepic |
| `coco_mini.py` | COCO dataset → mini NPZ converter |
| `fcn.py` | FCN-ResNet50 segmentation, panoptic visualisation |
| `mp_issues.py` | XOR and bullseye problem visualisation |
| `network_diagram.py` | SVG network architecture diagrams (drawsvg) |
| `add_captions.py` | Azure CV API caption overlay |

---

## 2. Gap Analysis

### 2a. CNN/DNN Gaps (`lecture_notes_1` targets)

`lecture_notes_1.ipynb` currently covers slides 23–59 (MNIST CNNs through semantic
segmentation). The following CNN/DNN topics from the course are **missing** from
`lecture_notes_1`:

| # | Slide(s) | Topic | What's Needed | Source Script |
|---|----------|-------|---------------|---------------|
| G1 | 60 | **Sampling the VAE latent space** | VAE training on MNIST, latent interpolation, sample generation | `vae.py` |
| G2 | 61 | **Object detection** (semantic vs instance vs panoptic segmentation) | Side-by-side comparison of COCO segmentation types | `fcn.py` (`panoptic_comparison`) |
| G3 | 62 | **Salience / Grad-CAM** | Already have `salience.py` but not exercised in notebook | `salience.py` (exists) |
| G4 | 63–64 | **YOLO** (real-time object detection) | Load pre-trained YOLO, run on COCO images, visualise bboxes + labels | New script: `yolo.py` |
| G5 | 65–66 | **Recognition / LFW dataset** | Display LFW samples, discuss identity recognition task | `facenet.py` |
| G6 | 67–68 | **Feature embedding / FaceNet** with triplet loss | Train FaceNet on LFW, visualise 2D embedding space, show triplet loss | `facenet.py` |
| G7 | 69–70 | **Adversarial attacks** | Generate adversarial perturbations on trained model, show FGSM | New code (partial in `multiperceptron.py`) |
| G8 | 57–59 | **Transposed convolution** (deeper treatment) | Currently in `filters.py` but not demonstrated in notebook | `filters.py` (exists) |
| G9 | 56 | **Autoencoding / VAE theory** (ELBO, KL divergence) | Markdown exposition + code walkthrough | `vae.py` |

### 2b. RNN & Transformer Gaps (`lecture_notes_2` targets)

**None** of the RNN or Transformer content is present in the repo. This will form
a completely new `lecture_notes_2.ipynb`.

| # | Slide(s) | Topic | What's Needed | Source Script |
|---|----------|-------|---------------|---------------|
| R1 | 71–72 | **RNN fundamentals** (Elman network, hidden state, pseudocode) | Markdown theory + simple RNN implementation | `rnn.py` |
| R2 | 72 | **Vanishing gradient in RNNs** | Visualise gradient magnitude vs sequence length | `rnn.py` |
| R3 | 73 | **Self-supervision** (next-token prediction) | Explain causal prediction, implement self-supervised RNN | `rnn.py` |
| R4 | 74 | **Sequential datasets** (tokenisation: word, character, subword) | Load language_6.json, show tokenisation strategies | `rnn.py` + data |
| R5 | 75 | **RNN results on language** | Train RNN on 6-language dataset, show per-character predictions | `rnn.py` |
| R6 | 76 | **Images as sequences** (patch-based) | EMNIST dataset, 7×7 grid of 4×4 patches | `rnn.py` |
| R7 | 77–79 | **Patch tokenisation** (learned embedding, Bernoulli mixture, binary CE) | Implement BernoulliMixture, PatchDecoder | `rnn.py`, `patch_clustering.py` |
| R8 | 80 | **RNN results on EMNIST** | Train and evaluate, show partial-image predictions | `rnn.py` |
| R9 | 81 | **Transfer learning / fine-tuning** | Freeze RNN, retrain classifier head | `rnn.py` |
| R10 | 82 | **LSTM** (gates, cell state) | Markdown explanation + diagram, LSTM implementation | `rnn.py` (extend) |
| R11 | 83 | **RNN scaling problems** ($O(NT)$ cost) | Discussion of computational cost, segue to transformers | Markdown only |
| T1 | 84–86 | **Positional encoding** (1D and 2D, Fourier features) | Train PositionalMLP on 1D function and 2D image | `transformers.py` |
| T2 | 87–88 | **Attention** (SDPA, multi-head self-attention) | Implement SDPA + MHSA from scratch, visualise attention weights | `transformers.py` |
| T3 | 89–90 | **Vision Transformer (ViT)** architecture | Build ViT, explain class token, layer norm, GELU | `transformers.py` |
| T4 | 91 | **Transformer results on EMNIST** | Train ViT on EMNIST, visualise attention maps | `transformers.py` |
| T5 | 92 | **Transformer results on language** | Language classification with transformer | `transformers.py` |
| T6 | 93 | **Comparison: ViT vs CNN** on CIFAR-10 | Head-to-head comparison, parameter efficiency, pretraining effects | `transformers.py` |
| T7 | 94 | **Decoder transformers** (masked self-attention, causal masking) | Implement decoder-only transformer | `transformers.py` |
| T8 | 95 | **Generative pretraining + fine-tuning** (GPT-style) | Train generative model, fine-tune for classification | `transformers.py` |
| T9 | 96–97 | **Patch tokenisation redux** (K-means dictionary, vector quantisation) | Implement K-means clustering on image patches | `patch_clustering.py` |
| T10 | 98 | **Decoder transformer on EMNIST** (with token dictionary) | Train and generate samples | `transformers.py` |
| T11 | 99–100 | **Sampling** (temperature, top-K, top-P, beam search) | Implement all sampling strategies, compare outputs | `rnn.py` (beam_search) |
| T12 | 101 | **Encoder-decoder transformer** (full seq2seq) | Implement complete transformer | `transformers.py` |
| T13 | 102–103 | **Vision-language model** (synthetic postcodes) | Train encoder-decoder on postcode OCR task | `transformers.py` |
| S1 | 104–105 | **VGGT** (3D reconstruction from images) | Markdown exposition + optional demo with pre-trained model | Markdown + optional code |
| S2 | 106–107 | **CLIP** (contrastive language-image pretraining) | Load pre-trained CLIP, zero-shot classification demo | New script: `clip.py` |
| S3 | 108–112 | **DINOv2 / self-supervised learning** (DINO loss, iBOT, features) | Load DINOv2, extract features, linear probe, patch tracking | `dino.py` (extend) |
| S4 | 112 | **Responsible AI** | Markdown discussion of ethics, compute cost, environmental impact | Markdown only |

---

## 3. Import Plan

### Phase 1: Complete `lecture_notes_1.ipynb` (CNN/DNN gaps)

#### New scripts to create

| Script | Purpose | Based On |
|--------|---------|----------|
| `yolo.py` | Pre-trained YOLO inference on images | New (use `ultralytics` or `torchvision` YOLO) |
| `facenet.py` | FaceNet triplet-loss training on LFW | Adapt from course `facenet.py` |
| `vae.py` | VAE with ELBO loss, latent sampling | Adapt from course `vae.py` |
| `adversarial.py` | FGSM adversarial attack generation | New (expand existing attack code in `multiperceptron.py`) |

#### Existing scripts to update

| Script | Changes |
|--------|---------|
| `salience.py` | No changes needed — integrate into notebook |
| `filters.py` | No changes needed — exercise transposed convolution in notebook |
| `cnn.py` | Add panoptic segmentation visualisation from course `fcn.py` |
| `datasets.py` | Ensure LFW loader works standalone |

#### Notebook additions (append to `lecture_notes_1.ipynb`)

1. **Autoencoding & VAEs** — Theory (ELBO, KL divergence) + VAE training + latent space sampling
2. **Transposed Convolution** — Exercise existing code in `filters.py`
3. **Object Detection** — Semantic vs instance vs panoptic segmentation discussion
4. **Salience / Grad-CAM** — Demo with `salience.py`
5. **YOLO** — Pre-trained YOLO on COCO images
6. **Recognition & FaceNet** — LFW dataset, triplet loss, embedding visualisation
7. **Adversarial Attacks** — FGSM demonstration

### Phase 2: Create `lecture_notes_2.ipynb` (RNNs & Transformers)

#### New scripts to create

| Script | Purpose | Based On |
|--------|---------|----------|
| `rnn.py` | Elman RNN, LSTM, training, beam search | Adapt from course `rnn.py` |
| `transformers.py` | Positional encoding, MHSA, ViT, decoder, encoder-decoder | Adapt from course `transformers.py` |
| `patch_clustering.py` | K-means patch dictionary | Adapt from course `patch_clustering.py` |
| `clip.py` | Pre-trained CLIP zero-shot classification | New (use `open_clip` or `transformers` library) |
| `dino.py` | DINOv2 feature extraction and linear probe | Adapt from course `dino.py` |
| `video.py` | PLTVideoWriter utility (if videos are used in notebooks) | Copy from course `video.py` |

#### Notebook structure

```
lecture_notes_2.ipynb
├── Recurrent Neural Networks
│   ├── RNN Fundamentals (Elman network, hidden state updates)
│   ├── Vanishing Gradients in RNNs
│   ├── Self-Supervision (next-token prediction)
│   ├── Sequential Datasets (tokenisation strategies)
│   ├── RNN on Language Data (6-language classification)
│   ├── Images as Sequences (EMNIST patch sequences)
│   ├── Patch Tokenisation (Bernoulli mixture, learned embeddings)
│   ├── RNN on EMNIST
│   ├── Transfer Learning / Fine-tuning
│   ├── LSTM (gates, cell state, architecture)
│   └── RNN Limitations (O(NT) scaling)
├── Transformers
│   ├── Positional Encoding (1D Fourier features, 2D image regression)
│   ├── Attention Mechanisms (SDPA, multi-head self-attention)
│   ├── Vision Transformer (ViT) architecture
│   ├── ViT on EMNIST (training, attention map visualisation)
│   ├── ViT on Language (6-language classification)
│   ├── ViT vs CNN (CIFAR-10 comparison)
│   ├── Decoder Transformers (masked self-attention)
│   ├── Generative Pre-training (GPT-style)
│   ├── Patch Tokenisation Redux (K-means dictionary)
│   ├── Decoder Transformer on EMNIST (generation + sampling)
│   ├── Sampling Strategies (temperature, top-K, top-P, beam search)
│   ├── Encoder-Decoder Transformer (full architecture)
│   └── Vision-Language Model (synthetic postcode OCR)
└── State of the Art
    ├── VGGT (3D vision from transformers)
    ├── CLIP (zero-shot classification demo)
    ├── DINOv2 (self-supervised features, linear probe)
    └── Responsible AI
```

### Phase 3: Supporting data and assets

| Item | Action |
|------|--------|
| `language_6.json` | Copy from course scripts |
| `beam_search.txt` | Copy from course scripts |
| EMNIST loader | Add to `datasets.py` (EMNIST letters+digits, 36 classes) |
| Postcode generator | Add synthetic postcode generation to `datasets.py` |
| Sample images | Copy `cat.jpg`, `locomotive.jpg`, `panda.jpg`, `stop_sign.jpg` etc. as needed |

---

## 4. LaTeX → Markdown Mapping

The course notes use a custom LaTeX slide format. Below is the mapping from LaTeX
constructs to their Jupyter Markdown equivalents.

### Structural elements

| LaTeX | Markdown (Jupyter) | Notes |
|-------|---------------------|-------|
| `\slide{Title}` | `## Title` | Each slide becomes a H2 section |
| `\slide{Title, cont.}` | (merge with previous section) | Continuation slides merge into parent |
| `\nin` (new indent) | New paragraph | Natural paragraph break |
| `\newpar` | Blank line | Paragraph separator |
| `\vs{Nmm}` | (omit) | Vertical spacing — not needed |
| `\begin{itemize} \item` | `- item` | Bullet list |
| `\begin{enumerate} \item` | `1. item` | Numbered list |
| `\emph{text}` | `*text*` | Italic emphasis |
| `\textbf{text}` | `**text**` | Bold emphasis |
| `\texttt{text}` | `` `text` `` | Monospace / code |
| `\cite{key}` | `[Author, Year]` or footnote link | Replace with inline citation |
| `\begin{wrapfigure}...\includegraphics` | `![caption](figures/file.png)` | Inline image |
| `\includegraphics[width=W]{file}` | `<img src="figures/file.png" width="W">` | Sized image |

### Mathematical elements

| LaTeX | Markdown (Jupyter) | Notes |
|-------|---------------------|-------|
| `$...$` | `$...$` | Inline math (identical) |
| `\[ ... \]` or `$$...$$` | `$$...$$` | Display math |
| `\mathbf{x}` | `\mathbf{x}` | KaTeX supports this directly |
| `\frac{a}{b}` | `\frac{a}{b}` | KaTeX fraction |
| `\sum_{i}` | `\sum_{i}` | KaTeX summation |
| `\begin{align}` | `$$\begin{aligned}` | Use `aligned` in KaTeX |
| Custom symbols from `cv_symbols.tex` | Expand inline | e.g. if `\vx` → `\mathbf{x}` |

### Algorithmic / pseudocode elements

| LaTeX | Markdown (Jupyter) | Notes |
|-------|---------------------|-------|
| `\begin{algorithmic}` | Python code cell or markdown pseudocode | Prefer executable Python |
| `\State`, `\For`, `\If` | Python `for`, `if` | Convert to real code where possible |
| Pseudocode-only algorithms | Fenced code block (```text) | When not directly executable |

### Figures

| LaTeX Source | Notebook Treatment |
|-------------|-------------------|
| PDF figures from `../figures/` | Regenerate from script code cells, or convert to PNG and embed |
| SVG diagrams (network architectures) | Render from code or embed as PNG |
| Photo assets (`.jpg`, `.png`) | Copy to repo, reference with relative path |
| Generated plots (training curves etc.) | Regenerate live in notebook code cells |

### Content transformation principles

1. **Slides → narrative flow**: Slide titles become section headings, but prose should
   read as continuous text rather than bullet-point slides
2. **Figures → live code**: Where a figure was generated from a script, the notebook
   should produce it from a code cell so students can modify parameters
3. **Continuation slides merge**: "Topic, cont." slides merge into the parent section
4. **Add interactivity**: Use `ipywidgets` sliders where appropriate (as done in
   `lecture_notes_0.ipynb` and `lecture_notes_1.ipynb`)
5. **Mathematical derivations**: Keep full derivations but add step-by-step annotation
   in markdown between equations

---

## 5. Modernisation Plan

### 5.1 Dependency updates

| Current | Issue | Action |
|---------|-------|--------|
| `requirements.txt` has no version pins | Fragile installs | Pin minimum versions for all packages |
| PyTorch not in `requirements.txt` | Must be installed separately | Add torch + torchvision with install note |
| `scenepic` (used in course `video.py`) | Niche dependency, only for video export | Evaluate if needed; if so, add to requirements |
| `scikit-image` | Used only for `resize` in `salience.py` | Keep, but could replace with `torchvision.transforms` |
| No `ipywidgets` in requirements | Needed for interactive notebook elements | Add `ipywidgets` |
| No `open-clip-torch` or `ultralytics` | Needed for new CLIP/YOLO scripts | Add to requirements |

Proposed `requirements.txt`:

```
jupyter
ipywidgets
matplotlib>=3.7
tqdm
numpy>=1.24
scipy>=1.10
scikit-image>=0.21
torch>=2.0
torchvision>=0.15
ultralytics>=8.0
open-clip-torch>=2.20
```

### 5.2 Code quality improvements

| File | Issue | Action |
|------|-------|--------|
| ~~`datasets.py`~~ | ~~pkl files cause NumPy 2.4 deprecation~~ | ~~Done: converted to npz format~~ |
| ~~`datasets.py`~~ | ~~OneDrive download URLs may break~~ | ~~Done: migrated to blob storage~~ |
| `datasets.py` | `generate_sin()` is a stub | Remove or implement |
| `cnn.py` | Mixes training utilities with model definitions | Consider separating, but keep if notebook flow requires it |
| `multiperceptron.py` | Uses both numpy and PyTorch models | Acceptable for pedagogical reasons (showing progression) |
| `resnet.py` | Uses `ResNet50_Weights.DEFAULT` (modern) | No change needed |
| `salience.py` | Uses `skimage.transform.resize` | Could modernise to `torchvision.transforms.Resize` |
| All `.py` files | No `__main__` guards | Add `if __name__ == "__main__"` where applicable |
| `api_key.txt` | **Security risk** — API key checked into repo | Remove from repo, add to `.gitignore` |

### 5.3 Structural improvements

| Action | Detail | Status |
|--------|--------|--------|
| ~~Add `results/` directory~~ | ~~Store trained model results in git-ignored `results/` folder~~ | **Done** |
| ~~Update `.gitignore`~~ | ~~Add `results/` and `data/*.npz`~~ | **Done** |
| ~~Fix `torch.load` calls~~ | ~~Add `weights_only=False` for PyTorch 2.6+ compat~~ | **Done** |
| ~~Convert pkl → npz~~ | ~~Eliminate pickle dependency and NumPy deprecation~~ | **Done** |
| ~~Add `data/` directory~~ | ~~Store datasets in `data/`; `*.npz` git-ignored, small assets (images, json) tracked~~ | **Done** |
| ~~Move `train.jpg`~~ | ~~Moved to `data/` directory~~ | **Done** |
| README updates | Add lecture_notes_2 link, update topic list, add YOLO/CLIP/transformer topics |

### 5.4 Course scripts requiring adaptation

When importing course scripts, the following adaptations are needed:

| Script | Adaptation Required |
|--------|-------------------|
| `rnn.py` | Remove video generation (scenepic dependency), refactor for notebook use, remove hardcoded figure save paths |
| `transformers.py` | Extensive — extract PositionalMLP, MHSA, ViT, Decoder as clean classes; remove figure-save code; add notebook-friendly train/eval functions |
| `facenet.py` | Remove video generation, add notebook-friendly training loop with progress display |
| `vae.py` | Remove file I/O for `.results`, save to `results/` dir; add inline visualisation |
| `patch_clustering.py` | Remove video generation, simplify to K-means + visualisation; save to `results/` |
| `dino.py` | Remove hardcoded ImageNet path, make configurable; add feature extraction and linear probe |
| `fcn.py` | Remove video/scenepic dependency; keep segmentation visualisation |
| `video.py` | Only import if video features are desired in notebooks (optional dependency) |

### 5.5 README Overhaul

The current README is outdated — it references only two lectures and eight topics, and
does not reflect the full course scope or the repo's current structure.

#### Current problems

| Issue | Detail |
|-------|--------|
| Stale lecture links | Only links two YouTube videos; no mention of RNN/Transformer content |
| Incomplete topic list | Lists only 8 topics; course covers ~25+ |
| No repo structure overview | No description of scripts, data/, results/, images/ directories |
| No notebook descriptions | Doesn't explain what each notebook covers or how to use them |
| Outdated setup instructions | Still says "install wheel" and points to PyTorch install page separately |
| No prerequisites section | Doesn't mention Python version, GPU recommendation, or expected background |
| No course context | Minimal mention of 4F12; no link to course page or academic year |

#### Proposed README structure

```markdown
# Deep Learning for Computer Vision

Brief description of the repo as companion material for 4F12.

## Course Overview
- Course name, department, academic year
- Links to lecture recordings (all three: MLPs, CNNs, RNNs/Transformers)
- Link to course page

## Notebooks
- lecture_notes_0: Perceptrons → MLPs (slides 1–22)
- lecture_notes_1: CNNs → Object Detection (slides 23–71)
- lecture_notes_2: RNNs → Transformers → SOTA (slides 72–112)

## Repository Structure
- Top-level scripts (perceptron.py, mlp.py, cnn.py, etc.)
- data/ — datasets (downloaded on first run)
- results/ — cached training outputs (git-ignored)
- images/ — figures used in notebooks

## Topics Covered
Full list organised by notebook, covering all ~25 topics from foundations
through SOTA.

## Getting Started
- Prerequisites (Python ≥3.10, pip, GPU optional but recommended)
- Installation (single pip install -r requirements.txt)
- Running notebooks (jupyter lab / VS Code)

## Requirements
Brief note on torch/torchvision install and the requirements.txt contents.

## License
Link to LICENSE file.

## Acknowledgements
Course staff, data sources, pre-trained model attributions.
```

#### Implementation notes

- Update the topic list as each notebook is completed (don't list topics not yet in repo)
- Keep Getting Started simple — one `pip install` command, no separate wheel step
- Add a badge or note if GPU is strongly recommended for certain notebooks
- Link each script description to the notebook section that uses it

### 5.6 Recommended implementation order

```
1. Fix security issue (api_key.txt)
2. Update requirements.txt
3. Import and adapt Phase 1 scripts (vae.py, facenet.py, yolo.py, adversarial.py)
   — all new scripts should use RESULTS_DIR pattern for cached results
4. Complete lecture_notes_1.ipynb with new sections
5. Import and adapt Phase 2 scripts (rnn.py, transformers.py, patch_clustering.py, clip.py, dino.py)
   — all new scripts should use RESULTS_DIR pattern for cached results
6. Create lecture_notes_2.ipynb
7. Update README.md
8. Clean up data/ directory structure
```

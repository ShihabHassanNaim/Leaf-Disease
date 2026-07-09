# Chilli Leaf Disease Classification

Fine-tuning a **DINOv2-small + LoRA** backbone to classify 6 categories of chilli
(chili pepper) leaf conditions from photographs. The model is trained on the
public [Mendeley Chilli Leaf Disease Image Dataset](https://data.mendeley.com/)
and reaches **~94% accuracy** on the held-out test split while training only a
small number of adapter parameters.

> Trained locally on CPU. With a single 4060 Ti (CUDA 12.x) the same recipe runs
> roughly 10� faster.

---

## Table of contents
1. [What this project does](#1-what-this-project-does)
2. [Repository layout](#2-repository-layout)
3. [Quick start (TL;DR)](#3-quick-start-tldr)
4. [Detailed setup](#4-detailed-setup)
   - 4.1 [Clone & create a virtualenv](#41-clone--create-a-virtualenv)
   - 4.2 [Install PyTorch (CPU or CUDA)](#42-install-pytorch-cpu-or-cuda)
   - 4.3 [Install the remaining dependencies](#43-install-the-remaining-dependencies)
   - 4.4 [Get the dataset](#44-get-the-dataset)
5. [How to run](#5-how-to-run)
   - 5.1 [Train](#51-train)
   - 5.2 [Evaluate](#52-evaluate)
   - 5.3 [Grad-CAM visualizations](#53-grad-cam-visualizations)
   - 5.4 [Run inference on your own image](#54-run-inference-on-your-own-image)
6. [Configuration](#6-configuration)
7. [Results](#7-results)
8. [Troubleshooting](#8-troubleshooting)
9. [Citation](#9-citation)

---

## 1. What this project does

* Loads a **frozen DINOv2-small** vision transformer from Hugging Face.
* Attaches a **LoRA adapter** (rank = 8) to the attention projections � the
  backbone itself stays frozen, so we only train the adapter + a small
  classification head.
* Trains on the Mendeley chilli leaf dataset with standard image augmentations
  (resize, random horizontal flip, color jitter, normalization).
* Saves the best checkpoint by validation accuracy, runs full evaluation,
  produces confusion-matrix PNGs, generates Grad-CAM heatmaps, and exposes a
  one-image inference CLI.

The exact same code also supports DINOv3 � just flip the `model_id` in
`configs/config.yaml` (see [Configuration](#6-configuration)).

## 2. Repository layout

```
project/
+-- README.md                         # this file
+-- LICENSE                           # MIT
+-- requirements.txt                  # pinned Python dependencies
+-- .gitignore
�
+-- configs/
�   +-- config.yaml                   # all training / model / data settings
�
+-- datasets/
�   +-- dataset.py                    # PyTorch Dataset + DataLoader factory
�
+-- models/
�   +-- dinov3_lora.py                # generic Vision + LoRA classifier
�   +-- model_utils.py                # checkpoint save / load helpers
�
+-- utils/
�   +-- early_stopping.py
�   +-- gradcam.py                    # Grad-CAM implementation
�   +-- helpers.py
�   +-- losses.py
�   +-- metrics.py
�
+-- train.py                          # training entry-point
+-- evaluate.py                       # evaluation entry-point
+-- gradcam.py                        # Grad-CAM entry-point
+-- inference.py                      # single-image inference entry-point
�
+-- scripts/
�   +-- download_data.py              # helper to fetch / verify the dataset
�
+-- notebooks/
�   +-- training_experiments.ipynb    # the original exploration notebook
�
+-- checkpoints/                      # *.pt files produced by train.py
+-- outputs/
�   +-- confusion_matrices/           # PNGs from evaluate.py
�   +-- gradcam_heatmaps/             # PNGs from gradcam.py
�   +-- metrics/                      # JSON metric files from evaluate.py
+-- logs/                             # training logs
```

`__pycache__/`, the `checkpoints/`, `outputs/` and `logs/` directories themselves
are tracked by `.gitkeep` files; the actual binary artefacts they hold are
ignored.

## 3. Quick start (TL;DR)

```powershell
# 1. clone and enter
git clone https://github.com/ShihabHassanNaim/Leaf-Disease.git
cd chilli-leaf-disease-classification

# 2. virtualenv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. install PyTorch (CPU build shown � see below for CUDA)
pip install --upgrade pip
pip install torch>=2.2.0 --index-url https://download.pytorch.org/whl/cpu

# 4. install everything else
pip install -r requirements.txt

# 5. put the dataset somewhere, then point the config at it
#    (see section 4.4)

# 6. train
python train.py

# 7. evaluate
python evaluate.py

# 8. look at Grad-CAM heatmaps
python gradcam.py

# 9. predict a single image
python inference.py --image path/to/leaf.jpg
```

## 4. Detailed setup

### 4.1 Clone & create a virtualenv

You need **Python 3.10+** (tested on 3.12).

```powershell
git clone https://github.com/ShihabHassanNaim/Leaf-Disease.git
cd chilli-leaf-disease-classification
python -m venv .venv
.\.venv\Scripts\Activate.ps1           # Windows PowerShell
# source .venv/bin/activate             # macOS / Linux
```

> VS Code tip: open the folder, press <kbd>Ctrl+Shift+P</kbd> ? "Python: Select
> Interpreter" ? pick `.venv\Scripts\python.exe`.

### 4.2 Install PyTorch (CPU or CUDA)

This repo pins `torch>=2.2.0` in `requirements.txt`, but the right PyTorch wheel
depends on your hardware. Pick **one** of the commands below and run it
**before** installing `requirements.txt`.

| Hardware | Command |
| --- | --- |
| CPU only | `pip install torch>=2.2.0 --index-url https://download.pytorch.org/whl/cpu` |
| CUDA 12.1 (RTX 30/40 series) | `pip install torch>=2.2.0 --index-url https://download.pytorch.org/whl/cu121` |
| CUDA 11.8 | `pip install torch>=2.2.0 --index-url https://download.pytorch.org/whl/cu118` |
| Apple Silicon (MPS) | `pip install torch>=2.2.0` |

Verify your install:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 4.3 Install the remaining dependencies

```powershell
pip install -r requirements.txt
```

`requirements.txt` covers: `transformers`, `peft`, `numpy`, `pandas`, `pillow`,
`tqdm`, `PyYAML`, `scikit-learn`, `matplotlib`, `seaborn`.

### 4.4 Get the dataset

The dataset is hosted on **Mendeley Data** and requires a (free) login to
download, so there is no fully automatic route. The bundled helper script
prints step-by-step instructions:

```powershell
python scripts/download_data.py --data-dir data/ChilliLeaf
```

It will either:

* ? say `[OK] Dataset already prepared at ...` if you already placed it there, or
* ?? print manual download instructions for Mendeley Data.

After downloading and extracting the ZIP, the directory tree under
`data/ChilliLeaf` must look like:

```
data/ChilliLeaf/
+-- Bacterial_Spot/
+-- Cercospora_Leaf_Spot/
+-- Curl_Virus/
+-- Healthy_Leaf/
+-- Nutrition_Deficiency/
+-- Powdery_Mildew/
```

> ?? The original Mendeley ZIP nests each class folder twice
> (`Chilli Leaf Disease Image Dataset for Classificati/Bacterial_Spot/Bacterial_Spot/...`).
> Either let `train.py` point at the **outer** folder or use the helper script
> with `--url` pointing at the ZIP � it auto-extracts.

Finally, edit `configs/config.yaml` so `data.root` matches your local path:

```yaml
data:
  root: "data/ChilliLeaf"
  train_split: 0.7
  val_split:   0.15
  test_split:  0.15
  image_size:  224
```

## 5. How to run

All four entry-points self-bootstrap the project root into `sys.path`, so you
can launch them from anywhere.

### 5.1 Train

```powershell
python train.py
```

What happens:

1. The config + dataset are loaded.
2. A frozen DINOv2-small backbone + LoRA adapter + classification head are
   built (see `models/dinov3_lora.py`).
3. The model is trained for `training.epochs` epochs (default: `5`).
4. The best checkpoint by validation accuracy is written to
   `checkpoints/<run_name>_best.pt`.
5. `class_to_idx.json` is written next to the checkpoint so downstream tools
   know the label mapping.

Useful flags:

```powershell
python train.py --epochs 10 --batch-size 32 --lr 1e-4
```

### 5.2 Evaluate

```powershell
python evaluate.py
```

Produces:

* `outputs/metrics/<run_name>_metrics.json` � accuracy, macro/weighted
  precision/recall/F1, per-class report.
* `outputs/confusion_matrices/<run_name>_cm.png` � confusion matrix.
* `outputs/confusion_matrices/<run_name>_cm_normalized.png` � normalized
  version.

### 5.3 Grad-CAM visualizations

```powershell
python gradcam.py
```

Generates a grid of Grad-CAM overlays for every class in
`outputs/gradcam_heatmaps/`. Use these to sanity-check what the model is
actually looking at.

### 5.4 Run inference on your own image

```powershell
python inference.py --image path/to/your_leaf.jpg
```

Optional flags:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--image` | _required_ | path to a single image |
| `--checkpoint` | `checkpoints/<latest>_best.pt` | override the model file |
| `--top-k` | `3` | print top-K predictions with confidences |

The script writes a JSON next to the image and prints a tidy table to stdout.

## 6. Configuration

Everything that matters lives in `configs/config.yaml`:

```yaml
model:
  model_id: "facebook/dinov2-small"   # also accepts "facebook/dinov3-..." ids
  num_classes: 6
  lora:
    r: 8
    alpha: 16
    dropout: 0.05
    target_modules: ["qkv"]           # attention projection names

data:
  root: "data/ChilliLeaf"
  image_size: 224
  train_split: 0.7
  val_split:   0.15
  test_split:  0.15

training:
  epochs: 5
  batch_size: 16
  lr: 1.0e-4
  weight_decay: 1.0e-4
  num_workers: 2
  seed: 42
```

To swap to DINOv3, change `model.model_id` to e.g.
`facebook/dinov3-vit-small-patch16-224` and tweak `target_modules` to match the
attention layer names exposed by that checkpoint.

## 7. Results

On the held-out test split (15 % of the Mendeley dataset, stratified):

| Metric | Value |
| --- | --- |
| Accuracy | **0.9395** |
| Macro F1 | **0.9406** |
| Weighted F1 | **0.9399** |
| Best validation accuracy | 0.9320 |

Per-class metrics and confusion matrices are in `outputs/`.

## 8. Troubleshooting

**`huggingface_hub` cannot download the backbone** � first run needs internet
access to pull `facebook/dinov2-small`. If you are offline, pre-download the
weights and set `HF_HOME` / `TRANSFORMERS_CACHE` to that directory.

**`OutOfMemoryError` on GPU** � drop `batch_size` in `configs/config.yaml` or
enable gradient accumulation (left as an exercise � the training loop already
exposes the optimizer step boundary).

**Dataset not found** � re-run `python scripts/download_data.py --data-dir <path>`
and double-check `data.root` in the config.

**`scripts/download_data.py` cannot auto-fetch** � that is expected, Mendeley
requires a free login. Follow the printed manual instructions.

**Windows path with trailing space / OneDrive sync** � keep the project in a
plain folder such as `E:\projects\chilli-leaf-disease-classification\` to avoid
sync conflicts and weird quoting bugs.

## 9. Citation

If this code helps your research, please cite the underlying dataset and
backbone:

```bibtex
@article{oquab2024dinov2,
  title  = {DINOv2: Learning Robust Visual Features without Supervision},
  author = {Oquab, Maxime and Darcet, Timoth{\'e} and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and H{\'e}naff, Daniel and Bronstein, Michael and Labatut, Pascal and others},
  journal= {Transactions on Machine Learning Research},
  year   = {2024}
}

@misc{hu2022lora,
  title  = {LoRA: Low-Rank Adaptation of Large Language Models},
  author = {Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  year   = {2022}
}

@misc{mendeley_chilli,
  title  = {Chilli Leaf Disease Image Dataset for Classification},
  author = {{Mendeley Data}},
  year   = {2023}
}
```

---

MIT � Your Name Here � see [LICENSE](LICENSE).
#

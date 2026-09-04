# Infant Cry Classification with SimCLR + 5-Fold Cross-Validation

Self-supervised (SimCLR) pretraining on infant cry spectrograms, followed
by 5-fold stratified cross-validation of a classifier built on the frozen
encoder.

## Pipeline

The full pipeline follows the generalized workflow used across recent
infant cry classification / SSL literature: raw audio → time–frequency
representation → contrastive pretraining → supervised fine-tuning →
evaluation & explainability.

```
┌─────────────────────┐
│  1. AUDIO PREPROCESSING
│  ─────────────────────
│  • Load audio (resample to target rate, e.g. 16 kHz)
│  • Segment into fixed-length windows (e.g. 1s, with overlap)
│  • Denoise (noisereduce / spectral subtraction / band-pass filtering)
│  • Normalize each segment (peak or RMS normalization)
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│  2. FEATURE EXTRACTION
│  ─────────────────────
│  • Convert each audio segment into a time–frequency image
│    - Continuous Wavelet Transform (CWT) scalogram (Morlet wavelet), or
│    - Spectrogram / log-Mel spectrogram
│  • Save images in ImageFolder-compatible layout (data/cry_images/<class>/)
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│  3. SELF-SUPERVISED PRETRAINING (SimCLR)
│  ─────────────────────
│  • Generate two augmented views per image (random crop, flip,
│    color jitter, grayscale) via SimCLRTransform
│  • Pass both views through a shared encoder (e.g. ResNet-18/34,
│    EfficientNet-B0) + projection head
│  • Maximize agreement between views with NT-Xent (contrastive) loss
│  • Save the pretrained encoder weights
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│  4. SUPERVISED FINE-TUNING / CLASSIFICATION
│  ─────────────────────
│  • Load the pretrained encoder (frozen or unfrozen)
│  • Attach a fully-connected classification head
│  • Fine-tune with cross-entropy loss on labeled cry categories
│  • Evaluate with 5-fold (stratified) cross-validation
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│  5. MODEL EVALUATION & EXPLAINABILITY
│  ─────────────────────
│  • Standard metrics: accuracy, precision, recall, F1-score (per class)
│  • Grad-CAM: visualize discriminative regions of the time–frequency image
│  • Edge-enhanced / sharpened saliency maps for finer interpretation
└─────────────────────┘
```

## Project structure

```
.
├── data/                  # Raw and processed data (gitignored, kept via .gitkeep)
├── notebooks/             # Exploratory / experiment notebooks
│   └── 5_fold_cv.ipynb
├── results/                # Checkpoints, metrics, plots (gitignored, kept via .gitkeep)
├── src/                    # Source code
│   ├── data_prep.py         # Audio loading, denoising, spectrogram generation
│   ├── simclr.py             # SimCLR model, augmentations, NT-Xent loss
│   ├── train_simclr.py       # Self-supervised pretraining loop
│   └── kfold_eval.py         # 5-fold CV finetuning/evaluation
├── config.yaml             # All hyperparameters and paths
├── requirements.txt
└── .gitignore
```

## Setup

```bash
git clone <your-repo-url>
cd infant-cry-classification
pip install -r requirements.txt
```

## Data

This project uses the [donateacry-corpus](https://github.com/gveres/donateacry-corpus)
dataset. Clone it and point `data_prep.py` at the raw audio directory:

```bash
git clone https://github.com/gveres/donateacry-corpus.git data/raw
python src/data_prep.py
```

This generates spectrogram images under `data/cry_images/<class>/*.png`,
in an `ImageFolder`-compatible layout.
## Preprocessed Cry Images

The `cry_images/` directory contains preprocessed spectrogram images organized
in ImageFolder format. Each subfolder corresponds to a cry category.

| Folder | Description | Link |
|---|---|---|
| `cry_images/` | Root directory containing all preprocessed spectrogram images | [Google Drive](https://drive.google.com/drive/folders/1xiflVsmocWhlAqTm4-cNn1ThzF1ECNpU?usp=sharing) |
| `cry_images/hungry/` | Spectrogram images corresponding to hungry cries | |
| `cry_images/pain/` | Spectrogram images corresponding to pain cries | |
| `cry_images/burping/` | Spectrogram images corresponding to burping cries | |
| `cry_images/discomfort/` | Spectrogram images corresponding to discomfort cries | |
| `cry_images/tired/` | Spectrogram images corresponding to tired cries | |

## Usage
**Pretrain the SimCLR encoder** on unlabeled spectrograms:
   ```bash
   python src/train_simclr.py
   ```
   Saves encoder weights to `results/checkpoints/simclr_encoder.pt`.
   

**Run 5-fold cross-validation** with the frozen encoder:
   ```bash
   python src/kfold_eval.py
   ```
   Prints per-fold accuracy and the mean ± std across folds.

All paths and hyperparameters (batch size, learning rate, epochs, number
of folds, etc.) are controlled from `config.yaml`.
### Training Checkpoints

The trained model checkpoints are organized by fold and training epoch.

| Folder/File | Description | Link |
|---|---|---|
| `results/checkpoints/` | Root directory containing all trained model checkpoints |  |
| `results/checkpoints/fold0/` | Fold 0 model checkpoints | |
| `results/checkpoints/fold0_epoch1.pth` | Fold 0 checkpoint after epoch 1 | [Google Drive](https://drive.google.com/file/d/1T6zEuu3rxXwCfwg9iPnCMtQeKst4gFNe/view?usp=sharing) |
| `results/checkpoints/fold0_epoch2.pth` | Fold 0 checkpoint after epoch 2 | [Google Drive](https://drive.google.com/file/d/1Nb6T6DIALIrrTu52cKKfd3F5S-Sr6NYl/view?usp=sharing) |
| `results/checkpoints/fold0_epoch3.pth` | Fold 0 checkpoint after epoch 3 |[Google Drive](https://drive.google.com/file/d/1PLzzEPzhkAhBbhqPUPHTIP7TfkomJVXq/view?usp=sharing) |
| `results/checkpoints/fold0_epoch4.pth` | Fold 0 checkpoint after epoch 4 | [Google Drive](https://drive.google.com/file/d/10cqIkvaGoo4mj4hut8ANF3W_1abKQkHQ/view?usp=sharing)|
| `results/checkpoints/fold0_epoch5.pth` | Fold 0 checkpoint after epoch 5 |[Google Drive](https://drive.google.com/file/d/1dKYNsBjaCEalwGxN5q2M1YH2T8RbLqSJ/view?usp=sharing) |

## Notebooks

`notebooks/5_fold_cv.ipynb` contains the original exploratory version of
this pipeline (dataset setup, SimCLR pretraining, and 5-fold CV in one
notebook) — useful for running end-to-end in Colab.

## Results

Cross-validation metrics and any generated plots/logs are saved to `results/`.

## License

Add a license of your choice (e.g. MIT) here.

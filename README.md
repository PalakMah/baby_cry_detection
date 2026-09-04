# Infant Cry Classification with SimCLR + 5-Fold Cross-Validation

Self-supervised (SimCLR) pretraining on infant cry spectrograms, followed
by 5-fold stratified cross-validation of a classifier built on the frozen
encoder.

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

## Usage

1. **Pretrain the SimCLR encoder** on unlabeled spectrograms:
   ```bash
   python src/train_simclr.py
   ```
   Saves encoder weights to `results/checkpoints/simclr_encoder.pt`.

2. **Run 5-fold cross-validation** with the frozen encoder:
   ```bash
   python src/kfold_eval.py
   ```
   Prints per-fold accuracy and the mean ± std across folds.

All paths and hyperparameters (batch size, learning rate, epochs, number
of folds, etc.) are controlled from `config.yaml`.

## Notebooks

`notebooks/5_fold_cv.ipynb` contains the original exploratory version of
this pipeline (dataset setup, SimCLR pretraining, and 5-fold CV in one
notebook) — useful for running end-to-end in Colab.

## Results

Cross-validation metrics and any generated plots/logs are saved to `results/`.

## License

Add a license of your choice (e.g. MIT) here.

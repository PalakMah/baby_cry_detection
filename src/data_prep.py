"""
Data preparation utilities for the infant cry classification pipeline.

Handles loading raw audio from the donateacry-corpus dataset, denoising,
and converting waveforms into spectrogram images used for SimCLR
pretraining and downstream classification.
"""

import os
import shutil

import librosa
import numpy as np
import noisereduce as nr
import pywt
from PIL import Image
from scipy.signal import butter
from tqdm import tqdm


def load_audio(path, sr=22050):
    """Load an audio file as a mono waveform."""
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def denoise(y, sr):
    """Apply noise reduction to a raw waveform."""
    return nr.reduce_noise(y=y, sr=sr)


def waveform_to_image(y, sr, out_path, size=(224, 224)):
    """Convert a waveform into a spectrogram image saved to disk."""
    raise NotImplementedError("Fill in with your spectrogram/CWT logic")


def build_image_dataset(raw_audio_dir, out_dir):
    """
    Walk a directory of raw audio clips (organized by class) and write
    out spectrogram images in an ImageFolder-compatible layout, e.g.:

        out_dir/
            class_a/xxx.png
            class_b/yyy.png
    """
    os.makedirs(out_dir, exist_ok=True)
    for cls in tqdm(os.listdir(raw_audio_dir)):
        cls_dir = os.path.join(raw_audio_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        out_cls_dir = os.path.join(out_dir, cls)
        os.makedirs(out_cls_dir, exist_ok=True)
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith(".wav"):
                continue
            y, sr = load_audio(os.path.join(cls_dir, fname))
            y = denoise(y, sr)
            out_path = os.path.join(out_cls_dir, fname.replace(".wav", ".png"))
            waveform_to_image(y, sr, out_path)


if __name__ == "__main__":
    build_image_dataset(raw_audio_dir="data/raw", out_dir="data/cry_images")

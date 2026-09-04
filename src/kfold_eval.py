"""
5-fold (stratified) cross-validation of a linear/finetuned classifier
on top of the frozen SimCLR encoder.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_classifier(encoder_ckpt, num_classes, freeze_encoder=True):
    encoder = models.resnet18(pretrained=False)
    num_ftrs = encoder.fc.in_features
    encoder.fc = nn.Identity()
    encoder.load_state_dict(torch.load(encoder_ckpt, map_location=DEVICE))

    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False

    classifier = nn.Sequential(encoder, nn.Linear(num_ftrs, num_classes))
    return classifier.to(DEVICE)


def run_kfold(cfg):
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    dataset = datasets.ImageFolder(root=cfg["data_dir"], transform=transform)
    labels = np.array([y for _, y in dataset.samples])

    skf = StratifiedKFold(
        n_splits=cfg["k_folds"], shuffle=True, random_state=cfg["seed"]
    )

    fold_accuracies = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
        train_loader = DataLoader(
            Subset(dataset, train_idx), batch_size=cfg["batch_size"], shuffle=True
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx), batch_size=cfg["batch_size"], shuffle=False
        )

        model = build_classifier(
            cfg["encoder_checkpoint"], num_classes=len(dataset.classes)
        )
        optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
        criterion = nn.CrossEntropyLoss()

        for epoch in range(cfg["finetune_epochs"]):
            model.train()
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total
        fold_accuracies.append(acc)
        print(f"Fold {fold}/{cfg['k_folds']} - accuracy: {acc:.4f}")

    print(f"Mean accuracy: {np.mean(fold_accuracies):.4f} +/- {np.std(fold_accuracies):.4f}")
    return fold_accuracies


if __name__ == "__main__":
    cfg = load_config()
    run_kfold(cfg)

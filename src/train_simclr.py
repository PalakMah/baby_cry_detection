"""
Self-supervised pretraining loop for the SimCLR encoder on infant cry
spectrogram images.
"""

import os

import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from simclr import SimCLR, SimCLRDataset, SimCLRTransform, nt_xent_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_simclr(cfg):
    dataset = SimCLRDataset(cfg["data_dir"], transform=SimCLRTransform())
    loader = DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=True
    )

    model = SimCLR().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for view1, view2 in loader:
            view1, view2 = view1.to(DEVICE), view2.to(DEVICE)

            _, z1 = model(view1)
            _, z2 = model(view2)
            loss = nt_xent_loss(z1, z2, temperature=cfg["temperature"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{cfg['epochs']} - loss: {avg_loss:.4f}")

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    ckpt_path = os.path.join(cfg["checkpoint_dir"], "simclr_encoder.pt")
    torch.save(model.encoder.state_dict(), ckpt_path)
    print(f"Saved encoder weights to {ckpt_path}")


if __name__ == "__main__":
    cfg = load_config()
    train_simclr(cfg)

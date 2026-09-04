"""
SimCLR self-supervised pretraining components: augmentation, dataset
wrapper, model, and NT-Xent contrastive loss.
"""

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms


class SimCLRTransform:
    """Produces two augmented views of the same image for contrastive learning."""

    def __init__(self):
        self.base_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)


class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.dataset = datasets.ImageFolder(root=root, transform=transform)

    def __getitem__(self, idx):
        (view1, view2), label = self.dataset[idx]
        return view1, view2

    def __len__(self):
        return len(self.dataset)


class SimCLR(nn.Module):
    def __init__(self, base_encoder=models.resnet18, out_dim=128):
        super().__init__()
        self.encoder = base_encoder(pretrained=False)
        num_ftrs = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs),
            nn.ReLU(),
            nn.Linear(num_ftrs, out_dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z


def nt_xent_loss(z_i, z_j, temperature=0.5):
    """Normalized temperature-scaled cross entropy loss (SimCLR)."""
    z_i = nn.functional.normalize(z_i, dim=1)
    z_j = nn.functional.normalize(z_j, dim=1)
    N = z_i.size(0)

    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T)

    mask = ~torch.eye(2 * N, dtype=torch.bool, device=z_i.device)
    similarity_matrix = similarity_matrix.masked_select(mask).view(2 * N, -1)

    positives = torch.cat(
        [torch.sum(z_i * z_j, dim=1), torch.sum(z_j * z_i, dim=1)]
    ).unsqueeze(1)

    logits = torch.cat([positives, similarity_matrix], dim=1) / temperature
    labels = torch.zeros(2 * N, dtype=torch.long, device=z_i.device)

    return nn.functional.cross_entropy(logits, labels)

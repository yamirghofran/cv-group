from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


class _BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet32(nn.Module):

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._stack(16, 16, stride=1)
        self.layer2 = self._stack(16, 32, stride=2)
        self.layer3 = self._stack(32, 64, stride=2)
        self.fc = nn.Linear(64, num_classes)

    @staticmethod
    def _stack(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            _BasicBlock(in_ch, out_ch, stride),
            *[_BasicBlock(out_ch, out_ch) for _ in range(4)],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer3(self.layer2(self.layer1(out)))
        return self.fc(F.adaptive_avg_pool2d(out, 1).flatten(1))


def _transforms(img_size: int, augment: bool) -> transforms.Compose:
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    base = [transforms.Resize((img_size, img_size))]
    if augment:
        base += [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomRotation(15),
        ]
    return transforms.Compose(base + [transforms.ToTensor(), transforms.Normalize(mean, std)])


def compute_class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    counts = np.bincount([lbl for _, lbl in dataset.samples], minlength=len(dataset.classes))
    counts = np.where(counts == 0, 1, counts).astype(float)
    weights = 1.0 / counts
    return torch.tensor(weights / weights.sum() * len(counts), dtype=torch.float32)


def load_datasets(
    data_dir: Path, img_size: int
) -> tuple[datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder | None]:
    train_ds = datasets.ImageFolder(data_dir / "train", transform=_transforms(img_size, augment=True))
    val_ds = datasets.ImageFolder(data_dir / "val", transform=_transforms(img_size, augment=False))
    test_ds = (
        datasets.ImageFolder(data_dir / "test", transform=_transforms(img_size, augment=False))
        if (data_dir / "test").exists()
        else None
    )
    return train_ds, val_ds, test_ds


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run_epoch(
    model: ResNet32,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (logits.detach().argmax(1) == labels).sum().item()
            n += images.size(0)
    return total_loss / n, correct / n


def train(
    data_dir: Path,
    output_path: Path,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    img_size: int = 64,
    device_str: str = "auto",
) -> dict[str, Any]:
    device = resolve_device(device_str)
    train_ds, val_ds, _ = load_datasets(data_dir, img_size)

    class_weights = compute_class_weights(train_ds).to(device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = ResNet32(num_classes=len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    history: dict[str, list[float]] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in tqdm(range(1, epochs + 1), desc="Training"):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, device, optimizer)
        vl_loss, vl_acc = _run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        tqdm.write(
            f"Epoch {epoch:3d} | train {tr_loss:.4f}/{tr_acc:.4f} | val {vl_loss:.4f}/{vl_acc:.4f}"
        )

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            save_checkpoint(output_path, model, train_ds.class_to_idx, img_size, vl_acc)
            tqdm.write(f"  → saved (val_acc={vl_acc:.4f})")

    print(f"Done — best val_acc={best_val_acc:.4f}")
    return {"best_val_acc": best_val_acc, "history": history}


def save_checkpoint(
    path: Path, model: ResNet32, class_to_idx: dict[str, int], img_size: int, val_acc: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
            "num_classes": len(class_to_idx),
            "img_size": img_size,
            "val_acc": val_acc,
        },
        path,
    )


def load_checkpoint(path: Path, device_str: str = "auto") -> tuple[ResNet32, dict[str, int], int]:
    device = resolve_device(device_str)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ResNet32(num_classes=ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    idx_to_class = {v: k for k, v in ckpt["class_to_idx"].items()}
    return model, idx_to_class, ckpt["img_size"]


class ResNetOCR:
    """Load a trained ResNet-32 checkpoint and predict jersey numbers from BGR crops."""

    def __init__(self, checkpoint_path: Path, device_str: str = "auto") -> None:
        self.device = resolve_device(device_str)
        self.model, self.idx_to_class, img_size = load_checkpoint(checkpoint_path, device_str)
        self.transform = _transforms(img_size, augment=False)

    def predict(self, bgr_image: np.ndarray) -> tuple[str, float]:
        from PIL import Image

        rgb = bgr_image[..., ::-1].copy()
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = F.softmax(self.model(tensor), dim=1)
        conf, idx = probs.max(dim=1)
        return self.idx_to_class[idx.item()], float(conf.item())

    def predict_batch(self, bgr_images: list[np.ndarray]) -> list[tuple[str, float]]:
        return [self.predict(img) for img in bgr_images]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune ResNet-32 on jersey number crops.")
    parser.add_argument("--data", required=True, help="Dataset root: must contain train/ and val/ subdirs.")
    parser.add_argument("--output", required=True, help="Checkpoint output (.pth).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    train(
        data_dir=Path(args.data),
        output_path=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.img_size,
        device_str=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

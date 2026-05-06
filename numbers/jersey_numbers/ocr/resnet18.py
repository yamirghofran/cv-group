from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


def build_model(num_classes: int) -> nn.Module:
    """Build ResNet18 with custom classification head."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes),
    )
    return model


def _transforms(img_size: int, augment: bool) -> transforms.Compose:
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    base = [transforms.Resize((img_size, img_size))]
    if augment:
        base += [
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
            transforms.RandomRotation(15),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
            transforms.RandomGrayscale(p=0.1),
        ]
    base += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if augment:
        base += [transforms.RandomErasing(p=0.3, scale=(0.02, 0.2))]
    return transforms.Compose(base)


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


def _mixup_batch(
    images: torch.Tensor, labels: torch.Tensor, alpha: float = 0.4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(images.size(0), device=images.device)
    return images * lam + images[idx] * (1 - lam), labels, labels[idx], lam


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any = None,
    mixup: bool = False,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training and mixup:
                images, labels_a, labels_b, lam = _mixup_batch(images, labels)
                logits = model(images)
                loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
                correct += (lam * (logits.detach().argmax(1) == labels_a).float()
                            + (1 - lam) * (logits.detach().argmax(1) == labels_b).float()).sum().item()
            else:
                logits = model(images)
                loss = criterion(logits, labels)
                correct += (logits.detach().argmax(1) == labels).sum().item()
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            total_loss += loss.item() * images.size(0)
            n += images.size(0)
    return total_loss / n, correct / n


def train(
    data_dir: Path,
    output_path: Path,
    frozen_epochs: int = 3,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-3,
    img_size: int = 224,
    device_str: str = "auto",
) -> dict[str, Any]:
    device = resolve_device(device_str)
    print(f"Device: {device}")
    train_ds, val_ds, _ = load_datasets(data_dir, img_size)

    class_weights = compute_class_weights(train_ds).to(device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(num_classes=len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    best_val_acc = 0.0
    history: dict[str, list[float]] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Phase 1: frozen backbone — train only the new head
    print(f"\nPhase 1: frozen backbone ({frozen_epochs} epochs)")
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=frozen_epochs
    )
    for epoch in tqdm(range(1, frozen_epochs + 1), desc="Frozen"):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, device, optimizer, scheduler, mixup=True)
        vl_loss, vl_acc = _run_epoch(model, val_loader, criterion, device, optimizer=None)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        tqdm.write(f"Epoch {epoch:3d} | train {tr_loss:.4f}/{tr_acc:.4f} | val {vl_loss:.4f}/{vl_acc:.4f}")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            save_checkpoint(output_path, model, train_ds.class_to_idx, img_size, vl_acc)
            tqdm.write(f"  -> saved (val_acc={vl_acc:.4f})")

    # Phase 2: unfreeze all — discriminative LRs (backbone gets 10x lower lr)
    print(f"\nPhase 2: unfrozen ({epochs} epochs)")
    for param in model.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam([
        {"params": [p for name, p in model.named_parameters() if "fc" not in name], "lr": lr / 10},
        {"params": model.fc.parameters(), "lr": lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[lr / 10, lr],
        steps_per_epoch=len(train_loader),
        epochs=epochs,
    )
    patience, no_improve = 20, 0
    for epoch in tqdm(range(1, epochs + 1), desc="Unfrozen"):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, device, optimizer, scheduler, mixup=True)
        vl_loss, vl_acc = _run_epoch(model, val_loader, criterion, device, optimizer=None)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        tqdm.write(f"Epoch {frozen_epochs + epoch:3d} | train {tr_loss:.4f}/{tr_acc:.4f} | val {vl_loss:.4f}/{vl_acc:.4f}")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            no_improve = 0
            save_checkpoint(output_path, model, train_ds.class_to_idx, img_size, vl_acc)
            tqdm.write(f"  -> saved (val_acc={vl_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                tqdm.write(f"Early stopping at epoch {frozen_epochs + epoch}")
                break

    print(f"Done — best val_acc={best_val_acc:.4f}")
    return {"best_val_acc": best_val_acc, "history": history}


def save_checkpoint(
    path: Path, model: nn.Module, class_to_idx: dict[str, int], img_size: int, val_acc: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
            "num_classes": len(class_to_idx),
            "img_size": img_size,
            "val_acc": val_acc,
            "arch": "resnet18",
        },
        path,
    )


def load_checkpoint(path: Path, device_str: str = "auto") -> tuple[nn.Module, dict[str, int], int]:
    device = resolve_device(device_str)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(num_classes=ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    idx_to_class = {v: k for k, v in ckpt["class_to_idx"].items()}
    return model, idx_to_class, ckpt["img_size"]


class ResNet18OCR:
    """Load a trained ResNet-18 checkpoint and predict jersey numbers from BGR crops."""

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

    def predict_proba(self, bgr_image: np.ndarray) -> tuple[np.ndarray, list[str]]:
        from PIL import Image

        rgb = bgr_image[..., ::-1].copy()
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = F.softmax(self.model(tensor), dim=1).cpu().numpy()[0]
        class_names = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]
        return probs, class_names

    def predict_batch(self, bgr_images: list[np.ndarray]) -> list[tuple[str, float]]:
        return [self.predict(img) for img in bgr_images]


def finetune(
    checkpoint_path: Path,
    data_dir: Path,
    output_path: Path,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    device_str: str = "auto",
) -> dict[str, Any]:
    device = resolve_device(device_str)
    print(f"Device: {device}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    img_size = ckpt["img_size"]

    train_ds, val_ds, _ = load_datasets(data_dir, img_size)
    class_weights = compute_class_weights(train_ds).to(device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(num_classes=ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    for param in model.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = float(ckpt.get("val_acc", 0.0))
    print(f"Resuming from val_acc={best_val_acc:.4f}")

    patience, no_improve = 15, 0
    for epoch in tqdm(range(1, epochs + 1), desc="Finetune"):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, device, optimizer, mixup=False)
        vl_loss, vl_acc = _run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step()
        tqdm.write(f"Epoch {epoch:3d} | train {tr_loss:.4f}/{tr_acc:.4f} | val {vl_loss:.4f}/{vl_acc:.4f}")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            no_improve = 0
            save_checkpoint(output_path, model, train_ds.class_to_idx, img_size, vl_acc)
            tqdm.write(f"  -> saved (val_acc={vl_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                tqdm.write(f"Early stopping at epoch {epoch}")
                break

    print(f"Done — best val_acc={best_val_acc:.4f}")
    return {"best_val_acc": best_val_acc}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune ResNet-18 on jersey number crops.")
    parser.add_argument("--data", required=True, help="Dataset root: must contain train/ and val/ subdirs.")
    parser.add_argument("--output", required=True, help="Checkpoint output (.pth).")
    parser.add_argument("--resume", help="Resume fine-tuning from this checkpoint at a lower LR.")
    parser.add_argument("--frozen-epochs", type=int, default=3, help="Epochs with backbone frozen.")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs after unfreezing.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.resume:
        finetune(
            checkpoint_path=Path(args.resume),
            data_dir=Path(args.data),
            output_path=Path(args.output),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device_str=args.device,
        )
    else:
        train(
            data_dir=Path(args.data),
            output_path=Path(args.output),
            frozen_epochs=args.frozen_epochs,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            img_size=args.img_size,
            device_str=args.device,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

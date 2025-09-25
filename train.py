#!/usr/bin/env python3
"""Train a CNN on a subset of the RadioML 2016.10a dataset."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Ensure the local ``src`` package is importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data.rml_dataset import RMLDataset  # type: ignore  # noqa: E402
from models.cnn import CNN1DClassifier  # type: ignore  # noqa: E402


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/RML2016.10a"),
        help="Directory containing the downloaded RadioML subset.",
    )
    parser.add_argument(
        "--snr",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of SNR levels to include during training.",
    )
    parser.add_argument(
        "--modulation",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of modulation classes to include (case-insensitive).",
    )
    parser.add_argument(
        "--max-examples-per-class",
        type=int,
        default=600,
        help="Limit the number of examples per modulation/SNR pair to speed up training.",
    )
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Optimizer learning rate."
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of the dataset reserved for validation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Torch device to use (e.g. 'cpu' or 'cuda'). Defaults to auto-detected GPU if available.",
    )
    parser.add_argument(
        "--auto-download",
        action="store_true",
        help=(
            "Automatically download the requested subset of the dataset when the directory "
            "is missing."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to store training artifacts (metrics, confusion matrix, etc.).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of worker threads for the PyTorch DataLoader.",
    )
    return parser.parse_args(argv)


def maybe_download_dataset(args: argparse.Namespace) -> None:
    if args.data_dir.exists():
        return
    if not args.auto_download:
        raise FileNotFoundError(
            f"Dataset directory {args.data_dir} does not exist. Run scripts/download_rml2016_subset.py "
            "manually or pass --auto-download."
        )

    cmd = [sys.executable, "scripts/download_rml2016_subset.py", "--destination", str(args.data_dir)]
    if args.snr:
        cmd.extend(["--snr", *map(str, args.snr)])
    if args.modulation:
        cmd.extend(["--modulation", *args.modulation])
    print("Dataset not found locally. Triggering download via:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def create_dataloaders(
    dataset: RMLDataset,
    *,
    batch_size: int,
    val_fraction: float,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    train_split, val_split = dataset.train_val_split(val_fraction=val_fraction)

    def to_dataset(split: dict) -> TensorDataset:
        samples = torch.from_numpy(split["samples"])  # shape: (N, 2, 128)
        labels = torch.from_numpy(split["labels"].astype(np.int64))
        return TensorDataset(samples, labels)

    train_dataset = to_dataset(train_split)
    val_dataset = to_dataset(val_split)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch, (inputs, targets) in enumerate(loader, start=1):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            running_loss += loss.item() * inputs.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    avg_loss = running_loss / total
    accuracy = correct / total
    preds_array = np.concatenate(all_preds)
    targets_array = np.concatenate(all_targets)
    return avg_loss, accuracy, preds_array, targets_array


def plot_confusion_matrix(cm: np.ndarray, labels: Sequence[str], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Validation confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    maybe_download_dataset(args)

    dataset = RMLDataset(
        args.data_dir,
        snrs=args.snr,
        modulations=args.modulation,
        max_examples_per_class=args.max_examples_per_class,
    )
    train_loader, val_loader = create_dataloaders(
        dataset,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
    )

    device = torch.device(args.device)
    model = CNN1DClassifier(num_classes=len(dataset.label_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_acc = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )
        tqdm.write(
            f"Epoch {epoch:02d}/{args.epochs}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_names": dataset.label_names,
                    "args": vars(args),
                },
                args.output_dir / "best_model.pt",
            )

    _, val_acc, y_pred, y_true = evaluate(model, val_loader, criterion, device)
    report = classification_report(
        y_true,
        y_pred,
        target_names=dataset.label_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    conf_mat = confusion_matrix(y_true, y_pred)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "history": history,
                "validation_accuracy": val_acc,
                "classification_report": report,
            },
            f,
            indent=2,
        )
    plot_confusion_matrix(conf_mat, dataset.label_names, args.output_dir / "confusion_matrix.png")

    print(f"Validation accuracy: {val_acc:.3f}")
    print(f"Saved metrics to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

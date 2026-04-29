#!/usr/bin/env python3
"""
Azure ML: Train U-Net Segmentation Model
- Prepared training data üzerinde MONAI U-Net eğit
- 60 epoch, mixed precision, cosine annealing scheduler
- Best model checkpoint'i kaydet
"""

import argparse
import json
from pathlib import Path
from typing import Optional
import traceback

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

try:
    from monai.networks.nets import UNet
    from monai.losses import DiceLoss
except ImportError:
    print("❌ MONAI bulunamadı!")
    sys.exit(1)


class AMOSVFAPMADataset(Dataset):
    """3-channel input/output training dataset"""

    def __init__(self, data_dir: Path, case_ids: list):
        self.data_dir = data_dir
        self.case_ids = case_ids

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        case_dir = self.data_dir / case_id

        input_npy = np.load(case_dir / "input.npy")  # (3, 512, 512)
        output_npy = np.load(case_dir / "output.npy")  # (3, 512, 512)

        return {
            "input": torch.from_numpy(input_npy).float(),
            "target": torch.from_numpy(output_npy).float(),
        }


def train_epoch(model, train_loader, optimizer, loss_fn, device, scaler):
    """Train one epoch"""
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate_epoch(model, val_loader, loss_fn, device):
    """Validate one epoch"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)

            with autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

            total_loss += loss.item()

    return total_loss / len(val_loader)


def main(args):
    print("=" * 60)
    print("🚀 Azure ML U-Net Training")
    print("=" * 60)

    train_data_dir = Path(args.train_data)
    output_dir = Path(args.output_dir)
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"📂 Training data: {train_data_dir}")
    print(f"📂 Output: {output_dir}")

    # Load split info
    split_path = train_data_dir / "split.json"
    if not split_path.exists():
        print("❌ split.json bulunamadı!")
        return

    with open(split_path, "r") as f:
        split_info = json.load(f)

    train_ids = split_info["train_ids"]
    val_ids = split_info["val_ids"]

    print(f"📊 Train: {len(train_ids)}, Val: {len(val_ids)}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")

    # Dataset ve DataLoader
    train_dataset = AMOSVFAPMADataset(train_data_dir, train_ids)
    val_dataset = AMOSVFAPMADataset(train_data_dir, val_ids)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=3,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.1,
    ).to(device)

    print(f"📊 Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Loss ve Optimizer
    dice_loss = DiceLoss(sigmoid=True)
    bce_loss = nn.BCEWithLogitsLoss()

    def combined_loss(pred, target):
        return 0.5 * dice_loss(pred, target) + 0.5 * bce_loss(pred, target)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler()

    # Training loop
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "lr": []}

    print("\n🚀 Starting training...")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, combined_loss, device, scaler)
        val_loss = validate_epoch(model, val_loader, combined_loss, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = output_dir / "best_unet.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                best_model_path,
            )
            print(f"  💾 Best model saved: {val_loss:.4f}")

    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Model saved: {output_dir / 'best_unet.pt'}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net segmentation model")
    parser.add_argument("--train_data", type=str, required=True, help="Prepared training data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=60, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")

    args = parser.parse_args()
    main(args)

#!/usr/bin/env python3
"""
Azure ML U-Net Eğitim Scripti
- AMOS22 NIfTI + TotalSegmentator maskeleri ile eğit
- Mixed precision + GPU optimization
- Checkpointing ve model saving
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import traceback

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import cv2
from tqdm import tqdm

try:
    from monai.networks.nets import UNet
    from monai.losses import DiceLoss
except ImportError:
    print("❌ MONAI kütüphanesi bulunamadı. Lütfen pip install monai çalıştırın.")
    sys.exit(1)

import SimpleITK as sitk


class AMOSVFAPMADataset(Dataset):
    """AMOS22 VFA/PMA training dataset"""

    def __init__(self, data_list: list, target_size: tuple = (512, 512)):
        """
        Args:
            data_list: [{'case_id': ..., 'hu_slice': ..., 'vfa_mask': ..., ...}, ...]
            target_size: (H, W) resize hedefi
        """
        self.data_list = data_list
        self.target_size = target_size

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        hu = torch.from_numpy(item["hu_slice"]).float()
        ts = torch.from_numpy(item["ts_seg"]).float()
        vb = torch.from_numpy(item["vertebra_mask"]).float()

        vfa = torch.from_numpy(item["vfa_mask"]).float()
        pma = torch.from_numpy(item["pma_mask"]).float()
        inner = torch.from_numpy(item["inner_abdomen_mask"]).float()

        return {
            "input": torch.stack([hu, ts, vb], dim=0),
            "target": torch.stack([vfa, pma, inner], dim=0),
            "case_id": item["case_id"],
        }


def prepare_training_data(
    amos_dir: Path, ts_dir: Path, max_cases: int = None
) -> list:
    """AMOS22 + TS maskeleriyle training data hazırla"""
    print("📦 Training data hazırlanıyor...")

    # Dinamik olarak core_mini import etmeyi dene
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from core_mini import (
            load_hu,
            find_l3_slice_index,
            detect_vertebra_center,
            compute_inner_abdomen_mask,
            compute_vfa,
            estimate_psoas_mask,
            compute_pma,
            get_pixel_area_mm2,
        )
    except ImportError:
        print("⚠️ core_mini bulunamadı")
        return []

    training_data = []
    nii_files = sorted(
        list(amos_dir.glob("*.nii.gz")) + list(amos_dir.glob("*.nii"))
    )

    if max_cases:
        nii_files = nii_files[:max_cases]

    print(f"🔄 {len(nii_files)} vaka işleniyor...")

    for amos_path in tqdm(nii_files):
        case_id = amos_path.stem.replace(".nii", "")

        try:
            # HU yükle
            hu_vol, spacing, _ = load_hu(str(amos_path))
            if hu_vol is None:
                continue

            # L3 bul
            z_l3, vb_conf = find_l3_slice_index(hu_vol)
            if vb_conf < 0.5:
                continue

            hu_l3 = hu_vol[z_l3, :, :]

            # TS seg yükle
            ts_seg = None
            ts_path = ts_dir / case_id / "abdominal_muscles.nii.gz"
            if ts_path.exists():
                try:
                    ts_img = sitk.ReadImage(str(ts_path))
                    ts_vol = sitk.GetArrayFromImage(ts_img)
                    if ts_vol.ndim == 3:
                        ts_seg = ts_vol[z_l3, :, :]
                except Exception:
                    pass

            # Vertebra
            vb_center, vb_mask = detect_vertebra_center(hu_l3)
            if vb_center is None:
                continue

            # Maskeleri hesapla
            inner_mask = compute_inner_abdomen_mask(hu_l3, ts_seg, vb_center)
            pixel_area = get_pixel_area_mm2(spacing[:2])

            vfa_result = compute_vfa(hu_l3, inner_mask, pixel_area)
            vfa_mask = vfa_result["vfa_mask"]

            psoas_mask = estimate_psoas_mask(hu_l3, vb_center)
            pma_result = compute_pma(hu_l3, psoas_mask, pixel_area)
            pma_mask = pma_result["pma_mask"]

            # Resize to 512x512
            target_size = (512, 512)
            hu_resized = cv2.resize(hu_l3, target_size, interpolation=cv2.INTER_LINEAR)
            ts_resized = (
                cv2.resize(
                    ts_seg.astype(np.float32), target_size, interpolation=cv2.INTER_NEAREST
                )
                if ts_seg is not None
                else np.zeros(target_size, dtype=np.float32)
            )
            vb_resized = cv2.resize(
                vb_mask.astype(np.float32), target_size, interpolation=cv2.INTER_NEAREST
            )
            vfa_resized = cv2.resize(
                vfa_mask.astype(np.float32), target_size, interpolation=cv2.INTER_NEAREST
            )
            pma_resized = cv2.resize(
                pma_mask.astype(np.float32), target_size, interpolation=cv2.INTER_NEAREST
            )
            inner_resized = cv2.resize(
                inner_mask.astype(np.float32), target_size, interpolation=cv2.INTER_NEAREST
            )

            training_data.append(
                {
                    "case_id": case_id,
                    "hu_slice": hu_resized,
                    "ts_seg": ts_resized,
                    "vertebra_mask": vb_resized,
                    "vfa_mask": vfa_resized,
                    "pma_mask": pma_resized,
                    "inner_abdomen_mask": inner_resized,
                }
            )

        except Exception as e:
            print(f"  ⚠️ {case_id}: {e}")
            continue

    print(f"✅ {len(training_data)} vaka hazırlandı")
    return training_data


def train_epoch(
    model,
    train_loader,
    device,
    optimizer,
    scheduler,
    scaler,
    dice_loss,
    bce_loss,
    epoch,
    num_epochs,
):
    """Bir epoch eğitimi"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch_idx, batch in enumerate(progress_bar):
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        # Mixed precision
        with autocast():
            outputs = model(inputs)
            loss_dice = dice_loss(outputs, targets)
            loss_bce = bce_loss(outputs, targets)
            loss = loss_dice + loss_bce

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, device, dice_loss, bce_loss):
    """Validation"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)

            with autocast():
                outputs = model(inputs)
                loss_dice = dice_loss(outputs, targets)
                loss_bce = bce_loss(outputs, targets)
                loss = loss_dice + loss_bce

            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss


def main(args):
    print("=" * 70)
    print("🚀 Azure ML U-Net Eğitim (AMOS22 VFA/PMA)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")
    print(f"💾 CUDA: {torch.cuda.is_available()}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Training data hazırla
    data_dir = Path(args.data_dir)
    ts_dir = data_dir.parent / "ts_masks"

    training_data = prepare_training_data(
        data_dir, ts_dir, max_cases=args.max_cases
    )

    if not training_data:
        print("❌ Training data hazırlama başarısız!")
        return

    # Dataset ve DataLoader
    dataset = AMOSVFAPMADataset(training_data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    print(f"📊 Train: {len(train_dataset)} | Val: {len(val_dataset)}")

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

    print(
        f"📈 Model params: {sum(p.numel() for p in model.parameters()):,}"
    )

    # Loss, optimizer, scheduler
    dice_loss = DiceLoss(sigmoid=True)
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = GradScaler()

    # Eğitim loop
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "lr": []}

    print(f"\n🎯 Starting {args.epochs} epoch training...")

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            device,
            optimizer,
            scheduler,
            scaler,
            dice_loss,
            bce_loss,
            epoch,
            args.epochs,
        )

        val_loss = validate(model, val_loader, device, dice_loss, bce_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        print(
            f"E{epoch+1}: Train={train_loss:.4f} Val={val_loss:.4f} LR={current_lr:.6f}"
        )

        # Best model kaydet
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
            print(f"  💾 Best model kaydedildi: {best_model_path}")

    # Final sonuçlar
    summary = {
        "total_epochs": args.epochs,
        "best_val_loss": float(best_val_loss),
        "final_train_loss": float(history["train_loss"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
        "training_data": len(training_data),
        "train_set_size": len(train_dataset),
        "val_set_size": len(val_dataset),
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "model_path": str(output_dir / "best_unet.pt"),
    }

    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("✅ Eğitim tamamlandı!")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Model: {output_dir / 'best_unet.pt'}")
    print(f"📊 Rapor: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net on AMOS22 dataset")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="AMOS22 NIfTI files directory"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for model"
    )
    parser.add_argument("--epochs", type=int, default=60, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--max_cases", type=int, default=None, help="Max training cases"
    )

    args = parser.parse_args()
    main(args)

#!/usr/bin/env python3
"""
Azure ML Full Pipeline: U-Net Training
Step 2: Train MONAI U-Net on TotalSegmentator teachers
"""
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# MONAI imports
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.transforms import Compose, ScaleIntensityRange, Resize
from monai.data import CacheDataset

class L3TeacherDataset(Dataset):
    """Dataset from TotalSegmentator teacher labels"""
    
    def __init__(self, case_dirs, transform=None, img_size=256):
        self.case_dirs = case_dirs
        self.transform = transform
        self.img_size = img_size
        
        # Load all cases
        self.samples = []
        for case_dir in case_dirs:
            metadata_path = case_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                self.samples.append({
                    "case_dir": case_dir,
                    "metadata": metadata
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        case_dir = sample["case_dir"]
        
        # Load HU slice
        hu_slice = np.load(case_dir / "hu_slice.npy")
        
        # Normalize HU to [0, 1]
        hu_normalized = np.clip((hu_slice + 1000) / 2000.0, 0, 1).astype(np.float32)
        
        # Add channel dimension [1, H, W]
        image = hu_normalized[np.newaxis, ...]
        
        # Load labels
        psoas_left = np.load(case_dir / "psoas_left.npy")[np.newaxis, ...]
        psoas_right = np.load(case_dir / "psoas_right.npy")[np.newaxis, ...]
        vat = np.load(case_dir / "vat.npy")[np.newaxis, ...]
        
        # Resize if needed
        if image.shape[1] != self.img_size or image.shape[2] != self.img_size:
            from scipy.ndimage import zoom
            scale = self.img_size / min(image.shape[1], image.shape[2])
            image = zoom(image, (1, scale, scale), order=1)
            psoas_left = zoom(psoas_left, (1, scale, scale), order=0)
            psoas_right = zoom(psoas_right, (1, scale, scale), order=0)
            vat = zoom(vat, (1, scale, scale), order=0)
            
            # Crop/pad to exact size
            h, w = image.shape[1:]
            if h > self.img_size:
                start = (h - self.img_size) // 2
                image = image[:, start:start+self.img_size, :]
                psoas_left = psoas_left[:, start:start+self.img_size, :]
                psoas_right = psoas_right[:, start:start+self.img_size, :]
                vat = vat[:, start:start+self.img_size, :]
            if w > self.img_size:
                start = (w - self.img_size) // 2
                image = image[:, :, start:start+self.img_size]
                psoas_left = psoas_left[:, :, start:start+self.img_size]
                psoas_right = psoas_right[:, :, start:start+self.img_size]
                vat = vat[:, :, start:start+self.img_size]
        
        # Convert to torch
        image = torch.from_numpy(image.copy())
        
        # Combined label: [psoas_left, psoas_right, vat]
        labels = torch.from_numpy(
            np.concatenate([psoas_left, psoas_right, vat], axis=0).astype(np.float32)
        )
        
        return {"image": image, "labels": labels, "case_id": sample["metadata"]["case_id"]}

def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train one epoch"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
    
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser(description='U-Net Training')
    parser.add_argument('--teacher-dir', required=True, help='Teacher labels directory')
    parser.add_argument('--output-dir', required=True, help='Output directory for model')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--img-size', type=int, default=256, help='Image size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    args = parser.parse_args()
    
    teacher_dir = Path(args.teacher_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("STEP 2: U-Net Training")
    print("=" * 70)
    print(f"Teacher dir: {teacher_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Find teacher label directories
    case_dirs = sorted([d for d in teacher_dir.glob("*") if d.is_dir() and (d / "metadata.json").exists()])
    print(f"Found {len(case_dirs)} teacher cases")
    
    if len(case_dirs) == 0:
        print("ERROR: No teacher labels found!")
        return 1
    
    # Split train/val (80/20)
    split_idx = int(len(case_dirs) * 0.8)
    train_dirs = case_dirs[:split_idx]
    val_dirs = case_dirs[split_idx:]
    
    print(f"Train: {len(train_dirs)} cases")
    print(f"Val: {len(val_dirs)} cases")
    print()
    
    # Create datasets
    train_dataset = L3TeacherDataset(train_dirs, img_size=args.img_size)
    val_dataset = L3TeacherDataset(val_dirs, img_size=args.img_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()
    
    # Create model
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=3,  # psoas_left, psoas_right, vat
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2
    ).to(device)
    
    # Loss and optimizer
    criterion = DiceCELoss(sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # AMP scaler for mixed precision (if GPU available)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Training loop
    best_val_loss = float('inf')
    history = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss)
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, output_dir / 'best_model.pth')
            print(f"  ✓ Best model saved!")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, output_dir / f'checkpoint_epoch{epoch}.pth')
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, output_dir / 'final_model.pth')
    
    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print()
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Models saved to: {output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

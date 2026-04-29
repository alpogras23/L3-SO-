#!/usr/bin/env python3
"""
L3 VFA/PMA için 2D U-Net Eğitimi.
4-channel output: VFA, Psoas Left, Psoas Right, L3 Vertebra
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import random

# Paths
DATA_DIR = Path('/home/azureuser/amos22_data/l3_slices_2d')
CT_DIR = DATA_DIR / 'ct_slices'
MASK_DIR = DATA_DIR / 'masks'
MODEL_OUT = Path('/home/azureuser/L3_SO_ANALYSIS/models')

# Training config
BATCH_SIZE = 4
EPOCHS = 100
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 256  # Resize target

class L3Dataset(Dataset):
    """L3 VFA/PMA eğitim dataset'i"""
    
    def __init__(self, ct_files, mask_files, augment=True):
        self.ct_files = ct_files
        self.mask_files = mask_files
        self.augment = augment
    
    def __len__(self):
        return len(self.ct_files)
    
    def __getitem__(self, idx):
        # CT slice yükle
        ct = np.load(self.ct_files[idx]).astype(np.float32)
        
        # HU normalizasyonu (-1024 to 3071 -> 0 to 1)
        ct = np.clip(ct, -1024, 3071)
        ct = (ct + 1024) / 4095.0
        
        # Mask yükle (4 channel: VFA, PsoasL, PsoasR, L3)
        mask = np.load(self.mask_files[idx]).astype(np.float32)
        
        # Resize to fixed size
        ct = self._resize(ct, IMG_SIZE)
        mask = self._resize_mask(mask, IMG_SIZE)
        
        # Augmentation
        if self.augment:
            ct, mask = self._augment(ct, mask)
        
        # Add channel dimension to CT
        ct = ct[np.newaxis, ...]  # (1, H, W)
        
        return torch.from_numpy(ct), torch.from_numpy(mask)
    
    def _resize(self, img, size):
        """Basit resize (nearest neighbor)"""
        import cv2
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    
    def _resize_mask(self, mask, size):
        """4-channel mask resize"""
        import cv2
        resized = np.zeros((4, size, size), dtype=np.float32)
        for i in range(4):
            resized[i] = cv2.resize(mask[i], (size, size), interpolation=cv2.INTER_NEAREST)
        return resized
    
    def _augment(self, ct, mask):
        """Basit augmentation: flip"""
        if random.random() > 0.5:
            ct = np.flip(ct, axis=1).copy()
            mask = np.flip(mask, axis=2).copy()
            # Psoas L/R swap
            mask[[1, 2]] = mask[[2, 1]]
        return ct, mask


class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) * 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """Basit U-Net for multi-class segmentation"""
    
    def __init__(self, in_channels=1, out_channels=4, features=[32, 64, 128, 256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        # Encoder
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # Decoder
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, 2, 2))
            self.ups.append(DoubleConv(feature * 2, feature))
        
        # Final conv
        self.final = nn.Conv2d(features[0], out_channels, 1)
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skip_connections[i // 2]
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)
        
        return self.final(x)


class DiceLoss(nn.Module):
    """Dice loss for multi-class segmentation"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(pred.size(0), pred.size(1), -1)
        target_flat = target.view(target.size(0), target.size(1), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """BCE + Dice combined loss"""
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        return self.bce_weight * self.bce(pred, target) + self.dice_weight * self.dice(pred, target)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for ct, mask in loader:
        ct = ct.to(device)
        mask = mask.to(device)
        
        optimizer.zero_grad()
        pred = model(ct)
        loss = criterion(pred, mask)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    dice_scores = {0: [], 1: [], 2: [], 3: []}  # VFA, PsoasL, PsoasR, L3
    
    with torch.no_grad():
        for ct, mask in loader:
            ct = ct.to(device)
            mask = mask.to(device)
            
            pred = model(ct)
            loss = criterion(pred, mask)
            total_loss += loss.item()
            
            # Per-channel dice
            pred_bin = (torch.sigmoid(pred) > 0.5).float()
            for ch in range(4):
                intersection = (pred_bin[:, ch] * mask[:, ch]).sum()
                union = pred_bin[:, ch].sum() + mask[:, ch].sum()
                if union > 0:
                    dice = (2 * intersection / union).item()
                    dice_scores[ch].append(dice)
    
    avg_dice = {ch: np.mean(scores) if scores else 0 for ch, scores in dice_scores.items()}
    return total_loss / len(loader), avg_dice


def main():
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    
    # Dataset dosyalarını bul
    ct_files = sorted(CT_DIR.glob('*.npy'))
    mask_files = sorted(MASK_DIR.glob('*.npy'))
    
    print(f"Toplam {len(ct_files)} eğitim örneği bulundu")
    
    if len(ct_files) < 10:
        print("Yeterli eğitim verisi yok!")
        return
    
    # Train/Val split (80/20)
    indices = list(range(len(ct_files)))
    random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx = indices[:split]
    val_idx = indices[split:]
    
    train_ct = [ct_files[i] for i in train_idx]
    train_mask = [mask_files[i] for i in train_idx]
    val_ct = [ct_files[i] for i in val_idx]
    val_mask = [mask_files[i] for i in val_idx]
    
    print(f"Eğitim: {len(train_ct)}, Validasyon: {len(val_ct)}")
    
    # DataLoaders
    train_dataset = L3Dataset(train_ct, train_mask, augment=True)
    val_dataset = L3Dataset(val_ct, val_mask, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Model
    model = UNet(in_channels=1, out_channels=4).to(DEVICE)
    print(f"Model device: {DEVICE}")
    print(f"Model parametreleri: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss & Optimizer
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_val_loss = float('inf')
    best_dice = 0
    training_log = []
    
    print("\nEğitim başlıyor...")
    print("-" * 60)
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, dice_scores = validate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)
        
        # Mean Dice
        mean_dice = np.mean(list(dice_scores.values()))
        
        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'dice_vfa': dice_scores[0],
            'dice_psoas_l': dice_scores[1],
            'dice_psoas_r': dice_scores[2],
            'dice_l3': dice_scores[3],
            'mean_dice': mean_dice
        }
        training_log.append(log_entry)
        
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"Dice: VFA={dice_scores[0]:.3f} PL={dice_scores[1]:.3f} PR={dice_scores[2]:.3f} L3={dice_scores[3]:.3f}")
        
        # Save best model
        if mean_dice > best_dice:
            best_dice = mean_dice
            best_val_loss = val_loss
            
            model_path = MODEL_OUT / 'psoas_vfa_best.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'dice_scores': dice_scores,
                'mean_dice': mean_dice
            }, model_path)
            print(f"  ★ Best model kaydedildi (Mean Dice: {mean_dice:.4f})")
    
    # Final model kaydet
    final_path = MODEL_OUT / 'psoas_vfa_final.pth'
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
        'dice_scores': dice_scores
    }, final_path)
    
    # Training log kaydet
    log_path = MODEL_OUT / 'training_log.json'
    with open(log_path, 'w') as f:
        json.dump({
            'config': {
                'batch_size': BATCH_SIZE,
                'epochs': EPOCHS,
                'learning_rate': LEARNING_RATE,
                'img_size': IMG_SIZE,
                'train_samples': len(train_ct),
                'val_samples': len(val_ct)
            },
            'best_dice': best_dice,
            'best_val_loss': best_val_loss,
            'training_log': training_log
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Eğitim tamamlandı!")
    print(f"En iyi Mean Dice: {best_dice:.4f}")
    print(f"Model: {MODEL_OUT / 'psoas_vfa_best.pth'}")

if __name__ == '__main__':
    main()

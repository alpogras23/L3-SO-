"""
TBCC Dataset VFA/PMA Eğitim Script (Radyolog Uyumluluğu Optimizasyonlu)

Bu script TBCC hasta verilerini (DICOM/NIfTI) ve CVAT/radyolog maskelerini
kullanarak L3 seviyesinde VFA/PMA segmentasyon modeli eğitir.

Radyolog ground truth ile karşılaştırmalı eğitim yaparak maksimum klinik doğruluk hedeflenir.

Kullanım:
    python psoas_ml/tbcc_train_vfa_pma.py \
        --data-csv data_prepped/ground_truth.csv \
        --img-root data/images \
        --mask-root data/masks \
        --epochs 60 \
        --batch-size 8 \
        --out-dir /content/drive/MyDrive/L3_RESULTS/tbcc_run
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from monai.networks.nets import UNet
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, ScaleIntensityRange,
    RandFlip, RandRotate, RandZoom, Resize
)


def parse_args():
    parser = argparse.ArgumentParser(description="TBCC L3 VFA/PMA Training")
    parser.add_argument("--data-csv", type=str, required=True,
                        help="Ground truth CSV (case_id,image_path,vfa,pma,height,sex)")
    parser.add_argument("--img-root", type=str, required=True,
                        help="Görüntü dosyaları kök dizini (DICOM veya NIfTI)")
    parser.add_argument("--mask-root", type=str, required=True,
                        help="CVAT/radyolog maskeleri kök dizini")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=10,
                        help="Her N epoch'ta checkpoint kaydet")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Çıktı dizini (model + QA + metrics)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-against-gt", action="store_true",
                        help="Her epoch sonunda ground truth ile MAE hesapla")
    return parser.parse_args()


class TBCCDataset(Dataset):
    """
    TBCC ground truth CSV + CVAT maskeleri ile eğitim dataset
    
    Mask formatı:
        - vfa_mask.png / vfa_mask.npy (visseral yağ)
        - pma_mask.png / pma_mask.npy (psoas kas)
        - sat_mask.png / sat_mask.npy (subkutan yağ, opsiyonel)
    """
    def __init__(self, df, img_root, mask_root, img_size=256, augment=False):
        self.df = df.reset_index(drop=True)
        self.img_root = Path(img_root)
        self.mask_root = Path(mask_root)
        self.img_size = img_size
        self.augment = augment
        
        self.load_img = LoadImage(image_only=True)
        self.tx_img = Compose([
            EnsureChannelFirst(),
            ScaleIntensityRange(a_min=-1000.0, a_max=1000.0, b_min=0.0, b_max=1.0, clip=True),
            Resize((img_size, img_size)),
        ])
        self.tx_mask = Resize((img_size, img_size), mode="nearest")
        self.aug = Compose([
            RandFlip(spatial_axis=1, prob=0.5),
            RandRotate(range_x=math.pi/36, prob=0.3),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.3),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        case_id = row["case_id"]
        
        # Görüntü yolu
        img_path = self.img_root / row["image_path"]
        if not img_path.exists():
            # Alternatif uzantı dene
            img_path = self.img_root / f"{case_id}.dcm"
            if not img_path.exists():
                img_path = self.img_root / f"{case_id}.nii.gz"
        
        # Maskeleri yükle
        mask_dir = self.mask_root / case_id
        
        def load_mask(name):
            for ext in [".png", ".npy", ".nii.gz"]:
                f = mask_dir / f"{name}{ext}"
                if f.exists():
                    if ext == ".npy":
                        return np.load(f)
                    else:
                        return self.load_img(f)
            return None
        
        vfa_mask = load_mask("vfa_mask")
        pma_mask = load_mask("pma_mask")
        sat_mask = load_mask("sat_mask")
        
        # Görüntü ve maskelerin None kontrolü
        try:
            img = self.load_img(img_path)
            img = self.tx_img(img)  # (1, H, W)
        except Exception as e:
            # Hata durumunda dummy veri dön
            print(f"⚠️ Görüntü yüklenemedi: {img_path} ({e})")
            img = torch.zeros(1, self.img_size, self.img_size)
        
        # Maskeleri resize ve birleştir
        H, W = self.img_size, self.img_size
        label = np.zeros((H, W), dtype=np.int64)
        
        if vfa_mask is not None:
            vfa_t = self.tx_mask(torch.from_numpy(np.array(vfa_mask)[None, None, ...].astype(np.float32)))
            label[vfa_t[0, 0].round().numpy() > 0] = 1
        
        if sat_mask is not None:
            sat_t = self.tx_mask(torch.from_numpy(np.array(sat_mask)[None, None, ...].astype(np.float32)))
            label[sat_t[0, 0].round().numpy() > 0] = 2
        
        if pma_mask is not None:
            pma_t = self.tx_mask(torch.from_numpy(np.array(pma_mask)[None, None, ...].astype(np.float32)))
            label[pma_t[0, 0].round().numpy() > 0] = 3
        
        img_t = img.float()
        lbl_t = torch.from_numpy(label)
        
        if self.augment:
            stacked = torch.cat([img_t, lbl_t.unsqueeze(0).float()], dim=0)
            stacked = self.aug(stacked)
            img_t = stacked[0:1]
            lbl_t = stacked[1].round().long()
        
        return img_t, lbl_t, case_id


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()
    
    running_loss = 0.0
    n_batches = 0
    
    with torch.set_grad_enabled(train):
        for batch in loader:
            if len(batch) == 3:
                img, lbl, _ = batch
            else:
                img, lbl = batch
            
            img = img.to(device)
            lbl = lbl.to(device)
            
            if train:
                optimizer.zero_grad()
            
            out = model(img)
            loss = criterion(out, lbl)
            
            if train:
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            n_batches += 1
    
    return running_loss / max(1, n_batches)


def compute_metrics_vs_gt(model, loader, device, df_gt):
    """
    Ground truth VFA/PMA değerleri ile model tahminlerini karşılaştır
    
    Returns:
        dict: MAE, bias, correlation metrikleri
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for img, lbl, case_ids in loader:
            img = img.to(device)
            out = model(img)
            pred = out.argmax(dim=1).cpu().numpy()  # (B, H, W)
            
            for i, case_id in enumerate(case_ids):
                vfa_pred = (pred[i] == 1).sum()
                pma_pred = (pred[i] == 3).sum()
                predictions.append({
                    "case_id": case_id,
                    "vfa_pred": vfa_pred,
                    "pma_pred": pma_pred,
                })
    
    pred_df = pd.DataFrame(predictions)
    merged = df_gt.merge(pred_df, on="case_id", how="inner")
    
    if len(merged) == 0:
        return None
    
    # Pixel sayısını mm² alanına çevir (basit proxy: pixel_spacing varsa kullan)
    # Şu an basit pixel sayısı karşılaştırması yapıyoruz
    vfa_gt = merged["vfa"].values
    vfa_pred = merged["vfa_pred"].values
    pma_gt = merged["pma"].values
    pma_pred = merged["pma_pred"].values
    
    metrics = {
        "vfa_mae": np.abs(vfa_gt - vfa_pred).mean(),
        "vfa_bias": (vfa_pred - vfa_gt).mean(),
        "pma_mae": np.abs(pma_gt - pma_pred).mean(),
        "pma_bias": (pma_pred - pma_gt).mean(),
        "n_cases": len(merged),
    }
    
    return metrics


def main():
    args = parse_args()
    
    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Paths
    img_root = Path(args.img_root)
    mask_root = Path(args.mask_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load ground truth CSV
    df = pd.read_csv(args.data_csv)
    print(f"✅ Ground truth yüklendi: {len(df)} vaka")
    
    # Train/val split
    df_shuffled = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    n_val = max(1, int(0.2 * len(df_shuffled)))
    df_val = df_shuffled.iloc[:n_val]
    df_train = df_shuffled.iloc[n_val:]
    
    train_ds = TBCCDataset(df_train, img_root, mask_root, img_size=args.img_size, augment=True)
    val_ds = TBCCDataset(df_val, img_root, mask_root, img_size=args.img_size, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                               num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                             num_workers=2, pin_memory=True)
    
    print(f"📊 Train: {len(train_ds)} | Val: {len(val_ds)}")
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Device: {device}")
    
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=4,  # bg, VFA, SAT, PMA
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Training loop
    best_val = float("inf")
    log_path = out_dir / "training_log.json"
    logs = []
    
    for epoch in range(1, args.epochs + 1):
        tr_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, None, device, train=False)
        
        log_entry = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": val_loss,
        }
        
        # Ground truth karşılaştırma (opsiyonel)
        if args.validate_against_gt and epoch % 5 == 0:
            metrics = compute_metrics_vs_gt(model, val_loader, device, df_val)
            if metrics:
                log_entry.update(metrics)
                print(f"[{epoch:03d}/{args.epochs}] train={tr_loss:.4f} val={val_loss:.4f} "
                      f"| VFA_MAE={metrics['vfa_mae']:.1f} PMA_MAE={metrics['pma_mae']:.1f}")
            else:
                print(f"[{epoch:03d}/{args.epochs}] train={tr_loss:.4f} val={val_loss:.4f}")
        else:
            print(f"[{epoch:03d}/{args.epochs}] train={tr_loss:.4f} val={val_loss:.4f}")
        
        logs.append(log_entry)
        
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_dir / "best_model.pth")
            print(f"  🔥 En iyi model kaydedildi (val_loss={val_loss:.4f})")
        
        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), out_dir / f"epoch_{epoch:03d}.pth")
    
    # Final save
    torch.save(model.state_dict(), out_dir / "last_model.pth")
    
    # Save logs
    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)
    
    # Final ground truth karşılaştırma
    if args.validate_against_gt:
        model.load_state_dict(torch.load(out_dir / "best_model.pth"))
        final_metrics = compute_metrics_vs_gt(model, val_loader, device, df_val)
        if final_metrics:
            print("\n" + "="*60)
            print("📊 FINAL RADYOLOG KARŞILAŞTIRMA METRİKLERİ")
            print("="*60)
            print(f"VFA MAE  : {final_metrics['vfa_mae']:.2f} mm² (hedef: <50)")
            print(f"VFA Bias : {final_metrics['vfa_bias']:+.2f} mm²")
            print(f"PMA MAE  : {final_metrics['pma_mae']:.2f} mm² (hedef: <15)")
            print(f"PMA Bias : {final_metrics['pma_bias']:+.2f} mm²")
            print(f"Vaka Sayısı: {final_metrics['n_cases']}")
            print("="*60)
            
            # Metrikleri kaydet
            with open(out_dir / "final_metrics.json", "w") as f:
                json.dump(final_metrics, f, indent=2)
    
    print(f"\n✅ Eğitim tamamlandı. En iyi val loss: {best_val:.4f}")
    print(f"📁 Çıktılar: {out_dir}")


if __name__ == "__main__":
    main()

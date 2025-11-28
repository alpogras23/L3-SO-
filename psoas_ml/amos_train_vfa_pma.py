"""
AMOS22 Dataset VFA/PMA Eğitim Script (Colab GPU Optimizasyonlu)

Bu script AMOS22 NIfTI verilerini TotalSegmentator ve Comp2Comp teacher
maskeleri ile birleştirerek L3 seviyesinde VFA/SAT/psoas segmentasyon
modeli eğitir.

Kullanım:
    python psoas_ml/amos_train_vfa_pma.py \
        --amos-root /content/drive/MyDrive/AMOS22 \
        --ts-root /content/drive/MyDrive/TS_teachers_AMOS22 \
        --c2c-root /content/drive/MyDrive/C2C_teachers_AMOS22 \
        --epochs 60 \
        --batch-size 4 \
        --use-c2c-override \
        --out-dir /content/drive/MyDrive/L3_RESULTS/amos_run
"""

import argparse
import json
import math
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from monai.networks.nets import UNet
from monai.transforms import (
    Compose, ScaleIntensityRange, EnsureChannelFirst, 
    RandFlip, RandRotate, RandZoom
)


def parse_args():
    parser = argparse.ArgumentParser(description="AMOS22 L3 VFA/PMA Training")
    parser.add_argument("--amos-root", type=str, required=True,
                        help="AMOS22 NIfTI kök dizini (imagesTr içeren)")
    parser.add_argument("--ts-root", type=str, required=True,
                        help="TotalSegmentator teacher çıktı kökü")
    parser.add_argument("--c2c-root", type=str, default=None,
                        help="Comp2Comp teacher .npy kökü (opsiyonel)")
    parser.add_argument("--use-c2c-override", action="store_true",
                        help="C2C maskelerini TS üzerine override et")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=10,
                        help="Her N epoch'ta checkpoint kaydet")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Çıktı dizini (model + QA)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_nifti(path: Path):
    img = nib.load(str(path))
    data = img.get_fdata()
    return data, img.affine, img.header


def pick_l3_slice(ts_case_dir: Path):
    """Vertebra maskesinden en geniş slice'ı bul (L3 proxy)"""
    cand = [
        ts_case_dir / "vertebrae_lumbar.nii.gz",
        ts_case_dir / "vertebrae_thoracic.nii.gz",
    ]
    for cf in cand:
        if cf.exists():
            vol, _, _ = load_nifti(cf)
            areas = vol.reshape(vol.shape[0], -1).sum(axis=1)
            if areas.max() > 0:
                return int(areas.argmax())
    return None


def extract_teacher_slice(ct_path, ts_root, c2c_root=None, use_c2c=False):
    """
    AMOS CT + TS + (VARSA) C2C birleşimi ile teacher maskeleri üret
    """
    case_id = ct_path.stem.replace(".nii", "")
    ts_dir = ts_root / case_id
    if not ts_dir.exists():
        return None

    hu, aff, hdr = load_nifti(ct_path)
    hu = hu.astype(np.float32)

    z = pick_l3_slice(ts_dir)
    if z is None or z < 0 or z >= hu.shape[0]:
        return None

    hu_slice = hu[z]

    # TS maskelerini oku
    def load_ts(name):
        f = ts_dir / f"{name}.nii.gz"
        if f.exists():
            vol, _, _ = load_nifti(f)
            return (vol[z] > 0).astype(np.uint8)
        return None

    fascia = load_ts("abdominal_wall")
    if fascia is None:
        fascia = (hu_slice > -300).astype(np.uint8)

    muscle = load_ts("muscle")
    if muscle is None:
        muscle = ((hu_slice >= -29) & (hu_slice <= 150)).astype(np.uint8)

    fat_band = ((hu_slice >= -190) & (hu_slice <= -30)).astype(np.uint8)

    sat_default = load_ts("subcutaneous_fat")
    if sat_default is None:
        sat_default = (fat_band * (1 - fascia)).astype(np.uint8)

    vat_default = (fat_band * fascia).astype(np.uint8)

    # Psoas band (geometrik heuristik)
    H, W = hu_slice.shape
    cx = W // 2
    y0, y1 = int(H * 0.55), int(H * 0.95)
    x_margin = int(W * 0.15)
    psoas_band = np.zeros_like(muscle, dtype=np.uint8)
    psoas_band[y0:y1, :] = 1
    psoas_band[:, cx - x_margin:cx + x_margin] = 0
    psoas_default = (muscle & psoas_band).astype(np.uint8)

    vat_mask = vat_default.copy()
    sat_mask = sat_default.copy()
    psoas_mask = psoas_default.copy()
    c2c_used = False

    # C2C override
    if use_c2c and c2c_root is not None:
        c2c_dir = Path(c2c_root)
        
        def load_c2c(name):
            f = c2c_dir / f"{case_id}_{name}.npy"
            if not f.exists():
                return None
            arr = np.load(f)
            if arr.ndim == 3:
                if z >= arr.shape[0]:
                    return None
                return (arr[z] > 0).astype(np.uint8)
            elif arr.ndim == 2:
                return (arr > 0).astype(np.uint8)
            return None

        c2c_vat = load_c2c("vat")
        c2c_sat = load_c2c("sat")
        c2c_pl = load_c2c("psoas_left")
        c2c_pr = load_c2c("psoas_right")

        if c2c_vat is not None:
            vat_mask = c2c_vat
            c2c_used = True
        if c2c_sat is not None:
            sat_mask = c2c_sat
            c2c_used = True
        if c2c_pl is not None or c2c_pr is not None:
            pl = c2c_pl if c2c_pl is not None else np.zeros_like(psoas_mask)
            pr = c2c_pr if c2c_pr is not None else np.zeros_like(psoas_mask)
            psoas_mask = ((pl > 0) | (pr > 0)).astype(np.uint8)
            c2c_used = True

    return {
        "case_id": case_id,
        "z": z,
        "hu_slice": hu_slice,
        "fascia": fascia,
        "vat": vat_mask,
        "sat": sat_mask,
        "psoas": psoas_mask,
        "c2c_used": c2c_used,
    }


class L3Dataset(Dataset):
    """
    4-sınıflı (bg, VAT, SAT, psoas) segmentasyon dataset
    """
    def __init__(self, items, img_size=256, augment=False):
        self.items = items
        self.img_size = img_size
        self.augment = augment
        
        from monai.transforms import Resize as MonaiResize
        self.tx_img = Compose([
            EnsureChannelFirst(),
            ScaleIntensityRange(a_min=-1000.0, a_max=1000.0, b_min=0.0, b_max=1.0, clip=True),
            MonaiResize((img_size, img_size)),
        ])
        self.tx_lbl = MonaiResize((img_size, img_size), mode="nearest")
        self.aug = Compose([
            RandFlip(spatial_axis=1, prob=0.5),
            RandRotate(range_x=math.pi/36, prob=0.3),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.3),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        itm = self.items[idx]
        hu = itm["hu_slice"]
        vat = itm["vat"]
        sat = itm["sat"]
        psoas = itm["psoas"]

        img = self.tx_img(hu.astype(np.float32))

        vat_t = self.tx_lbl(torch.from_numpy(vat[None, None, ...].astype(np.float32)))
        sat_t = self.tx_lbl(torch.from_numpy(sat[None, None, ...].astype(np.float32)))
        psoas_t = self.tx_lbl(torch.from_numpy(psoas[None, None, ...].astype(np.float32)))

        vat_r = vat_t[0, 0].round().numpy().astype(np.uint8)
        sat_r = sat_t[0, 0].round().numpy().astype(np.uint8)
        psoas_r = psoas_t[0, 0].round().numpy().astype(np.uint8)

        label = np.zeros_like(vat_r, dtype=np.int64)
        label[vat_r > 0] = 1
        label[sat_r > 0] = 2
        label[psoas_r > 0] = 3

        img_t = torch.from_numpy(img.astype(np.float32))
        lbl_t = torch.from_numpy(label)

        if self.augment:
            stacked = torch.cat([img_t, lbl_t.unsqueeze(0).float()], dim=0)
            stacked = self.aug(stacked)
            img_t = stacked[0:1]
            lbl_t = stacked[1].round().long()

        return img_t, lbl_t


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()
    
    running_loss = 0.0
    n_batches = 0
    
    with torch.set_grad_enabled(train):
        for img, lbl in loader:
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


def main():
    args = parse_args()
    
    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Paths
    amos_root = Path(args.amos_root)
    ts_root = Path(args.ts_root)
    c2c_root = Path(args.c2c_root) if args.c2c_root else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Find NIfTI files
    nifti_root = amos_root / "imagesTr" if (amos_root / "imagesTr").exists() else amos_root
    ct_files = sorted(list(nifti_root.glob("*.nii.gz")))
    print(f"✅ AMOS CT dosyaları: {len(ct_files)}")
    
    # Extract teacher slices
    teacher_slices = []
    for ct_path in tqdm(ct_files, desc="Teacher slice çıkarılıyor"):
        res = extract_teacher_slice(ct_path, ts_root, c2c_root, args.use_c2c_override)
        if res is not None:
            teacher_slices.append(res)
    
    print(f"✅ Kullanılabilir teacher slice: {len(teacher_slices)}")
    c2c_count = sum(int(x["c2c_used"]) for x in teacher_slices)
    print(f"   (C2C override kullanılan: {c2c_count})")
    
    if len(teacher_slices) < 4:
        print("⚠️ Yetersiz veri, eğitim sonlandırılıyor.")
        return
    
    # Train/val split
    random.shuffle(teacher_slices)
    n_val = max(1, int(0.2 * len(teacher_slices)))
    val_items = teacher_slices[:n_val]
    train_items = teacher_slices[n_val:]
    
    train_ds = L3Dataset(train_items, img_size=args.img_size, augment=True)
    val_ds = L3Dataset(val_items, img_size=args.img_size, augment=False)
    
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
        out_channels=4,
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
        
        print(f"[{epoch:03d}/{args.epochs}] train={tr_loss:.4f}  val={val_loss:.4f}")
        
        logs.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss})
        
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
    
    print(f"✅ Eğitim tamamlandı. En iyi val loss: {best_val:.4f}")
    print(f"📁 Çıktılar: {out_dir}")


if __name__ == "__main__":
    main()

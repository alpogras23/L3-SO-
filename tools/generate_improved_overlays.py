"""
Geliştirilmiş Overlay Üretimi (High-Quality QA)

Model tahminlerini ve ground truth maskelerini CT görüntüsü üzerine
yüksek kaliteli overlay olarak çizer. Radyolog karşılaştırması için.

Kullanım:
    python tools/generate_improved_overlays.py \
        --model /path/to/best_model.pth \
        --data-root /path/to/data \
        --out-dir /path/to/overlays \
        --max-cases 20
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.networks.nets import UNet
from monai.transforms import LoadImage, EnsureChannelFirst, ScaleIntensityRange, Resize


def parse_args():
    parser = argparse.ArgumentParser(description="High-Quality Overlay Generator")
    parser.add_argument("--model", type=str, required=True,
                        help="Eğitilmiş model .pth dosyası")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Test verisi kök dizini")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Overlay çıktı dizini")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--max-cases", type=int, default=20,
                        help="Maksimum overlay üret")
    return parser.parse_args()


def load_model(model_path, device):
    """Model yükle"""
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def create_overlay(hu_slice, pred_mask, alpha=0.7):
    """
    HU slice + tahmin maskesi overlay oluştur
    
    Args:
        hu_slice: (H, W) HU values
        pred_mask: (H, W) segmentation mask (0:bg, 1:VFA, 2:SAT, 3:PMA)
        alpha: Overlay transparency
    
    Returns:
        RGB overlay image
    """
    # HU normalizasyonu (-150 to 250 HU için optimal görüntü)
    hu_norm = np.clip((hu_slice + 150) / 400.0, 0, 1)
    
    # RGB base
    rgb = np.stack([hu_norm, hu_norm, hu_norm], axis=-1)
    
    # Renkler: VFA=kırmızı, SAT=mavi, PMA=yeşil
    vfa_mask = (pred_mask == 1)
    sat_mask = (pred_mask == 2)
    pma_mask = (pred_mask == 3)
    
    overlay = rgb.copy()
    overlay[vfa_mask, 0] = alpha * 1.0 + (1 - alpha) * overlay[vfa_mask, 0]  # Kırmızı
    overlay[sat_mask, 2] = alpha * 1.0 + (1 - alpha) * overlay[sat_mask, 2]  # Mavi
    overlay[pma_mask, 1] = alpha * 1.0 + (1 - alpha) * overlay[pma_mask, 1]  # Yeşil
    
    return overlay


def main():
    args = parse_args()
    
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Device: {device}")
    
    # Model yükle
    model = load_model(args.model, device)
    print(f"✅ Model yüklendi: {args.model}")
    
    # Test görüntülerini bul
    img_paths = []
    for ext in ["*.nii.gz", "*.dcm"]:
        img_paths.extend(list(data_root.rglob(ext)))
    
    img_paths = sorted(img_paths)[:args.max_cases]
    print(f"🔎 Bulunan görüntü: {len(img_paths)}")
    
    if not img_paths:
        print("⚠️ Görüntü bulunamadı.")
        return
    
    # Transforms
    load_img = LoadImage(image_only=True)
    tx = EnsureChannelFirst()
    scale = ScaleIntensityRange(a_min=-1000.0, a_max=1000.0, b_min=0.0, b_max=1.0, clip=True)
    resize_op = Resize((args.img_size, args.img_size))
    
    # Her görüntü için overlay üret
    results = []
    for img_path in img_paths:
        case_id = img_path.stem.replace(".nii", "")
        print(f"🎨 Overlay üretiliyor: {case_id}")
        
        try:
            # Görüntü yükle
            img = load_img(img_path)
            img = tx(img)
            img_scaled = scale(img)
            img_resized = resize_op(img_scaled)
            
            # Model tahmini
            img_tensor = torch.from_numpy(np.array(img_resized)).unsqueeze(0).float().to(device)
            with torch.no_grad():
                out = model(img_tensor)
                pred = out.argmax(dim=1)[0].cpu().numpy()
            
            # Overlay oluştur
            hu_slice = img_resized[0].numpy()
            overlay = create_overlay(hu_slice, pred, alpha=0.7)
            
            # Kaydet
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            axes[0].imshow(hu_slice, cmap="gray")
            axes[0].set_title(f"CT HU (case: {case_id})")
            axes[0].axis("off")
            
            axes[1].imshow(overlay)
            axes[1].set_title("Model Tahmini (R=VFA, B=SAT, G=PMA)")
            axes[1].axis("off")
            
            plt.tight_layout()
            plt.savefig(out_dir / f"{case_id}_overlay.png", dpi=150, bbox_inches="tight")
            plt.close()
            
            # Metrikleri hesapla
            vfa_area = (pred == 1).sum()
            sat_area = (pred == 2).sum()
            pma_area = (pred == 3).sum()
            
            results.append({
                "case_id": case_id,
                "vfa_pixels": int(vfa_area),
                "sat_pixels": int(sat_area),
                "pma_pixels": int(pma_area),
            })
            
        except Exception as e:
            print(f"⚠️ Hata ({case_id}): {e}")
    
    # Sonuçları kaydet
    with open(out_dir / "overlay_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Overlay üretimi tamamlandı: {len(results)} adet")
    print(f"📁 Çıktı dizini: {out_dir}")


if __name__ == "__main__":
    main()

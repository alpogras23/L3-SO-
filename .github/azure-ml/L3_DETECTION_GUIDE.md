# 🔍 L3 SEVİYESİ NASIL BULUNUR? - Detaylı Rehber

## 📊 GENEL MANTIK

L3 vertebra seviyesini bulma işlemi **3 adımda** çalışır:

```
CT Görüntü (3D Volume)
    ↓
TotalSegmentator Segmentasyonu
    ↓
Vertebra Maske Analizi
    ↓
L3 Slice İndeksi Tespit
    ↓
2D Extraction (256×256)
```

---

## 1️⃣ STEP 1: TotalSegmentator Segmentasyonu

### TotalSegmentator Nedir?
- **Foundation Model**: FastSAM + 3D U-Net tabanlı
- **Trained on**: 150+ organ / anatomical structures
- **Output**: Organ labels (her organ için binary mask)

### TS'den Çıkan Vertebra Labels

TotalSegmentator çalıştığında `segmentations/` klasöründe şu dosyalar üretilir:

```
segmentations/
├─ vertebrae.nii.gz          ← Tüm vertebraların mask'ı
├─ vertebra_L1.nii.gz        ← L1 (Lumbar 1)
├─ vertebra_L2.nii.gz        ← L2 (Lumbar 2)
├─ vertebra_L3.nii.gz        ← L3 (Lumbar 3) ✅
├─ vertebra_L4.nii.gz        ← L4
├─ vertebra_L5.nii.gz        ← L5
├─ vertebra_S1.nii.gz        ← S1 (Sacral)
└─ ... (diğer organlar)
```

**Key Point**: TotalSegmentator L3'ü **otomatik olarak segmente eder!**

---

## 2️⃣ STEP 2: Vertebra Maske Analizi

### Approach A: Doğrudan L3 Maskesini Kullan (En İyi)

```python
# Load the specific L3 mask
l3_mask = nib.load("segmentations/vertebra_L3.nii.gz").get_fdata()

# Find slices that contain L3 vertebra
l3_z_slices = np.where(l3_mask.max(axis=(0, 1)) > 0)[0]
# l3_z_slices = [87, 88, 89, 90, 91] (örneğin)

# Use median slice (center of vertebra)
l3_slice = int(np.median(l3_z_slices))  # = 89
```

**Avantajları**:
- ✅ Anatomically accurate (L3 kemiği tam olarak tespit)
- ✅ Vertebra thickness otomatik handle edilir
- ✅ Robust to variations

### Approach B: Tüm Vertebra Maskesinden Median Kullan (Fallback)

```python
# Load all vertebrae (if L3 separate dosya yoksa)
vertebrae_mask = nib.load("segmentations/vertebrae.nii.gz").get_fdata()

# Find all vertebra-containing slices
all_vert_z = np.where(vertebrae_mask.max(axis=(0, 1)) > 0)[0]
# all_vert_z = [50, 51, ..., 150] (tüm spine)

# L3 genellikle lower-middle spine'da
# Yaklaşık %40-50'sinde
l3_slice = all_vert_z[len(all_vert_z) // 2]
```

**Avantajları**:
- ✅ Fallback (L3 dosyası yoksa)
- ✅ Simple
- ❌ Daha az accurate

---

## 3️⃣ STEP 3: L3 Slice Özellikleri

### L3 Slice Nedir?

```
AXIAL PLANE (İçeriden bakış)
┌──────────────────────────┐
│    Skin & Fat            │
│  ┌────────────────────┐  │
│  │  Peritoneal Fat    │  │
│  │  ┌──────────────┐  │  │
│  │  │  Visceral    │  │  │
│  │  │  Fat (VAT)   │  │  │
│  │  │ ┌──────────┐ │  │  │
│  │  │ │ Organs   │ │  │  │
│  │  │ └──────────┘ │  │  │
│  │  │          ┌──┴──┐  │  │ ← Vertebra (L3)
│  │  │ Psoas┌───┘     │  │  │ ← Psoas Muscles
│  │  │     └───┐      │  │  │   (Left & Right)
│  │  └────────┘└─────┘   │  │
│  └────────────────────┘  │
│                          │
└──────────────────────────┘
```

### L3 Slice İçeriği (Değeri)

| Yapı | HU Değeri | Açıklama |
|------|-----------|----------|
| **Psoas Muscle** | 30-80 HU | Dark gray (muscle) |
| **Vertebra** | 200-1000+ HU | Very bright (bone) |
| **VAT (Yağ)** | -50 to -100 HU | Dark (fat) |
| **SAT (Subcutaneous Fat)** | -80 to -120 HU | Lighter gray (fat) |
| **Air** | -1000 HU | Black |
| **Water** | 0 HU | Gray |

---

## 4️⃣ KODDA L3 BULMA - STEP-BY-STEP

### Step 1: TotalSegmentator Çalıştır

```python
from totalsegmentator.python_api import totalsegmentator

# CT image üzerinde organ segmentasyonu yap
totalsegmentator(
    "ct_image.nii.gz",
    output_dir="ts_output",
    task="total",
    fast=True
)

# Çıktı: ts_output/segmentations/ klasörü
```

### Step 2: Vertebra Maskesini Yükle

```python
import nibabel as nib
import numpy as np

# L3-specific mask
l3_nib = nib.load("ts_output/segmentations/vertebra_L3.nii.gz")
l3_mask = l3_nib.get_fdata()  # Shape: (512, 512, 180) örneğin

# Maske format: 
# - 0 = L3 yok
# - 1 = L3 var
```

### Step 3: L3 Slice'ı Tespit Et

```python
# Z-ekseni boyunca L3 içeren slices'ları bul
z_indices = np.where(l3_mask.max(axis=(0, 1)) > 0)[0]

# Örnek output:
# z_indices = [85, 86, 87, 88, 89, 90, 91, 92]

# L3 vertebra center'ı (median)
l3_slice_idx = int(np.median(z_indices))
# l3_slice_idx = 89

print(f"L3 found at slice: {l3_slice_idx}")
```

### Step 4: L3 Slice'ı CT'den Çıkar

```python
# CT görüntüsünü yükle
ct_nib = nib.load("ct_image.nii.gz")
ct_data = ct_nib.get_fdata()  # Shape: (512, 512, 180)

# L3 slice'ı al
l3_hu_slice = ct_data[:, :, l3_slice_idx]  # Shape: (512, 512)

# HU değerleri range: -1000 to 3000+
print(f"L3 slice HU range: {l3_hu_slice.min()} to {l3_hu_slice.max()}")
```

### Step 5: Maskeleri Extract Et

```python
# Psoas masks
psoas_left_mask = nib.load("ts_output/segmentations/psoas_major_left.nii.gz").get_fdata()
psoas_right_mask = nib.load("ts_output/segmentations/psoas_major_right.nii.gz").get_fdata()

# L3 slice'ta psoas'ı al
psoas_left_l3 = psoas_left_mask[:, :, l3_slice_idx]      # Binary: 0 or 1
psoas_right_l3 = psoas_right_mask[:, :, l3_slice_idx]    # Binary: 0 or 1

# VAT mask
vat_mask = ... # visceral fat mask (eğer var)
vat_l3 = vat_mask[:, :, l3_slice_idx] if vat_mask else np.zeros_like(psoas_left_l3)
```

---

## 5️⃣ GÜÇ VE ZAYIF YÖNLER

### ✅ Güçlü Yönler

| Yön | Açıklama |
|-----|----------|
| **Automated** | Manuel işarretleme yok |
| **Consistent** | Aynı anatomiye her seferinde ulaşır |
| **Fast** | TS birkaç dakikada çalışır |
| **Anatomically Sound** | L3 vertebra kesinlikle buluyor |
| **Organ-Specific** | Her organa ayrı mask'ı var |

### ❌ Potansiyel Sorunlar

| Sorun | Çözüm |
|-------|-------|
| **Vertebra Fusion** | L3 + L4 birleşmişse → fallback median kullan |
| **Metal Artifacts** | CT'de metal implant → TS robust, genellikle çalışır |
| **Deformity** | Kyphosis / scoliosis → TS anatomik varyasyon handle ediyor |
| **Image Quality** | Very noisy CT → TS hala güzel, trained on diverse data |
| **Missing L3** | Çok nadiren (post-surgical) → fallback: center slice |

---

## 6️⃣ VALIDATION - L3'ü DOĞRU BULDUĞUNU NASIL KONTROL EDERIZ?

### Method 1: Vertebra Pixellerini Say

```python
# L3 slice'ta vertebra pixel count
vertebra_pixels = np.sum(l3_mask[:, :, l3_slice_idx] > 0)

# L3 vertebra ~5000-15000 pixel (typical)
# Eğer 1000 altsı → yanlış slice olabilir
if vertebra_pixels < 1000:
    print("⚠️ Too few vertebra pixels, might be wrong slice")
```

### Method 2: Psoas Lokasyonunu Kontrol Et

```python
# Psoas left ve right'ın koordinatlarını kontrol et
psoas_left_coords = np.where(psoas_left_l3 > 0)
psoas_right_coords = np.where(psoas_right_l3 > 0)

# Psoas left'in X coordinates (solda olmalı)
psoas_left_x_mean = np.mean(psoas_left_coords[1])
psoas_right_x_mean = np.mean(psoas_right_coords[1])

# Left < Right (anatomik doğruluk)
if psoas_left_x_mean < psoas_right_x_mean:
    print("✅ Psoas anatomy is correct")
else:
    print("⚠️ Psoas coordinates might be wrong")
```

### Method 3: Overlay Kontrol

```python
# L3 slice'ı ve mask'ları görselle
import cv2
import matplotlib.pyplot as plt

# Normalize HU to 0-255
hu_display = np.clip((l3_hu_slice + 100) / 300 * 255, 0, 255).astype(np.uint8)

# Maskeleri color'la
overlay = cv2.cvtColor(hu_display, cv2.COLOR_GRAY2BGR)
overlay[psoas_left_l3 > 0] = [255, 0, 0]      # Red (left)
overlay[psoas_right_l3 > 0] = [0, 255, 0]     # Green (right)

# Göster
plt.imshow(overlay)
plt.title(f"L3 Slice at index {l3_slice_idx}")
plt.show()
```

---

## 7️⃣ KODDA CURRENT IMPLEMENTATION

### Current `pick_l3_slice()` Fonksiyonu

```python
def pick_l3_slice(ts_output_dir: str) -> Optional[int]:
    """Pick L3 slice using TotalSegmentator output."""
    
    segmentations_dir = Path(ts_output_dir) / "segmentations"
    
    # Try vertebra_L3.nii.gz first (best)
    l3_file = segmentations_dir / "vertebra_L3.nii.gz"
    if l3_file.exists():
        l3_data = nib.load(l3_file).get_fdata()
        z_slices = np.where(l3_data.max(axis=(0, 1)) > 0)[0]
        if len(z_slices) > 0:
            return int(np.median(z_slices))
    
    # Fallback: use all vertebrae (generic)
    vertebra_file = segmentations_dir / "vertebrae.nii.gz"
    if vertebra_file.exists():
        vert_data = nib.load(vertebra_file).get_fdata()
        z_slices = np.where(vert_data.max(axis=(0, 1)) > 0)[0]
        if len(z_slices) > 0:
            return int(np.median(z_slices))
    
    # Final fallback: center slice
    return None
```

**Avantajlar**:
- ✅ L3-specific mask kullanır (best accuracy)
- ✅ Fallback strategy var (robust)
- ✅ Median slice kullanır (vertebra center)
- ✅ Simple and reliable

---

## 🎯 SUMMARY

| Aşama | Metod | Çıktı |
|-------|-------|-------|
| 1. Segmentasyon | TotalSegmentator | organ masks |
| 2. L3 Detection | vertebra_L3.nii.gz analizi | z-indices |
| 3. Slice Selection | np.median(z_indices) | l3_slice_idx |
| 4. Extraction | CT[x, y, l3_slice_idx] | 2D HU slice |
| 5. Validation | pixel count + anatomy check | ✅ Verified |

**Result**: Anatomically accurate, automated L3 level detection! ✅

---

## 📚 REFERANSLAR

- **TotalSegmentator Paper**: https://arxiv.org/abs/2208.05868
- **L3 in Medical Imaging**: L3 cross-sectional area is standard for body composition analysis
- **Vertebra Anatomy**: L3 = 3rd lumbar vertebra, typical in abdomen mid-section

# Colab Pro+ Tek Tık AMOS22 Eğitim Playbook

Bu kılavuz, Google Drive’daki AMOS22 verisiyle Colab Pro/Pro+ üzerinde tek tık eğitim akışını açıklar: öğretmen üretimi (TS/C2C), L3 PNG export, manifest ve 60 epoch eğitim, ardından overlay görselleştirme.

## 0) Ön Koşullar
- Google Drive içinde `AMOS22/` klasörü (NIfTI `.nii.gz` hacimler)
- Colab Pro/Pro+ (T4/L4/A100 GPU uygun)
- Runtime > Change runtime type > GPU

## 1) Colab’te Ortam Kurulumu
```python
# Colab: ortam ve bağımlılıklar
!pip -q install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip -q install monai-weekly[all] pytorch-lightning==2.* SimpleITK nibabel opencv-python pandas seaborn
# TotalSegmentator (opsiyonel öğretmen)
!pip -q install TotalSegmentator nnunetv2
```

## 2) Drive Bağlama ve Repo Alma
```python
from google.colab import drive
drive.mount('/content/drive')

# Projeyi alın (veya kendi çatallanmış deponuzu)
!git clone https://github.com/alpogras23/L3-SO-.git /content/L3_SO
%cd /content/L3_SO
```

## 3) Yol Tanımları
```python
from pathlib import Path
DRIVE_ROOT = Path('/content/drive/MyDrive')
AMOS_ROOT = DRIVE_ROOT/'AMOS22'               # AMOS22 NIfTI klasörü
WORK_ROOT = Path('/content/work')            # Çalışma alanı
WORK_ROOT.mkdir(parents=True, exist_ok=True)
```

## 4) (Opsiyonel) Öğretmen Kurulumu
- TotalSegmentator: `TotalSegmentator -i <dicom_or_nifti> -o <out>`
- C2C: Harici CLI ya da modülünüz varsa `C2C_CLI` ortam değişkenini ayarlayın.

```python
import os
os.environ['C2C_CLI'] = 'c2c_infer'  # varsa; yoksa öğretmen C2C adımı atlanır
```

## 5) Uçtan Uca Pipeline (Tek Komut)
Aşağıdaki komut AMOS NIfTI -> DICOM, öğretmen(ler), L3 PNG export, manifest ve eğitim (60 epoch) adımlarını orkestra eder.

```python
!python scripts/run_amos_multiteacher_pipeline.py \
  --amos-root "$DRIVE_ROOT/AMOS22" \
  --out-root "/content/work/pipeline" \
  --images-out "/content/work/images" \
  --gt-out "/content/work/gt_masks" \
  --c2c-out "/content/work/c2c_masks" \
  --ts-out "/content/work/ts_masks" \
  --labels-config "scripts/labels_config_example.json" \
  --run-teachers \
  --epochs 60 \
  --threads 2 \
  --throttle-ms 0 \
  --train
```

Notlar:
- `--labels-config` ile ID/pattern/weights gibi ayarları tek yerden yönetirsiniz.
- `--run-teachers` yoksa mevcut TS/C2C label haritaları kullanılır.
- Eğitim çıktıları `psoas_ml/ckpts/` altında (`best_mt.ckpt`).

## 6) Eğitim Sonrası Metrikler ve Grafikler
```python
# Lightning CSV logger klasörlerini verin
!python scripts/visualize_training.py --runs psoas_ml/ckpts --out training_plots
```

## 7) Overlay Üretimi
- Eğitimden bağımsız olarak L3 overlay görselleri üretmek için `scripts/build_overlays.py` veya GUI/Headless değerlendirme yollarını kullanın.

```python
# Örnek: manifest’teki ilk 10 görüntü için overlay
!python scripts/build_overlays.py \
  --manifest psoas_ml/data/multiteacher_manifest.csv \
  --out training_plots/overlays --max 10
```

## 8) Sık Karşılaşılan Sorunlar
- DICOM üretimi için en sağlıklı yol `plastimatch`tir. Colab’te doğrudan kullanmak zordur; bu nedenle NIfTI akışı tercih edilir.
- TS için `nnUNetv2` veri dosyaları indirme süresi uzun olabilir; Pro+ runtime’ta hızlanır. Gerekirse TS adımını kapatıp sadece GT/C2C kullanın.
- VRAM yetersizliği: `batch-size` ya da ağ genişliğini düşürün.

## 9) İsteğe Bağlı: Sadece Export (Eğitimsiz)
```python
!python scripts/run_amos_multiteacher_pipeline.py \
  --amos-root "$DRIVE_ROOT/AMOS22" \
  --out-root "/content/work/pipeline" \
  --images-out "/content/work/images" \
  --gt-out "/content/work/gt_masks" \
  --c2c-out "/content/work/c2c_masks" \
  --labels-config "scripts/labels_config_example.json"
```

---
Bu dosya, Drive’daki AMOS22 verisiyle Colab’ta “tek tık” eğitim akışını pratik hale getirir. Sorun yaşarsanız `scripts/run_amos_multiteacher_pipeline.py` yardım mesajına (`-h`) bakın.

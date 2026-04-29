# Multi-Teacher Pipeline Hızlı Başlangıç Rehberi

Bu rehber, AMOS CT + TS + C2C ile CPU üzerinde 60 epoch eğitim yapmak için gereken adımları sıralıyor.

## Önkoşullar

- Python 3.9+, `.venv` sanal ortamı aktif
- `pip install SimpleITK numpy torch pytorch-lightning monai` (gerekli paketler)
- AMOS 22 NIfTI verileri (CT hacmi + opsiyonel GT segmentasyon)
- TotalSegmentator ve/veya C2C label haritaları (multi-label NIfTI veya per-class)

## 1. Veri hazırlık (etiket ID'leri)

Etiket NIfTI'lerinizden ID'leri keşfet:

```bash
python scripts/inspect_label_ids.py --nii /path/to/c2c_labels.nii.gz
python scripts/inspect_label_ids.py --nii /path/to/ts_labels.nii.gz
```

Bulduğunuz ID'leri preset dosyasına yaz:

```bash
python scripts/generate_preset_from_labels.py \
  --nii /path/to/c2c_labels.nii.gz \
  --nii /path/to/ts_labels.nii.gz \
  --out-json scripts/label_presets/tbcc.json \
  --set-c2c-l3-id 55 \
  --set-c2c-psoas-ids 55,56 \
  --set-c2c-vat-id 100 --set-c2c-sat-id 101 \
  --set-gt-psoas-ids 55,56 \
  --set-ts-psoas-ids 55,56 \
  --set-w-gt 1.0 --set-w-c2c 0.3 --set-w-ts 0.3
```

## 2. (Opsiyonel) TS per-class birleştirme

TotalSegmentator per-class NIfTI üretiyorsa, önce bunları tek multi-label haritaya dönüştür:

```bash
python scripts/merge_nifti_labels.py \
  --out /path/to/ts_merged/CASE_001/labels.nii.gz \
  --label 55:/path/to/ts_perclass/CASE_001/psoas_left.nii.gz \
  --label 56:/path/to/ts_perclass/CASE_001/psoas_right.nii.gz \
  --label 100:/path/to/ts_perclass/CASE_001/vat.nii.gz \
  --label 101:/path/to/ts_perclass/CASE_001/sat.nii.gz
```

Alternatif: Pipeline içinde otomatik birleştirme için `--ts-perclass-root` ve `--ts-merge-label` kullan (adım 3'te).

## 3. Uçtan uca pipeline çalıştırma

AMOS→DICOM→L3 export→Manifest→Eğitim tek komutla:

```bash
python scripts/run_amos_multiteacher_pipeline.py \
  --labels-config scripts/label_presets/tbcc.json \
  --amos-root /path/to/AMOS22 \
  --out-root build/pipeline_full \
  --images-out build/pipeline_full/images \
  --gt-out build/pipeline_full/gt \
  --c2c-out build/pipeline_full/c2c \
  --ts-out build/pipeline_full/ts \
  --gt-labels-root /path/to/amos_gt \
  --gt-label-pattern "{stem}_gt.nii.gz" \
  --c2c-labels-root /path/to/c2c_labels \
  --c2c-label-pattern "{stem}/labels.nii.gz" \
  --ts-labels-root /path/to/ts_merged \
  --ts-label-pattern "{stem}/labels.nii.gz" \
  --weights 1.0,0.3,0.3 \
  --epochs 60 --threads 1 --throttle-ms 10 \
  --min-usable 10 --train
```

**TS per-class otomatik birleştirme için** (yukarıdaki yerine):

```bash
python scripts/run_amos_multiteacher_pipeline.py \
  --labels-config scripts/label_presets/tbcc.json \
  --amos-root /path/to/AMOS22 \
  --out-root build/pipeline_full \
  --images-out build/pipeline_full/images \
  --gt-out build/pipeline_full/gt \
  --c2c-out build/pipeline_full/c2c \
  --ts-out build/pipeline_full/ts \
  --gt-labels-root /path/to/amos_gt \
  --gt-label-pattern "{stem}_gt.nii.gz" \
  --c2c-labels-root /path/to/c2c_labels \
  --c2c-label-pattern "{stem}/labels.nii.gz" \
  --ts-perclass-root /path/to/ts_perclass \
  --ts-perclass-out-root build/ts_merged \
  --ts-merge-label 55:psoas_left*.nii.gz \
  --ts-merge-label 56:psoas_right*.nii.gz \
  --ts-merge-label 100:vat*.nii.gz \
  --ts-merge-label 101:sat*.nii.gz \
  --weights 1.0,0.3,0.3 \
  --epochs 60 --threads 1 --throttle-ms 10 \
  --min-usable 10 --train
```

## 4. Görsel QA: Overlay üretimi

Export edilen PNG'lerden renkli overlay üret:

```bash
python scripts/build_overlays.py \
  --images-root build/pipeline_full/images \
  --gt-root build/pipeline_full/gt \
  --ts-root build/pipeline_full/ts \
  --c2c-root build/pipeline_full/c2c \
  --out-root build/pipeline_full/overlays \
  --alpha 0.6
```

Renkler: C2C=kırmızı, GT=yeşil, TS=mavi.

## 5. QA özet raporu

Her örneğin boyut ve nonzero piksel bilgisini CSV'ye dökmek için:

```bash
python scripts/qa_pipeline_summary.py \
  --images-root build/pipeline_full/images \
  --gt-root build/pipeline_full/gt \
  --ts-root build/pipeline_full/ts \
  --c2c-root build/pipeline_full/c2c \
  --out build/pipeline_full/qa_summary.csv \
  --print-head 20
```

## 6. Eğitim sonrası

Eğitim tamamlandığında:
- Best checkpoint: `psoas_ml/ckpts/best_mt.ckpt`
- CSV log: `psoas_ml/ckpts/version_*/metrics.csv`

Checkpointı inference için kullanabilir veya doğruluk metriklerini analiz edebilirsiniz.

## CPU ısı yönetimi

- `--threads 1`: tek çekirdek (düşük sıcaklık)
- `--throttle-ms 10-30`: her batch sonrası bekleme (soğuma)
- `--batch-size 2`: bellek ve sıcaklığı dengeleyin

## Sorun giderme

- **DICOM dönüşümü başarısız**: plastimatch yoksa SimpleITK fallback devreye girer (minimal tag seti).
- **L3 dilimi bulunamıyor**: `l3_selector.py` içindeki eşikleri kontrol edin veya `export_l3_pngs.py`'yi manuel dilim indeksi ile çalıştırın.
- **Manifest usable < min-usable**: En az bir kaynak (GT/TS/C2C) maske sağlamalısınız; ID/pattern ayarlarını gözden geçirin.
- **Eğitim loss NaN**: Maskelerin doğru etiket ID'lerine sahip olduğunu, 0/255 binary PNG'ler olduğunu doğrulayın.

## Daha fazla bilgi

- Detaylı açıklama: `README_MULTITEACHER.md`
- Yardımcı scriptler: `scripts/` dizininde
- Preset şablon: `scripts/label_presets/tbcc.json`

# Multi-Teacher Pipeline Araçlar Özeti

Bu dizindeki scriptler, AMOS + TS + C2C multi-teacher eğitim iş akışını destekler.

## Veri hazırlık

### `prepare_amos_dicom.py`
AMOS NIfTI → DICOM dönüştürücü (plastimatch veya SimpleITK fallback).

```bash
python scripts/prepare_amos_dicom.py \
  --amos-root /path/AMOS22 \
  --out-root data/amos_dicom
```

### `merge_nifti_labels.py`
TS per-class NIfTI → tek multi-label NIfTI birleştirici.

```bash
python scripts/merge_nifti_labels.py \
  --out merged.nii.gz \
  --label 55:psoas_left.nii.gz \
  --label 56:psoas_right.nii.gz \
  --label 100:vat.nii.gz \
  --label 101:sat.nii.gz
```

### `inspect_label_ids.py`
NIfTI label histogram (ID keşfi).

```bash
python scripts/inspect_label_ids.py --nii /path/labels.nii.gz
```

### `generate_preset_from_labels.py`
Label histogram + preset JSON oluşturma/güncelleme.

```bash
python scripts/generate_preset_from_labels.py \
  --nii /path/c2c.nii.gz \
  --nii /path/ts.nii.gz \
  --out-json scripts/label_presets/tbcc.json \
  --set-c2c-l3-id 55 \
  --set-c2c-psoas-ids 55,56 \
  --set-gt-psoas-ids 55,56 \
  --set-ts-psoas-ids 55,56 \
  --set-w-gt 1.0 --set-w-c2c 0.3 --set-w-ts 0.3
```

## L3 dilim export

### `export_l3_pngs.py`
DICOM seri + multi-label NIfTI → L3 PNG export (görüntü + maskeler).

```bash
python scripts/export_l3_pngs.py \
  --dicom-series data/amos_dicom/CASE_001/series \
  --out-images images \
  --out-c2c c2c_masks \
  --out-gt gt_masks \
  --out-ts ts_masks \
  --prefix CASE_001 \
  --c2c-label-map labels.nii.gz \
  --label-c2c-l3 55 \
  --label-c2c-psoas 55,56 \
  --label-c2c-vat 100 \
  --label-c2c-sat 101 \
  --gt-label-map gt.nii.gz \
  --label-gt-psoas 55,56 \
  --ts-label-map ts.nii.gz \
  --label-ts-psoas 55,56
```

## Manifest

### `build_multiteacher_manifest.py`
images + (gt/ts/c2c) mask klasörlerinden eğitim manifest CSV üretir.

```bash
python scripts/build_multiteacher_manifest.py \
  --images-root images \
  --gt-root gt_masks \
  --ts-root ts_masks \
  --c2c-root c2c_masks \
  --out manifest.csv \
  --w-gt 1.0 --w-ts 0.3 --w-c2c 0.3
```

## Uçtan uca pipeline

### `run_amos_multiteacher_pipeline.py`
AMOS→DICOM→(öğretmenler)→L3 export→Manifest→(eğitim) tek komutla.

```bash
python scripts/run_amos_multiteacher_pipeline.py \
  --labels-config scripts/label_presets/tbcc.json \
  --amos-root /path/AMOS22 \
  --out-root build/pipeline \
  --images-out build/pipeline/images \
  --gt-out build/pipeline/gt \
  --c2c-out build/pipeline/c2c \
  --ts-out build/pipeline/ts \
  --gt-labels-root /path/amos_gt \
  --gt-label-pattern "{stem}_gt.nii.gz" \
  --c2c-labels-root /path/c2c \
  --c2c-label-pattern "{stem}/labels.nii.gz" \
  --ts-labels-root /path/ts \
  --ts-label-pattern "{stem}/labels.nii.gz" \
  --weights 1.0,0.3,0.3 \
  --epochs 60 --threads 1 --throttle-ms 10 \
  --min-usable 10 --train
```

**TS per-class otomatik birleştirme:**
```bash
  --ts-perclass-root /path/ts_perclass \
  --ts-perclass-out-root build/ts_merged \
  --ts-merge-label 55:psoas_left*.nii.gz \
  --ts-merge-label 56:psoas_right*.nii.gz \
  --ts-merge-label 100:vat*.nii.gz \
  --ts-merge-label 101:sat*.nii.gz
```

## Görsel QA

### `build_overlays.py`
Görüntü + maskeler → renkli overlay PNG (C2C=kırmızı, GT=yeşil, TS=mavi).

```bash
python scripts/build_overlays.py \
  --images-root build/pipeline/images \
  --gt-root build/pipeline/gt \
  --ts-root build/pipeline/ts \
  --c2c-root build/pipeline/c2c \
  --out-root build/pipeline/overlays \
  --alpha 0.6
```

### `qa_pipeline_summary.py`
Export edilmiş verilerden QA özet CSV (boyut, nonzero piksel, var/yok).

```bash
python scripts/qa_pipeline_summary.py \
  --images-root build/pipeline/images \
  --gt-root build/pipeline/gt \
  --ts-root build/pipeline/ts \
  --c2c-root build/pipeline/c2c \
  --out qa_summary.csv \
  --print-head 20
```

## Eğitim

### `psoas_ml/multiteacher_train.py`
Manifest tabanlı multi-teacher UNet eğitimi (CPU-safe).

```bash
python psoas_ml/multiteacher_train.py \
  --manifest manifest.csv \
  --epochs 60 \
  --batch-size 2 \
  --threads 1 \
  --throttle-ms 10 \
  --out psoas_ml/ckpts
```

## Öğretmen çalıştırma (opsiyonel)

### `run_teachers.py`
TotalSegmentator ve C2C'yi DICOM üzerinde çalıştırır (CLI'ler kurulu olmalı).

```bash
python scripts/run_teachers.py \
  --dicom-root data/amos_dicom \
  --out-root data/teachers \
  --c2c-cmd c2c_infer
```

Not: Öğretmenleri pipeline dışında kendiniz çalıştırıp label haritalarını sağlarsanız bu adım opsiyoneldir.

## Label preset şablonu

`scripts/label_presets/tbcc.json`:
```json
{
  "name": "TBCC-like default",
  "c2c": {
    "l3_id": 55,
    "psoas_ids": [55, 56],
    "vat_id": 100,
    "sat_id": 101
  },
  "gt": {
    "psoas_ids": [55, 56]
  },
  "ts": {
    "psoas_ids": [55, 56]
  },
  "weights": {
    "w_gt": 1.0,
    "w_ts": 0.3,
    "w_c2c": 0.3
  }
}
```

## Sık kullanılan iş akışı

1. ID keşfi: `inspect_label_ids.py` → `generate_preset_from_labels.py`
2. (Opsiyonel) Per-class merge: `merge_nifti_labels.py` veya pipeline içi auto-merge
3. Pipeline: `run_amos_multiteacher_pipeline.py --train`
4. QA: `build_overlays.py` + `qa_pipeline_summary.py`
5. Eğitim tamamlandığında: `psoas_ml/ckpts/best_mt.ckpt` kullanıma hazır

## Hata ayıklama

- DICOM conversion fail: plastimatch ve SimpleITK varlığını kontrol edin
- L3 export empty: L3 dilim tespiti hatası; manuel dilim index deneyin
- Manifest usable < min-usable: GT/TS/C2C'den en az birinin mevcut olduğunu doğrulayın
- Loss NaN: Maskelerin binary (0/255) PNG ve doğru ID'lere sahip olduğunu kontrol edin

## Daha fazla dökümantasyon

- `README_MULTITEACHER.md`: Kapsamlı açıklama
- `QUICKSTART_MULTITEACHER.md`: Hızlı başlangıç adımları
- `.github/copilot-instructions.md`: Proje yapısı ve best practices

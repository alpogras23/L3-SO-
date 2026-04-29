# AMOS + TS + C2C Multi-Teacher Eğitim (CPU)

Bu belge, AMOS yerel GT (yüksek ağırlık) ile TS/C2C pseudo-label (düşük ağırlık) kullanarak CPU üzerinde 60 epoch eğitim iş akışını özetler.

## 1) AMOS'u DICOM'a çevir (plastimatch veya SimpleITK fallback)

Öncelik: plastimatch. Eğer yoksa aynı komut SimpleITK fallback ile de çalışır.

macOS plastimatch kurulumu (opsiyonel):

- `brew install plastimatch`

Çalıştırma (plastimatch yoksa otomatik SimpleITK fallback):

```
python scripts/prepare_amos_dicom.py --amos-root /Users/<you>/Desktop/AMOS22 --out-root data/amos_dicom
```

Not: SimpleITK fallback ile üretilen DICOM serileri minimal tag setine sahiptir (araştırma amaçlı uygundur). Klinik-grade DICOM üretimi için plastimatch önerilir.

## 2) Öğretmenleri DICOM üzerinde çalıştır (opsiyonel)

- TS (TotalSegmentator): `pip install TotalSegmentator` ve nnUNetv2 kurulu olmalı.
- C2C: sisteminizdeki C2C CLI'yi `c2c_infer` veya `C2C_CLI` env ile belirtin.

```
python scripts/run_teachers.py --dicom-root data/amos_dicom --out-root data/teachers --c2c-cmd c2c_infer
```

Not: Bu betik yalnızca öğretmen çıkışlarını üretir. Maskeleri PNG dilimlerine map etmek ve L3 dilimini seçmek proje akışınıza göre yapılmalıdır.

## 3) Manifest oluştur

Eğitime girdi olarak PNG görüntüler ve karşılık gelen maskeler kullanılır. Aşağıdaki dizinleri örnekleyin:

- `images_root`: Gri seviyeli dilim görüntüleri (PNG)
- `gt_root`: AMOS GT'den üretilmiş psoas/organ maskeleri (PNG)
- `ts_root`: TS'ten elde edilen maskeler (PNG)
- `c2c_root`: C2C'ten elde edilen maskeler (PNG)

```
python scripts/build_multiteacher_manifest.py \
  --images-root psoas_ml/data/images \
  --gt-root psoas_ml/data/gt_masks \
  --ts-root psoas_ml/data/ts_masks \
  --c2c-root psoas_ml/data/c2c_masks \
  --out psoas_ml/data/multiteacher_manifest.csv \
  --w-gt 1.0 --w-ts 0.3 --w-c2c 0.3
```

PNG’leri üretmek için: DICOM serisi ve C2C/GT çok-etiketli NIfTI ile L3 dilimini seçip 2D maske çıkarmak adına aşağıdaki scripti kullanabilirsiniz.

```
python scripts/export_l3_pngs.py \
  --dicom-series data/amos_dicom/CASE_001/series_X \
  --out-images psoas_ml/data/images \
  --out-c2c psoas_ml/data/c2c_masks \
  --out-gt psoas_ml/data/gt_masks \
  --prefix CASE_001 \
  --c2c-label-map data/teachers/c2c/CASE_001/labels.nii.gz \
  --label-c2c-l3 55 \
  --label-c2c-psoas 55,56 \
  --label-c2c-vat 100 \
  --label-c2c-sat 101 \
  --c2c-base-from-psoas \
  --gt-label-map data/amos_labels/CASE_001_gt.nii.gz \
  --label-gt-psoas 55
```

Notlar:
- `--label-*-*` değerleri veri setinizdeki etiket id’lerine göre ayarlanmalıdır.
- `--c2c-base-from-psoas` bayrağı, `c2c_masks/CASE_001.png` dosyasını psoas maskesine eşit yazar; manifest eşleştirmesi için kolaylık sağlar.

### TS Per-class -> Multi-label Birleştirme

TotalSegmentator bazı kurulumlarda sınıf başına ayrı NIfTI üretebilir. Aşağıdaki araçla bunları tek çok-etiketli haritaya birleştirebilirsiniz:

```
python scripts/merge_nifti_labels.py \
  --out data/teachers/ts/CASE_001/labels.nii.gz \
  --label 55:data/teachers/ts/CASE_001/psoas_left.nii.gz \
  --label 56:data/teachers/ts/CASE_001/psoas_right.nii.gz \
  --label 100:data/teachers/ts/CASE_001/vat.nii.gz \
  --label 101:data/teachers/ts/CASE_001/sat.nii.gz
```

Çakışmalarda son giren etiket kazanır. `--reference` ile geometriyi sabitleyebilirsiniz.

#### Pipeline içinde otomatik birleştirme (opsiyonel)

Per-class TS dosyalarını pipeline’da otomatik birleştirmek için şu bayrakları kullanın:

```
python scripts/run_amos_multiteacher_pipeline.py \
  --labels-config scripts/label_presets/tbcc.json \
  --amos-root /path/AMOS22 \
  --out-root build/pipeline_ts \
  --ts-perclass-root /path/TS_PERCLASS \
  --ts-perclass-out-root build/ts_merged \
  --ts-merge-label 55:psoas_left*.nii.gz \
  --ts-merge-label 56:psoas_right*.nii.gz \
  --ts-merge-label 100:vat*.nii.gz \
  --ts-merge-label 101:sat*.nii.gz \
  --min-usable 10 --train
```

Notlar:
- Varsayılan olarak case klasör adı `{stem}` kabul edilir ve `--ts-perclass-root/{stem}/` altında glob uygulanır.
- Çakışmalarda son giren etiket kazanır. `--ts-merge-reference` ile (case içinde glob) referans geometri belirtebilirsiniz.

### Label ID keşfi ve preset güncelleme (yeni)

Elinizdeki NIfTI etiket haritalarındaki ID'leri hızlıca görmek için:

```
python scripts/inspect_label_ids.py --nii /path/to/labels.nii.gz
```

Bulduğunuz ID'leri preset dosyasına yazmayı otomatikleştirmek için:

```
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

Komut, mevcut dosyayı günceller veya yoksa oluşturur. `--set-*` bayraklarını ihtiyacınıza göre değiştirebilirsiniz.

## 4) Eğitim (CPU, 60 epoch, ısınmayı azaltma)

```
python psoas_ml/multiteacher_train.py \
  --manifest psoas_ml/data/multiteacher_manifest.csv \
  --epochs 60 \
  --batch-size 2 \
  --threads 2 \
  --throttle-ms 0 \
  --out psoas_ml/ckpts
```

İpucu: `--threads 1` ve `--throttle-ms 20` gibi değerlerle CPU yükünü daha da düşürebilirsiniz.
## 5) Tek komutta uçtan uca pipeline (opsiyonel)

Aşağıdaki komut, AMOS→DICOM (plastimatch veya fallback), (opsiyonel) öğretmenler, L3 PNG export, manifest ve (opsiyonel) eğitimi tek komutta çalıştırır:

```
python scripts/run_amos_multiteacher_pipeline.py \
  --amos-root "$HOME/Desktop/AMOS22" \
  --out-root data/pipeline \
  --images-out psoas_ml/data/images \
  --gt-out psoas_ml/data/gt_masks \
  --c2c-out psoas_ml/data/c2c_masks \
  --ts-out psoas_ml/data/ts_masks \
  --c2c-l3-id 55 --c2c-psoas-ids 55,56 --c2c-vat-id 100 --c2c-sat-id 101 \
  --gt-psoas-ids 55 \
  --gt-labels-root data/amos_labels \
  --gt-label-pattern "{stem}_gt.nii.gz" \
  --c2c-labels-root data/teachers/c2c \
  --c2c-label-pattern "{stem}/labels.nii.gz" \
  --weights 1.0,0.3,0.3 \
  --epochs 60 --threads 1 --throttle-ms 10 --min-usable 10 --train
```

Notlar:
- Öğretmenleri bu pipeline içinde çalıştırmak için `--run-teachers` ekleyin. TotalSegmentator/C2C CLI kurulu değilse mevcut label haritaları kullanılır.
- `--min-usable` ile, manifestte en az kaç adet etiketli/pseudo etiketli satır olması gerektiğini şart koşabilirsiniz. Aksi halde eğitim başlamaz (boş loss önlenir).

### Görsel QA: Overlay üretimi (opsiyonel)

Export edilmiş PNG görüntüleri ve maskelerden renkli overlay üretmek için:

```
python scripts/build_overlays.py \
  --images-root build/pipeline_demo_cfg4/images \
  --gt-root build/pipeline_demo_cfg4/gt \
  --ts-root build/pipeline_demo_cfg4/ts \
  --c2c-root build/pipeline_demo_cfg4/c2c \
  --out-root build/pipeline_demo_cfg4/overlays \
  --alpha 0.6
```

Renkler: C2C=kırmızı, GT=yeşil, TS=mavi. `--alpha` ile şeffaflığı ayarlayabilirsiniz.

## Notlar

- AMOS GT kas ve vertebra için altın etiket; TS/C2C daha düşük ağırlıkla düzenleyici.
- L3 düzeyinin oturtulması: Mevcut çekirdekteki `l3_selector.py` ve/veya TS/C2C çıktılarından seviye belirleme ile entegre edilebilir.
- Bu repo içinde örnek eğitim girdi klasörü `psoas_ml/data/images` ve `psoas_ml/data/masks` şeklindedir; çoklu öğretmen eğitiminde yeni `gt_masks/ts_masks/c2c_masks` dizinleri kullanılır.

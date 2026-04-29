# AMOS22 + TotalSegmentator Eğitim Planı

## Durum: TotalSegmentator Çalışıyor ⏳

Şu anda 3 AMOS vakası için TotalSegmentator segmentasyonu arka planda çalışıyor:
- `amos_0001.nii.gz` → `data/ts_labels_amos/amos_0001/`
- `amos_0004.nii.gz` → `data/ts_labels_amos/amos_0004/`
- `amos_0005.nii.gz` → `data/ts_labels_amos/amos_0005/`

CPU üzerinde her vaka ~5-10 dakika sürebilir.

## TotalSegmentator Çıktıları

TS her vaka için per-class NIfTI dosyaları üretir:
- `psoas_major_left.nii.gz`
- `psoas_major_right.nii.gz`
- `vertebrae_L3.nii.gz`
- `subcutaneous_fat.nii.gz`
- `torso_fat.nii.gz`
- + diğer anatomik yapılar

## Label ID Eşleştirmesi

Bizim pipeline için gerekli ID'ler:
```json
{
  "ts": {
    "l3_id": 1,           // TS vertebrae_L3.nii.gz (binary: 0/1)
    "psoas_ids": [1, 1],  // Her iki psoas binary (left=1, right=1)
    "vat_id": 1,          // torso_fat binary
    "sat_id": 1           // subcutaneous_fat binary
  }
}
```

**Not**: TS per-class dosyaları binary (0/1) olduğu için, merge ederken her sınıfa farklı ID atamamız gerekecek.

## Adım Adım İş Akışı

### 1. TS Segmentasyonlarının Tamamlanmasını Bekle

```bash
# İlerlemeyi kontrol et
ps aux | grep TotalSegmentator

# Çıktıları kontrol et (bittiğinde)
ls data/ts_labels_amos/amos_0001/
```

### 2. TS Per-class → Multi-label Birleştirme

Her vaka için ayrı ayrı:

```bash
cd "$HOME/Desktop/L3_SO_ANALYSIS"

# amos_0001 için
.venv/bin/python scripts/merge_nifti_labels.py \
  --out data/ts_merged_amos/amos_0001_labels.nii.gz \
  --label 55:data/ts_labels_amos/amos_0001/psoas_major_left.nii.gz \
  --label 56:data/ts_labels_amos/amos_0001/psoas_major_right.nii.gz \
  --label 100:data/ts_labels_amos/amos_0001/torso_fat.nii.gz \
  --label 101:data/ts_labels_amos/amos_0001/subcutaneous_fat.nii.gz \
  --reference data/ts_labels_amos/amos_0001/psoas_major_left.nii.gz

# amos_0004 için (aynı şekilde)
# amos_0005 için (aynı şekilde)
```

### 3. Preset Güncelleme

TS etiketlerini preset'e ekle:

```bash
.venv/bin/python scripts/generate_preset_from_labels.py \
  --nii data/ts_merged_amos/amos_0001_labels.nii.gz \
  --out-json scripts/label_presets/amos_ts.json \
  --set-ts-psoas-ids 55,56 \
  --set-gt-psoas-ids 1,2,3  \
  --set-w-gt 1.0 --set-w-ts 0.3
```

**Not**: AMOS GT'de psoas ID'si yok, sadece organ segmentasyonları var. Bu yüzden:
- **GT**: AMOS organlarını (liver, kidney vb.) kullanacağız (anatomik referans için)
- **TS**: Psoas kaslarını pseudo-label olarak kullanacağız

### 4. Pipeline Çalıştırma (İlk 3 Vaka Test)

```bash
.venv/bin/python scripts/run_amos_multiteacher_pipeline.py \
  --labels-config scripts/label_presets/amos_ts.json \
  --amos-root "$HOME/Desktop/amos22/imagesTr" \
  --out-root build/amos_pipeline_test \
  --images-out build/amos_pipeline_test/images \
  --gt-out build/amos_pipeline_test/gt \
  --ts-out build/amos_pipeline_test/ts \
  --gt-labels-root "$HOME/Desktop/amos22/labelsTr" \
  --gt-label-pattern "{stem}.nii.gz" \
  --ts-labels-root data/ts_merged_amos \
  --ts-label-pattern "{stem}_labels.nii.gz" \
  --case-glob "amos_000[145].nii.gz" \
  --weights 1.0,0.3,0.0 \
  --epochs 5 --threads 1 --throttle-ms 10 \
  --min-usable 1 --train
```

### 5. QA ve Overlay Kontrolü

```bash
# Overlay üret
.venv/bin/python scripts/build_overlays.py \
  --images-root build/amos_pipeline_test/images \
  --gt-root build/amos_pipeline_test/gt \
  --ts-root build/amos_pipeline_test/ts \
  --out-root build/amos_pipeline_test/overlays \
  --alpha 0.6

# QA özeti
.venv/bin/python scripts/qa_pipeline_summary.py \
  --images-root build/amos_pipeline_test/images \
  --gt-root build/amos_pipeline_test/gt \
  --ts-root build/amos_pipeline_test/ts \
  --out build/amos_pipeline_test/qa_summary.csv
```

### 6. Tam Dataset Eğitimi (60 Epoch)

Test başarılıysa, tüm AMOS training set için:

```bash
# Önce tüm vak

alar için TS çalıştır (paralel batch ile)
for i in {1..240}; do
  case_num=$(printf "%04d" $i)
  input="$HOME/Desktop/amos22/imagesTr/amos_${case_num}.nii.gz"
  [ -f "$input" ] && .venv/bin/TotalSegmentator -i "$input" \
    -o "data/ts_labels_amos/amos_${case_num}" --ml --fast &
  
  # Her 4 vakada bekle (CPU koruması)
  if (( i % 4 == 0 )); then
    wait
  fi
done
wait

# Tüm TS çıktılarını merge et
python scripts/batch_merge_ts.py  # (yeni script gerekir)

# Pipeline ile 60 epoch eğitim
.venv/bin/python scripts/run_amos_multiteacher_pipeline.py \
  --labels-config scripts/label_presets/amos_ts.json \
  --amos-root "$HOME/Desktop/amos22/imagesTr" \
  --out-root build/amos_full_train \
  --gt-labels-root "$HOME/Desktop/amos22/labelsTr" \
  --ts-labels-root data/ts_merged_amos \
  --epochs 60 --threads 1 --throttle-ms 10 \
  --min-usable 50 --train
```

## Önemli Notlar

1. **AMOS GT**: Psoas kası yok, sadece organlar var. Bu yüzden GT'yi anatomik bağlam için kullanacağız (opsiyonel).

2. **TS Pseudo-labels**: Psoas segmentasyonu için ana kaynak. Yüksek kaliteli ama yine de pseudo-label (w=0.3).

3. **C2C Yok**: Şu an C2C kullanmıyoruz çünkü elimizde yok. Sadece GT (organlar) + TS (psoas).

4. **CPU Sınırlamaları**: 
   - TS segmentasyon: ~5-10 dk/vaka
   - 240 vaka için toplam: ~20-40 saat
   - Paralel batch önerilir (4 vaka aynı anda)

5. **Eğitim Süresi**: 
   - 60 epoch, 240 vaka, CPU: ~24-48 saat
   - `--throttle-ms 10-30` ile ısı kontrolü

## Sonraki Adımlar

- [ ] TS segmentasyonlarının tamamlanmasını bekle (3 vaka)
- [ ] Per-class → multi-label birleştirme
- [ ] 3 vaka ile test pipeline çalıştır (5 epoch)
- [ ] Overlay + QA kontrol
- [ ] Başarılıysa: Tüm dataset için TS batch işleme
- [ ] 60 epoch tam eğitim

## Kaynaklar

- [TotalSegmentator Dökümantasyonu](https://github.com/wasserth/TotalSegmentator)
- [AMOS22 Dataset](https://amos22.grand-challenge.org/)
- Pipeline araçları: `scripts/PIPELINE_SUMMARY.md`
- Hızlı başlangıç: `QUICKSTART_MULTITEACHER.md`

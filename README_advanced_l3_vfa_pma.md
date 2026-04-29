# Gelişmiş L3 VFA/PMA Hesaplama Aracı

Bu araç DICOM görüntülerinden L3 vertebra seviyesinde Visseral Yağ Alanı (VFA) ve Psoas Kas Alanı (PMA) hesaplaması yapar. Psoas maskesi çıkarımı ve fasya sınır tespiti ile en doğru klinik sonuçları üretir.

## Özellikler

- **DICOM Serisi İşleme**: Otomatik slice sıralama ve HU dönüşümü
- **L3 Vertebra Tespiti**: Yoğunluk tabanlı otomatik vertebra bulma
- **Dokü Segmentasyonu**: HU threshold tabanlı kas ve yağ ayrımı
- **Psoas Kas Çıkarımı**: Vertebra merkezine göre sol/sağ psoas ayrımı
- **Fasya Sınır Tespiti**: Abdominal duvar sınır belirleme
- **Alan Hesaplaması**: mm² cinsinden doğru ölçümler
- **Görsel Overlay**: Segmentasyon sonuçlarının görselleştirilmesi
- **Konfigürasyon Desteği**: settings.json ile parametre ayarlama

## Kurulum

```bash
pip install pydicom numpy scipy scikit-image opencv-python
```

## Kullanım

### Temel Kullanım

```bash
python advanced_l3_vfa_pma.py --input-dicom /path/to/dicom/directory --output-dir ./results
```

### Çoklu Vaka İşleme

```bash
# Birden fazla hasta dizini için
python advanced_l3_vfa_pma.py --input-dicom /data/patient1 /data/patient2 --output-dir ./batch_results
```

### Komut Satırı Seçenekleri

- `--input-dicom`: DICOM dosyalarının bulunduğu dizin(ler)
- `--output-dir`: Sonuçların kaydedileceği dizin
- `--verbose, -v`: Detaylı log çıktısı

## Çıktılar

Her vaka için aşağıdaki dosyalar oluşturulur:

- `l3_vfa_pma_results.json`: Ölçüm sonuçları ve metadata
- `l3_vfa_pma_overlay.png`: Segmentasyon overlay görseli

### JSON Sonuç Formatı

```json
{
  "case_id": "patient001",
  "l3_slice_idx": 2,
  "status": "success",
  "pma_mm2": 1250.5,
  "vfa_mm2": 850.2,
  "sat_mm2": 245.8,
  "total_fat_mm2": 1096.0,
  "pixel_area_mm2": 1.5,
  "hu_stats": {
    "hu_min": -1024.0,
    "hu_max": 3071.0,
    "hu_mean": 25.3,
    "hu_std": 450.2
  },
  "metadata": {
    "pixel_spacing": [1.5, 1.5],
    "slice_thickness": 5.0,
    "study_uid": "...",
    "series_uid": "..."
  }
}
```

## Konfigürasyon

`settings.json` dosyası ile HU threshold'ları ve işlem parametreleri ayarlanabilir:

```json
{
  "VFA_HU_LOW": -180,
  "VFA_HU_HIGH": -20,
  "PSOAS_HU_MIN": -10,
  "PSOAS_HU_MAX": 100,
  "VB_HU_MIN": 150,
  "VB_HU_MAX": 4000
}
```

## Teknik Detaylar

### HU Threshold'ları

- **Psoas Kas**: -10 to 100 HU
- **Visseral Yağ**: -180 to -20 HU
- **Vertebra**: 150 to 4000 HU

### İşleme Hattı

1. **DICOM Yükleme**: Pixel spacing ve slice sıralama
2. **L3 Bulma**: Maksimum vertebra yoğunluğu
3. **Segmentasyon**: HU tabanlı doku ayrımı
4. **Psoas Çıkarımı**: Vertebra merkezine göre bölme
5. **Fasya Tespiti**: Kas-yağ sınır belirleme
6. **Alan Hesaplama**: Pixel counting ve mm² dönüşümü

### DICOM Uyumluluğu

- Otomatik pixel spacing tespiti (PixelSpacing, NominalScannedPixelSpacing, ImagerPixelSpacing)
- Slice sıralama (SliceLocation, ImagePositionPatient, InstanceNumber, dosya adı)
- HU rescale uygulama (RescaleSlope, RescaleIntercept)

## Sorun Giderme

### Yaygın Hatalar

**"No DICOM files found"**
- DICOM dizin yolunu kontrol edin
- Dosyaların .dcm uzantılı olduğundan emin olun

**"operands could not be broadcast"**
- Pixel spacing bilgisi eksik, varsayılan 1.5mm kullanılıyor

**"No valid DICOM files loaded"**
- DICOM dosyalarının bozuk olmadığından emin olun

### Debug Modu

Detaylı log için verbose flag kullanın:

```bash
python advanced_l3_vfa_pma.py --input-dicom /path/to/dicom --output-dir ./results --verbose
```

## Performans

- Tipik işlem süresi: 2-5 saniye/vaka
- Hafıza kullanımı: ~50MB/vaka
- GPU gerektirmez, CPU tabanlı

## Lisans

Bu araç akademik ve klinik araştırma amaçlıdır.
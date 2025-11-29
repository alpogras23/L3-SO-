# L3 SO Analizi Projesi - AI Asistan Rehberi

## Proje Amacı

Bu proje, **radyolog ölçümleriyle maksimum uyumluluk** sağlayan otomatik L3 vertebra seviyesi VFA (visseral yağ alanı) ve PMA (psoas kas alanı) hesaplama uygulamasıdır. Temel hedef: **En doğru klinik sonuçlar** elde etmek için sürekli parametre optimizasyonu ve doğruluk iyileştirmesi.

### Proje Geliştirme Geçmişi
- ChatGPT, Codex ve GitHub ile geliştirildi
- Masaüstü kısayolu ile çalıştırılıyor (run komutu sorunlu)
- Radyolog ground truth verilerine karşı sürekli validasyon yapılıyor

## Temel Mimari

### 1. İkili Çekirdek Sistemi
- **`core_mini.py`**: Tüm görüntü işleme operasyonları için kararlı, üretim çekirdek motoru
- **`l3_vfa_pma_core_v3_7.py`**: `core_mini.py`'ye delege eden eski uyumluluk katmanı
- Uygun çekirdeği dinamik olarak yüklemek için `tools/core_selector.py` kullanın - bu tüm çekirdek operasyonları için tek giriş noktasıdır

### 2. Ana Giriş Noktaları
- **GUI**: `run_gui_bootstrap.py` → `desktop_project/l3_vfa_pma_gui.py`
- **Toplu İşleme**: Çoklu hasta analizi için `desktop_project/batch_process.py`
- **Tek Vaka**: Bireysel vakaları test etmek için `desktop_project/quick_test.py`
- **CLI Değerlendirme**: Komut satırı işleme için `tools/eval_one.py`

### 3. İşleme Hattı Akışı
1. **DICOM Girişi** → `load_hu()` Hounsfield Birimlerini çıkarır
2. **L3 Seçimi** → `l3_selector.py` seriden optimal dilimi bulur
3. **Segmentasyon** → `inner_abdomen_via_wall()` fasya sınırını oluşturur
4. **VFA Hesaplama** → Fasya içinde adaptif yağ eşikleme
5. **PMA Hesaplama** → Vertebra etrafında psoas kas segmentasyonu
6. **Kalite Güvencesi** → Güven skorlaması ve doğrulama bayrakları

## Kritik Doğruluk Parametreleri

### Radyolog Uyumluluğu İçin Temel Ayarlar
- **VFA Eşikleri**: `VFA_HU_LOW/HIGH` - yağ dokusunu doğru tanımlamak için kritik
- **PMA Kas Tespiti**: `PSOAS_*` parametreleri - kas-yağ ayrımı için
- **Adaptif Eşikleme**: Her hastanın anatomik farklılıklarına uyum
- **Kalite Kontrol**: `CONFIDENCE_SCORE`, `LEAK_FLAG` - güvenilirlik metrikleri

### Ground Truth Validasyon Sistemi
- **Primary config**: Proje kökünde `settings.json`
- **Override hiyerarşisi**: Ortam değişkenleri → settings.json → varsayılanlar
- **Sürekli optimizasyon**: Radyolog ölçümleri ile karşılaştırmalı ayarlama
- İşleme operasyonlarından önce her zaman `load_settings()` kullanın

### Preset Yönetimi
- İşleme modları arasında geçiş yapmak için `scripts/select_preset.py {fast|full|eval}` kullanın
- Çalışma zamanında otomatik yüklenen `site_preset.json` oluşturur
- Farklı presetler hız vs doğruluk vs değerlendirme tutarlılığı için optimize eder

## Geliştirme İş Akışları

### GUI'yi Çalıştırma
```bash
# ÖNEMLI: Uygulama masaüstü kısayolu ile çalıştırılmalı (run komutu sorunlu)
# Geliştirme sırasında sadece test amaçlı:
python run_gui_bootstrap.py
# Veya doğrudan (bazı ortamlarda görüntü sorunları olabilir)
python desktop_project/l3_vfa_pma_gui.py
```

### Doğruluk Odaklı Geliştirme
```bash
# Radyolog ölçümleriyle karşılaştırma (EN ÖNEMLİ)
python desktop_project/optimize_params.py --ref ground_truth.csv --cases-dir DATA_ROOT
# Parametre optimizasyonu bias penalty ile
python optimize_params.py --ref gt.csv --cases-dir DATA --bias-penalty 0.2 --pearson-weight 0.1
```

### Toplu İşleme
```bash
# Tüm hasta dizin yapısını işle
python desktop_project/batch_process.py --input-root DATA_ROOT --out-dir results
# Ground truth karşılaştırması ile belirli dosyaları analiz et
python desktop_project/optimize_params.py --ref ground_truth.csv --cases-dir DATA_ROOT
```

### ML Bileşenini Eğitme
```bash
# Psoas segmentasyon U-Net'ini eğit
python -m psoas_ml.train --csv data_prepped/ground_truth.csv --img_root data/images --mask_root data/masks
```

## Projeye Özgü Desenler

### Hata İşleme
- Tüm işleme fonksiyonları tuple `(results_dict, overlay_image)` döndürür
- DICOM operasyonları etrafında `try/except` kullanın - dosyalar bozuk olabilir
- Segmentasyon fonksiyonlarından `None` dönüşlerini kontrol edin (başarısızlığı gösterir)

### Veri Akışı Kuralları
- **Girdi**: DICOM yolları, hasta metaverisi (boy, cinsiyet)
- **Çıktı**: Metrics dataclass + **yüksek kalite overlay görüntüleri**
- **Hata Ayıklama**: Overlay'leri her zaman `DEBUG_OUT/` veya `qa_out/` kaydedin
- **Görsel QA**: Overlay kalitesi radyolog karşılaştırması için kritik

### Overlay Optimizasyon Teknikleri
```python
# Yüksek kalite overlay üretimi için örnek kod yapısı
def build_high_quality_overlay(hu_array, vfa_mask, pma_mask, vertebra_mask, fascia_contour):
    # 1. HU normalizasyonu (-150 to 250 HU aralığı)
    normalized_hu = normalize_hu_for_display(hu_array, window_low=-150, window_high=250)
    
    # 2. Renk kodlaması
    overlay = create_colored_overlay(
        base_image=normalized_hu,
        vfa_mask=vfa_mask,      # Kırmızı
        pma_mask=pma_mask,      # Mavi  
        vertebra_mask=vertebra_mask,  # Sarı
        fascia_contour=fascia_contour,  # Yeşil
        alpha_blend=0.75
    )
    
    # 3. Anotasyonlar ekle
    overlay = add_measurements_annotation(overlay, vfa_mm2, pma_mm2, confidence_score)
    
    return overlay
```

### Dosya Organizasyonu
- `/desktop_project/`: Ana işleme kodu ve GUI
- `/psoas_ml/`: Sinir ağı eğitimi ve çıkarım
- `/tools/`: Değerlendirme betikleri ve yardımcı programlar
- `/qa_out/`: Kalite güvencesi çıktıları ve toplu sonuçlar
- `/out_eval/`: Tek vakalar için değerlendirme çıktıları

## Radyolog Uyumluluk Kontrol Süreci

### 1. Ground Truth Karşılaştırma İş Akışı
```bash
# Mevcut parametrelerle test
python desktop_project/optimize_params.py --ref ground_truth.csv --cases-dir DATA_ROOT --out baseline_results.json

# En iyi parametreleri bul
python desktop_project/optimize_params.py --ref ground_truth.csv --cases-dir DATA_ROOT --random 100 --bias-penalty 0.15 --pearson-weight 0.1

# Sonuçları analiz et
python desktop_project/metrics_report.py --results-dir results --ref ground_truth.csv --out detailed_report.json
```

### 2. Kalite Kontrol Kriterleri
- **MAE Hedefi**: VFA için <50mm², PMA için <15mm²
- **Bias Limiti**: ±5% içinde kalmalı
- **Korelasyon Hedefi**: r > 0.90 (radyolog ölçümleri ile)
- **Confidence Threshold**: >0.75 güvenlik skoru

### 3. Problemli Vakaları İyileştirme
```bash
# Düşük confidence skorlu vakaları bul
grep '"CONFIDENCE_SCORE": [0-4]' qa_out/*.json

# Leak flag olan vakaları incele  
grep '"LEAK_FLAG": 1' qa_out/*.json

# Vertebra hizalama sorunları
grep '"VB_CENTER_OFFSET_RATIO": [1-9]' qa_out/*.json
```

### 4. Iteratif İyileştirme Döngüsü
1. **Baseline ölçüm** → Mevcut doğruluk seviyesini belirle
2. **Problem analizi** → Hangi vakalarda sapma var?
3. **Parametre ayarlama** → Hedeflenen iyileştirmeler
4. **Validasyon** → Radyolog ölçümleri ile karşılaştır
5. **Döngü** → Hedef doğruluğa ulaşana kadar tekrarla

## Yaygın Görevler

### 5. **Radyolog Uyumluluğunu Artırma Adımları**
- Overlay görsellerini radyolog ölçümleriyle karşılaştır
- Sistematik bias tespiti
- Anatomik varyasyon adaptasyonu
- Edge case yönetimi
- Cross-validation

## Yüksek Kalite Overlay Üretimi

### Overlay Kalite Kriterleri
- **Renk Kodlaması**: VFA (kırmızı), PMA (mavi), Vertebra (sarı), Fasya (yeşil)
- **Şeffaflık**: Alpha blending ile %70-80 şeffaflık
- **Kontrast**: HU normalizasyonu ile net görünüm
- **Anotasyon**: Ölçüm değerleri, confidence skorları
- **Çözünürlük**: Orijinal DICOM çözünürlüğünü koru

### Overlay Geliştirme Komutları
```bash
# Overlay kalitesini test et
python scripts/overlay_quality_check.py --input qa_out/ --ref ground_truth.csv

# Overlay ayarlarını optimize et
python core_mini.py --overlay-alpha 0.75 --overlay-colors vfa:red,pma:blue

# Batch overlay üretimi
python desktop_project/batch_process.py --save-overlays --overlay-high-quality
```

### Kritik Overlay Parametreleri
- `OVERLAY_ALPHA`: Şeffaflık oranı (0.7-0.8 optimal)
- `OVERLAY_NORMALIZE_HU`: HU normalizasyonu (-150 to 250)
- `OVERLAY_SHOW_CONFIDENCE`: Confidence skorunu göster
- `OVERLAY_SHOW_MEASUREMENTS`: VFA/PMA değerlerini göster
- `OVERLAY_VERTEBRA_OUTLINE`: Vertebra konturunu belirginleştir

### Overlay Kalite Kontrolleri
1. **Anatomik Doğruluk**: Segmentasyon sınırları anatomik yapılara uygun mu?
2. **Renk Ayrımı**: Farklı dokular net şekilde ayırt edilebiliyor mu?
3. **Ölçüm Görünürlüğü**: Text overlayler okunabilir mi?
4. **Vertebra Hizalama**: Merkez tespit doğru gösteriliyor mu?
5. **Fasya Sınırları**: İç abdominal duvar net çiziliyor mu?

## Acil Doğruluk Problemi Çözüm Rehberi

### Kritik Doğruluk Sorunları Tespiti
**Problem Belirtileri:**
- PMA ölçümü radyologdan 5-10x düşük
- VFA ölçümü radyologdan 20-50x düşük  
- VB_conf < 0.75 (vertebra güven skoru düşük)
- WARN: vb_center_low_conf

### Acil Müdahale Adımları

#### 1. Vertebra Merkez Tespit Problemini Çöz
```bash
# Vertebra tespit parametrelerini optimize et
python optimize_params.py --param VB_HU_MIN=150,200,250 --param VB_HU_MAX=3000,4000,5000

# Vertebra morfoloji filtrelerini ayarla
python optimize_params.py --param VB_MIN_AREA_MM2=100,150,200 --param VB_MAX_AREA_MM2=2000,3000,4000
```

#### 2. Fasya Segmentasyon Alanını Genişlet
```bash
# Fasya halka bandını artır (daha geniş alan yakalamak için)
python optimize_params.py --param FASCIA_RING_BAND_MM=15,20,25 --param GUARD_BAND_MM=25,30,35

# Snap to muscle edge mesafesini artır
python optimize_params.py --param SNAP_SEARCH_MM=15,20,25
```

#### 3. HU Threshold Aralıklarını Genişlet
```bash
# VFA için daha geniş yağ HU aralığı
python optimize_params.py --param VFA_HU_LOW=-190,-180,-170 --param VFA_HU_HIGH=-30,-20,-10

# Psoas için kas HU aralığını genişlet
python optimize_params.py --param PSOAS_HU_MIN=-10,0,10 --param PSOAS_HU_MAX=80,100,120
```

#### 4. Bu Spesifik Vaka İçin Debug
```bash
# Case ID: 334379490 için detaylı analiz
python scripts/debug_case.py --case-id 334379490 --ref-pma 1055.7 --ref-vfa 13068

# Overlay ile görsel kontrol
python quick_test.py path/to/334379490.dcm M 1.7 --save-debug --show-overlay
```

### Beklenen Değer Aralıkları
- **PMA**: 800-1500 mm² (tipik yetişkin erkek)
- **VFA**: 8000-20000 mm² (obez hasta profili)
- **VB_conf**: >0.75 (güvenilir vertebra tespiti)

### Acil Parametre Override
`settings.json`'a şu değerleri ekle:
```json
{
  "VB_HU_MIN": 150,
  "VB_HU_MAX": 4000,
  "FASCIA_RING_BAND_MM": 20,
  "GUARD_BAND_MM": 30,
  "VFA_HU_LOW": -180,
  "VFA_HU_HIGH": -20,
  "PSOAS_HU_MIN": -10,
  "PSOAS_HU_MAX": 100
}
```

### Yeni Parametreler Ekleme
1. `core_mini.py`'de `CFG` varsayılanlarını güncelle
2. `settings.json` şemasına ekle
3. Tip güvenli erişim için `cfg_float()`, `cfg_range()` yardımcılarını kullan
4. `scripts/smoke_core.py` ile test et

### İşleme Doğruluğunu Artırma
1. Radyolog ölçümleri ile overlay görüntülerini karşılaştır
2. `QA_*.json` dosyalarında kalite metrikleri kontrol et
3. `CONFIDENCE_SCORE` < 0.7 olan vakaları manuel incele
4. `LEAK_FLAG` = 1 olan segmentasyonları gözden geçir
5. `VB_CENTER_OFFSET_RATIO` > 0.15 vertebra hizalama sorunlarını işaretle

### Radyolog Uyumluluğu İçin Kritik Kontroller
- **VFA değerleri**: Tipik aralık 50-500 cm² arası
- **PMA değerleri**: Tipik aralık 10-50 cm² arası  
- **HU threshold optimizasyonu**: Her vaka için adaptif ayarlama
- **Fasya segmentasyonu**: Manuel kontrol gerektiren kritik bölge

## Gelişmiş Optimizasyon Teknikleri

### HU Threshold Ayarlama Rehberi
```bash
# Adaptif yağ HU bandı optimizasyonu
# VFA_HU_LOW: -190 ile -120 arası test et
# VFA_HU_HIGH: -50 ile -20 arası test et
python optimize_params.py --param VFA_HU_LOW=-150,-140,-130 --param VFA_HU_HIGH=-50,-40,-30
```

**Kritik HU Parametreleri:**
- `VFA_HU_LOW`: Yağ dokusunun alt sınırı (tipik: -150 HU)
- `VFA_HU_HIGH`: Yağ dokusunun üst sınırı (tipik: -50 HU)
- `ADAPT_FAT_ALPHA`: Otsu vs percentil karışım oranı (0.3-0.7)
- `ADAPT_FAT_P_LOW/HIGH`: Percentil tabanlı adaptif eşikleme

### Fasya Segmentasyon Optimizasyonu
```bash
# Fasya sıkılaştırma parametreleri
python optimize_params.py --param FASCIA_STRICT=true,false --param FASCIA_STRICT_CLOSE_MM=4,6,8
```

**Fasya Kritik Parametreleri:**
- `FASCIA_STRICT`: Morfolojik sıkılaştırma (true önerilir)
- `FASCIA_RING_BAND_MM`: Fasya halka bandı genişliği (8-12mm)
- `SNAP_SEARCH_MM`: Kas kenarına yaslanma mesafesi (6-12mm)
- `GUARD_BAND_MM`: Koruma bandı genişliği (15-20mm)

### Psoas Kas Segmentasyon İyileştirme
```bash
# Psoas parametrelerini optimize et
python optimize_params.py --param PSOAS_MIN_AREA_MM2=150,200,250 --param PSOAS_SCORE_W_DIST=0.5,0.6,0.7
```

**Psoas Kritik Parametreleri:**
- `PSOAS_MIN_AREA_MM2`: Minimum psoas alanı (150-300mm²)
- `PSOAS_MAX_AREA_MM2`: Maksimum psoas alanı (2500-3500mm²)
- `PSOAS_SCORE_W_DIST`: Mesafe ağırlığı (0.4-0.8)
- `PSOAS_SCORE_W_CONV`: Konvekslik ağırlığı (0.2-0.6)

### Doğruluk Optimizasyonu
```bash
# Ground truth'a karşı parametre uzayını ara (EN ÖNEMLİ İŞ AKIŞI)
python optimize_params.py --ref gt.csv --cases-dir DATA --random 50 --bias-penalty 0.2
# Radyolog ölçümleriyle uyumluluk kontrolü
python scripts/validate_folder.py --input DATA --labels ground_truth.csv
# Ayarlar konfigürasyonunu doğrula
python settings_validate.py --settings settings.json
```

### Klinik Doğruluk Metrikleri
1. **MAE (Mean Absolute Error)**: Radyolog ölçümleri ile fark
2. **Bias kontrolü**: Sistematik sapma tespiti ve düzeltmesi
3. **Pearson korelasyonu**: Ölçümler arası doğrusal ilişki
4. **Confidence scoring**: Güvenilirlik değerlendirmesi

## Entegrasyon Noktaları

- **DICOM Kütüphaneleri**: Tıbbi görüntü I/O için `pydicom` kullanır
- **Görüntü İşleme**: Bilgisayarlı görü operasyonları için OpenCV (`cv2`)
- **ML Framework**: Tıbbi görüntüleme ML için PyTorch Lightning + MONAI
- **GUI Framework**: Masaüstü arayüzü için Tkinter
- **Çıktı Formatları**: Sonuçlar için JSON, overlay'ler için PNG, toplu analiz için CSV

## Test Stratejisi

- **Smoke Testler**: Hızlı doğrulama için `make smoke-{fast|full|eval}`
- **Determinizm**: Tekrarlanabilirlik kontrolleri için `scripts/determinism_smoke.py`
- **Görsel QA**: Segmentasyon kalitesi için overlay çıktılarını her zaman incele
- **Ground Truth**: Doğruluk değerlendirmesi için `scripts/validate_folder.py` kullan
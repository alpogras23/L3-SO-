# Comp2Comp-Enhanced L3 VFA/PMA Calculator

Bu proje, Comp2Comp'un gelişmiş segmentation algoritmalarını kullanarak L3 vertebra seviyesinde Visceral Fat Area (VFA) ve Psoas Muscle Area (PMA) hesaplaması yapan bir araçtır.

## Özellikler

- **Comp2Comp-İlhamlı Segmentation**: Stanford'un Comp2Comp kütüphanesinden esinlenilen gelişmiş doku segmentasyonu
- **Çoklu Format Desteği**: NIfTI (.nii.gz) ve DICOM (.dcm) dosyalarını destekler
- **Gelişmiş HU Thresholding**: Kas (-10 to 100 HU) ve yağ (-180 to -20 HU) için optimize edilmiş eşikleme
- **Morfolojik İşlemler**: Bağlantılı bileşen filtreleme ve morfolojik temizleme
- **Fasya Sınır Tespiti**: Distance transform tabanlı gelişmiş sınır tespiti
- **Görsel Çıktılar**: Renk kodlu segmentasyon görselleri
- **Batch İşleme**: Çoklu dosya işleme desteği

## Kurulum

```bash
# Gereksinimleri yükleyin
pip install -r requirements.txt

# Veya spesifik paketleri yükleyin
pip install nibabel pydicom numpy scipy scikit-image opencv-python
```

## Kullanım

### Tek Dosya İşleme

```bash
# DICOM dosyası için
python comp2comp_enhanced_l3.py --input path/to/file.dcm --output ./results

# NIfTI dosyası için
python comp2comp_enhanced_l3.py --input path/to/file.nii.gz --output ./results
```

### Batch İşleme

```bash
# Klasördeki tüm dosyaları işle
python comp2comp_enhanced_l3.py --input /path/to/input_directory --output ./results --batch
```

## Çıktılar

Her vaka için aşağıdaki dosyalar oluşturulur:

- `*_results.json`: Ölçüm sonuçları (PMA, VFA, SAT, toplam yağ)
- `*_l3_vfa_pma.png`: Renk kodlu segmentasyon görseli
- `*_masks.nii.gz`: Segmentasyon maskeleri (5 katmanlı NIfTI)

### Renk Kodlaması

- **Kırmızı**: Kas dokusu
- **Yeşil**: Yağ dokusu
- **Mavi**: Psoas kasları
- **Sarı**: Vertebra
- **Magenta**: Fasya sınırı

## Algoritma Detayları

### 1. Dokü Segmentasyonu
- **Kas**: HU -10 to 100 aralığında
- **Yağ**: HU -180 to -20 aralığında
- Morfolojik açma/kapama işlemleri ile gürültü temizleme
- Bağlantılı bileşen filtreleme ile küçük bölgelerin çıkarılması

### 2. Vertebra Tespiti
- HU 200-1500 aralığında kemik dokusu tespiti
- En büyük bağlantılı bileşen olarak vertebra seçimi

### 3. Psoas Kas Çıkarımı
- Vertebra merkezine göre sol/sağ ayrımı
- Her yarıda en büyük kas bölgesi olarak psoas tespiti

### 4. Fasya Sınır Hesaplama
- Kas ve yağ maskeleri arasındaki geçiş bölgesi
- Distance transform ile sınır belirleme
- Morfolojik işlemler ile sınır temizleme

### 5. Alan Hesaplaması
- Piksel sayımı ve piksel alanı çarpımı
- VFA: Fasya sınırı içindeki yağ
- PMA: Sol + sağ psoas alanı
- SAT: Fasya dışı yağ (subcutaneous)

## Yapılandırma

`settings.json` dosyasında parametreleri özelleştirebilirsiniz:

```json
{
  "PSOAS_HU_MIN": -10,
  "PSOAS_HU_MAX": 100,
  "VFA_HU_LOW": -180,
  "VFA_HU_HIGH": -20
}
```

## Test

```bash
# Mevcut test verisi ile
python comp2comp_enhanced_l3.py --input comp2comp_test_input/CT_small.dcm --output ./test_results
```

## Sonuç Örneği

```
📊 Results:
PMA: 2160.13 mm²
VFA: 254.65 mm²
SAT: 595.93 mm²
Total Fat: 850.58 mm²
```

## Teknik Detaylar

- **Dil**: Python 3.11+
- **Kütüphaneler**: nibabel, pydicom, numpy, scipy, scikit-image, opencv
- **Platform**: macOS, Linux, Windows
- **Bellek**: Tipik vaka için ~100MB
- **İşlem Süresi**: Tek vaka için ~5-10 saniye

## Geliştirme

Bu kod, Comp2Comp segmentation pipeline'ından esinlenilerek geliştirilmiştir. Ana iyileştirmeler:

1. **Adaptive Thresholding**: HU değerlerine göre dinamik eşikleme
2. **Morphological Refinement**: Açma/kapama işlemleri ile segmentasyon iyileştirme
3. **Connected Components**: Küçük bölgelerin filtrelenmesi
4. **Boundary Detection**: Distance transform tabanlı sınır tespiti

## Lisans

Bu proje açık kaynak kodludur ve akademik kullanım için uygundur.
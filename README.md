# L3 SO Analyzer (Python)

Bu proje, L3 seviyesinde viseral yağ (VFA) ve psoas kas alanı (PMA) ölçümlerini basit ve kararlı bir "mini çekirdek" (core_mini.py) üzerinden yapar. GUI (`l3_vfa_pma_gui.py`) dinamik olarak bu çekirdeği yükler.

## Hızlı Başlangıç

1) Ortamı kur

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2) Hızlı test çalıştır

```bash
# Örnek kullanım: quick_test <dicom> <Sex: M|F> <Height_m>
python quick_test.py /path/to/L3_slice.dcm M 1.70
```

3) GUI’yi çalıştır

```bash
# Doğrudan .py üzerinden çalıştırma (önerilir)
python -X dev -u l3_vfa_pma_gui.py
```

> Not: macOS’ta bir .app paketi mevcut olabilir; fakat geliştirme sırasında .py dosyasını kullanmanız önerilir.

## Çekirdek Mimari: core_mini.py

- Tek dosyalık kararlı motor; `process_one` iki giriş biçimini destekler:
  - (a) DICOM yolu: `process_one(dicom_path, sex, height_m)`
  - (b) HU numpy dizisi + DICOM dataset: `process_one(hu_array, ds, sex, height_m)`
- Çıktı, ölçümler ve bayraklar içeren bir sözlük ve opsiyonel bir overlay görüntüsüdür.
- `settings.json` içindeki değerler (eşikler, parametreler) `load_settings()` ve yardımcıları (`cfg_float`, `cfg_range`, `cfg_dict`) ile okunur.

## VS Code Kolaylıkları

- Görevler
  - Format (black)
  - Sort imports (isort)
  - Lint (pylint quick)
- Debug/Çalıştır
  - Run GUI (mini)
  - Quick Test (sample DICOM)

## Sorun Giderme

- OpenCV hataları: `cv2.connectedComponentsWithStats` için `connectivity=8`, morfolojik işlemler için `iterations=` kullanın; `floodFill` maskesi `h+2, w+2` boyutunda olmalıdır.
- DICOM okuma: HU dönüşümü için `RescaleSlope` ve `RescaleIntercept` kontrol edilir; eksikse varsayılan 1 ve 0 kullanılır.
- Görüntüleme: Overlay görüntüsü BGR’dir; `cv2.imshow` kullanırken renk beklentisini dikkate alın.
- Lint/Format: Aşağıdaki komut sırası çoğu durumda yeterlidir:

```bash
ruff check . --fix
isort .
black .
```

## Bakım: Yedek Dosyaları Arşivleme

Dağınık .bak/.save yedeklerini güvenle arşivlemek için:

```bash
# Sadece ne yapacağını gösterir
python scripts/archive_backups.py --dry-run

# Gerçekten arşivler (_archive_all_copies/<tarih-saat>/ altına taşır)
python scripts/archive_backups.py
```

## Lisans

Bu depo, dahili/araştırma amaçlı kullanım için hazırlanmıştır.

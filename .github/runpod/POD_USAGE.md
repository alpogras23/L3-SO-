# RunPod Pod Bağlantı ve Kullanım Rehberi

## Pod Bilgileri
- **Pod ID**: 7kp6lp145lkj08
- **GPU**: 1x RTX 4090 (10 vCPU, 83GB RAM)
- **Maliyet**: $0.34/saat (~$2-3 toplam 240 vaka için)
- **Durum**: RUNNING ✅

## SSH Bağlantısı

1. **RunPod Console'a git**: https://www.runpod.io/console/pods
2. Pod **"l3-teacher-gen"** satırında **"Connect"** butonuna tıkla
3. SSH bilgilerini not et:
   ```bash
   ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
   # Veya şifre ile:
   ssh root@<pod-ip> -p <port>
   # Şifre: <password>
   ```

## Kurulum Adımları

### 1. Pod'a Bağlan ve Ortamı Hazırla
```bash
# SSH ile bağlan
ssh root@<pod-ip> -p <port>

# Setup scriptini çalıştır (eğer upload ettiysek)
bash /workspace/setup_pod.sh

# Veya manuel kurulum:
pip install --upgrade pip
pip install totalsegmentator nibabel SimpleITK pydicom opencv-python-headless \
    scikit-image scipy numpy tqdm matplotlib monai

# GPU kontrolü
nvidia-smi

# Dizinleri oluştur
mkdir -p /workspace/amos22 /workspace/teacher_labels
```

### 2. Dosyaları Upload Et

**Seçenek A: runpodctl ile (Önerilen)**
```bash
# Local makineden:
cd ~/Desktop/L3_SO_ANALYSIS

# Setup scriptini upload et
runpodctl send .github/runpod/setup_pod.sh 7kp6lp145lkj08:/workspace/setup_pod.sh

# Python scriptini upload et
runpodctl send .github/azure-ml/step1_real_teachers_all.py 7kp6lp145lkj08:/workspace/step1_real_teachers_all.py

# AMOS22 verilerini upload et (BÜYÜK DOSYA - zaman alacak!)
runpodctl send /path/to/amos22/ 7kp6lp145lkj08:/workspace/amos22/
```

**Seçenek B: rsync ile**
```bash
# SSH bilgilerini al, sonra:
rsync -avz --progress -e "ssh -p <port>" \
    /path/to/amos22/ root@<pod-ip>:/workspace/amos22/

rsync -avz -e "ssh -p <port>" \
    .github/azure-ml/step1_real_teachers_all.py \
    root@<pod-ip>:/workspace/
```

**Seçenek C: Web Terminal ile (Küçük dosyalar için)**
RunPod Console'da "Connect" → "Start Web Terminal" → dosyaları kopyala-yapıştır

### 3. Teacher Generation'ı Çalıştır

```bash
# Pod'a SSH ile bağlan
ssh root@<pod-ip> -p <port>

# Script parametrelerini kontrol et
cd /workspace
python step1_real_teachers_all.py --help

# AMOS22 verilerinin yüklü olduğunu doğrula
ls -lh amos22/ | head -20
echo "Toplam DICOM dosyası: $(find amos22/ -name "*.dcm" -o -name "*.nii.gz" | wc -l)"

# Teacher generation'ı başlat (UZUN SÜREÇ - 4-6 saat)
python step1_real_teachers_all.py \
    --input-dir /workspace/amos22 \
    --output-dir /workspace/teacher_labels \
    --num-workers 4 \
    --batch-size 1 \
    2>&1 | tee teacher_generation.log

# Veya screen/tmux ile (SSH bağlantısı kapansa da çalışmaya devam eder):
screen -S teacher
python step1_real_teachers_all.py \
    --input-dir /workspace/amos22 \
    --output-dir /workspace/teacher_labels \
    --num-workers 4 \
    --batch-size 1 \
    2>&1 | tee teacher_generation.log
# Detach: Ctrl+A, D
# Re-attach: screen -r teacher
```

### 4. İlerlemeyi İzle

```bash
# Başka bir terminal'den bağlan veya screen detach yap
ssh root@<pod-ip> -p <port>

# Çıktı dosyalarını say
watch -n 30 'ls teacher_labels/ | wc -l'

# GPU kullanımını izle
watch -n 5 nvidia-smi

# Log'u takip et
tail -f teacher_generation.log

# Disk kullanımını kontrol et
df -h
```

### 5. Sonuçları İndir

```bash
# Local makineden:
cd ~/Desktop/L3_SO_ANALYSIS

# runpodctl ile
runpodctl receive 7kp6lp145lkj08:/workspace/teacher_labels/ ./amos22_teachers/

# rsync ile
rsync -avz --progress -e "ssh -p <port>" \
    root@<pod-ip>:/workspace/teacher_labels/ \
    ./amos22_teachers/

# Sonuçları doğrula
ls -lh amos22_teachers/
echo "Toplam teacher label: $(ls amos22_teachers/*.nii.gz 2>/dev/null | wc -l)"
```

### 6. Pod'u Durdur (Önemli!)

```bash
# İş bitince pod'u durdur (maliyet kesmeye devam eder!)
runpodctl stop pod 7kp6lp145lkj08

# Veya tamamen sil
runpodctl remove pod 7kp6lp145lkj08
```

## Beklenen Sonuçlar
- **240 AMOS22 vakası** → **240 teacher segmentation label** (.nii.gz formatında)
- **Süre**: 4-6 saat (RTX 4090'da ~1-2 dakika/vaka)
- **Maliyet**: $1.36 - $2.04 (4-6 saat x $0.34/saat)
- **Çıktı**: `teacher_labels/amos_0001.nii.gz`, `teacher_labels/amos_0002.nii.gz`, ...

## Sorun Giderme

### CUDA Out of Memory
```bash
# Batch size'ı küçült
python step1_real_teachers_all.py --batch-size 1 --num-workers 2
```

### Disk Doluyor
```bash
# Scratch temizle
rm -rf /tmp/scratch/*
rm -rf /tmp/totalseg*
```

### Script Hataları
```bash
# Python ve bağımlılıkları kontrol et
python -c "import totalsegmentator; print(totalsegmentator.__version__)"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Yardımcı Komutlar

```bash
# Pod durumunu kontrol
runpodctl get pod 7kp6lp145lkj08

# Pod'u durdur/başlat
runpodctl stop pod 7kp6lp145lkj08
runpodctl start pod 7kp6lp145lkj08

# Pod loglarını izle
runpodctl logs 7kp6lp145lkj08

# Pod'u sil
runpodctl remove pod 7kp6lp145lkj08
```

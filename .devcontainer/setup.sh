#!/bin/bash
set -e

echo "🚀 L3 SO Analysis DevContainer Kurulumu Başlatılıyor..."

# Paket yöneticisi güncellemesi
pip install -U pip setuptools wheel

# PyTorch CUDA 12.1
echo "📦 PyTorch CUDA kurulumu..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Tıbbi görüntüleme kütüphaneleri
echo "🏥 Tıbbi görüntüleme kütüphaneleri..."
pip install monai pytorch-lightning==2.4.* SimpleITK nibabel pydicom scikit-image

# TotalSegmentator + nnUNetv2
echo "🧠 TotalSegmentator ve nnUNet..."
pip install TotalSegmentator==1.7.4 nnunetv2==2.3.1

# CV ve veri bilimi
echo "📊 Görüntü işleme ve analiz..."
pip install opencv-python-headless tqdm rich seaborn pandas matplotlib

# Plastimatch (DICOM dönüşümü için)
echo "🔧 Plastimatch (opsiyonel)..."
sudo apt-get update -qq
sudo apt-get install -y plastimatch || echo "⚠️  Plastimatch kurulamadı (SimpleITK fallback kullanılacak)"

# Jupyter kurulumu
echo "📓 Jupyter..."
pip install jupyter jupyterlab ipywidgets

# GPU kontrolü
echo "🎮 GPU Kontrolü:"
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Count:', torch.cuda.device_count())"

echo "✅ Kurulum tamamlandı!"
echo "📖 Notebook'u açmak için: notebooks/AMOS22_L3_TS_C2C_VFA_PMA_tek_tik.ipynb"

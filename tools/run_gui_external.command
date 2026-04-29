#!/bin/zsh
# Bu script, nereden çağrılırsa çağrılsın, kendi konumuna göre repo köküne cd eder
SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1

# Varsa sanal ortamı aktifleştir
source .venv/bin/activate 2>/dev/null || true

# GUI çalıştırma için ortam değişkenleri
export TK_SILENCE_DEPRECATION=1 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Ayar yolunu workspace'e sabitle
export SETTINGS_PATH="$REPO_ROOT/settings.json"
# Başlığa bir etiket ekleyerek kaynak workspace olduğunu göster
export L3_GUI_TITLE_TAG="WS"

# GUI bootstrap üzerinden başlat (platform bağımsız ve proje yapısına uygun)
python run_gui_bootstrap.py

read -n 1 -s -r -p "Kapatmak için tuşa bas..."

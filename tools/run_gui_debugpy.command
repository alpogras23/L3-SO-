#!/bin/zsh
# Bu script, nereden çağrılırsa çağrılsın, kendi konumuna göre repo köküne cd eder
SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1

# Varsa sanal ortamı aktifleştir
source .venv/bin/activate 2>/dev/null || true

export TK_SILENCE_DEPRECATION=1 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Zorunlu çekirdek ve ayar yollarını workspace'e sabitle
export FORCE_CORE_PATH="$REPO_ROOT/desktop_project/l3_vfa_pma_core_v3_7_FIXED.py"
export SETTINGS_PATH="$REPO_ROOT/desktop_project/settings.json"
export L3_GUI_TITLE_TAG="WS-DBG"

# Offline debugpy varsa kullan; yoksa normal çalıştır.
python - <<'PY'
import importlib.util, sys
sys.exit(0) if importlib.util.find_spec("debugpy") else sys.exit(1)
PY
EC=$?
if [ $EC -ne 0 ]; then
  WHEEL=$(ls tools/wheels/debugpy*.whl 2>/dev/null | head -n1)
  if [ -n "$WHEEL" ]; then
    echo "📦 debugpy offline kurulum: $WHEEL"
    pip install "$WHEEL" || true
  else
    echo "⚠️ debugpy bulunamadı (tools/wheels/ altına .whl koyarsan otomatik kurulur)."
  fi
fi

# Tekrar kontrol:
python - <<'PY'
import importlib.util, sys
sys.exit(0) if importlib.util.find_spec("debugpy") else sys.exit(1)
PY
EC=$?
if [ $EC -eq 0 ]; then
  echo "✅ debugpy hazır → 5678 portunda bekleniyor (VS Code: Attach GUI)"
  python -X faulthandler -X tracemalloc=5 -m debugpy --listen 5678 --wait-for-client run_gui_bootstrap.py
else
  echo "ℹ️ debugpy yok → normal başlatılıyor"
  python run_gui_bootstrap.py
fi

read -n 1 -s -r -p "Kapatmak için tuşa bas..."

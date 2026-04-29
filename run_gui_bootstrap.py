import os
import runpy
import sys
from pathlib import Path

# Clear variables that can break Tk when launched from VS Code/Electron
for key in (
    "OBJC_DISABLE_INITIALIZE_FORK_SAFETY",
    "DYLD_LIBRARY_PATH",
    "DYLD_FRAMEWORK_PATH",
    "QT_API",
    "QT_PLUGIN_PATH",
):
    os.environ.pop(key, None)

# Silence noisy Tk warnings and avoid buffered stdout/stderr
os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# Keep BLAS thread usage in check when running inside VS Code
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import tkinter as tk

    root = tk.Tk()
    root.destroy()
except Exception as exc:
    print(
        "🚫 Tk GUI başlatılamadı (muhtemelen VS Code altında headless):",
        type(exc).__name__,
        str(exc)[:200],
        flush=True,
    )
    print(
        "ℹ️ Lütfen Terminal.app üzerinden çalıştırın veya VS Code 'console': 'externalTerminal' kullanın.",
        flush=True,
    )
    sys.exit(42)

repo_root = Path(__file__).resolve().parent
# Eğer VS Code debug ile doğrudan çağrıldıysa ve env değişkenleri yoksa, workspace yollarını varsayılan olarak ata
os.environ.setdefault("SETTINGS_PATH", str(repo_root / "settings.json"))
os.environ.setdefault("L3_GUI_TITLE_TAG", "WS")

print(f"[BOOT] repo_root={repo_root}")
print(f"[BOOT] SETTINGS_PATH={os.environ.get('SETTINGS_PATH')}")

# GUI dosyası: Önce proje kökünde, yoksa desktop_project'te ara
gui_path = repo_root / "vfa_pma_gui.py"
if not gui_path.exists():
    gui_path = repo_root / "desktop_project" / "l3_vfa_pma_gui.py"
print(f"[BOOT] gui_path={gui_path}")

if not gui_path.exists():
    print("🚫 GUI betiği bulunamadı:", gui_path, flush=True)
    sys.exit(43)

gui_dir = gui_path.parent
print(f"[BOOT] cwd_before={Path.cwd()}")
os.chdir(gui_dir)
print(f"[BOOT] cwd_after={Path.cwd()}")
sys.argv = ["l3_vfa_pma_gui.py"]
runpy.run_path(str(gui_path), run_name="__main__")

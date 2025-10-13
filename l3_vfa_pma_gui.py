import importlib
import importlib.util
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# OpenCV opsiyonel
try:
    import cv2
except Exception:
    cv2 = None

APP_TITLE = "L3 VFA / PMA / PMI – Mini GUI"
WINDOW_W, WINDOW_H = 900, 650


def _force_load_core():
    """
    Çekirdeği benzersiz modül adıyla (cache bypass) dinamik yükler.
    Öncelik sırası: v3_7 -> core (eski)
    __pycache__ dikkate alınmaz; her çağrıda yeni modül adı kullanılır.
    """
    candidates = [
        ("core_mini.py", "mini"),
        ("l3_vfa_pma_core_v3_7.py", "v3_7"),
        ("l3_vfa_pma_core.py", "legacy"),
    ]
    root = Path(__file__).resolve().parent
    chosen = None
    for fname, tag in candidates:
        p = root / fname
        if p.exists():
            chosen = (p, tag)
            break
    if not chosen:
        raise FileNotFoundError("Çekirdek dosyası bulunamadı.")

    path, tag = chosen
    modname = f"core_live_{tag}_{int(time.time()*1000)}"

    # spec oluştur ve yükle
    spec = importlib.util.spec_from_file_location(modname, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Spec oluşturulamadı: {path}")
    mod = importlib.util.module_from_spec(spec)
    try:
        # dataclass ve benzeri decorator'lar için __module__ kaydı gerekli
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    except Exception as e:
        raise ImportError(f"Çekirdek yüklenemedi: {path}: {e}")

    print(f"[INFO] forced core load: {path} as {modname}")
    return mod


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry(f"{WINDOW_W}x{WINDOW_H}")
        self.resizable(False, False)

        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="x")

        # DICOM
        ttk.Label(frm, text="DICOM (.dcm):").grid(row=0, column=0, sticky="w")
        self.dcm_var = tk.StringVar(value="")
        ttk.Entry(frm, textvariable=self.dcm_var, width=64).grid(
            row=0, column=1, padx=6, pady=4, sticky="w"
        )
        ttk.Button(frm, text="Seç...", command=self.pick_dcm).grid(
            row=0, column=2, padx=4
        )

        # PNG (opsiyonel)
        ttk.Label(frm, text="PNG (aynı kesit, opsiyonel):").grid(
            row=1, column=0, sticky="w"
        )
        self.png_var = tk.StringVar(value="")
        ttk.Entry(frm, textvariable=self.png_var, width=64).grid(
            row=1, column=1, padx=6, pady=4, sticky="w"
        )
        ttk.Button(frm, text="Seç...", command=self.pick_png).grid(
            row=1, column=2, padx=4
        )

        # PNG kullan (kapatılırsa DICOM-only)
        self.use_png_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            frm, text="PNG kullan (kapat: DICOM-only)", variable=self.use_png_var
        ).grid(row=2, column=1, sticky="w")

        # Cinsiyet
        ttk.Label(frm, text="Cinsiyet:").grid(row=3, column=0, sticky="w")
        self.sex_var = tk.StringVar(value="M")
        ttk.Combobox(
            frm, textvariable=self.sex_var, values=["M", "F"], state="readonly", width=5
        ).grid(row=3, column=1, sticky="w")

        # Boy
        ttk.Label(frm, text="Boy (cm):").grid(row=4, column=0, sticky="w")
        self.h_var = tk.StringVar(value="170")
        ttk.Entry(frm, textvariable=self.h_var, width=10).grid(
            row=4, column=1, sticky="w"
        )

        # Komşu dilim füzyonu
        self.fuse_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            frm, text="Çok-dilim füzyon (L3±1)", variable=self.fuse_var
        ).grid(row=5, column=1, sticky="w")

        # Butonlar
        rowb = ttk.Frame(self)
        rowb.pack(fill="x", padx=10, pady=(6, 2))
        ttk.Button(rowb, text="Analiz et", command=self._run).pack(side="left")

        ttk.Label(self, text="Sonuçlar:").pack(anchor="w", padx=10)
        self.txt = tk.Text(self, height=18, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=10, pady=6)

        ttk.Label(
            self,
            text="Not: Çekirdek her RUN’da taze yüklenir; ayarlar anında sonuçlara yansır.",
        ).pack(anchor="w", padx=10, pady=(0, 8))

        self._last = None
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def pick_dcm(self):
        f = filedialog.askopenfilename(
            title="DICOM seç", filetypes=[("DICOM", "*.dcm"), ("Tümü", "*.*")]
        )
        if f:
            self.dcm_var.set(f)

    def pick_png(self):
        f = filedialog.askopenfilename(
            title="PNG seç", filetypes=[("PNG", "*.png"), ("Tümü", "*.*")]
        )
        if f:
            self.png_var.set(f)

    def show_res(self, res: dict):
        self.txt.delete("1.0", "end")

        vfa_key = f"VFA≥{int(res.get('VFA_OBESITY_CUTOFF', 100)) if 'VFA_OBESITY_CUTOFF' in res else '100'}cm2"
        ratio_val = res.get("VFA_PMA_ratio", "")
        pmi_val = res.get("PMI_cm2_per_m2", "")
        ratio_str = "NA" if ratio_val in ("", None) else f"{ratio_val:.2f}"
        pmi_str = "NA" if pmi_val in ("", None) else f"{pmi_val:.2f}"

        def w(s=""):
            self.txt.insert("end", s + "\n")

        w("=== RESULT @ MINI GUI ===")
        w(
            f"ID: {res.get('ID','')} | Sex: {res.get('Sex','')} | Height(m): {res.get('Height_m','')}"
        )
        w(
            f"PMA: {res.get('PMA_mm2',0):.1f} mm²   VFA: {res.get('VFA_mm2',0):.1f} mm² ({res.get('VFA_cm2',0):.1f} cm²)"
        )
        w(f"VFA/PMA: {ratio_str}   PMI: {pmi_str} cm²/m²")
        w(
            f"Sarcopenia(PMI<cutoff): {res.get('Sarcopenia(PMI<cutoff)',0)}  {vfa_key}: {res.get(vfa_key, res.get('Obesity_VFA_based',0))}  SO: {res.get('SO_Flag(1=yes)',0)}"
        )

    def _run(self):
        try:
            core = _force_load_core()

            dcm_str = self.dcm_var.get().strip()
            png_str = self.png_var.get().strip()
            if not dcm_str:
                messagebox.showerror("Hata", "Geçerli DICOM dosyası seçin.")
                return

            dcm_path = Path(dcm_str)
            png_path = Path(png_str) if (png_str and self.use_png_var.get()) else None
            sex = (self.sex_var.get() or "M").upper()

            try:
                h_cm = float(self.h_var.get().replace(",", "."))
            except Exception:
                messagebox.showerror("Hata", "Boy (cm) sayısal olmalı.")
                return

            use_png = bool(self.use_png_var.get())
            fuse_neighbors = bool(self.fuse_var.get())

            # Çekirdeği çağır
            try:
                out = core.process_one(
                    dcm_path,
                    png_path,
                    h_cm / 100.0,
                    sex,
                    use_png=use_png,
                    fuse_neighbors=fuse_neighbors,
                )
            except TypeError:
                out = core.process_one(dcm_path, png_path, h_cm / 100.0, sex)

            # (res, overlay) bekliyoruz
            if isinstance(out, tuple) and len(out) >= 2:
                res, overlay = out[0], out[1]
            else:
                messagebox.showerror("Hata", "Çekirdekten beklenmeyen çıktı.")
                return

            # Yazdır
            self.show_res(res)

            # Overlay gösterimi (varsa ve OpenCV yüklüyse)
            try:
                if overlay is not None and cv2 is not None:
                    vis = cv2.resize(
                        overlay,
                        (int(overlay.shape[1] * 0.85), int(overlay.shape[0] * 0.85)),
                    )
                    cv2.imshow(f"{res.get('ID','overlay')}", vis)
                    cv2.waitKey(1)
            except Exception:
                pass

        except Exception as e:
            messagebox.showerror("Hata", f"İşleme hatası: {e}")

    def on_close(self):
        try:
            if cv2 is not None:
                cv2.destroyAllWindows()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()

#!/usr/bin/env python3
"""
L3 VFA/PMA Hesaplama GUI - ML Tabanlı
Tek DICOM dosyasından VFA ve PMA hesaplar.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from pathlib import Path
import threading
import sys
import os

# Proje kökünü path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut
except ImportError:
    messagebox.showerror("Hata", "pydicom kurulu değil!\npip install pydicom")
    sys.exit(1)

try:
    import cv2
    from PIL import Image, ImageTk
except ImportError:
    messagebox.showerror("Hata", "opencv-python ve pillow kurulu değil!")
    sys.exit(1)

# SADECE AMOS U-Net modelini kullan (hybrid değil)
try:
    from psoas_ml.ml_inference import VFAPMAPredictor, create_overlay
    PREDICTOR_TYPE = 'amos_unet'
    ML_AVAILABLE = True
    print("AMOS U-Net modeli yükleniyor...")
except ImportError as e:
    print(f"ML modülü yüklenemedi: {e}")
    ML_AVAILABLE = False
    PREDICTOR_TYPE = None


class VFAPMACalculator:
    """DICOM'dan VFA/PMA hesaplayan ana sınıf"""
    
    def __init__(self):
        self.predictor = None
        self.load_model()
    
    def load_model(self):
        """AMOS U-Net Predictor'ı yükle"""
        if not ML_AVAILABLE:
            print("ML modülü mevcut değil, rule-based mod kullanılacak")
            return
        
        try:
            self.predictor = VFAPMAPredictor()
            print(f"Predictor tipi: {PREDICTOR_TYPE}")
        except Exception as e:
            print(f"Predictor yüklenemedi: {e}, rule-based mod kullanılacak")
            self.predictor = None
    
    def load_dicom(self, dicom_path):
        """DICOM dosyasını yükle ve HU değerlerine çevir"""
        ds = pydicom.dcmread(dicom_path)
        
        # Pixel array al
        pixel_array = ds.pixel_array.astype(np.float32)
        
        # HU'ya çevir
        hu_array = apply_modality_lut(pixel_array, ds)
        
        # Pixel spacing
        if hasattr(ds, 'PixelSpacing'):
            ps = ds.PixelSpacing
            pixel_spacing = (float(ps[0]), float(ps[1]))
        else:
            pixel_spacing = (1.0, 1.0)
        
        return hu_array, pixel_spacing, ds
    
    def calculate(self, dicom_path):
        """VFA ve PMA hesapla"""
        # DICOM yükle
        hu_array, pixel_spacing, ds = self.load_dicom(dicom_path)
        
        if self.predictor is not None:
            # ML-based hesaplama
            results = self.predictor.predict(hu_array, pixel_spacing)
            overlay = create_overlay(hu_array, results['masks'])
        else:
            # Rule-based fallback
            results = self._rule_based_calculate(hu_array, pixel_spacing)
            overlay = self._create_basic_overlay(hu_array)
        
        # Metadata ekle
        results['patient_id'] = getattr(ds, 'PatientID', 'Unknown')
        results['study_date'] = getattr(ds, 'StudyDate', 'Unknown')
        results['pixel_spacing'] = pixel_spacing
        
        return results, overlay
    
    def _rule_based_calculate(self, hu_array, pixel_spacing):
        """Basit rule-based hesaplama (ML yoksa fallback)"""
        # VFA: -190 to -30 HU
        vfa_mask = (hu_array >= -190) & (hu_array <= -30)
        
        # Basit PMA tahmini: vertebra çevresinde kas dokusu
        pma_mask = (hu_array >= -29) & (hu_array <= 150)
        
        pixel_area = pixel_spacing[0] * pixel_spacing[1]
        
        return {
            'vfa_mm2': np.sum(vfa_mask) * pixel_area,
            'vfa_cm2': np.sum(vfa_mask) * pixel_area / 100,
            'pma_total_mm2': np.sum(pma_mask) * pixel_area * 0.1,  # Tahmini
            'pma_cm2': np.sum(pma_mask) * pixel_area * 0.1 / 100,
            'masks': {'vfa': vfa_mask.astype(np.uint8)},
            'confidence': 0.3  # Düşük güven - rule-based
        }
    
    def _create_basic_overlay(self, hu_array):
        """Basit grayscale overlay"""
        hu_norm = np.clip(hu_array, -150, 250)
        hu_norm = ((hu_norm + 150) / 400 * 255).astype(np.uint8)
        return cv2.cvtColor(hu_norm, cv2.COLOR_GRAY2RGB)


class VFAPMAGUI:
    """Tkinter GUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("L3 VFA/PMA Hesaplama - ML Tabanlı")
        self.root.geometry("1000x700")
        
        self.calculator = VFAPMACalculator()
        self.current_overlay = None
        
        self.setup_ui()
    
    def setup_ui(self):
        # Ana frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Üst panel - kontroller
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(control_frame, text="DICOM Aç", 
                   command=self.open_dicom).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Klasör Seç", 
                   command=self.open_folder).pack(side=tk.LEFT, padx=5)
        
        self.status_label = ttk.Label(control_frame, text="DICOM dosyası seçin")
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # ML durumu
        ml_status = "✓ ML Model Aktif" if self.calculator.predictor else "⚠ Rule-Based Mod"
        ml_label = ttk.Label(control_frame, text=ml_status,
                            foreground="green" if self.calculator.predictor else "orange")
        ml_label.pack(side=tk.RIGHT, padx=10)
        
        # Orta panel - görüntü ve sonuçlar
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Sol - görüntü
        img_frame = ttk.LabelFrame(content_frame, text="Görüntü", padding=5)
        img_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.image_label = ttk.Label(img_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Sağ - sonuçlar
        results_frame = ttk.LabelFrame(content_frame, text="Sonuçlar", padding=10)
        results_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        # VFA
        ttk.Label(results_frame, text="VFA (Visseral Yağ)", 
                  font=('Helvetica', 12, 'bold')).pack(anchor=tk.W)
        self.vfa_var = tk.StringVar(value="-- cm²")
        ttk.Label(results_frame, textvariable=self.vfa_var,
                  font=('Helvetica', 24)).pack(anchor=tk.W, pady=(0, 20))
        
        # PMA
        ttk.Label(results_frame, text="PMA (Psoas Kas Alanı)", 
                  font=('Helvetica', 12, 'bold')).pack(anchor=tk.W)
        self.pma_var = tk.StringVar(value="-- cm²")
        ttk.Label(results_frame, textvariable=self.pma_var,
                  font=('Helvetica', 24)).pack(anchor=tk.W, pady=(0, 20))
        
        # Confidence
        ttk.Label(results_frame, text="Güven Skoru", 
                  font=('Helvetica', 10)).pack(anchor=tk.W)
        self.conf_var = tk.StringVar(value="--%")
        ttk.Label(results_frame, textvariable=self.conf_var,
                  font=('Helvetica', 14)).pack(anchor=tk.W, pady=(0, 20))
        
        # Progress
        self.progress = ttk.Progressbar(results_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=10)
        
        # Alt panel - hasta bilgileri
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.patient_var = tk.StringVar(value="Hasta: -")
        ttk.Label(info_frame, textvariable=self.patient_var).pack(side=tk.LEFT)
        
        self.date_var = tk.StringVar(value="Tarih: -")
        ttk.Label(info_frame, textvariable=self.date_var).pack(side=tk.LEFT, padx=20)
    
    def open_dicom(self):
        """DICOM dosyası aç"""
        filetypes = [
            ("DICOM files", "*.dcm"),
            ("All files", "*.*")
        ]
        filepath = filedialog.askopenfilename(filetypes=filetypes)
        
        if filepath:
            self.process_file(filepath)
    
    def open_folder(self):
        """DICOM klasörü seç"""
        folder = filedialog.askdirectory()
        
        if folder:
            # İlk DICOM dosyasını bul
            dcm_files = list(Path(folder).glob("*.dcm"))
            if dcm_files:
                self.process_file(str(dcm_files[0]))
            else:
                messagebox.showwarning("Uyarı", "Klasörde DICOM dosyası bulunamadı")
    
    def process_file(self, filepath):
        """Dosyayı işle"""
        self.status_label.config(text=f"İşleniyor: {Path(filepath).name}")
        self.progress.start()
        
        # Arka planda işle
        thread = threading.Thread(target=self._process_thread, args=(filepath,))
        thread.start()
    
    def _process_thread(self, filepath):
        """İşleme thread'i"""
        try:
            results, overlay = self.calculator.calculate(filepath)
            
            # GUI güncelle (main thread'de)
            self.root.after(0, lambda: self._update_results(results, overlay))
            
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: self._show_error(msg))
    
    def _update_results(self, results, overlay):
        """Sonuçları göster"""
        self.progress.stop()
        self.status_label.config(text="Tamamlandı")
        
        # VFA/PMA
        self.vfa_var.set(f"{results['vfa_cm2']:.1f} cm²")
        self.pma_var.set(f"{results['pma_cm2']:.1f} cm²")
        self.conf_var.set(f"{results['confidence']*100:.0f}%")
        
        # Hasta bilgileri
        self.patient_var.set(f"Hasta: {results.get('patient_id', '-')}")
        self.date_var.set(f"Tarih: {results.get('study_date', '-')}")
        
        # Overlay görüntü
        self._display_overlay(overlay)
    
    def _display_overlay(self, overlay):
        """Overlay görüntüyü göster"""
        # Resize to fit
        max_size = 500
        h, w = overlay.shape[:2]
        scale = min(max_size / w, max_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        overlay_resized = cv2.resize(overlay, (new_w, new_h))
        
        # BGR to RGB
        overlay_rgb = cv2.cvtColor(overlay_resized, cv2.COLOR_BGR2RGB)
        
        # PIL Image
        img = Image.fromarray(overlay_rgb)
        self.current_overlay = ImageTk.PhotoImage(img)
        
        self.image_label.config(image=self.current_overlay)
    
    def _show_error(self, message):
        """Hata göster"""
        self.progress.stop()
        self.status_label.config(text="Hata!")
        messagebox.showerror("Hata", message)


def main():
    # Tk başlat
    root = tk.Tk()
    
    # macOS için
    if sys.platform == 'darwin':
        os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")
    
    app = VFAPMAGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()

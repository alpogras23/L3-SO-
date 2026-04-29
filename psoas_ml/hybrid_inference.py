#!/usr/bin/env python3
"""
Hybrid VFA/PMA Inference Module - V2
HU-tabanlı VFA + Doğru anatomik psoas tespiti.
Klinik DICOM verileri için optimize edilmiş.

Anatomi:
- L3 CT'lerde vertebra POSTERIOR (arkada, y büyük)
- Psoas kasları vertebranın ANTERIOR-LATERAL'inde (önde ve yanda)
- VFA: Fasya sınırı içinde, subkutan yağ hariç
"""
import numpy as np
import cv2
from scipy import ndimage
from scipy.spatial import ConvexHull

# HU Thresholds - Optimize edilmiş
VFA_HU_LOW = -190
VFA_HU_HIGH = -30
PSOAS_HU_LOW = 0      # Psoas için dar aralık (daha spesifik)
PSOAS_HU_HIGH = 100
MUSCLE_HU_LOW = -29
MUSCLE_HU_HIGH = 150
BONE_HU_MIN = 200


class HybridVFAPMAPredictor:
    """HU-tabanlı VFA ve anatomik psoas tespiti - V2"""
    
    def __init__(self):
        print("Hybrid Predictor V2 yüklendi (Anatomik Psoas + HU-based VFA)")
    
    def predict(self, hu_array, pixel_spacing=(1.0, 1.0)):
        """VFA ve PMA hesapla."""
        h, w = hu_array.shape
        pixel_area = pixel_spacing[0] * pixel_spacing[1]
        
        # 1. Body mask
        body_mask = self._create_body_mask(hu_array)
        
        # 2. Vertebra tespiti
        vertebra_mask, vertebra_center = self._detect_vertebra(hu_array, body_mask)
        
        # 3. Psoas tespiti - anatomik olarak doğru
        psoas_left, psoas_right = self._detect_psoas_anatomic(
            hu_array, vertebra_center, body_mask, pixel_spacing
        )
        
        # 4. Fasya sınırı
        fascia_mask = self._create_fascia_boundary(hu_array, body_mask)
        
        # 5. VFA hesapla
        vfa_mask = self._calculate_vfa(hu_array, fascia_mask, body_mask, pixel_spacing)
        
        # 6. Alanları hesapla
        vfa_mm2 = np.sum(vfa_mask) * pixel_area
        pma_left_mm2 = np.sum(psoas_left) * pixel_area
        pma_right_mm2 = np.sum(psoas_right) * pixel_area
        pma_total_mm2 = pma_left_mm2 + pma_right_mm2
        l3_mm2 = np.sum(vertebra_mask) * pixel_area
        
        confidence = self._calculate_confidence(
            vfa_mm2, pma_total_mm2, l3_mm2, vertebra_center, (h, w)
        )
        
        return {
            'vfa_mm2': vfa_mm2,
            'vfa_cm2': vfa_mm2 / 100,
            'pma_left_mm2': pma_left_mm2,
            'pma_right_mm2': pma_right_mm2,
            'pma_total_mm2': pma_total_mm2,
            'pma_cm2': pma_total_mm2 / 100,
            'l3_mm2': l3_mm2,
            'masks': {
                'vfa': vfa_mask,
                'psoas_left': psoas_left,
                'psoas_right': psoas_right,
                'l3': vertebra_mask,
                'fascia': fascia_mask
            },
            'confidence': confidence,
            'vertebra_center': vertebra_center
        }
    
    def _create_body_mask(self, hu_array):
        """Body mask oluştur"""
        body = hu_array > -500
        body = ndimage.binary_fill_holes(body)
        body = ndimage.binary_opening(body, iterations=3)
        return body.astype(np.uint8)
    
    def _detect_vertebra(self, hu_array, body_mask):
        """Vertebra tespiti"""
        h, w = hu_array.shape
        
        bone_mask = (hu_array > BONE_HU_MIN) & (body_mask > 0)
        labeled, num = ndimage.label(bone_mask)
        
        if num == 0:
            return np.zeros((h, w), dtype=np.uint8), (h // 2, w // 2)
        
        # En büyük kemik bölgesi (vertebra)
        sizes = ndimage.sum(bone_mask, labeled, range(1, num + 1))
        largest = np.argmax(sizes) + 1
        vertebra_mask = (labeled == largest).astype(np.uint8)
        
        # Vertebra merkezi
        vy, vx = ndimage.center_of_mass(vertebra_mask)
        
        return vertebra_mask, (int(vy), int(vx))
    
    def _detect_psoas_anatomic(self, hu_array, vertebra_center, body_mask, pixel_spacing):
        """
        Anatomik olarak doğru psoas tespiti.
        
        L3 CT'de:
        - Vertebra POSTERIOR (arkada, büyük y değeri)
        - Psoas vertebranın ANTERIOR-LATERAL'inde (küçük y, yanlarda)
        - Psoas vertebra gövdesinin hemen yanında başlar
        """
        h, w = hu_array.shape
        vy, vx = vertebra_center
        pixel_area = pixel_spacing[0] * pixel_spacing[1]
        
        # Psoas için kas HU maskesi (dar aralık, daha spesifik)
        muscle_mask = (hu_array >= PSOAS_HU_LOW) & (hu_array <= PSOAS_HU_HIGH)
        muscle_mask = muscle_mask & (body_mask > 0)
        
        # Psoas boyutları (mm cinsinden, piksel cinsine çevir)
        search_anterior_mm = 80   # Vertebradan öne doğru
        search_posterior_mm = 20  # Vertebradan arkaya doğru (biraz overlap)
        search_lateral_mm = 100   # Yana doğru
        search_medial_mm = 20     # Merkeze doğru minimum mesafe
        
        # Piksel cinsine çevir
        anterior = int(search_anterior_mm / pixel_spacing[0])
        posterior = int(search_posterior_mm / pixel_spacing[0])
        lateral = int(search_lateral_mm / pixel_spacing[1])
        medial = int(search_medial_mm / pixel_spacing[1])
        
        # Y aralığı: vertebranın önünde (y < vy) ve biraz arkasında
        y_start = max(0, vy - anterior)
        y_end = min(h, vy + posterior)
        
        # SOL PSOAS: x < vx (sol taraf)
        left_x_start = max(0, vx - lateral)
        left_x_end = max(0, vx - medial)
        
        left_roi = np.zeros((h, w), dtype=bool)
        left_roi[y_start:y_end, left_x_start:left_x_end] = True
        left_muscle = muscle_mask & left_roi
        
        # SAĞ PSOAS: x > vx (sağ taraf)
        right_x_start = min(w, vx + medial)
        right_x_end = min(w, vx + lateral)
        
        right_roi = np.zeros((h, w), dtype=bool)
        right_roi[y_start:y_end, right_x_start:right_x_end] = True
        right_muscle = muscle_mask & right_roi
        
        # En büyük bağlı bileşenleri çıkar
        psoas_left = self._extract_psoas_component(left_muscle, pixel_area)
        psoas_right = self._extract_psoas_component(right_muscle, pixel_area)
        
        return psoas_left, psoas_right
    
    def _extract_psoas_component(self, muscle_mask, pixel_area, 
                                  min_area_mm2=200, max_area_mm2=4000):
        """Psoas bileşenini çıkar - en uygun boyut ve şekil"""
        h, w = muscle_mask.shape
        
        # Morfolojik temizlik
        muscle_clean = ndimage.binary_opening(muscle_mask, iterations=1)
        muscle_clean = ndimage.binary_closing(muscle_clean, iterations=2)
        
        labeled, num = ndimage.label(muscle_clean)
        if num == 0:
            return np.zeros((h, w), dtype=np.uint8)
        
        best_region = None
        best_score = 0
        
        for i in range(1, num + 1):
            region = (labeled == i)
            area_mm2 = np.sum(region) * pixel_area
            
            # Boyut filtresi
            if area_mm2 < min_area_mm2 or area_mm2 > max_area_mm2:
                continue
            
            # Şekil skoru: kompaktlık + eliptiklik
            # Psoas tipik olarak oval şekilli
            try:
                # Bounding box oranı
                rows = np.any(region, axis=1)
                cols = np.any(region, axis=0)
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                
                bbox_h = rmax - rmin + 1
                bbox_w = cmax - cmin + 1
                aspect_ratio = max(bbox_h, bbox_w) / (min(bbox_h, bbox_w) + 1)
                
                # Doluluk oranı
                fill_ratio = np.sum(region) / (bbox_h * bbox_w + 1)
                
                # Psoas için ideal: aspect ratio 1-3, fill ratio 0.5-0.9
                aspect_score = 1.0 if 1.0 <= aspect_ratio <= 3.0 else 0.5
                fill_score = fill_ratio
                
                score = area_mm2 * aspect_score * fill_score
                
                if score > best_score:
                    best_score = score
                    best_region = region
            except:
                continue
        
        if best_region is None:
            return np.zeros((h, w), dtype=np.uint8)
        
        return best_region.astype(np.uint8)
    
    def _create_fascia_boundary(self, hu_array, body_mask):
        """Fasya sınırı - abdominal organların convex hull'u"""
        h, w = hu_array.shape
        
        # Yumuşak doku (organlar, kaslar)
        soft_tissue = (hu_array > -100) & (hu_array < 200) & (body_mask > 0)
        soft_tissue = ndimage.binary_opening(soft_tissue, iterations=2)
        
        points = np.argwhere(soft_tissue > 0)
        if len(points) < 10:
            return body_mask
        
        try:
            hull = ConvexHull(points)
            hull_mask = np.zeros((h, w), dtype=np.uint8)
            hull_points = points[hull.vertices]
            cv2.fillPoly(hull_mask, [hull_points[:, ::-1].astype(np.int32)], 1)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            hull_mask = cv2.morphologyEx(hull_mask, cv2.MORPH_CLOSE, kernel)
            
            return hull_mask
        except:
            return body_mask
    
    def _calculate_vfa(self, hu_array, fascia_mask, body_mask, pixel_spacing):
        """VFA hesapla - visseral yağ alanı"""
        # Yağ HU aralığı
        fat_mask = (hu_array >= VFA_HU_LOW) & (hu_array <= VFA_HU_HIGH)
        
        # Fasya içinde
        vfa_mask = fat_mask & (fascia_mask > 0)
        
        # Subkutan yağı çıkar (kenardan 15mm)
        erosion_mm = 15
        erosion_px = int(erosion_mm / pixel_spacing[0])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_px, erosion_px))
        eroded_body = cv2.erode(body_mask, kernel)
        subcutaneous = body_mask - eroded_body
        
        vfa_mask = vfa_mask & (subcutaneous == 0)
        
        # Morfolojik temizlik
        vfa_mask = ndimage.binary_opening(vfa_mask, iterations=2)
        
        return vfa_mask.astype(np.uint8)
    
    def _calculate_confidence(self, vfa_mm2, pma_mm2, l3_mm2, vertebra_center, shape):
        """Güvenilirlik skoru"""
        h, w = shape
        vy, vx = vertebra_center
        conf = 0.0
        
        # Vertebra pozisyonu (posterior olmalı, y > h*0.4)
        if vy > h * 0.4 and 0.3 < vx/w < 0.7:
            conf += 0.25
        
        # L3 alanı (tipik: 500-2500 mm²)
        if 300 < l3_mm2 < 3500:
            conf += 0.25
        
        # VFA alanı (tipik: 2000-40000 mm²)
        if 1000 < vfa_mm2 < 50000:
            conf += 0.25
        
        # PMA alanı (tipik: 800-4000 mm²)
        if 500 < pma_mm2 < 6000:
            conf += 0.25
        
        return min(conf, 1.0)


def create_overlay(hu_array, masks, alpha=0.5):
    """Segmentasyon overlay görüntüsü"""
    hu_norm = np.clip(hu_array, -150, 250)
    hu_norm = ((hu_norm + 150) / 400 * 255).astype(np.uint8)
    overlay = cv2.cvtColor(hu_norm, cv2.COLOR_GRAY2RGB)
    
    colors = {
        'vfa': (0, 0, 255),         # Kırmızı
        'psoas_left': (255, 0, 0),  # Mavi
        'psoas_right': (255, 0, 0), # Mavi
        'l3': (0, 255, 255),        # Sarı
        'fascia': (0, 255, 0)       # Yeşil
    }
    
    for name, mask in masks.items():
        if mask is None or not np.any(mask):
            continue
        color = colors.get(name, (0, 255, 0))
        
        if name == 'fascia':
            contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                           cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)
        else:
            overlay[mask > 0] = (
                overlay[mask > 0] * (1 - alpha) + 
                np.array(color) * alpha
            ).astype(np.uint8)
    
    return overlay


if __name__ == '__main__':
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut
    from pathlib import Path
    
    predictor = HybridVFAPMAPredictor()
    
    dcm_dir = Path('/Users/alperenogras/Desktop/görüntüler')
    dcm_files = sorted(dcm_dir.glob('*.dcm'))[:5]
    
    print(f"{'Dosya':<25} {'VFA(cm2)':<10} {'PMA(cm2)':<10} {'L3(mm2)':<10} {'Conf':<6}")
    print("=" * 65)
    
    for dcm_path in dcm_files:
        ds = pydicom.dcmread(str(dcm_path))
        hu = apply_modality_lut(ds.pixel_array, ds).astype(np.float32)
        ps = [float(x) for x in getattr(ds, 'PixelSpacing', [1.0, 1.0])]
        
        results = predictor.predict(hu, ps)
        name = dcm_path.name[:24]
        print(f"{name:<25} {results['vfa_cm2']:>8.1f}   {results['pma_cm2']:>8.1f}   {results['l3_mm2']:>8.0f}   {results['confidence']:.2f}")

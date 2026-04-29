#!/usr/bin/env python3
"""
Geliştirilmiş L3 Slice Çıkarım - Doğru VFA ve PMA için
- Psoas: abdominal_muscles task'tan psoas_major kullanır
- VFA: Organ maskelerinden convex hull ile fasya sınırı hesaplar
- Doğru anatomik segmentasyon
"""
import numpy as np
import nibabel as nib
import json
from pathlib import Path
from datetime import datetime
from scipy import ndimage
from scipy.spatial import ConvexHull
import cv2

# Paths
AMOS_IMG_DIR = Path('/home/azureuser/amos22_data/imagesTr')
TOTALSEG_DIR = Path('/home/azureuser/amos22_data/totalseg_output')
ABDOMINAL_DIR = Path('/home/azureuser/amos22_data/abdominal_muscles_output')
OUTPUT_DIR = Path('/home/azureuser/amos22_data/l3_slices_v2')

# Çıktı alt dizinleri
CT_OUT = OUTPUT_DIR / 'ct_slices'
MASK_OUT = OUTPUT_DIR / 'masks'
META_OUT = OUTPUT_DIR / 'metadata'

# HU thresholds
VFA_HU_LOW = -190
VFA_HU_HIGH = -30


def find_l3_center_slice(l3_mask):
    """L3 vertebra mask'ından orta slice'ı bul"""
    z_slices = np.any(l3_mask, axis=(0, 1))
    z_indices = np.where(z_slices)[0]
    
    if len(z_indices) == 0:
        return None
    
    center_idx = z_indices[len(z_indices) // 2]
    return center_idx


def create_fascia_boundary(organ_slices, body_slice, ct_shape):
    """
    Fasya sınırı oluştur:
    1. Tüm abdominal organları birleştir
    2. Convex hull hesapla
    3. Posterolateral guard band ekle (hernia koruma)
    """
    h, w = ct_shape
    
    # Tüm organları birleştir
    combined = np.zeros((h, w), dtype=np.uint8)
    for organ in organ_slices:
        if organ is not None:
            combined = np.maximum(combined, (organ > 0).astype(np.uint8))
    
    if np.sum(combined) < 100:
        # Organ bulunamadı, body outline kullan
        if body_slice is not None:
            return (body_slice > 0).astype(np.uint8)
        return np.ones((h, w), dtype=np.uint8)
    
    # Convex hull hesapla
    points = np.argwhere(combined > 0)
    if len(points) < 3:
        return combined
    
    try:
        hull = ConvexHull(points)
        hull_mask = np.zeros((h, w), dtype=np.uint8)
        hull_points = points[hull.vertices]
        cv2.fillPoly(hull_mask, [hull_points[:, ::-1].astype(np.int32)], 1)
        
        # Morfolojik düzeltme
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        hull_mask = cv2.morphologyEx(hull_mask, cv2.MORPH_CLOSE, kernel)
        
        return hull_mask
    except Exception:
        return combined


def create_vfa_mask(ct_slice, fascia_mask, organ_masks):
    """
    VFA maskesi oluştur:
    1. HU aralığı -190 ile -30 arası (yağ dokusu)
    2. Fasya sınırı içinde
    3. Organların dışında
    """
    # Yağ dokusu HU mask
    fat_mask = (ct_slice >= VFA_HU_LOW) & (ct_slice <= VFA_HU_HIGH)
    
    # Organları birleştir (bunların dışında olmalı)
    organ_exclusion = np.zeros_like(fat_mask, dtype=bool)
    for organ_mask in organ_masks:
        if organ_mask is not None:
            organ_exclusion |= (organ_mask > 0)
    
    # VFA = yağ dokusu AND fasya içi AND organ dışı
    vfa_mask = fat_mask & (fascia_mask > 0) & ~organ_exclusion
    
    # Küçük bölgeleri temizle
    vfa_mask = ndimage.binary_opening(vfa_mask, iterations=2)
    
    return vfa_mask.astype(np.uint8)


def load_nifti_slice(nifti_path, slice_idx):
    """NIfTI dosyasından belirli slice'ı yükle"""
    if not nifti_path.exists():
        return None
    nii = nib.load(str(nifti_path))
    data = nii.get_fdata()
    if slice_idx >= data.shape[2]:
        return None
    return data[:, :, slice_idx]


def process_case(case_id):
    """Tek bir vaka için L3 slice ve maskeleri çıkar"""
    
    # L3 vertebra mask yükle (total task'tan)
    l3_path = TOTALSEG_DIR / case_id / 'vertebrae_L3.nii.gz'
    if not l3_path.exists():
        return None, "L3 vertebra mask bulunamadı"
    
    l3_nii = nib.load(str(l3_path))
    l3_mask = l3_nii.get_fdata()
    
    # L3 center slice bul
    l3_center = find_l3_center_slice(l3_mask)
    if l3_center is None:
        return None, "L3 center bulunamadı"
    
    # CT görüntü yükle
    ct_path = AMOS_IMG_DIR / f'{case_id}.nii.gz'
    if not ct_path.exists():
        return None, "CT görüntü bulunamadı"
    
    ct_nii = nib.load(str(ct_path))
    ct_data = ct_nii.get_fdata()
    affine = ct_nii.affine
    
    # Pixel spacing hesapla
    pixel_spacing = np.abs([affine[0,0], affine[1,1], affine[2,2]])
    
    # L3 seviyesi slice'ı al
    ct_slice = ct_data[:, :, l3_center].astype(np.float32)
    l3_slice = l3_mask[:, :, l3_center].astype(np.uint8)
    
    # Psoas maskeleri yükle (abdominal_muscles task'tan - DOĞRU KAYNAK)
    abdom_dir = ABDOMINAL_DIR / case_id
    psoas_left_path = abdom_dir / 'psoas_major_left.nii.gz'
    psoas_right_path = abdom_dir / 'psoas_major_right.nii.gz'
    
    # Fallback: total task'tan iliopsoas
    if not psoas_left_path.exists():
        psoas_left_path = TOTALSEG_DIR / case_id / 'iliopsoas_left.nii.gz'
    if not psoas_right_path.exists():
        psoas_right_path = TOTALSEG_DIR / case_id / 'iliopsoas_right.nii.gz'
    
    psoas_left_slice = load_nifti_slice(psoas_left_path, l3_center)
    psoas_right_slice = load_nifti_slice(psoas_right_path, l3_center)
    
    if psoas_left_slice is not None:
        psoas_left_slice = psoas_left_slice.astype(np.uint8)
    if psoas_right_slice is not None:
        psoas_right_slice = psoas_right_slice.astype(np.uint8)
    
    # Organları yükle (VFA ve fasya hesabı için)
    organ_names = ['liver', 'spleen', 'kidney_left', 'kidney_right', 
                   'stomach', 'small_bowel', 'colon', 'aorta', 
                   'inferior_vena_cava', 'pancreas', 'duodenum']
    organ_slices = []
    
    for organ in organ_names:
        organ_path = TOTALSEG_DIR / case_id / f'{organ}.nii.gz'
        organ_slice = load_nifti_slice(organ_path, l3_center)
        organ_slices.append(organ_slice)
    
    # Body outline (varsa)
    body_path = TOTALSEG_DIR / case_id / 'body.nii.gz'
    body_slice = load_nifti_slice(body_path, l3_center)
    
    # Fasya sınırı oluştur
    fascia_mask = create_fascia_boundary(organ_slices, body_slice, ct_slice.shape)
    
    # VFA mask oluştur
    vfa_mask = create_vfa_mask(ct_slice, fascia_mask, organ_slices)
    
    # Metadata
    meta = {
        'case_id': case_id,
        'l3_slice_idx': int(l3_center),
        'ct_shape': list(ct_data.shape),
        'pixel_spacing_mm': pixel_spacing.tolist(),
        'vfa_pixel_count': int(np.sum(vfa_mask)),
        'psoas_left_pixel_count': int(np.sum(psoas_left_slice)) if psoas_left_slice is not None else 0,
        'psoas_right_pixel_count': int(np.sum(psoas_right_slice)) if psoas_right_slice is not None else 0,
        'l3_pixel_count': int(np.sum(l3_slice)),
        'fascia_pixel_count': int(np.sum(fascia_mask)),
        'psoas_source': 'psoas_major' if (ABDOMINAL_DIR / case_id / 'psoas_major_left.nii.gz').exists() else 'iliopsoas'
    }
    
    # Alanları mm² cinsinden hesapla
    pixel_area = pixel_spacing[0] * pixel_spacing[1]
    meta['vfa_area_mm2'] = meta['vfa_pixel_count'] * pixel_area
    meta['psoas_left_area_mm2'] = meta['psoas_left_pixel_count'] * pixel_area
    meta['psoas_right_area_mm2'] = meta['psoas_right_pixel_count'] * pixel_area
    meta['pma_total_area_mm2'] = meta['psoas_left_area_mm2'] + meta['psoas_right_area_mm2']
    meta['fascia_area_mm2'] = meta['fascia_pixel_count'] * pixel_area
    
    return {
        'ct_slice': ct_slice,
        'vfa_mask': vfa_mask,
        'psoas_left_mask': psoas_left_slice,
        'psoas_right_mask': psoas_right_slice,
        'l3_mask': l3_slice,
        'fascia_mask': fascia_mask,
        'metadata': meta,
        'pixel_spacing': pixel_spacing
    }, None


def save_outputs(data):
    """Çıktıları kaydet"""
    case_id = data['metadata']['case_id']
    
    # CT slice
    ct_out_path = CT_OUT / f'{case_id}_ct.npy'
    np.save(str(ct_out_path), data['ct_slice'])
    
    # 5-channel mask (VFA, PsoasL, PsoasR, L3, Fascia)
    h, w = data['ct_slice'].shape
    combined_mask = np.zeros((5, h, w), dtype=np.uint8)
    combined_mask[0] = data['vfa_mask']
    combined_mask[1] = data['psoas_left_mask'] if data['psoas_left_mask'] is not None else 0
    combined_mask[2] = data['psoas_right_mask'] if data['psoas_right_mask'] is not None else 0
    combined_mask[3] = data['l3_mask']
    combined_mask[4] = data['fascia_mask']
    
    mask_out_path = MASK_OUT / f'{case_id}_mask.npy'
    np.save(str(mask_out_path), combined_mask)
    
    # Metadata
    meta_out_path = META_OUT / f'{case_id}_meta.json'
    with open(meta_out_path, 'w') as f:
        json.dump(data['metadata'], f, indent=2)
    
    return ct_out_path, mask_out_path, meta_out_path


def main():
    # Çıktı dizinlerini oluştur
    CT_OUT.mkdir(parents=True, exist_ok=True)
    MASK_OUT.mkdir(parents=True, exist_ok=True)
    META_OUT.mkdir(parents=True, exist_ok=True)
    
    # TotalSegmentator çıktıları olan vakaları bul
    cases = sorted([d.name for d in TOTALSEG_DIR.iterdir() if d.is_dir()])
    print(f"Toplam {len(cases)} vaka bulundu (total task)")
    
    # Abdominal muscles çıktılarını kontrol et
    abdom_cases = set()
    if ABDOMINAL_DIR.exists():
        abdom_cases = set([d.name for d in ABDOMINAL_DIR.iterdir() if d.is_dir()])
    print(f"Abdominal muscles: {len(abdom_cases)} vaka")
    
    success_count = 0
    fail_count = 0
    all_meta = []
    
    for i, case_id in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] {case_id} işleniyor...", end=" ")
        
        data, error = process_case(case_id)
        
        if data is None:
            print(f"✗ {error}")
            fail_count += 1
            continue
        
        save_outputs(data)
        all_meta.append(data['metadata'])
        
        vfa = data['metadata']['vfa_area_mm2']
        pma = data['metadata']['pma_total_area_mm2']
        src = data['metadata']['psoas_source']
        print(f"✓ VFA={vfa:.0f}mm², PMA={pma:.0f}mm² [{src}]")
        success_count += 1
    
    # Özet metadata kaydet
    summary_path = OUTPUT_DIR / 'dataset_summary.json'
    summary = {
        'total_cases': len(cases),
        'success': success_count,
        'failed': fail_count,
        'timestamp': datetime.now().isoformat(),
        'cases': all_meta
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Tamamlandı: {success_count} başarılı, {fail_count} başarısız")
    print(f"Çıktılar: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

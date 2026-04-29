#!/usr/bin/env python3
"""
TotalSegmentator 3D çıktılarından L3 seviyesi 2D slice'lar çıkarır.
VFA ve PMA eğitimi için teacher mask ve CT görüntü çiftleri oluşturur.
"""
import numpy as np
import nibabel as nib
import json
from pathlib import Path
from datetime import datetime
import cv2

# Paths
AMOS_IMG_DIR = Path('/home/azureuser/amos22_data/imagesTr')
TOTALSEG_DIR = Path('/home/azureuser/amos22_data/totalseg_output')
OUTPUT_DIR = Path('/home/azureuser/amos22_data/l3_slices_2d')

# Çıktı alt dizinleri
CT_OUT = OUTPUT_DIR / 'ct_slices'
MASK_OUT = OUTPUT_DIR / 'masks'
META_OUT = OUTPUT_DIR / 'metadata'

# HU thresholds for VFA (yağ dokusu)
VFA_HU_LOW = -190
VFA_HU_HIGH = -30

# Psoas kas HU aralığı
PSOAS_HU_LOW = -29
PSOAS_HU_HIGH = 150

def find_l3_center_slice(l3_mask):
    """L3 vertebra mask'ından orta slice'ı bul"""
    # Z ekseni boyunca nonzero slice'ları bul
    z_slices = np.any(l3_mask, axis=(0, 1))
    z_indices = np.where(z_slices)[0]
    
    if len(z_indices) == 0:
        return None
    
    # Ortadaki slice'ı al
    center_idx = z_indices[len(z_indices) // 2]
    return center_idx

def create_vfa_mask(ct_slice, organ_masks):
    """
    VFA maskesi oluştur:
    1. HU aralığı -190 ile -30 arası (yağ dokusu)
    2. Organların dışında (liver, spleen, kidney, stomach, bowel)
    """
    # Yağ dokusu HU mask
    fat_mask = (ct_slice >= VFA_HU_LOW) & (ct_slice <= VFA_HU_HIGH)
    
    # Organları birleştir (bunların dışında olmalı)
    organ_exclusion = np.zeros_like(fat_mask, dtype=bool)
    for organ_mask in organ_masks:
        if organ_mask is not None:
            organ_exclusion |= (organ_mask > 0)
    
    # VFA = yağ dokusu AND organ dışı
    vfa_mask = fat_mask & ~organ_exclusion
    
    return vfa_mask.astype(np.uint8)

def process_case(case_id, totalseg_dir, amos_img_dir):
    """Tek bir vaka için L3 slice ve maskeleri çıkar"""
    
    # L3 vertebra mask yükle
    l3_path = totalseg_dir / case_id / 'vertebrae_L3.nii.gz'
    if not l3_path.exists():
        return None, "L3 vertebra mask bulunamadı"
    
    l3_nii = nib.load(str(l3_path))
    l3_mask = l3_nii.get_fdata()
    
    # L3 center slice bul
    l3_center = find_l3_center_slice(l3_mask)
    if l3_center is None:
        return None, "L3 center bulunamadı"
    
    # CT görüntü yükle
    ct_path = amos_img_dir / f'{case_id}.nii.gz'
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
    
    # Psoas maskeleri yükle
    psoas_left_path = totalseg_dir / case_id / 'iliopsoas_left.nii.gz'
    psoas_right_path = totalseg_dir / case_id / 'iliopsoas_right.nii.gz'
    
    psoas_left_slice = None
    psoas_right_slice = None
    
    if psoas_left_path.exists():
        psoas_left = nib.load(str(psoas_left_path)).get_fdata()
        psoas_left_slice = psoas_left[:, :, l3_center].astype(np.uint8)
    
    if psoas_right_path.exists():
        psoas_right = nib.load(str(psoas_right_path)).get_fdata()
        psoas_right_slice = psoas_right[:, :, l3_center].astype(np.uint8)
    
    # Organları yükle (VFA hesabı için hariç tutulacak)
    organ_names = ['liver', 'spleen', 'kidney_left', 'kidney_right', 
                   'stomach', 'small_bowel', 'colon']
    organ_masks = []
    
    for organ in organ_names:
        organ_path = totalseg_dir / case_id / f'{organ}.nii.gz'
        if organ_path.exists():
            organ_data = nib.load(str(organ_path)).get_fdata()
            organ_slice = organ_data[:, :, l3_center]
            organ_masks.append(organ_slice)
        else:
            organ_masks.append(None)
    
    # VFA mask oluştur (HU tabanlı + organ exclusion)
    vfa_mask = create_vfa_mask(ct_slice, organ_masks)
    
    # Metadata
    meta = {
        'case_id': case_id,
        'l3_slice_idx': int(l3_center),
        'ct_shape': list(ct_data.shape),
        'pixel_spacing_mm': pixel_spacing.tolist(),
        'vfa_pixel_count': int(np.sum(vfa_mask)),
        'psoas_left_pixel_count': int(np.sum(psoas_left_slice)) if psoas_left_slice is not None else 0,
        'psoas_right_pixel_count': int(np.sum(psoas_right_slice)) if psoas_right_slice is not None else 0,
        'l3_pixel_count': int(np.sum(l3_slice))
    }
    
    # VFA/PMA alanı mm² cinsinden hesapla
    pixel_area = pixel_spacing[0] * pixel_spacing[1]
    meta['vfa_area_mm2'] = meta['vfa_pixel_count'] * pixel_area
    meta['psoas_left_area_mm2'] = meta['psoas_left_pixel_count'] * pixel_area
    meta['psoas_right_area_mm2'] = meta['psoas_right_pixel_count'] * pixel_area
    meta['pma_total_area_mm2'] = meta['psoas_left_area_mm2'] + meta['psoas_right_area_mm2']
    
    return {
        'ct_slice': ct_slice,
        'vfa_mask': vfa_mask,
        'psoas_left_mask': psoas_left_slice,
        'psoas_right_mask': psoas_right_slice,
        'l3_mask': l3_slice,
        'metadata': meta,
        'pixel_spacing': pixel_spacing
    }, None

def save_outputs(data, output_dir):
    """Çıktıları kaydet"""
    case_id = data['metadata']['case_id']
    
    # CT slice (NumPy formatında)
    ct_out_path = CT_OUT / f'{case_id}_ct.npy'
    np.save(str(ct_out_path), data['ct_slice'])
    
    # 4-channel mask (VFA, PsoasL, PsoasR, L3)
    h, w = data['ct_slice'].shape
    combined_mask = np.zeros((4, h, w), dtype=np.uint8)
    combined_mask[0] = data['vfa_mask']
    combined_mask[1] = data['psoas_left_mask'] if data['psoas_left_mask'] is not None else 0
    combined_mask[2] = data['psoas_right_mask'] if data['psoas_right_mask'] is not None else 0
    combined_mask[3] = data['l3_mask']
    
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
    print(f"Toplam {len(cases)} vaka bulundu")
    
    success_count = 0
    fail_count = 0
    all_meta = []
    
    for i, case_id in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] {case_id} işleniyor...", end=" ")
        
        data, error = process_case(case_id, TOTALSEG_DIR, AMOS_IMG_DIR)
        
        if data is None:
            print(f"✗ {error}")
            fail_count += 1
            continue
        
        save_outputs(data, OUTPUT_DIR)
        all_meta.append(data['metadata'])
        
        vfa = data['metadata']['vfa_area_mm2']
        pma = data['metadata']['pma_total_area_mm2']
        print(f"✓ VFA={vfa:.0f}mm², PMA={pma:.0f}mm²")
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
    
    print(f"\n{'='*50}")
    print(f"Tamamlandı: {success_count} başarılı, {fail_count} başarısız")
    print(f"Çıktılar: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()

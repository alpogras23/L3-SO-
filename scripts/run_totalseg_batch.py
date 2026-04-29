#!/usr/bin/env python3
"""
TotalSegmentator ile AMOS22 CT'lerden teacher mask üretimi.
Hedef: VFA, PMA ve L3 tespiti için gerekli segmentasyonlar
"""
import os
import subprocess
import glob
import json
from pathlib import Path
from datetime import datetime

# Paths
AMOS_DIR = Path('/home/azureuser/amos22_data/imagesTr')
OUTPUT_DIR = Path('/home/azureuser/amos22_data/totalseg_output')
PROGRESS_FILE = OUTPUT_DIR / 'progress.json'

def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {'completed': [], 'failed': []}

def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def run_totalseg(nifti_path, out_dir):
    """Tek bir CT için TotalSegmentator çalıştır"""
    cmd = [
        'TotalSegmentator',
        '-i', str(nifti_path),
        '-o', str(out_dir),
        '-ta', 'total',  # Tüm anatomik yapılar
        '-f',  # Fast mode (hızlı)
        '-d', 'gpu',  # GPU kullan
        '-q'   # Quiet mode
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return result.returncode == 0, result.stderr
    except Exception as e:
        return False, str(e)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Tüm NIfTI dosyalarını bul
    nifti_files = sorted(glob.glob(str(AMOS_DIR / '*.nii.gz')))
    print(f'Toplam {len(nifti_files)} CT dosyası bulundu')
    
    progress = load_progress()
    completed = set(progress['completed'])
    
    for i, nifti_path in enumerate(nifti_files):
        case_id = Path(nifti_path).stem.replace('.nii', '')
        
        if case_id in completed:
            print(f'[{i+1}/{len(nifti_files)}] {case_id} - ATLANDI (zaten işlendi)')
            continue
        
        out_dir = OUTPUT_DIR / case_id
        print(f'[{i+1}/{len(nifti_files)}] {case_id} işleniyor... {datetime.now().strftime("%H:%M:%S")}')
        
        success, msg = run_totalseg(nifti_path, out_dir)
        
        if success:
            progress['completed'].append(case_id)
            print(f'  ✓ Başarılı')
        else:
            progress['failed'].append(case_id)
            print(f'  ✗ Başarısız: {msg[:100]}')
        
        save_progress(progress)
    
    print(f'\nTamamlandı: {len(progress["completed"])}, Başarısız: {len(progress["failed"])}')

if __name__ == '__main__':
    main()

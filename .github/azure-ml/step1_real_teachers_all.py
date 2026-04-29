#!/usr/bin/env python3
"""
Generate teacher labels from AMOS22 dataset using TotalSegmentator.
Processes ALL AMOS files found in the input directory.

Input: AMOS22 NII.GZ files (any quantity)
Output: teacher_labels/ with HU slices + binary masks (psoas_left, psoas_right, vat, inner_abdomen)
"""

import sys
import json
import argparse
import tempfile
from pathlib import Path
from typing import Optional
import subprocess

import numpy as np
import nibabel as nib
import cv2

# TotalSegmentator
try:
    from totalsegmentator.python_api import totalsegmentator
except ImportError:
    totalsegmentator = None
    print("⚠️  TotalSegmentator not installed - install with: pip install TotalSegmentator")

# Import core_mini for inner_abdomen_via_wall
from pathlib import Path as PathLib
# Add project root to path if not already there
project_root = PathLib(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from core_mini import inner_abdomen_via_wall
    CORE_MINI_AVAILABLE = True
    print("✅ core_mini imported successfully - inner_abdomen masks will be generated")
except ImportError:
    CORE_MINI_AVAILABLE = False
    print("⚠️  core_mini not available - inner_abdomen mask will not be generated")

# Import HU calibration and leak prevention modules
try:
    from hu_calibration import HUCalibrator, apply_calibrated_fat_mask
    from leak_prevention import PosterolateralLeakPrevention
    HU_CALIBRATION_AVAILABLE = True
    print("✅ HU calibration and leak prevention modules imported")
except ImportError:
    HU_CALIBRATION_AVAILABLE = False
    print("⚠️  HU calibration modules not available - using default HU bands")


def locate_segmentation_file(ts_output_dir: Path, filename: str) -> Optional[Path]:
    """Locate a particular segmentation file anywhere inside the TotalSegmentator output tree."""
    # Check common direct locations first for speed
    preferred_roots = [ts_output_dir / "segmentations", ts_output_dir]
    for root in preferred_roots:
        candidate = root / filename
        if candidate.exists():
            return candidate
    # Fallback: brute force search
    matches = list(ts_output_dir.rglob(filename))
    if matches:
        return matches[0]
    return None


def run_totalsegmentator(ct_path: str, output_dir: str, fast: bool = True, roi_subset: Optional[str] = None) -> bool:
    """
    Run TotalSegmentator on CT volume with multiple tasks.
    
    Args:
        ct_path: Path to CT NII.GZ file
        output_dir: Directory to save TS output
        fast: Use fast mode
        
        if 'TOTALSEG_SCRATCH' not in os.environ:
            os.environ['TOTALSEG_SCRATCH'] = tempfile.gettempdir()
    Returns:
        True if successful, False otherwise
    """
    if totalsegmentator is None:
        print(f"⚠️  Skipping TS (not installed): {ct_path}")
        return False
    
    try:
        # Set environment variable for TotalSegmentator scratch directory
        import os
        scratch_dir = tempfile.gettempdir()
        # Ensure scratch envs are set for both api and CLI
        os.environ.setdefault('SCRATCH', scratch_dir)
        os.environ.setdefault('TOTALSEG_SCRATCH', scratch_dir)
        # Run TotalSegmentator with abdominal_muscles task for psoas segmentation
        # task="abdominal_muscles": Contains psoas_major_left/right (labels 19-20)
        # This task specifically segments abdominal muscles including psoas
        # fast=False: Required for abdominal_muscles task (fast=True not supported)
        # NOTE: Requires significant RAM (16GB minimum, may need 32GB for large volumes)
        print("  🔄 Running TotalSegmentator (abdominal_muscles, fast=False, HIGH RAM)...")
        ts_kwargs = dict(
            input=ct_path,
            output=output_dir,
            task="abdominal_muscles",  # FIXED: Use abdominal_muscles for psoas
            ml=True,
            fast=False,  # REQUIRED: fast=True not supported with abdominal_muscles
            quiet=False,
        )
        if roi_subset:
            print("  ⚠️  ROI subset only supported for 'total' tasks; ignoring for abdominal_muscles")

        totalsegmentator(**ts_kwargs)
        print("  ✅ TotalSegmentator completed (abdominal_muscles, fast=False)")
        
        return True
    except Exception as e:
        print(f"  ❌ TotalSegmentator failed (python_api): {e}")
        # Fallback: CLI invocation to avoid python_api internal bugs (e.g., task_id)
        try:
            cmd = [
                "TotalSegmentator",
                "-i", ct_path,
                "-o", output_dir,
                "-ta", "abdominal_muscles",
                "--ml",
                "--quiet",
            ]
            if roi_subset:
                cmd += ["--roi_subset", roi_subset]
                print(f"  🪄 CLI ROI subset: {roi_subset}")
            print("  🔄 Fallback: running TotalSegmentator CLI ...")
            subprocess.run(cmd, check=True)
            print("  ✅ TotalSegmentator CLI completed")
            return True
        except Exception as e2:
            print(f"  ❌ TotalSegmentator CLI failed: {e2}")
            return False


def pick_l3_slice(ts_output_dir: str) -> Optional[int]:
    """
    Pick L3 vertebra slice from TotalSegmentator output.
    
    Strategy:
    1. Load vertebra label map (usually contains lumbar vertebra masks)
    2. Find slices where lumbar vertebra 3 is present
    3. Return center slice
    
    Args:
        ts_output_dir: TotalSegmentator output directory
        
    Returns:
        L3 slice index or None
    """
    ts_path = Path(ts_output_dir)

    # Look for vertebra files
    vertebra_file = None
    for vfile in ["vertebrae.nii.gz", "vertebra.nii.gz", "spine.nii.gz"]:
        candidate = locate_segmentation_file(ts_path, vfile)
        if candidate is not None:
            vertebra_file = str(candidate)
            break
    
    if not vertebra_file:
        print("  ⚠️  No vertebra segmentation found, using center slice")
        return None
    
    try:
        vertebra_nib = nib.load(vertebra_file)
        dataobj = vertebra_nib.dataobj
        z_indices = []
        depth = dataobj.shape[2]
        for z in range(depth):
            slice_data = np.asarray(dataobj[:, :, z])
            if slice_data.max() > 0:
                z_indices.append(z)
        if not z_indices:
            print("  ⚠️  No vertebra found, using center slice")
            return None
        l3_slice = int(np.median(z_indices))
        print(f"  📍 Estimated L3 slice: {l3_slice}")
        return l3_slice
    except Exception as e:
        print(f"  ⚠️  Failed to parse vertebra file: {e}")
        return None


def extract_teacher_labels(ct_path: str, ts_output_dir: str, output_dir: str) -> bool:
    """
    Extract 2D teacher labels at L3 from CT and TS segmentations.
    
    Args:
        ct_path: Path to CT NII.GZ file
        ts_output_dir: TotalSegmentator output directory
        output_dir: Output directory for teacher slices
        
    Returns:
        True if successful, False otherwise
    """
    try:
        case_id = Path(ct_path).stem.replace(".nii", "")
        case_output_dir = Path(output_dir) / case_id
        case_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load CT
        print("  📖 Loading CT...")
        ct_nib = nib.load(ct_path)
        ct_dataobj = ct_nib.dataobj
        ct_shape = ct_dataobj.shape[:3]
        print(f"  📊 CT shape: {ct_shape}")
        
        # Pick L3 slice
        l3_idx = pick_l3_slice(ts_output_dir)
        if l3_idx is None:
            # Fallback: use center slice
            l3_idx = ct_shape[2] // 2
            print(f"  📍 Using fallback center slice: {l3_idx}")
        
        # Ensure valid slice
        l3_idx = max(0, min(l3_idx, ct_shape[2] - 1))
        
        # Extract HU slice at L3 without loading full volume
        hu_slice = np.asarray(ct_dataobj[:, :, l3_idx], dtype=np.float32)
        
        ts_root = Path(ts_output_dir)

        # Create binary masks for psoas and VAT
        psoas_left = np.zeros_like(hu_slice, dtype=np.uint8)
        psoas_right = np.zeros_like(hu_slice, dtype=np.uint8)
        vat = np.zeros_like(hu_slice, dtype=np.uint8)
        
        # Try to load psoas segmentations if available
        psoas_left_file = locate_segmentation_file(ts_root, "psoas_major_left.nii.gz")
        psoas_right_file = locate_segmentation_file(ts_root, "psoas_major_right.nii.gz")
        
        if psoas_left_file is not None:
            try:
                psoas_left_nib = nib.load(psoas_left_file)
                psoas_left_data = psoas_left_nib.dataobj
                if psoas_left_data.shape[2] > l3_idx:
                    psoas_left = (np.asarray(psoas_left_data[:, :, l3_idx]) > 0).astype(np.uint8)
                print("  ✅ Loaded psoas_major_left")
            except Exception as e:
                print(f"  ⚠️  Failed to load psoas_left: {e}")
        else:
            print("  ⚠️  psoas_major_left not found in TS output")
        
        if psoas_right_file is not None:
            try:
                psoas_right_nib = nib.load(psoas_right_file)
                psoas_right_data = psoas_right_nib.dataobj
                if psoas_right_data.shape[2] > l3_idx:
                    psoas_right = (np.asarray(psoas_right_data[:, :, l3_idx]) > 0).astype(np.uint8)
                print("  ✅ Loaded psoas_major_right")
            except Exception as e:
                print(f"  ⚠️  Failed to load psoas_right: {e}")
        else:
            print("  ⚠️  psoas_major_right not found in TS output")

        # VAT approximation: Use torso_fat from tissue_types task
        vat_candidates = [
            "torso_fat.nii.gz",          # From tissue_types task - BEST for visceral fat
            "adipose_visceral.nii.gz",    # Rare, if available
            "adipose_intra_abdominal.nii.gz",
            "visceral_fat.nii.gz",
            "vat.nii.gz",
        ]
        vat_seg_file = None
        for candidate_name in vat_candidates:
            vat_seg_file = locate_segmentation_file(ts_root, candidate_name)
            if vat_seg_file is not None:
                print(f"  ✅ Found {Path(candidate_name).stem} for VAT")
                break
        
        if vat_seg_file is not None:
            try:
                vat_nib = nib.load(vat_seg_file)
                vat_data = vat_nib.dataobj
                if vat_data.shape[2] > l3_idx:
                    vat = (np.asarray(vat_data[:, :, l3_idx]) > 0).astype(np.uint8)
                print(f"  ✅ Loaded VAT mask with {vat.sum()} pixels")
            except Exception as e:
                print(f"  ⚠️  Failed to load VAT mask: {e}")
        else:
            print("  ⚠️  VAT mask not found, using HU-based fallback...")
            # Fallback: Use HU thresholding for visceral fat (-190 to -30 HU)
            # This is a rough approximation but better than nothing
            vat = ((hu_slice >= -190) & (hu_slice <= -30)).astype(np.uint8)
            print(f"  ✅ Generated VAT mask via HU thresholding: {vat.sum()} pixels")
        
        # ========== NEW: Generate INNER_ABDOMEN mask (fascia proxy) ==========
        inner_abdomen = np.zeros_like(hu_slice, dtype=np.uint8)
        leak_flag = 0
        leak_score = 0.0
        hu_calibration_report = {}
        
        if CORE_MINI_AVAILABLE:
            try:
                print("  🔄 Generating inner_abdomen mask via core_mini...")
                
                # Create minimal ds object (pixel spacing)
                # Extract from NIfTI header
                spacing = ct_nib.header.get_zooms()
                
                class SimpleDS:
                    """Minimal DICOM-like object for core_mini compatibility"""
                    def __init__(self, pixel_spacing):
                        self.PixelSpacing = [pixel_spacing[0], pixel_spacing[1]]
                
                ds = SimpleDS(spacing)
                
                # Run inner_abdomen_via_wall from core_mini
                inner_mask, wall_edges, vertebra_mask, center = inner_abdomen_via_wall(hu_slice, ds)
                
                # Convert to uint8 binary mask
                inner_abdomen = (inner_mask > 0).astype(np.uint8)
                
                # ========== LEAK PREVENTION: 4-layer constraint system ==========
                if HU_CALIBRATION_AVAILABLE:
                    try:
                        print("  🔄 Applying posterolateral leak prevention...")
                        
                        # Generate body mask from HU
                        body_mask = ((hu_slice > -500) & (hu_slice < 3000)).astype(np.uint8)
                        body_mask = cv2.morphologyEx(
                            body_mask,
                            cv2.MORPH_CLOSE,
                            np.ones((5, 5), np.uint8)
                        )
                        
                        # Try to load abdominal wall mask from TS
                        abdominal_wall_mask = None
                        wall_candidates = [
                            "abdominal_muscles.nii.gz",
                            "muscle.nii.gz",
                        ]
                        for wall_name in wall_candidates:
                            wall_file = locate_segmentation_file(ts_root, wall_name)
                            if wall_file:
                                try:
                                    wall_nib = nib.load(wall_file)
                                    wall_data = wall_nib.dataobj
                                    if wall_data.shape[2] > l3_idx:
                                        abdominal_wall_mask = (np.asarray(wall_data[:, :, l3_idx]) > 0).astype(np.uint8)
                                        print("  ✅ Loaded abdominal wall mask for leak prevention")
                                        break
                                except Exception as e:
                                    print(f"  ⚠️  Failed to load abdominal wall: {e}")
                        
                        # Apply leak prevention
                        leak_preventer = PosterolateralLeakPrevention()
                        leak_result = leak_preventer.detect_and_correct_leaks(
                            inner_abdomen,
                            body_mask,
                            vertebra_mask,
                            abdominal_wall_mask,
                            pixel_spacing=(spacing[0], spacing[1])
                        )
                        
                        # Use corrected mask
                        if leak_result.correction_applied:
                            # Get corrected mask (original minus leak regions)
                            inner_abdomen_corrected = cv2.bitwise_and(
                                inner_abdomen,
                                cv2.bitwise_not(leak_result.leak_regions)
                            )
                            
                            pixels_removed = inner_abdomen.sum() - inner_abdomen_corrected.sum()
                            print(f"  ✅ Leak prevention removed {pixels_removed} pixels ({leak_result.correction_method})")
                            
                            inner_abdomen = inner_abdomen_corrected
                        
                        leak_flag = 1 if leak_result.leak_detected else 0
                        leak_score = leak_result.leak_score
                        
                        print(f"  ✅ Leak detection: score={leak_score:.2f}, QC={'PASS' if leak_result.qc_pass else 'FAIL'}")
                        
                    except Exception as e:
                        print(f"  ⚠️  Leak prevention failed: {e}")
                        import traceback
                        traceback.print_exc()
                # ================================================================
                
                # ========== HU CALIBRATION for improved VAT ==========
                if HU_CALIBRATION_AVAILABLE and inner_abdomen.sum() > 0:
                    try:
                        print("  🔄 Calibrating HU bands...")
                        
                        calibrator = HUCalibrator()
                        
                        # Calibrate fat band based on inner abdomen ROI
                        calibration = calibrator.calibrate_fat_band(
                            hu_slice,
                            inner_abdomen,
                            protocol_info=None  # TODO: Extract from DICOM if available
                        )
                        
                        # Generate improved VAT using calibrated bands
                        fat_hu_mask = (
                            (hu_slice >= calibration.fat_low) & 
                            (hu_slice <= calibration.fat_high)
                        ).astype(np.uint8)
                        vat_improved = fat_hu_mask & inner_abdomen
                        
                        # Use improved VAT if it has reasonable coverage
                        if vat_improved.sum() > vat.sum() * 0.3:  # At least 30% of original
                            vat = vat_improved
                            print(f"  ✅ VAT improved using calibrated HU bands [{calibration.fat_low:.0f}, {calibration.fat_high:.0f}]: {vat.sum()} pixels")
                        
                        # Generate calibration report
                        hu_calibration_report = calibrator.generate_calibration_report(calibration, case_id)
                        print(f"  ✅ HU calibration: {calibration.adjustment_reason}, confidence={calibration.confidence:.2f}")
                        
                    except Exception as e:
                        print(f"  ⚠️  HU calibration failed: {e}")
                        import traceback
                        traceback.print_exc()
                # =====================================================
                
                # QC: Check for posterolateral leak (legacy simple check)
                # Simple heuristic: inner_abdomen shouldn't extend too far posteriorly
                h, w = inner_abdomen.shape
                center_x, center_y = center
                
                # Check posterior region (behind vertebra)
                posterior_region = inner_abdomen[center_y:, :]
                posterior_ratio = posterior_region.sum() / max(inner_abdomen.sum(), 1)
                
                if posterior_ratio > 0.4:  # More than 40% in posterior region
                    leak_flag = 1
                    print(f"  ⚠️  Possible posterolateral leak detected (posterior ratio: {posterior_ratio:.1%})")
                
                # Improved VAT using inner_abdomen intersection
                fat_hu_mask = ((hu_slice >= -190) & (hu_slice <= -30)).astype(np.uint8)
                vat_improved = fat_hu_mask & inner_abdomen
                
                # If improved VAT has reasonable coverage, use it
                if vat_improved.sum() > vat.sum() * 0.3:  # At least 30% of original
                    vat = vat_improved
                    print(f"  ✅ VAT improved using inner_abdomen intersection: {vat.sum()} pixels")
                
                print(f"  ✅ inner_abdomen mask generated: {inner_abdomen.sum()} pixels, leak_flag={leak_flag}")
                
            except Exception as e:
                print(f"  ⚠️  Failed to generate inner_abdomen mask: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("  ⚠️  core_mini not available, skipping inner_abdomen generation")
        # ====================================================================
        
        # Save HU slice as .npy (for training) and PNG (for visualization)
        np.save(str(case_output_dir / "hu_slice.npy"), hu_slice)
        hu_slice_normalized = np.clip((hu_slice + 100) / 300 * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(str(case_output_dir / "hu_slice.png"), hu_slice_normalized)
        
        # Save masks as .npy (for training) and PNG (for visualization)
        np.save(str(case_output_dir / "psoas_left.npy"), psoas_left)
        np.save(str(case_output_dir / "psoas_right.npy"), psoas_right)
        np.save(str(case_output_dir / "vat.npy"), vat)
        np.save(str(case_output_dir / "inner_abdomen.npy"), inner_abdomen)  # NEW
        
        cv2.imwrite(str(case_output_dir / "psoas_left.png"), psoas_left * 255)
        cv2.imwrite(str(case_output_dir / "psoas_right.png"), psoas_right * 255)
        cv2.imwrite(str(case_output_dir / "vat.png"), vat * 255)
        cv2.imwrite(str(case_output_dir / "inner_abdomen.png"), inner_abdomen * 255)  # NEW
        
        # Save metadata
        metadata = {
            "case_id": case_id,
            "l3_slice_index": int(l3_idx),
            "ct_shape": list(ct_shape),
            "pixel_spacing": [float(spacing[0]), float(spacing[1]), float(spacing[2])],
            "hu_min": float(hu_slice.min()),
            "hu_max": float(hu_slice.max()),
            "hu_mean": float(hu_slice.mean()),
            "psoas_left_pixels": int(psoas_left.sum()),
            "psoas_right_pixels": int(psoas_right.sum()),
            "vat_pixels": int(vat.sum()),
            "inner_abdomen_pixels": int(inner_abdomen.sum()),  # NEW
            "leak_flag": int(leak_flag),  # NEW: QC flag (0=ok, 1=detected)
            "leak_score": float(leak_score),  # NEW: Leak severity (0-1)
        }
        
        # Add HU calibration info if available
        if hu_calibration_report and 'calibration' in hu_calibration_report:
            metadata["hu_calibration"] = hu_calibration_report["calibration"]
            metadata["calibration_log"] = hu_calibration_report.get("calibration_log", [])
        
        metadata_file = case_output_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✅ Teacher labels extracted: {case_output_dir}")
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to extract teacher labels: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_amos_case(amos_file: Path, output_dir: str, fast: bool = True, roi_subset: Optional[str] = None) -> bool:
    """
    Process single AMOS case: TotalSegmentator + teacher extraction.
    
    Args:
        amos_file: Path to AMOS NII.GZ file
        output_dir: Output directory for teacher labels
        fast: Use fast mode
        
    Returns:
        True if successful, False otherwise
    """
    case_id = amos_file.stem.replace(".nii", "")
    print(f"\n{'='*70}")
    print(f"📌 Processing: {case_id}")
    print(f"{'='*70}")
    
    with tempfile.TemporaryDirectory(prefix=f"ts_{case_id}_") as tmpdir:
        ts_success = run_totalsegmentator(str(amos_file), tmpdir, fast=fast, roi_subset=roi_subset)
        
        if not ts_success:
            print(f"⚠️  TotalSegmentator failed for {case_id}, skipping")
            return False
        
        teacher_success = extract_teacher_labels(str(amos_file), tmpdir, output_dir)
        return teacher_success


def main():
    """Main: Process ALL AMOS cases with TotalSegmentator and extract L3 teachers."""
    parser = argparse.ArgumentParser(
        description="Generate teacher labels from AMOS22 with TotalSegmentator (processes ALL files)"
    )
    parser.add_argument("--input-data", type=str, default="/Users/alperenogras/Desktop/amos22/imagesTr",
                        help="Path to AMOS22 imagesTr directory")
    parser.add_argument("--output-dir", type=str, default="./teacher_labels",
                        help="Output directory for teacher labels")
    parser.add_argument("--fast", action="store_true", help="Use fast mode for TotalSegmentator")
    parser.add_argument("--roi-subset", type=str, default="body", help="ROI subset for TotalSegmentator to reduce RAM (e.g., body)")
    parser.add_argument("--start-idx", type=int, default=0, help="Start from this case index (for resuming)")
    parser.add_argument("--max-cases", type=int, default=None, help="Max cases to process (for testing)")
    args = parser.parse_args()
    
    input_dir = Path(args.input_data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find ALL AMOS nii.gz files (not limited by default)
    amos_files = sorted(input_dir.glob("amos_*.nii.gz"))
    
    if not amos_files:
        print(f"❌ No AMOS files found in {input_dir}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"🚀 AMOS22 TEACHER GENERATION (Processing ALL {len(amos_files)} files)")
    print(f"{'='*70}")
    print(f"📁 Input:  {input_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"⚡ Fast mode: {args.fast}")
    print(f"🪄 ROI subset: {args.roi_subset}")
    
    # Apply max_cases limit if specified
    if args.max_cases:
        print(f"⚠️  Limiting to {args.max_cases} cases (--max-cases)")
        amos_files = amos_files[:args.max_cases]
    
    # Apply start index for resuming
    if args.start_idx > 0:
        print(f"⏭️  Starting from index {args.start_idx}")
        amos_files = amos_files[args.start_idx:]
    
    print(f"📊 Will process {len(amos_files)} cases\n")
    
    # Process all cases
    successful = 0
    failed = 0
    
    for i, amos_file in enumerate(amos_files, start=args.start_idx + 1):
        success = process_amos_case(amos_file, str(output_dir), fast=args.fast, roi_subset=args.roi_subset)
        
        if success:
            successful += 1
        else:
            failed += 1
        
        print(f"📈 Progress: {i}/{len(amos_files) + args.start_idx} | Success: {successful} | Failed: {failed}")
    
    # Summary
    print(f"\n{'='*70}")
    print("✅ COMPLETED")
    print(f"{'='*70}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Output directory: {output_dir}")
    print(f"{'='*70}\n")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

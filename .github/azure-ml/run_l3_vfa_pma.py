#!/usr/bin/env python3
"""
Azure ML Pipeline Script: L3 VFA/PMA Batch Processing
- AMOS22 NIfTI ve TotalSegmentator maskelerini işle
- L3 slice'ını tespit et
- Rule-based VFA/PMA hesapla
- DL model (varsa) ile hybrid predictions
- Sonuçları JSON/PNG olarak kaydet
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
import traceback

import numpy as np
import cv2
import SimpleITK as sitk
from tqdm import tqdm

# Kodu workspace'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core_mini import (
        load_hu,
        find_l3_slice_index,
        detect_vertebra_center,
        compute_inner_abdomen_mask,
        compute_vfa,
        estimate_psoas_mask,
        compute_pma,
        get_pixel_area_mm2,
    )
except ImportError:
    print("⚠️ core_mini modülü bulunamadı, fallback mode")
    from l3_vfa_pma_core_v3_7 import (
        load_hu,
        find_l3_slice_index,
        detect_vertebra_center,
        compute_inner_abdomen_mask,
        compute_vfa,
        estimate_psoas_mask,
        compute_pma,
        get_pixel_area_mm2,
    )

try:
    import torch
    from psoas_ml.infer_gui import create_dl_pipeline
    DL_AVAILABLE = True
except Exception as e:
    print(f"⚠️ DL model yükleme başarısız: {e}")
    DL_AVAILABLE = False


class L3VFAPMAProcessor:
    """AMOS22 + TotalSegmentator batch processing"""

    def __init__(self, use_dl: bool = False, model_path: Optional[Path] = None):
        self.use_dl = use_dl and DL_AVAILABLE
        self.dl_pipeline = None

        if self.use_dl and model_path:
            try:
                from psoas_ml.infer_gui import create_dl_pipeline
                self.dl_pipeline = create_dl_pipeline(Path(model_path))
                if self.dl_pipeline:
                    print("✅ DL model yüklendi")
                else:
                    self.use_dl = False
            except Exception as e:
                print(f"⚠️ DL model yüklenemedi: {e}")
                self.use_dl = False

    def process_case(
        self,
        amos_nii_path: Path,
        ts_mask_path: Optional[Path] = None,
        blend_alpha: float = 0.5,
    ) -> Tuple[Dict, Optional[np.ndarray]]:
        """
        Bir AMOS22 vakasını işle

        Args:
            amos_nii_path: AMOS22 NIfTI dosya yolu
            ts_mask_path: TotalSegmentator abdominal_muscles maskesi yolu
            blend_alpha: DL blend factor (0=rule-based, 1=DL only, 0.5=hybrid)

        Returns:
            (results_dict, overlay_image)
        """
        results = {
            "case_id": amos_nii_path.stem.replace(".nii", ""),
            "timestamp": datetime.now().isoformat(),
            "status": "processing",
            "errors": [],
        }

        overlay = None

        try:
            # Load HU volume
            hu_vol, spacing, affine = load_hu(str(amos_nii_path))
            if hu_vol is None:
                results["status"] = "failed"
                results["errors"].append("HU volume yüklenemedi")
                return results, overlay

            # L3 slice tespit et
            z_l3, vb_conf = find_l3_slice_index(hu_vol)
            results["L3_index"] = int(z_l3)
            results["VB_confidence"] = float(vb_conf)

            if vb_conf < 0.5:
                results["status"] = "low_confidence"
                results["errors"].append(f"Düşük vertebra confidence: {vb_conf:.2f}")
                return results, overlay

            # L3 slice'ını çıkart
            hu_l3 = hu_vol[z_l3, :, :]
            original_shape = hu_l3.shape

            # TotalSegmentator maskesi yükle (varsa)
            ts_seg = None
            if ts_mask_path and ts_mask_path.exists():
                try:
                    ts_img = sitk.ReadImage(str(ts_mask_path))
                    ts_vol = sitk.GetArrayFromImage(ts_img)
                    if ts_vol.ndim == 3:
                        ts_seg = ts_vol[z_l3, :, :]
                    else:
                        ts_seg = ts_vol
                except Exception as e:
                    print(f"  ⚠️ TS mask yüklenemedi: {e}")

            # Vertebra merkez tespit
            vb_center, vb_mask = detect_vertebra_center(hu_l3)
            if vb_center is None:
                results["status"] = "vertebra_detection_failed"
                results["errors"].append("Vertebra merkezi tespit edilemedi")
                return results, overlay

            results["vertebra_center"] = [float(vb_center[0]), float(vb_center[1])]

            # Inner abdomen maskesi
            inner_mask = compute_inner_abdomen_mask(hu_l3, ts_seg, vb_center)

            # Pixel area (mm²)
            pixel_area = get_pixel_area_mm2(spacing[:2])

            # VFA hesapla (rule-based)
            vfa_result = compute_vfa(hu_l3, inner_mask, pixel_area)
            vfa_mm2_rule = vfa_result["vfa_mm2"]
            vfa_mask_rule = vfa_result["vfa_mask"]

            # PMA hesapla (rule-based)
            psoas_mask = estimate_psoas_mask(hu_l3, vb_center)
            pma_result = compute_pma(hu_l3, psoas_mask, pixel_area)
            pma_mm2_rule = pma_result["pma_mm2"]
            pma_mask_rule = pma_result["pma_mask"]

            # DL model (varsa) ile hybrid predictions
            vfa_mm2_final = vfa_mm2_rule
            pma_mm2_final = pma_mm2_rule
            vfa_mask_final = vfa_mask_rule
            pma_mask_final = pma_mask_rule

            if self.use_dl and self.dl_pipeline:
                try:
                    # DL predictions
                    vfa_mask_dl, pma_mask_dl, inner_dl = self.dl_pipeline.predict_masks(
                        hu_l3, ts_seg, vb_mask
                    )

                    # Hybrid blend
                    if blend_alpha > 0:
                        vfa_prob_rule = vfa_mask_rule.astype(np.float32)
                        vfa_prob_dl = vfa_mask_dl.astype(np.float32)
                        vfa_mask_final = (
                            (1 - blend_alpha) * vfa_prob_rule + blend_alpha * vfa_prob_dl
                        ) > 0.5

                        pma_prob_rule = pma_mask_rule.astype(np.float32)
                        pma_prob_dl = pma_mask_dl.astype(np.float32)
                        pma_mask_final = (
                            (1 - blend_alpha) * pma_prob_rule + blend_alpha * pma_prob_dl
                        ) > 0.5

                    # Final VFA/PMA hesapla
                    vfa_mm2_final = np.sum(vfa_mask_final) * pixel_area
                    pma_mm2_final = np.sum(pma_mask_final) * pixel_area

                    results["method"] = "hybrid_dl"
                    results["blend_alpha"] = float(blend_alpha)
                except Exception as e:
                    print(f"  ⚠️ DL inference başarısız: {e}")
                    results["method"] = "rule_based"
            else:
                results["method"] = "rule_based"

            # Sonuçlar
            results["VFA_mm2"] = float(vfa_mm2_final)
            results["VFA_cm2"] = float(vfa_mm2_final / 100)
            results["PMA_mm2"] = float(pma_mm2_final)
            results["PMA_cm2"] = float(pma_mm2_final / 100)
            results["VFA_PMA_ratio"] = (
                float(vfa_mm2_final / pma_mm2_final)
                if pma_mm2_final > 0
                else None
            )

            # Overlay oluştur
            overlay = self._create_overlay(
                hu_l3, vfa_mask_final, pma_mask_final, inner_mask, vb_center, vb_mask
            )

            results["status"] = "success"

        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(str(e))
            results["traceback"] = traceback.format_exc()
            print(f"  ❌ İşlem başarısız: {e}")

        return results, overlay

    def _create_overlay(
        self,
        hu_slice: np.ndarray,
        vfa_mask: np.ndarray,
        pma_mask: np.ndarray,
        inner_mask: np.ndarray,
        vb_center: Tuple[float, float],
        vb_mask: np.ndarray,
    ) -> np.ndarray:
        """Görsel overlay oluştur"""
        try:
            # HU normalizasyonu
            hu_norm = np.clip(hu_slice, -150, 250)
            hu_norm = ((hu_norm + 150) / 400 * 255).astype(np.uint8)
            overlay = cv2.cvtColor(hu_norm, cv2.COLOR_GRAY2BGR)

            # Renk kodlaması
            if vfa_mask is not None:
                overlay[vfa_mask > 0] = [0, 0, 255]  # Kırmızı
            if pma_mask is not None:
                overlay[pma_mask > 0] = [255, 0, 0]  # Mavi
            if vb_mask is not None:
                overlay[vb_mask > 0] = [0, 255, 255]  # Sarı

            # Vertebra merkezi
            if vb_center:
                cv2.circle(overlay, tuple(map(int, vb_center)), 5, (255, 255, 0), 2)

            return overlay
        except Exception as e:
            print(f"  ⚠️ Overlay oluşturma başarısız: {e}")
            return None


def find_amos_cases(data_dir: Path) -> list:
    """AMOS22 NIfTI dosyalarını bul"""
    nii_files = list(data_dir.glob("*.nii.gz")) + list(data_dir.glob("*.nii"))
    return sorted(nii_files)


def find_ts_mask(case_id: str, ts_dir: Path) -> Optional[Path]:
    """TotalSegmentator maskesini bul"""
    candidates = [
        ts_dir / case_id / "abdominal_muscles.nii.gz",
        ts_dir / case_id / "abdominal_muscles.nii",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def main(args):
    print("=" * 60)
    print("🚀 Azure ML L3 VFA/PMA Batch Processing")
    print("=" * 60)

    input_dir = Path(args.input_data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"📂 Input: {input_dir}")
    print(f"📂 Output: {output_dir}")

    # Processor oluştur
    processor = L3VFAPMAProcessor(
        use_dl=args.use_dl,
        model_path=Path(args.model_path) if args.model_path else None,
    )

    # AMOS cases bul
    amos_cases = find_amos_cases(input_dir)
    print(f"📦 {len(amos_cases)} AMOS22 vakası bulundu")

    if not amos_cases:
        print("❌ AMOS22 dosyası bulunamadı!")
        return

    # Batch processing
    results_list = []
    successful = 0
    failed = 0

    for idx, amos_path in enumerate(tqdm(amos_cases, desc="Processing"), 1):
        case_id = amos_path.stem.replace(".nii", "")
        print(f"\n[{idx}/{len(amos_cases)}] {case_id}")

        # TS maskesi bul (opsiyonel)
        ts_path = (
            find_ts_mask(case_id, input_dir.parent / "ts_masks")
            if (input_dir.parent / "ts_masks").exists()
            else None
        )

        # Process
        results, overlay = processor.process_case(amos_path, ts_path)
        results_list.append(results)

        # Sonuçları kaydet
        results_json = output_dir / f"{case_id}_results.json"
        with open(results_json, "w") as f:
            json.dump(results, f, indent=2)

        if overlay is not None:
            overlay_path = output_dir / f"{case_id}_overlay.png"
            cv2.imwrite(str(overlay_path), overlay)

        if results["status"] == "success":
            successful += 1
            print(
                f"  ✅ VFA: {results.get('VFA_cm2', 0):.1f} cm² | PMA: {results.get('PMA_cm2', 0):.1f} cm²"
            )
        else:
            failed += 1
            print(f"  ❌ Başarısız: {results.get('status', 'unknown')}")

    # Summary raporu
    summary = {
        "total_cases": len(amos_cases),
        "successful": successful,
        "failed": failed,
        "timestamp": datetime.now().isoformat(),
        "method": processor.use_dl and "hybrid_dl" or "rule_based",
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"✅ İşlem tamamlandı!")
    print(f"  Başarılı: {successful}/{len(amos_cases)}")
    print(f"  Başarısız: {failed}/{len(amos_cases)}")
    print(f"  Yöntem: {summary['method']}")
    print(f"📊 Sonuçlar: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="L3 VFA/PMA batch processing for AMOS22 dataset"
    )
    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="AMOS22 NIfTI dosyalarının bulunduğu dizin",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Sonuçların kaydedileceği dizin",
    )
    parser.add_argument(
        "--use_dl",
        action="store_true",
        help="DL model ile hybrid predictions kullan",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="U-Net checkpoint yolu (varsa)",
    )
    parser.add_argument(
        "--blend_alpha",
        type=float,
        default=0.5,
        help="DL blend factor (0=rule-based, 1=DL only)",
    )

    args = parser.parse_args()
    main(args)

#!/usr/bin/env python3
"""
Gelişmiş L3 VFA/PMA Hesaplama - DICOM İşleme ile

Bu script DICOM görüntülerinden L3 seviyesinde VFA ve PMA hesaplaması yapar.
Psoas maskesi çıkarımı ve fasya sınır tespiti ile en doğru sonuçları üretir.

Kullanım:
    python advanced_l3_vfa_pma.py --input-dicom /path/to/dicom --output-dir ./results
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pydicom
from scipy import ndimage
from skimage import measure


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_settings():
    """Load settings from settings.json."""
    settings_path = Path(__file__).parent / "settings.json"
    if settings_path.exists():
        with open(settings_path, 'r') as f:
            return json.load(f)
    else:
        logger.warning("settings.json not found, using defaults")
        return {}


# Global settings
SETTINGS = load_settings()


class AdvancedL3VFA_PMA:
    """Gelişmiş L3 VFA/PMA hesaplama sınıfı."""

    def __init__(self):
        """Initialize calculator."""
        logger.info("Advanced L3 VFA/PMA calculator initialized")

    def load_dicom_series(self, dicom_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Load DICOM series and extract HU volume.

        Args:
            dicom_path: Path to DICOM directory

        Returns:
            Tuple of (hu_volume, metadata)
        """
        dicom_files = list(Path(dicom_path).glob("*.dcm"))
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {dicom_path}")

        # Sort by slice position (try multiple attributes)
        def get_slice_position(dcm_file):
            ds = pydicom.dcmread(dcm_file)
            # Try SliceLocation first, then ImagePositionPatient, then InstanceNumber
            if hasattr(ds, 'SliceLocation') and ds.SliceLocation is not None:
                return ds.SliceLocation
            elif hasattr(ds, 'ImagePositionPatient') and ds.ImagePositionPatient:
                return ds.ImagePositionPatient[2]  # Z coordinate
            elif hasattr(ds, 'InstanceNumber') and ds.InstanceNumber is not None:
                return ds.InstanceNumber
            else:
                return 0  # Default if no positioning info

        dicom_files.sort(key=get_slice_position)

        slices = []
        metadata = {}

        for dcm_file in dicom_files:
            ds = pydicom.dcmread(dcm_file)

            # Extract HU values
            hu_slice = ds.pixel_array.astype(np.float32)
            
            # Apply rescale with fallbacks
            slope = getattr(ds, 'RescaleSlope', 1.0)
            intercept = getattr(ds, 'RescaleIntercept', 0.0)
            hu_slice = hu_slice * slope + intercept

            slices.append(hu_slice)

            # Store metadata from first slice
            if not metadata:
                # Extract pixel spacing with fallbacks
                pixel_spacing = None
                if hasattr(ds, 'PixelSpacing'):
                    pixel_spacing = ds.PixelSpacing
                elif hasattr(ds, 'NominalScannedPixelSpacing'):
                    pixel_spacing = ds.NominalScannedPixelSpacing
                elif hasattr(ds, 'ImagerPixelSpacing'):
                    pixel_spacing = ds.ImagerPixelSpacing
                else:
                    pixel_spacing = [1.5, 1.5]  # Default
                
                metadata = {
                    'pixel_spacing': pixel_spacing,
                    'slice_thickness': getattr(ds, 'SliceThickness', 5.0),
                    'rows': ds.Rows,
                    'columns': ds.Columns,
                    'study_uid': getattr(ds, 'StudyInstanceUID', 'unknown'),
                    'series_uid': getattr(ds, 'SeriesInstanceUID', 'unknown')
                }

        hu_volume = np.stack(slices, axis=0)
        logger.info(f"Loaded DICOM series: {hu_volume.shape} {len(dicom_files)} slices")

        return hu_volume, metadata

    def find_l3_slice(self, hu_volume: np.ndarray) -> int:
        """
        Find L3 vertebra slice using HU intensity analysis.

        Args:
            hu_volume: 3D HU volume

        Returns:
            L3 slice index
        """
        # Simple heuristic: find slice with highest bone density around center
        bone_scores = []

        for i in range(hu_volume.shape[0]):
            slice_data = hu_volume[i]

            # Bone HU range: >200 HU
            bone_mask = slice_data > 200
            bone_score = np.sum(bone_mask)

            bone_scores.append(bone_score)

        # Find slice with maximum bone content (likely L3 region)
        l3_slice_idx = np.argmax(bone_scores)

        # Ensure it's not at the edges
        l3_slice_idx = np.clip(l3_slice_idx, hu_volume.shape[0] // 4, 3 * hu_volume.shape[0] // 4)

        logger.info(f"L3 slice found at index {l3_slice_idx}")
        return l3_slice_idx

    def segment_vertebra(self, hu_slice: np.ndarray) -> np.ndarray:
        """
        Segment vertebra in L3 slice.

        Args:
            hu_slice: 2D HU slice

        Returns:
            Vertebra mask
        """
        # Bone HU thresholds from settings
        BONE_MIN = SETTINGS.get('VB_HU_MIN', 200)
        BONE_MAX = SETTINGS.get('VB_HU_MAX', 3000)

        # Bone segmentation
        bone_mask = ((hu_slice >= BONE_MIN) & (hu_slice <= BONE_MAX)).astype(np.uint8)

        # Clean up mask
        bone_mask = ndimage.binary_opening(bone_mask, iterations=1)
        bone_mask = ndimage.binary_closing(bone_mask, iterations=2)

        # Find largest connected component (main vertebra)
        labeled = measure.label(bone_mask)
        regions = measure.regionprops(labeled)

        if regions:
            largest_region = max(regions, key=lambda r: r.area)
            vertebra_mask = (labeled == largest_region.label).astype(np.uint8)
        else:
            vertebra_mask = bone_mask

        return vertebra_mask

    def segment_muscle_fat(self, hu_slice: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment muscle and fat tissues using Comp2Comp-inspired methods.

        Enhanced segmentation with:
        - Adaptive HU thresholding
        - Morphological operations
        - Connected components filtering

        Args:
            hu_slice: 2D HU slice

        Returns:
            Tuple of (muscle_mask, fat_mask)
        """
        # Comp2Comp style HU thresholds (more refined)
        MUSCLE_MIN = SETTINGS.get('PSOAS_HU_MIN', -10)  # Comp2Comp uses -10 to 100
        MUSCLE_MAX = SETTINGS.get('PSOAS_HU_MAX', 100)
        FAT_MIN = SETTINGS.get('VFA_HU_LOW', -180)      # Comp2Comp uses -180 to -20
        FAT_MAX = SETTINGS.get('VFA_HU_HIGH', -20)

        # Initial segmentation
        muscle_mask = ((hu_slice >= MUSCLE_MIN) & (hu_slice <= MUSCLE_MAX)).astype(np.uint8)
        fat_mask = ((hu_slice >= FAT_MIN) & (hu_slice <= FAT_MAX)).astype(np.uint8)

        # Comp2Comp style morphological operations
        # Remove small noise with opening
        muscle_mask = ndimage.binary_opening(muscle_mask, structure=np.ones((3,3)), iterations=1)
        fat_mask = ndimage.binary_opening(fat_mask, structure=np.ones((3,3)), iterations=1)

        # Remove small holes with closing
        muscle_mask = ndimage.binary_closing(muscle_mask, structure=np.ones((3,3)), iterations=1)
        fat_mask = ndimage.binary_closing(fat_mask, structure=np.ones((3,3)), iterations=1)

        # Filter connected components (keep only significant regions)
        muscle_mask = self._filter_connected_components(muscle_mask, min_area=50)
        fat_mask = self._filter_connected_components(fat_mask, min_area=100)

        return muscle_mask, fat_mask

    def _filter_connected_components(self, mask: np.ndarray, min_area: int = 50) -> np.ndarray:
        """
        Filter connected components by minimum area.

        Args:
            mask: Binary mask
            min_area: Minimum area threshold

        Returns:
            Filtered mask
        """
        if mask.sum() == 0:
            return mask

        # Label connected components
        labeled_mask = measure.label(mask)
        regions = measure.regionprops(labeled_mask)

        # Keep only regions above minimum area
        filtered_mask = np.zeros_like(mask)
        for region in regions:
            if region.area >= min_area:
                filtered_mask[labeled_mask == region.label] = 1

        return filtered_mask.astype(np.uint8)

    def extract_psoas_muscles(self, muscle_mask: np.ndarray,
                            vertebra_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract left and right psoas muscles based on vertebra position.

        Args:
            muscle_mask: Muscle segmentation mask
            vertebra_mask: Vertebra mask for reference

        Returns:
            Tuple of (psoas_left, psoas_right)
        """
        # Get vertebra centroid
        vertebra_centroid = self._get_centroid(vertebra_mask)

        # Split image into left and right
        height, width = muscle_mask.shape
        midline = vertebra_centroid[1]

        # Ensure midline is within bounds
        midline = np.clip(midline, 1, width - 1)

        left_half = muscle_mask[:, :int(midline)]
        right_half = muscle_mask[:, int(midline):]

        # Extract largest muscle region in each half
        psoas_left = self._extract_largest_region(left_half)
        psoas_right = self._extract_largest_region(right_half)

        # Pad masks back to full width
        psoas_left_full = np.zeros_like(muscle_mask)
        psoas_left_full[:, :int(midline)] = psoas_left

        psoas_right_full = np.zeros_like(muscle_mask)
        psoas_right_full[:, int(midline):] = psoas_right

        return psoas_left_full, psoas_right_full

    def _get_centroid(self, mask: np.ndarray) -> Tuple[int, int]:
        """Get centroid of binary mask."""
        indices = np.where(mask > 0)
        if len(indices[0]) == 0:
            return (mask.shape[0] // 2, mask.shape[1] // 2)
        return (int(np.mean(indices[0])), int(np.mean(indices[1])))

    def _extract_largest_region(self, mask: np.ndarray) -> np.ndarray:
        """Extract largest connected component."""
        if mask.sum() == 0:
            return mask

        labeled = measure.label(mask)
        regions = measure.regionprops(labeled)

        if not regions:
            return mask

        largest_region = max(regions, key=lambda r: r.area)
        largest_mask = (labeled == largest_region.label).astype(np.uint8)

        return largest_mask

    def calculate_fascia_boundary(self, muscle_mask: np.ndarray,
                                fat_mask: np.ndarray) -> np.ndarray:
        """
        Calculate abdominal fascia boundary using Comp2Comp-inspired methods.

        Enhanced boundary detection with:
        - Morphological operations
        - Distance transforms
        - Adaptive boundary finding

        Args:
            muscle_mask: Muscle mask
            fat_mask: Fat mask

        Returns:
            Fascia boundary mask
        """
        # Comp2Comp style fascia boundary detection
        # Find boundary between abdominal wall muscles and visceral fat

        # Create combined abdominal wall mask (all muscles)
        abdominal_wall = muscle_mask.copy()

        # Dilate abdominal wall to create search region
        wall_dilated = ndimage.binary_dilation(abdominal_wall, iterations=5)

        # Find visceral fat within dilated region
        visceral_fat = fat_mask & wall_dilated

        # Remove visceral fat from abdominal wall region to create cavity
        abdominal_cavity = wall_dilated & ~abdominal_wall

        # Fascia boundary is the interface between abdominal wall and cavity
        # Use distance transform to find the actual boundary
        distance_from_wall = ndimage.distance_transform_edt(abdominal_cavity)
        distance_from_fat = ndimage.distance_transform_edt(visceral_fat)

        # Boundary is where distances are minimal (interface)
        boundary_region = (distance_from_wall < 3) & (distance_from_fat < 3) & wall_dilated

        # Clean up boundary
        boundary = ndimage.binary_opening(boundary_region, structure=np.ones((3,3)), iterations=1)
        boundary = ndimage.binary_closing(boundary, structure=np.ones((3,3)), iterations=1)

        # Ensure boundary is connected and forms a reasonable contour
        boundary = self._filter_connected_components(boundary, min_area=20)

        return boundary.astype(np.uint8)

    def calculate_areas(self, psoas_left: np.ndarray, psoas_right: np.ndarray,
                       fascia_boundary: np.ndarray, fat_mask: np.ndarray,
                       pixel_spacing: Tuple[float, float]) -> Dict[str, float]:
        """
        Calculate VFA and PMA areas.

        Args:
            psoas_left: Left psoas mask
            psoas_right: Right psoas mask
            fascia_boundary: Fascia boundary mask
            fat_mask: Total fat mask
            pixel_spacing: Pixel spacing in mm

        Returns:
            Dictionary with area measurements
        """
        pixel_area = pixel_spacing[0] * pixel_spacing[1]

        # PMA (Psoas Muscle Area)
        pma_mask = psoas_left | psoas_right
        pma_pixels = np.sum(pma_mask)
        pma_mm2 = pma_pixels * pixel_area

        # VFA (Visceral Fat Area) - fat inside fascia boundary
        if fascia_boundary.sum() > 0:
            # Fill the area inside fascia boundary
            fascia_filled = ndimage.binary_fill_holes(fascia_boundary)
            vfa_mask = fat_mask & fascia_filled
            vfa_pixels = np.sum(vfa_mask)
            vfa_mm2 = vfa_pixels * pixel_area
        else:
            vfa_mm2 = 0

        # SAT (Subcutaneous Fat Area) - fat outside fascia boundary
        if fascia_boundary.sum() > 0:
            fascia_filled = ndimage.binary_fill_holes(fascia_boundary)
            sat_mask = fat_mask & ~fascia_filled
            sat_pixels = np.sum(sat_mask)
            sat_mm2 = sat_pixels * pixel_area
        else:
            sat_pixels = np.sum(fat_mask)
            sat_mm2 = sat_pixels * pixel_area
            vfa_mm2 = 0

        total_fat_mm2 = vfa_mm2 + sat_mm2

        return {
            'pma_mm2': pma_mm2,
            'vfa_mm2': vfa_mm2,
            'sat_mm2': sat_mm2,
            'total_fat_mm2': total_fat_mm2,
            'pixel_area_mm2': pixel_area,
            'pma_pixels': pma_pixels,
            'vfa_pixels': vfa_pixels if 'vfa_pixels' in locals() else 0,
            'sat_pixels': sat_pixels
        }

    def process_case(self, dicom_path: str, output_dir: Path) -> Dict[str, any]:
        """
        Process a single DICOM case.

        Args:
            dicom_path: Path to DICOM series
            output_dir: Output directory

        Returns:
            Results dictionary
        """
        case_id = Path(dicom_path).name
        logger.info(f"Processing case: {case_id}")

        # Create output directory
        case_output_dir = output_dir / case_id
        case_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Load DICOM series
            hu_volume, metadata = self.load_dicom_series(dicom_path)

            # Find L3 slice
            l3_slice_idx = self.find_l3_slice(hu_volume)
            l3_hu_slice = hu_volume[l3_slice_idx]

            # Segment vertebra
            vertebra_mask = self.segment_vertebra(l3_hu_slice)

            # Segment muscle and fat
            muscle_mask, fat_mask = self.segment_muscle_fat(l3_hu_slice)

            # Extract psoas muscles
            psoas_left, psoas_right = self.extract_psoas_muscles(muscle_mask, vertebra_mask)

            # Calculate fascia boundary
            fascia_boundary = self.calculate_fascia_boundary(muscle_mask, fat_mask)

            # Calculate areas
            areas = self.calculate_areas(
                psoas_left, psoas_right, fascia_boundary, fat_mask,
                metadata['pixel_spacing']
            )

            # Calculate HU statistics
            hu_stats = {
                'hu_min': float(np.min(l3_hu_slice)),
                'hu_max': float(np.max(l3_hu_slice)),
                'hu_mean': float(np.mean(l3_hu_slice)),
                'hu_std': float(np.std(l3_hu_slice))
            }

            # Prepare results
            results = {
                'case_id': case_id,
                'l3_slice_idx': l3_slice_idx,
                'status': 'success',
                **areas,
                'hu_stats': hu_stats,
                'metadata': metadata
            }

            # Create visualization
            self._create_visualization(
                l3_hu_slice, vertebra_mask, psoas_left, psoas_right,
                fascia_boundary, fat_mask, results, case_output_dir
            )

            # Save results
            self._save_results(results, case_output_dir)

            logger.info(f"Successfully processed case {case_id}")
            return results

        except Exception as e:
            logger.error(f"Failed to process case {case_id}: {str(e)}")
            return {
                'case_id': case_id,
                'status': 'error',
                'error': str(e)
            }

    def _create_visualization(self, hu_slice: np.ndarray, vertebra_mask: np.ndarray,
                            psoas_left: np.ndarray, psoas_right: np.ndarray,
                            fascia_boundary: np.ndarray, fat_mask: np.ndarray,
                            results: Dict, output_dir: Path):
        """Create visualization overlay."""
        # Normalize HU for display (-150 to 250 HU range)
        hu_display = np.clip(hu_slice, -150, 250)
        hu_display = ((hu_display + 150) / 400 * 255).astype(np.uint8)

        # Convert to RGB
        overlay = cv2.cvtColor(hu_display, cv2.COLOR_GRAY2RGB)

        # Add color overlays
        # Vertebra: Yellow
        overlay[vertebra_mask > 0] = [255, 255, 0]  # Yellow

        # Psoas muscles: Blue
        psoas_combined = psoas_left | psoas_right
        overlay[psoas_combined > 0] = [255, 0, 0]  # Blue

        # Fascia boundary: Green
        overlay[fascia_boundary > 0] = [0, 255, 0]  # Green

        # Fat: Red (semi-transparent)
        fat_overlay = overlay.copy()
        fat_overlay[fat_mask > 0] = [0, 0, 255]  # Red for fat
        overlay = cv2.addWeighted(overlay, 0.7, fat_overlay, 0.3, 0)

        # Add text annotations
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay, '.1f', (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay, '.1f', (10, 60), font, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay, '.1f', (10, 90), font, 0.7, (255, 255, 255), 2)

        # Save overlay
        overlay_path = output_dir / "l3_vfa_pma_overlay.png"
        cv2.imwrite(str(overlay_path), overlay)

    def _save_results(self, results: Dict, output_dir: Path):
        """Save results to JSON."""
        results_path = output_dir / "l3_vfa_pma_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="Advanced L3 VFA/PMA Calculator")
    parser.add_argument("--input-dicom", required=True,
                       help="Path to DICOM series directory")
    parser.add_argument("--output-dir", default="./advanced_l3_results",
                       help="Output directory")

    args = parser.parse_args()

    # Initialize calculator
    calculator = AdvancedL3VFA_PMA()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process case
    results = calculator.process_case(args.input_dicom, output_dir)

    # Print summary
    if results['status'] == 'success':
        print("\n✅ Advanced L3 VFA/PMA Analysis Complete")
        print(f"Case ID: {results['case_id']}")
        print(".1f")
        print(".1f")
        print(".1f")
        print(".1f")
        print(f"HU Range: {results['hu_stats']['hu_min']:.0f} to {results['hu_stats']['hu_max']:.0f}")
        print(f"Pixel Area: {results['pixel_area_mm2']:.3f} mm²")
    else:
        print(f"❌ Processing failed: {results.get('error', 'Unknown error')}")

    print(f"\nResults saved to: {output_dir / results['case_id']}")


if __name__ == "__main__":
    main()
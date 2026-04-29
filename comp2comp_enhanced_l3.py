#!/usr/bin/env python3
"""
Comp2Comp-Enhanced L3 VFA/PMA Calculator

Bu script Comp2Comp'un gelişmiş segmentation algoritmalarını kullanarak
L3 seviyesinde VFA ve PMA hesaplaması yapar.

Kullanım:
    python comp2comp_enhanced_l3.py --input /path/to/amos.nii.gz --output ./results
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import cv2
import nibabel as nib
import numpy as np
import pydicom
from scipy import ndimage
from skimage import measure, morphology

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


class Comp2CompEnhancedL3Calculator:
    """Comp2Comp-enhanced L3 VFA/PMA calculator."""

    def __init__(self):
        """Initialize calculator."""
        logger.info("Comp2Comp-Enhanced L3 VFA/PMA calculator initialized")

    def load_nifti(self, nifti_path: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Load NIfTI file and extract volume and spacing.

        Args:
            nifti_path: Path to NIfTI file

        Returns:
            Tuple of (volume, spacing)
        """
        img = nib.load(nifti_path)
        volume = img.get_fdata()
        spacing = img.header.get_zooms()

        logger.info(f"Loaded NIfTI: {volume.shape}, spacing: {spacing}")
        return volume, spacing

    def load_dicom(self, dicom_path: str) -> Tuple[np.ndarray, Tuple[float, float], str]:
        """
        Load DICOM file and extract slice and pixel spacing.

        Args:
            dicom_path: Path to DICOM file

        Returns:
            Tuple of (slice_data, pixel_spacing, protocol_name)
        """
        dcm = pydicom.dcmread(dicom_path)
        slice_data = dcm.pixel_array.astype(np.float32)

        # Apply rescale if available
        if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
            slice_data = slice_data * dcm.RescaleSlope + dcm.RescaleIntercept

        # Get pixel spacing
        pixel_spacing = (float(dcm.PixelSpacing[0]), float(dcm.PixelSpacing[1]))

        # Get protocol information
        protocol_name = getattr(dcm, 'ProtocolName', '')
        if not protocol_name:
            protocol_name = getattr(dcm, 'SeriesDescription', '')
        if not protocol_name:
            protocol_name = getattr(dcm, 'StudyDescription', 'Unknown')

        logger.info(f"Loaded DICOM: {slice_data.shape}, spacing: {pixel_spacing}, HU range: {slice_data.min():.1f} to {slice_data.max():.1f}, Protocol: {protocol_name}")
        return slice_data, pixel_spacing, protocol_name

    def find_l3_slice(self, volume: np.ndarray, spacing: Tuple[float, float, float]) -> int:
        """
        Find L3 vertebra slice using simple heuristics.

        Args:
            volume: 3D CT volume
            spacing: Voxel spacing

        Returns:
            L3 slice index
        """
        # Simple heuristic: find slice with most bone-like structures in middle third
        start_slice = volume.shape[2] // 3
        end_slice = 2 * volume.shape[2] // 3

        best_slice = start_slice
        max_bone_pixels = 0

        for z in range(start_slice, end_slice):
            slice_data = volume[:, :, z]
            # Count bone-like voxels (HU > 200)
            bone_pixels = np.sum(slice_data > 200)
            if bone_pixels > max_bone_pixels:
                max_bone_pixels = bone_pixels
                best_slice = z

        logger.info(f"L3 slice found at index: {best_slice}")
        return best_slice

    def segment_muscle_fat_comp2comp_style(self, hu_slice: np.ndarray, protocol_name: str = "") -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment muscle and fat tissues using Comp2Comp-inspired methods with adaptive thresholds.

        Enhanced segmentation with:
        - Adaptive HU thresholding based on CT protocol
        - Morphological operations
        - Connected components filtering

        Args:
            hu_slice: HU slice
            protocol_name: CT protocol name for adaptive thresholding

        Returns:
            Tuple of (muscle_mask, fat_mask)
        """
        # Adaptive HU thresholds based on protocol
        if "COLONOGRAPHY" in protocol_name.upper():
            # Colonography protocol - different HU ranges
            MUSCLE_MIN = SETTINGS.get('PSOAS_HU_MIN', -10) - 20  # More lenient
            MUSCLE_MAX = SETTINGS.get('PSOAS_HU_MAX', 100) + 50
            FAT_MIN = SETTINGS.get('VFA_HU_LOW', -180) - 50     # Lower fat threshold
            FAT_MAX = SETTINGS.get('VFA_HU_HIGH', -20) + 20
        else:
            # Standard abdominal CT
            MUSCLE_MIN = SETTINGS.get('PSOAS_HU_MIN', -10)
            MUSCLE_MAX = SETTINGS.get('PSOAS_HU_MAX', 100)
            FAT_MIN = SETTINGS.get('VFA_HU_LOW', -180)
            FAT_MAX = SETTINGS.get('VFA_HU_HIGH', -20)

        # Initial segmentation
        muscle_mask = ((hu_slice >= MUSCLE_MIN) & (hu_slice <= MUSCLE_MAX)).astype(np.uint8)
        fat_mask = ((hu_slice >= FAT_MIN) & (hu_slice <= FAT_MAX)).astype(np.uint8)

        # Enhanced morphological operations for DICOMNET data
        # Remove small noise with opening (larger kernel for robustness)
        muscle_mask = ndimage.binary_opening(muscle_mask, structure=np.ones((5,5)), iterations=1)
        fat_mask = ndimage.binary_opening(fat_mask, structure=np.ones((5,5)), iterations=1)

        # Remove small holes with closing
        muscle_mask = ndimage.binary_closing(muscle_mask, structure=np.ones((3,3)), iterations=1)
        fat_mask = ndimage.binary_closing(fat_mask, structure=np.ones((3,3)), iterations=1)

        # Filter connected components (keep only significant regions)
        muscle_mask = self._filter_connected_components(muscle_mask, min_area=100)  # Larger min area
        fat_mask = self._filter_connected_components(fat_mask, min_area=200)       # Larger min area

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

    def segment_vertebra(self, hu_slice: np.ndarray) -> np.ndarray:
        """
        Segment vertebra using HU thresholding.

        Args:
            hu_slice: HU slice

        Returns:
            Vertebra mask
        """
        # Bone HU range
        bone_mask = ((hu_slice >= 200) & (hu_slice <= 1500)).astype(np.uint8)

        # Clean up
        bone_mask = ndimage.binary_opening(bone_mask, iterations=2)

        # Get largest connected component (vertebra)
        labeled = measure.label(bone_mask)
        regions = measure.regionprops(labeled)

        if regions:
            largest_region = max(regions, key=lambda r: r.area)
            vertebra_mask = (labeled == largest_region.label).astype(np.uint8)
        else:
            vertebra_mask = bone_mask

        return vertebra_mask

    def extract_psoas_muscles(self, muscle_mask: np.ndarray,
                            vertebra_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract left and right psoas muscles based on vertebra position with advanced anatomical approach.

        Enhanced extraction:
        - Use vertebra bounding box for precise anatomical regions
        - Apply size and shape filtering for psoas muscles
        - Use morphological refinement for better accuracy

        Args:
            muscle_mask: Muscle segmentation mask
            vertebra_mask: Vertebra mask for reference

        Returns:
            Tuple of (psoas_left, psoas_right)
        """
        height, width = muscle_mask.shape

        # Get vertebra bounding box for anatomical reference
        vert_props = measure.regionprops(measure.label(vertebra_mask))[0]
        vert_bbox = vert_props.bbox  # (min_row, min_col, max_row, max_col)
        vert_centroid = vert_props.centroid
        vert_x = int(vert_centroid[1])

        # Define anatomical search regions based on vertebra position
        # Psoas muscles are typically 1-2 vertebra widths lateral to spine
        vert_width = vert_bbox[3] - vert_bbox[1]
        vert_height = vert_bbox[2] - vert_bbox[0]

        search_margin_x = int(vert_width * 1.5)  # 1.5x vertebra width
        search_margin_y = int(vert_height * 0.8)  # 0.8x vertebra height

        # Left psoas region: left of vertebra centroid
        left_x_min = max(0, vert_x - search_margin_x)
        left_x_max = max(0, vert_x - int(vert_width * 0.3))

        # Right psoas region: right of vertebra centroid
        right_x_min = min(width, vert_x + int(vert_width * 0.3))
        right_x_max = min(width, vert_x + search_margin_x)

        # Y range: around vertebra with anatomical margin
        y_min = max(0, vert_bbox[0] - int(search_margin_y * 0.5))
        y_max = min(height, vert_bbox[2] + int(search_margin_y * 0.5))

        # Extract muscle candidates in anatomical regions
        left_candidates = muscle_mask[y_min:y_max, left_x_min:left_x_max]
        right_candidates = muscle_mask[y_min:y_max, right_x_min:right_x_max]

        # Apply morphological refinement to remove noise
        left_candidates = morphology.binary_opening(left_candidates, morphology.disk(2))
        right_candidates = morphology.binary_opening(right_candidates, morphology.disk(2))

        # Find and filter psoas candidates by anatomical criteria
        psoas_left = self._filter_psoas_candidates(left_candidates, vert_height, vert_width)
        psoas_right = self._filter_psoas_candidates(right_candidates, vert_height, vert_width)

        # Create full-size masks
        left_full = np.zeros_like(muscle_mask)
        left_full[y_min:y_max, left_x_min:left_x_max] = psoas_left

        right_full = np.zeros_like(muscle_mask)
        right_full[y_min:y_max, right_x_min:right_x_max] = psoas_right

        return left_full, right_full

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

    def _filter_psoas_candidates(self, candidates: np.ndarray, vert_height: int, vert_width: int) -> np.ndarray:
        """
        Filter muscle candidates to identify psoas muscles based on anatomical criteria.

        Args:
            candidates: Binary mask of muscle candidates in psoas region
            vert_height: Height of vertebra for size reference
            vert_width: Width of vertebra for size reference

        Returns:
            Filtered mask containing likely psoas muscle
        """
        if candidates.sum() == 0:
            return candidates

        # Label connected components
        labeled = measure.label(candidates)
        regions = measure.regionprops(labeled)

        if not regions:
            return candidates

        # Filter criteria for psoas muscles:
        # 1. Size: Should be substantial but not too large (relative to vertebra)
        # 2. Shape: Should be reasonably compact (circularity)
        # 3. Position: Should be in expected anatomical location

        min_area = int(vert_height * vert_width * 0.1)  # At least 10% of vertebra area
        max_area = int(vert_height * vert_width * 3.0)  # At most 3x vertebra area

        valid_regions = []
        for region in regions:
            # Size filter
            if not (min_area <= region.area <= max_area):
                continue

            # Shape filter: prefer more circular/elongated shapes typical of psoas
            # Psoas muscles are typically elongated vertically
            bbox = region.bbox
            bbox_height = bbox[2] - bbox[0]
            bbox_width = bbox[3] - bbox[1]

            # Aspect ratio filter: psoas should be taller than wide
            aspect_ratio = bbox_height / max(bbox_width, 1)
            if aspect_ratio < 1.2:  # Should be at least 1.2x taller than wide
                continue

            # Circularity filter (compactness)
            perimeter = region.perimeter
            area = region.area
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            if circularity < 0.3:  # Too irregular shape
                continue

            valid_regions.append(region)

        # Select best candidate (largest valid region)
        if valid_regions:
            best_region = max(valid_regions, key=lambda r: r.area)
            return (labeled == best_region.label).astype(np.uint8)

        # Fallback: return largest region if no valid candidates found
        largest_region = max(regions, key=lambda r: r.area)
        return (labeled == largest_region.label).astype(np.uint8)

    def calculate_fascia_boundary_comp2comp_style(self, muscle_mask: np.ndarray,
                                fat_mask: np.ndarray) -> np.ndarray:
        """
        Calculate abdominal fascia boundary using advanced morphological approach.

        Enhanced boundary detection with:
        - Abdominal wall detection
        - Morphological operations for refinement
        - Distance-based boundary finding
        - Connected component filtering

        Args:
            muscle_mask: Muscle mask
            fat_mask: Fat mask

        Returns:
            Fascia boundary mask (inner abdominal region)
        """
        # Find abdominal wall by dilating muscle regions
        abdominal_wall = ndimage.binary_dilation(muscle_mask, iterations=2)

        # Fill small holes in abdominal wall
        abdominal_wall = ndimage.binary_fill_holes(abdominal_wall)

        # Create distance map from abdominal wall
        distance_from_wall = ndimage.distance_transform_edt(~abdominal_wall)

        # Find visceral fat regions
        visceral_fat = fat_mask & (distance_from_wall < 15)  # Within 15 pixels of wall

        # Create refined abdominal cavity mask
        # Start with dilated wall and remove areas far from muscle
        cavity_candidate = abdominal_wall.copy()

        # Erode to get inner region, but ensure it contains visceral fat
        for i in range(1, 5):  # Try different erosion levels
            eroded = ndimage.binary_erosion(abdominal_wall, iterations=i)
            if (eroded & visceral_fat).sum() > (fat_mask.sum() * 0.3):  # At least 30% of fat
                cavity_candidate = eroded
                break

        # Ensure fascia boundary contains significant visceral fat
        fascia_boundary = cavity_candidate & (visceral_fat.sum() > 0).astype(np.uint8)

        # If no boundary found, fall back to simpler approach
        if fascia_boundary.sum() == 0:
            fascia_boundary = ndimage.binary_erosion(abdominal_wall, iterations=1)
            fascia_boundary = fascia_boundary & fat_mask.astype(np.uint8)

        # Clean up with morphological operations
        fascia_boundary = ndimage.binary_opening(fascia_boundary, structure=np.ones((3,3)), iterations=1)
        fascia_boundary = ndimage.binary_closing(fascia_boundary, structure=np.ones((3,3)), iterations=1)

        # Filter to keep only the largest connected component
        if fascia_boundary.sum() > 0:
            labeled = measure.label(fascia_boundary)
            regions = measure.regionprops(labeled)
            if regions:
                largest_region = max(regions, key=lambda r: r.area)
                fascia_boundary = (labeled == largest_region.label).astype(np.uint8)

        return fascia_boundary.astype(np.uint8)

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
            pixel_spacing: Pixel spacing (mm)
            vfa_only: If True, skip PMA calculation

        Returns:
            Dictionary with calculated areas
        """
        pixel_area = pixel_spacing[0] * pixel_spacing[1]

        # PMA calculation
        pma_pixels = psoas_left.sum() + psoas_right.sum()
        pma_mm2 = pma_pixels * pixel_area

        # VFA calculation (fat within fascia boundary)
        vfa_mask = fat_mask & fascia_boundary.astype(bool)
        vfa_pixels = vfa_mask.sum()
        vfa_mm2 = vfa_pixels * pixel_area

        # SAT calculation (subcutaneous fat - outside fascia)
        sat_mask = fat_mask & ~fascia_boundary.astype(bool)
        sat_pixels = sat_mask.sum()
        sat_mm2 = sat_pixels * pixel_area

        # Total fat
        total_fat_pixels = fat_mask.sum()
        total_fat_mm2 = total_fat_pixels * pixel_area

        return {
            'pma_mm2': pma_mm2,
            'vfa_mm2': vfa_mm2,
            'sat_mm2': sat_mm2,
            'total_fat_mm2': total_fat_mm2,
            'pixel_area_mm2': pixel_area,
            'pma_pixels': pma_pixels,
            'vfa_pixels': vfa_pixels,
            'sat_pixels': sat_pixels,
            'total_fat_pixels': total_fat_pixels
        }

    def create_visualization(self, hu_slice: np.ndarray, muscle_mask: np.ndarray,
                           fat_mask: np.ndarray, vertebra_mask: np.ndarray,
                           psoas_left: np.ndarray, psoas_right: np.ndarray,
                           fascia_boundary: np.ndarray, results: Dict = None,
                           protocol_name: str = "Unknown") -> np.ndarray:
        """
        Create high-quality visualization overlay with improved colors and contrast.

        Args:
            hu_slice: Original HU slice
            muscle_mask: Muscle segmentation mask
            fat_mask: Fat mask
            vertebra_mask: Vertebra mask
            psoas_left: Left psoas mask
            psoas_right: Right psoas mask
            fascia_boundary: Fascia boundary mask

        Returns:
            RGB visualization image
        """
        # Enhanced HU normalization for better contrast
        # Use wider window for DICOMNET data
        hu_min, hu_max = -200, 400  # Abdominal CT window
        hu_norm = np.clip(hu_slice, hu_min, hu_max)
        hu_norm = ((hu_norm - hu_min) / (hu_max - hu_min) * 255).astype(np.uint8)

        # Convert to RGB
        vis_image = cv2.cvtColor(hu_norm, cv2.COLOR_GRAY2RGB)

        # Improved color scheme with better visibility
        # Background CT image (grayscale with slight blue tint)
        vis_image = vis_image.astype(np.float32)

        # Vertebra: Bright yellow with high opacity
        vertebra_overlay = np.zeros_like(vis_image)
        vertebra_overlay[vertebra_mask > 0] = [255, 255, 0]  # Bright yellow
        vis_image = cv2.addWeighted(vis_image, 1.0, vertebra_overlay, 0.8, 0)

        # Muscle: Red with medium opacity
        muscle_overlay = np.zeros_like(vis_image)
        muscle_overlay[muscle_mask > 0] = [255, 0, 0]  # Red
        vis_image = cv2.addWeighted(vis_image, 1.0, muscle_overlay, 0.6, 0)

        # Fat: Green with medium opacity
        fat_overlay = np.zeros_like(vis_image)
        fat_overlay[fat_mask > 0] = [0, 255, 0]  # Green
        vis_image = cv2.addWeighted(vis_image, 1.0, fat_overlay, 0.5, 0)

        # Psoas muscles: Bright blue with high opacity
        psoas_combined = psoas_left | psoas_right
        psoas_overlay = np.zeros_like(vis_image)
        psoas_overlay[psoas_combined > 0] = [0, 100, 255]  # Bright blue
        vis_image = cv2.addWeighted(vis_image, 1.0, psoas_overlay, 0.9, 0)

        # Fascia boundary: Magenta with high opacity
        fascia_overlay = np.zeros_like(vis_image)
        fascia_overlay[fascia_boundary > 0] = [255, 0, 255]  # Magenta
        vis_image = cv2.addWeighted(vis_image, 1.0, fascia_overlay, 0.8, 0)

        # Convert back to uint8
        vis_image = np.clip(vis_image, 0, 255).astype(np.uint8)

        # Add measurement annotations
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # Create text background
        overlay_h, overlay_w = vis_image.shape[:2]

        # Semi-transparent background for text
        text_bg = vis_image.copy()
        cv2.rectangle(text_bg, (10, 10), (320, 120), (0, 0, 0), -1)
        vis_image = cv2.addWeighted(vis_image, 0.7, text_bg, 0.3, 0)

        # Add measurements text
        cv2.putText(vis_image, f"PMA: {results.get('pma_mm2', 0):.1f} mm²", (15, 35),
                   font, font_scale, (255, 255, 255), thickness)
        cv2.putText(vis_image, f"VFA: {results.get('vfa_mm2', 0):.1f} mm²", (15, 60),
                   font, font_scale, (255, 255, 255), thickness)
        cv2.putText(vis_image, f"SAT: {results.get('sat_mm2', 0):.1f} mm²", (15, 85),
                   font, font_scale, (255, 255, 255), thickness)
        cv2.putText(vis_image, f"Protocol: {protocol_name}", (15, 110),
                   font, 0.4, (200, 200, 200), 1)

        # Add color legend
        legend_y = overlay_h - 120
        cv2.putText(vis_image, "LEGEND:", (15, legend_y),
                   font, 0.5, (255, 255, 255), 2)

        # Color indicators
        cv2.rectangle(vis_image, (15, legend_y + 10), (35, legend_y + 30), (255, 255, 0), -1)
        cv2.putText(vis_image, "Vertebra", (45, legend_y + 25),
                   font, 0.4, (255, 255, 255), 1)

        cv2.rectangle(vis_image, (15, legend_y + 35), (35, legend_y + 55), (255, 0, 0), -1)
        cv2.putText(vis_image, "Muscle", (45, legend_y + 50),
                   font, 0.4, (255, 255, 255), 1)

        cv2.rectangle(vis_image, (15, legend_y + 60), (35, legend_y + 80), (0, 255, 0), -1)
        cv2.putText(vis_image, "Fat", (45, legend_y + 75),
                   font, 0.4, (255, 255, 255), 1)

        cv2.rectangle(vis_image, (15, legend_y + 85), (35, legend_y + 105), (0, 100, 255), -1)
        cv2.putText(vis_image, "Psoas", (45, legend_y + 100),
                   font, 0.4, (255, 255, 255), 1)

        return vis_image

    def process_case(self, input_path: str, output_dir: str) -> Dict:
        """
        Process a single case (NIfTI or DICOM).

        Args:
            input_path: Path to NIfTI file or DICOM file
            output_dir: Output directory
            vfa_only: If True, only calculate VFA (skip PMA)

        Returns:
            Results dictionary
        """
        logger.info(f"Processing: {input_path}")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        input_path_obj = Path(input_path)

        if input_path_obj.suffix.lower() == '.gz' and input_path_obj.name.endswith('.nii.gz'):
            # NIfTI file
            volume, spacing = self.load_nifti(input_path)
            l3_slice_idx = self.find_l3_slice(volume, spacing)
            hu_slice = volume[:, :, l3_slice_idx]
            pixel_spacing = spacing[:2]
            protocol_name = "NIfTI"
        elif input_path_obj.suffix.lower() in ['.dcm', '.dicom']:
            # DICOM file
            hu_slice, pixel_spacing, protocol_name = self.load_dicom(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path_obj.suffix}")

        # Segment tissues with protocol-aware segmentation
        muscle_mask, fat_mask = self.segment_muscle_fat_comp2comp_style(hu_slice, protocol_name)

        # Segment vertebra
        vertebra_mask = self.segment_vertebra(hu_slice)

        # Extract psoas muscles
        psoas_left, psoas_right = self.extract_psoas_muscles(muscle_mask, vertebra_mask)

        # Calculate fascia boundary
        fascia_boundary = self.calculate_fascia_boundary_comp2comp_style(muscle_mask, fat_mask)

        # Calculate areas
        results = self.calculate_areas(psoas_left, psoas_right, fascia_boundary, fat_mask, pixel_spacing)

        # Create visualization
        vis_image = self.create_visualization(
            hu_slice, muscle_mask, fat_mask, vertebra_mask,
            psoas_left, psoas_right, fascia_boundary, results, protocol_name
        )

        # Save results
        case_name = input_path_obj.stem

        # Save visualization
        vis_path = output_path / f"{case_name}_l3_vfa_pma.png"
        cv2.imwrite(str(vis_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

        # Save masks as NIfTI (for DICOM, create single slice volume)
        if input_path_obj.suffix.lower() in ['.dcm', '.dicom']:
            mask_volume = np.zeros((*hu_slice.shape, 5), dtype=np.uint8)
        else:
            mask_volume = np.zeros((*hu_slice.shape, 5), dtype=np.uint8)

        mask_volume[:, :, 0] = muscle_mask
        mask_volume[:, :, 1] = fat_mask
        mask_volume[:, :, 2] = vertebra_mask
        mask_volume[:, :, 3] = psoas_left | psoas_right
        mask_volume[:, :, 4] = fascia_boundary

        mask_img = nib.Nifti1Image(mask_volume, np.eye(4))
        mask_path = output_path / f"{case_name}_masks.nii.gz"
        nib.save(mask_img, str(mask_path))

        # Save results as JSON
        results_path = output_path / f"{case_name}_results.json"
        # Convert numpy types to Python types for JSON serialization
        json_results = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v for k, v in results.items()}
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)

        logger.info(f"Results saved to: {output_path}")
        return results


def main():
    parser = argparse.ArgumentParser(description="Comp2Comp-Enhanced L3 VFA/PMA Calculator")
    parser.add_argument("--input", "-i", required=True, help="Input NIfTI file path")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--batch", "-b", action="store_true", help="Process all NIfTI files in input directory")

    args = parser.parse_args()

    calculator = Comp2CompEnhancedL3Calculator()

    if args.batch:
        # Batch processing
        input_path = Path(args.input)
        if input_path.is_dir():
            # Support both NIfTI and DICOM files
            nifti_files = list(input_path.glob("*.nii.gz"))
            dicom_files = list(input_path.glob("*.dcm")) + list(input_path.glob("*.dicom"))
            all_files = nifti_files + dicom_files

            logger.info(f"Found {len(nifti_files)} NIfTI and {len(dicom_files)} DICOM files for batch processing")

            for input_file in all_files:
                try:
                    results = calculator.process_case(str(input_file), args.output)
                    logger.info(f"Processed {input_file.name}: PMA={results['pma_mm2']:.1f}, VFA={results['vfa_mm2']:.1f}")
                except Exception as e:
                    logger.error(f"Failed to process {input_file.name}: {e}")
        else:
            logger.error("Batch mode requires input directory")
    else:
        # Single file processing
        results = calculator.process_case(args.input, args.output)

        print("\n📊 Results:")
        print(f"PMA: {results['pma_mm2']:.2f} mm²")
        print(f"VFA: {results['vfa_mm2']:.2f} mm²")
        print(f"SAT: {results['sat_mm2']:.2f} mm²")
        print(f"Total Fat: {results['total_fat_mm2']:.2f} mm²")


if __name__ == "__main__":
    main()
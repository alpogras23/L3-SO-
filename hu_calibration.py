#!/usr/bin/env python3
"""
HU Band Calibration for VFA/PMA Accuracy

This module provides automatic HU band calibration to handle:
- Different CT protocols (kVp, mAs, contrast phase)
- Scanner variations
- Reconstruction kernel differences

Critical for measurement accuracy: Small HU band shifts directly affect VFA values.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class HUBandCalibration:
    """Results of HU band calibration"""
    fat_low: float
    fat_high: float
    muscle_low: float
    muscle_high: float
    fat_pixel_ratio: float
    muscle_pixel_ratio: float
    adjustment_reason: str
    confidence: float  # 0-1, higher is better


class HUCalibrator:
    """
    Automatic HU band calibration for fat and muscle segmentation.
    
    Approach:
    1. Start with anatomical default bands
    2. Analyze HU histogram in ROI
    3. Apply adaptive corrections based on dataset characteristics
    4. Report calibration for reproducibility
    """
    
    # Default HU bands (literature values)
    DEFAULT_FAT_LOW = -190
    DEFAULT_FAT_HIGH = -30
    DEFAULT_MUSCLE_LOW = -29
    DEFAULT_MUSCLE_HIGH = 150
    
    # Adaptive thresholds
    MIN_FAT_RATIO = 0.01  # At least 1% fat pixels expected in VFA ROI
    MAX_FAT_RATIO = 0.70  # At most 70% fat pixels (sanity check)
    
    def __init__(self):
        self.calibration_log = []
    
    def calibrate_fat_band(
        self,
        hu_slice: np.ndarray,
        inner_abdomen_mask: np.ndarray,
        protocol_info: Optional[Dict] = None
    ) -> HUBandCalibration:
        """
        Calibrate fat HU band based on inner abdomen ROI histogram.
        
        Args:
            hu_slice: HU values for L3 slice
            inner_abdomen_mask: Binary mask of inner abdomen (VFA ROI)
            protocol_info: Optional CT protocol metadata (contrast phase, kVp, etc.)
            
        Returns:
            HUBandCalibration with adjusted bands and reasoning
        """
        # Start with defaults
        fat_low = self.DEFAULT_FAT_LOW
        fat_high = self.DEFAULT_FAT_HIGH
        muscle_low = self.DEFAULT_MUSCLE_LOW
        muscle_high = self.DEFAULT_MUSCLE_HIGH
        
        adjustment_reason = "default"
        confidence = 1.0
        
        # Extract HU values in ROI
        roi_hu = hu_slice[inner_abdomen_mask > 0]
        
        if len(roi_hu) == 0:
            return HUBandCalibration(
                fat_low, fat_high, muscle_low, muscle_high,
                0.0, 0.0, "empty_roi", 0.0
            )
        
        # Calculate histogram
        roi_min, roi_max = roi_hu.min(), roi_hu.max()
        
        # Test default band
        fat_pixels = ((roi_hu >= fat_low) & (roi_hu <= fat_high)).sum()
        fat_ratio = fat_pixels / len(roi_hu)
        
        muscle_pixels = ((roi_hu >= muscle_low) & (roi_hu <= muscle_high)).sum()
        muscle_ratio = muscle_pixels / len(roi_hu)
        
        # Adaptive corrections based on fat ratio
        if fat_ratio < self.MIN_FAT_RATIO:
            # Too few fat pixels - band might be too narrow
            # Check if there's tissue in -50 to -10 range (contrast/phase shift)
            extended_fat = ((roi_hu >= fat_low) & (roi_hu <= -10)).sum()
            extended_ratio = extended_fat / len(roi_hu)
            
            if extended_ratio > fat_ratio * 2:  # Significant improvement
                fat_high = -10
                fat_ratio = extended_ratio
                adjustment_reason = "fat_pixels_low_extended_upper"
                confidence = 0.8
                self.calibration_log.append(
                    f"Fat band extended to -10 HU (original ratio: {fat_ratio:.1%}, new: {extended_ratio:.1%})"
                )
        
        elif fat_ratio > self.MAX_FAT_RATIO:
            # Too many fat pixels - might be including non-fat tissue
            # Tighten upper bound
            stricter_fat = ((roi_hu >= fat_low) & (roi_hu <= -50)).sum()
            stricter_ratio = stricter_fat / len(roi_hu)
            
            if stricter_ratio > self.MIN_FAT_RATIO:
                fat_high = -50
                fat_ratio = stricter_ratio
                adjustment_reason = "fat_pixels_high_tightened_upper"
                confidence = 0.9
                self.calibration_log.append(
                    f"Fat band tightened to -50 HU (original ratio: {fat_ratio:.1%}, new: {stricter_ratio:.1%})"
                )
        
        # Check for contrast phase (arterial/venous)
        if protocol_info and protocol_info.get("contrast_phase"):
            phase = protocol_info["contrast_phase"].lower()
            if "arterial" in phase or "venous" in phase:
                # Contrast can shift fat HU upward slightly
                if fat_high < -20:
                    fat_high = -20
                    adjustment_reason += "_contrast_adjusted"
                    confidence *= 0.95
                    self.calibration_log.append(
                        f"Contrast phase detected ({phase}), fat upper bound adjusted to -20 HU"
                    )
        
        # Muscle band rarely needs adjustment, but check for sanity
        if muscle_ratio < 0.05:  # Less than 5% muscle seems wrong
            confidence *= 0.8
            self.calibration_log.append(
                f"Warning: Low muscle ratio {muscle_ratio:.1%} in ROI"
            )
        
        return HUBandCalibration(
            fat_low=fat_low,
            fat_high=fat_high,
            muscle_low=muscle_low,
            muscle_high=muscle_high,
            fat_pixel_ratio=fat_ratio,
            muscle_pixel_ratio=muscle_ratio,
            adjustment_reason=adjustment_reason,
            confidence=confidence
        )
    
    def calibrate_muscle_band(
        self,
        hu_slice: np.ndarray,
        psoas_mask: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Calibrate muscle HU band based on psoas ROI.
        
        Args:
            hu_slice: HU values for L3 slice
            psoas_mask: Binary mask of psoas muscles
            
        Returns:
            Tuple of (muscle_low, muscle_high, mean_hu)
        """
        psoas_hu = hu_slice[psoas_mask > 0]
        
        if len(psoas_hu) == 0:
            return self.DEFAULT_MUSCLE_LOW, self.DEFAULT_MUSCLE_HIGH, 0.0
        
        # Psoas HU statistics
        mean_hu = float(np.mean(psoas_hu))
        std_hu = float(np.std(psoas_hu))
        
        # Expected range for healthy psoas: 30-60 HU
        # If mean is significantly different, might indicate:
        # - Fatty infiltration (lower)
        # - Contrast enhancement (higher)
        # - Scanner calibration difference
        
        muscle_low = self.DEFAULT_MUSCLE_LOW
        muscle_high = self.DEFAULT_MUSCLE_HIGH
        
        if mean_hu < 20:  # Low muscle HU
            self.calibration_log.append(
                f"Warning: Low psoas HU mean {mean_hu:.1f} (expected 30-60). "
                "Possible fatty infiltration or scanner difference."
            )
        elif mean_hu > 80:  # High muscle HU
            self.calibration_log.append(
                f"Warning: High psoas HU mean {mean_hu:.1f} (expected 30-60). "
                "Possible contrast enhancement."
            )
            # Extend upper bound if enhanced
            muscle_high = 180
        
        return muscle_low, muscle_high, mean_hu
    
    def generate_calibration_report(
        self,
        calibration: HUBandCalibration,
        case_id: str
    ) -> Dict:
        """
        Generate calibration report for reproducibility.
        
        Args:
            calibration: HUBandCalibration result
            case_id: Case identifier
            
        Returns:
            Dict with calibration details
        """
        report = {
            "case_id": case_id,
            "calibration": {
                "fat_band": [calibration.fat_low, calibration.fat_high],
                "muscle_band": [calibration.muscle_low, calibration.muscle_high],
                "fat_pixel_ratio": f"{calibration.fat_pixel_ratio:.3f}",
                "muscle_pixel_ratio": f"{calibration.muscle_pixel_ratio:.3f}",
                "adjustment_reason": calibration.adjustment_reason,
                "confidence": f"{calibration.confidence:.2f}"
            },
            "calibration_log": self.calibration_log.copy()
        }
        
        return report


def apply_calibrated_fat_mask(
    hu_slice: np.ndarray,
    inner_abdomen_mask: np.ndarray,
    calibration: HUBandCalibration
) -> np.ndarray:
    """
    Apply calibrated fat HU band to generate VFA mask.
    
    Args:
        hu_slice: HU values
        inner_abdomen_mask: Inner abdomen ROI
        calibration: Calibrated HU bands
        
    Returns:
        Binary VFA mask
    """
    # Apply calibrated fat band
    fat_hu_mask = (
        (hu_slice >= calibration.fat_low) & 
        (hu_slice <= calibration.fat_high)
    ).astype(np.uint8)
    
    # Intersect with inner abdomen (critical for VFA accuracy)
    vfa_mask = fat_hu_mask & inner_abdomen_mask
    
    return vfa_mask


def apply_calibrated_muscle_mask(
    hu_slice: np.ndarray,
    psoas_mask: np.ndarray,
    muscle_low: float,
    muscle_high: float
) -> np.ndarray:
    """
    Apply calibrated muscle HU band for QC.
    
    Args:
        hu_slice: HU values
        psoas_mask: Psoas ROI
        muscle_low: Lower HU bound
        muscle_high: Upper HU bound
        
    Returns:
        Binary muscle mask (for QC verification)
    """
    muscle_hu_mask = (
        (hu_slice >= muscle_low) & 
        (hu_slice <= muscle_high)
    ).astype(np.uint8)
    
    # Intersect with psoas (should match well if calibration correct)
    muscle_qc_mask = muscle_hu_mask & psoas_mask
    
    return muscle_qc_mask


# Example usage in teacher generation or inference:
"""
# In step1_real_teachers_all.py or inference pipeline:

from hu_calibration import HUCalibrator, apply_calibrated_fat_mask

calibrator = HUCalibrator()

# Calibrate fat band
calibration = calibrator.calibrate_fat_band(
    hu_slice,
    inner_abdomen_mask,
    protocol_info={"contrast_phase": "venous"}  # Optional
)

# Generate VFA mask with calibrated bands
vfa_mask = apply_calibrated_fat_mask(hu_slice, inner_abdomen_mask, calibration)

# Generate calibration report
report = calibrator.generate_calibration_report(calibration, case_id)

# Save to metadata
metadata["hu_calibration"] = report["calibration"]
metadata["calibration_log"] = report["calibration_log"]
"""

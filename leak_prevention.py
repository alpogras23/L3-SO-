#!/usr/bin/env python3
"""
Posterolateral Leak Prevention for Inner Abdomen Segmentation

This module implements 4-layer constraint system to prevent fascia boundary
from extending into posterolateral subcutaneous/superficial fat regions.

Critical for VFA accuracy: Leaks cause SAT/SFA to be misclassified as VFA.
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class LeakDetectionResult:
    """Results of posterolateral leak detection"""
    leak_detected: bool
    leak_score: float  # 0-1, higher means more severe leak
    leak_regions: np.ndarray  # Binary mask of detected leak areas
    correction_applied: bool
    correction_method: str
    qc_pass: bool


class PosterolateralLeakPrevention:
    """
    4-layer constraint system to prevent posterolateral leaks in inner abdomen mask.
    
    Layers:
    1. Body constraint (absolute boundary)
    2. Abdominal wall constraint (muscle wall boundary)
    3. Vertebra anchor (midline reference)
    4. Morphological refinement (stability)
    """
    
    def __init__(self):
        self.leak_detection_log = []
    
    def apply_body_constraint(
        self,
        inner_abdomen: np.ndarray,
        body_mask: np.ndarray
    ) -> np.ndarray:
        """
        Layer 1: Force inner abdomen to stay within body contour.
        
        Args:
            inner_abdomen: Inner abdomen mask
            body_mask: Body contour mask
            
        Returns:
            Constrained inner abdomen mask
        """
        # Simple intersection - absolute constraint
        constrained = cv2.bitwise_and(inner_abdomen, body_mask)
        
        if constrained.sum() < inner_abdomen.sum():
            removed = inner_abdomen.sum() - constrained.sum()
            self.leak_detection_log.append(
                f"Body constraint removed {removed} pixels outside body contour"
            )
        
        return constrained
    
    def apply_abdominal_wall_constraint(
        self,
        inner_abdomen: np.ndarray,
        abdominal_wall_mask: Optional[np.ndarray],
        pixel_spacing: Tuple[float, float]
    ) -> np.ndarray:
        """
        Layer 2: Prune regions extending beyond abdominal wall boundary.
        
        Args:
            inner_abdomen: Inner abdomen mask
            abdominal_wall_mask: Abdominal muscles mask (from TotalSegmentator)
            pixel_spacing: (row_spacing, col_spacing) in mm
            
        Returns:
            Pruned inner abdomen mask
        """
        if abdominal_wall_mask is None:
            self.leak_detection_log.append(
                "Warning: No abdominal wall mask available for constraint"
            )
            return inner_abdomen
        
        # Create wall boundary band (dilate wall outward by ~5mm)
        wall_dilation_mm = 5.0
        kernel_size = int(wall_dilation_mm / pixel_spacing[0])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        wall_boundary = cv2.dilate(abdominal_wall_mask, kernel, iterations=1)
        
        # Everything outside wall boundary is potential leak
        outside_wall = cv2.bitwise_and(
            inner_abdomen,
            cv2.bitwise_not(wall_boundary)
        )
        
        if outside_wall.sum() > 0:
            # Remove posterolateral regions outside wall
            # Keep only largest connected component (main cavity)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                cv2.bitwise_and(inner_abdomen, wall_boundary),
                connectivity=8
            )
            
            if num_labels > 1:
                # Find largest component (excluding background=0)
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                main_cavity = (labels == largest_label).astype(np.uint8)
                
                removed = inner_abdomen.sum() - main_cavity.sum()
                self.leak_detection_log.append(
                    f"Wall constraint removed {removed} pixels outside abdominal wall"
                )
                
                return main_cavity
        
        return inner_abdomen
    
    def apply_vertebra_anchor(
        self,
        inner_abdomen: np.ndarray,
        vertebra_mask: np.ndarray,
        pixel_spacing: Tuple[float, float]
    ) -> Tuple[np.ndarray, float]:
        """
        Layer 3: Use vertebra as midline anchor to detect posterior leaks.
        
        Args:
            inner_abdomen: Inner abdomen mask
            vertebra_mask: Vertebra mask
            pixel_spacing: (row_spacing, col_spacing) in mm
            
        Returns:
            Tuple of (corrected mask, leak score)
        """
        # Find vertebra centroid as anchor
        vertebra_moments = cv2.moments(vertebra_mask)
        if vertebra_moments['m00'] == 0:
            self.leak_detection_log.append("Warning: Empty vertebra mask")
            return inner_abdomen, 0.0
        
        vb_cx = int(vertebra_moments['m10'] / vertebra_moments['m00'])
        vb_cy = int(vertebra_moments['m01'] / vertebra_moments['m00'])
        
        # Get vertebra bounding box
        y_indices, x_indices = np.where(vertebra_mask > 0)
        vb_y_min, vb_y_max = y_indices.min(), y_indices.max()
        
        # Define posterior region (behind/below vertebra)
        h, w = inner_abdomen.shape
        posterior_region = np.zeros_like(inner_abdomen)
        posterior_region[vb_cy:, :] = 1  # Everything below vertebra center
        
        # Check posterior leak
        posterior_inner = cv2.bitwise_and(inner_abdomen, posterior_region.astype(np.uint8))
        posterior_pixels = posterior_inner.sum()
        total_pixels = inner_abdomen.sum()
        
        posterior_ratio = posterior_pixels / max(total_pixels, 1)
        
        # Leak score based on posterior ratio
        # Expected: Most of inner abdomen should be anterior/lateral to vertebra
        # If > 40% is posterior, likely leak
        leak_score = max(0.0, (posterior_ratio - 0.40) / 0.30)  # 0 at 40%, 1 at 70%
        
        if posterior_ratio > 0.40:
            self.leak_detection_log.append(
                f"Vertebra anchor: High posterior ratio {posterior_ratio:.1%} "
                f"(leak score: {leak_score:.2f})"
            )
            
            # Correction: Keep only components connected to anterior region
            anterior_region = np.zeros_like(inner_abdomen)
            anterior_region[:vb_cy, :] = 1
            
            anterior_inner = cv2.bitwise_and(inner_abdomen, anterior_region.astype(np.uint8))
            
            # Dilate anterior slightly to allow continuity
            dilation_mm = 10.0
            kernel_size = int(dilation_mm / pixel_spacing[0])
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            anterior_dilated = cv2.dilate(anterior_inner, kernel, iterations=1)
            
            # Keep only parts connected to anterior
            corrected = cv2.bitwise_and(inner_abdomen, anterior_dilated)
            
            removed = inner_abdomen.sum() - corrected.sum()
            if removed > 0:
                self.leak_detection_log.append(
                    f"Vertebra anchor removed {removed} posterior leak pixels"
                )
            
            return corrected, leak_score
        
        return inner_abdomen, leak_score
    
    def apply_morphological_refinement(
        self,
        inner_abdomen: np.ndarray,
        pixel_spacing: Tuple[float, float]
    ) -> np.ndarray:
        """
        Layer 4: Morphological operations to ensure cavity stability.
        
        Operations:
        - Hole filling (close internal gaps)
        - Closing (smooth boundary, connect nearby regions)
        - Largest component (remove disconnected leak tails)
        
        Args:
            inner_abdomen: Inner abdomen mask
            pixel_spacing: (row_spacing, col_spacing) in mm
            
        Returns:
            Refined mask
        """
        # Hole filling
        filled = inner_abdomen.copy()
        contours, _ = cv2.findContours(filled, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(filled, [contour], 0, 1, -1)
        
        # Closing to smooth boundary
        close_mm = 5.0
        kernel_size = int(close_mm / pixel_spacing[0])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Keep only largest connected component
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            closed,
            connectivity=8
        )
        
        if num_labels > 2:  # More than 1 component (excluding background)
            # Find largest
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            refined = (labels == largest_label).astype(np.uint8)
            
            removed_components = num_labels - 2
            self.leak_detection_log.append(
                f"Morphological refinement removed {removed_components} disconnected components"
            )
            
            return refined
        
        return closed
    
    def detect_and_correct_leaks(
        self,
        inner_abdomen: np.ndarray,
        body_mask: np.ndarray,
        vertebra_mask: np.ndarray,
        abdominal_wall_mask: Optional[np.ndarray],
        pixel_spacing: Tuple[float, float]
    ) -> LeakDetectionResult:
        """
        Apply full 4-layer constraint system and detect leaks.
        
        Args:
            inner_abdomen: Initial inner abdomen mask
            body_mask: Body contour
            vertebra_mask: Vertebra mask
            abdominal_wall_mask: Abdominal wall mask (optional)
            pixel_spacing: Pixel spacing in mm
            
        Returns:
            LeakDetectionResult with corrected mask and QC status
        """
        self.leak_detection_log = []  # Reset log
        
        original_pixels = inner_abdomen.sum()
        corrected = inner_abdomen.copy()
        
        # Layer 1: Body constraint
        corrected = self.apply_body_constraint(corrected, body_mask)
        
        # Layer 2: Abdominal wall constraint
        corrected = self.apply_abdominal_wall_constraint(
            corrected,
            abdominal_wall_mask,
            pixel_spacing
        )
        
        # Layer 3: Vertebra anchor
        corrected, leak_score = self.apply_vertebra_anchor(
            corrected,
            vertebra_mask,
            pixel_spacing
        )
        
        # Layer 4: Morphological refinement
        corrected = self.apply_morphological_refinement(corrected, pixel_spacing)
        
        # Calculate leak metrics
        final_pixels = corrected.sum()
        pixels_removed = original_pixels - final_pixels
        removal_ratio = pixels_removed / max(original_pixels, 1)
        
        # Detect leak regions (what was removed)
        leak_regions = cv2.bitwise_and(
            inner_abdomen,
            cv2.bitwise_not(corrected)
        )
        
        leak_detected = leak_score > 0.3 or removal_ratio > 0.15
        correction_applied = pixels_removed > 0
        
        # QC decision
        qc_pass = (
            leak_score < 0.5 and  # Moderate leak acceptable
            removal_ratio < 0.30 and  # Less than 30% removed
            final_pixels > 1000  # Reasonable cavity size
        )
        
        correction_method = ", ".join([
            "body" if "Body constraint" in log else ""
            for log in self.leak_detection_log
        ] + [
            "wall" if "Wall constraint" in log else ""
            for log in self.leak_detection_log
        ] + [
            "vertebra" if "Vertebra anchor" in log else ""
            for log in self.leak_detection_log
        ] + [
            "morph" if "Morphological" in log else ""
            for log in self.leak_detection_log
        ])
        correction_method = correction_method.strip(", ") or "none"
        
        return LeakDetectionResult(
            leak_detected=leak_detected,
            leak_score=leak_score,
            leak_regions=leak_regions,
            correction_applied=correction_applied,
            correction_method=correction_method,
            qc_pass=qc_pass
        )


# Example usage in teacher generation or inference:
"""
from leak_prevention import PosterolateralLeakPrevention

leak_preventer = PosterolateralLeakPrevention()

# Apply full constraint system
result = leak_preventer.detect_and_correct_leaks(
    inner_abdomen_mask,
    body_mask,
    vertebra_mask,
    abdominal_wall_mask,  # Optional, from TotalSegmentator
    pixel_spacing=(0.8, 0.8)
)

# Use corrected mask
inner_abdomen_corrected = (result.leak_regions == 0).astype(np.uint8) & inner_abdomen_mask

# QC reporting
metadata["leak_detection"] = {
    "leak_detected": result.leak_detected,
    "leak_score": result.leak_score,
    "correction_applied": result.correction_applied,
    "correction_method": result.correction_method,
    "qc_pass": result.qc_pass,
    "log": leak_preventer.leak_detection_log
}
"""

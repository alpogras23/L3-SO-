#!/usr/bin/env python3
"""
HU (Hounsfield Unit) Threshold-based Tissue Classification

For L3 level analysis:
- PMA (Psoas Muscle Area): -29 to +150 HU
- VAT (Visceral Fat Area): -150 to -50 HU
- SAT (Subcutaneous Fat Area): -190 to -30 HU
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class HUThresholds:
    """HU value ranges for tissue classification."""
    
    # Muscle (PMA - Psoas Muscle Area)
    MUSCLE_HU_MIN: float = -29
    MUSCLE_HU_MAX: float = 150
    
    # Visceral Fat (VAT)
    VAT_HU_MIN: float = -150
    VAT_HU_MAX: float = -50
    
    # Subcutaneous Fat (SAT)
    SAT_HU_MIN: float = -190
    SAT_HU_MAX: float = -30
    
    # Bone (for exclusion)
    BONE_HU_MIN: float = 200
    BONE_HU_MAX: float = 3000
    
    # Reference tissues
    AIR_HU: float = -1000
    WATER_HU: float = 0
    FAT_HU: float = -100
    MUSCLE_REF_HU: float = 40


class HUClassifier:
    """Classify tissues in CT slice by HU value."""
    
    def __init__(self, thresholds: Optional[HUThresholds] = None):
        """
        Initialize classifier with HU thresholds.
        
        Args:
            thresholds: HUThresholds object (uses defaults if None)
        """
        self.thresholds = thresholds or HUThresholds()
    
    def classify_muscle(self, hu_slice: np.ndarray) -> np.ndarray:
        """
        Classify muscle tissue (PMA range).
        
        Args:
            hu_slice: 2D array of HU values
            
        Returns:
            Binary mask (True where muscle is detected)
        """
        muscle_mask = (
            (hu_slice >= self.thresholds.MUSCLE_HU_MIN) &
            (hu_slice <= self.thresholds.MUSCLE_HU_MAX)
        )
        
        # Exclude bone
        bone_mask = hu_slice >= self.thresholds.BONE_HU_MIN
        muscle_mask = muscle_mask & ~bone_mask
        
        return muscle_mask.astype(np.uint8)
    
    def classify_vat(self, hu_slice: np.ndarray, 
                     peritoneal_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Classify visceral fat (VAT range).
        
        Args:
            hu_slice: 2D array of HU values
            peritoneal_mask: Optional binary mask of peritoneal cavity
                           If provided, VAT only inside this region
            
        Returns:
            Binary mask (True where VAT is detected)
        """
        vat_mask = (
            (hu_slice >= self.thresholds.VAT_HU_MIN) &
            (hu_slice <= self.thresholds.VAT_HU_MAX)
        )
        
        # Constrain to peritoneal cavity if provided
        if peritoneal_mask is not None:
            vat_mask = vat_mask & peritoneal_mask
        
        return vat_mask.astype(np.uint8)
    
    def classify_sat(self, hu_slice: np.ndarray,
                     peritoneal_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Classify subcutaneous fat (SAT range).
        
        Args:
            hu_slice: 2D array of HU values
            peritoneal_mask: Optional binary mask of peritoneal cavity
                           SAT is where peritoneal_mask is False
            
        Returns:
            Binary mask (True where SAT is detected)
        """
        sat_mask = (
            (hu_slice >= self.thresholds.SAT_HU_MIN) &
            (hu_slice <= self.thresholds.SAT_HU_MAX)
        )
        
        # Exclude regions inside peritoneal cavity (those are VAT)
        if peritoneal_mask is not None:
            sat_mask = sat_mask & ~peritoneal_mask
        
        return sat_mask.astype(np.uint8)
    
    def classify_all(self, hu_slice: np.ndarray,
                     peritoneal_mask: Optional[np.ndarray] = None
                     ) -> Dict[str, np.ndarray]:
        """
        Classify all tissues in one call.
        
        Args:
            hu_slice: 2D array of HU values
            peritoneal_mask: Optional binary mask
            
        Returns:
            Dictionary with keys: 'muscle', 'vat', 'sat'
        """
        return {
            'muscle': self.classify_muscle(hu_slice),
            'vat': self.classify_vat(hu_slice, peritoneal_mask),
            'sat': self.classify_sat(hu_slice, peritoneal_mask),
        }


class AreaCalculator:
    """Calculate tissue areas from classified masks."""
    
    def __init__(self, pixel_spacing_x: float = 0.977,
                 pixel_spacing_y: float = 0.977):
        """
        Initialize with pixel spacing.
        
        Args:
            pixel_spacing_x: Pixel size in X direction (mm)
            pixel_spacing_y: Pixel size in Y direction (mm)
                           Default: 0.977 mm (typical DICOM)
        """
        self.pixel_spacing_x = pixel_spacing_x
        self.pixel_spacing_y = pixel_spacing_y
        self.pixel_area_mm2 = pixel_spacing_x * pixel_spacing_y
    
    def calculate_area(self, mask: np.ndarray) -> float:
        """
        Calculate total area from binary mask.
        
        Args:
            mask: Binary mask (0/1)
            
        Returns:
            Area in mm²
        """
        pixel_count = np.sum(mask > 0)
        area_mm2 = pixel_count * self.pixel_area_mm2
        return area_mm2
    
    def calculate_areas(self, masks: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate areas for multiple tissues.
        
        Args:
            masks: Dictionary with keys: 'muscle', 'vat', 'sat'
            
        Returns:
            Dictionary with tissue areas in mm²
        """
        return {
            tissue: self.calculate_area(mask)
            for tissue, mask in masks.items()
        }


class HUAnalyzer:
    """Complete HU-based analysis pipeline."""
    
    def __init__(self, 
                 thresholds: Optional[HUThresholds] = None,
                 pixel_spacing_x: float = 0.977,
                 pixel_spacing_y: float = 0.977):
        """
        Initialize analyzer.
        
        Args:
            thresholds: HU threshold configuration
            pixel_spacing_x: Pixel size in mm
            pixel_spacing_y: Pixel size in mm
        """
        self.classifier = HUClassifier(thresholds)
        self.calculator = AreaCalculator(pixel_spacing_x, pixel_spacing_y)
    
    def analyze_slice(self, hu_slice: np.ndarray,
                      peritoneal_mask: Optional[np.ndarray] = None
                      ) -> Dict[str, float]:
        """
        Complete analysis of L3 slice.
        
        Args:
            hu_slice: 2D array of HU values at L3
            peritoneal_mask: Optional fascia/peritoneal boundary
            
        Returns:
            Dictionary with:
            - pma_mm2: Psoas muscle area
            - vat_mm2: Visceral fat area
            - sat_mm2: Subcutaneous fat area
            - total_fat_mm2: VAT + SAT
        """
        # Classify tissues
        masks = self.classifier.classify_all(hu_slice, peritoneal_mask)
        
        # Calculate areas
        areas = self.calculator.calculate_areas(masks)
        
        # Add totals
        areas['total_fat_mm2'] = areas['vat'] + areas['sat']
        areas['pma_mm2'] = areas['muscle']
        areas['vat_mm2'] = areas['vat']
        areas['sat_mm2'] = areas['sat']
        
        return areas
    
    def get_tissue_statistics(self, hu_slice: np.ndarray) -> Dict[str, float]:
        """
        Get HU statistics for quality control.
        
        Args:
            hu_slice: 2D array of HU values
            
        Returns:
            Dictionary with min, max, mean HU values
        """
        return {
            'hu_min': float(hu_slice.min()),
            'hu_max': float(hu_slice.max()),
            'hu_mean': float(hu_slice.mean()),
            'hu_std': float(hu_slice.std()),
        }


# Example usage and validation
if __name__ == "__main__":
    print("HU Threshold Configuration for L3 Analysis")
    print("=" * 60)
    
    # Show default thresholds
    thresholds = HUThresholds()
    print(f"\nMuscle (PMA):      {thresholds.MUSCLE_HU_MIN:+.0f} to {thresholds.MUSCLE_HU_MAX:+.0f} HU")
    print(f"Visceral Fat (VAT): {thresholds.VAT_HU_MIN:+.0f} to {thresholds.VAT_HU_MAX:+.0f} HU")
    print(f"Subcut Fat (SAT):  {thresholds.SAT_HU_MIN:+.0f} to {thresholds.SAT_HU_MAX:+.0f} HU")
    
    print("\n" + "=" * 60)
    print("Reference HU Values (for validation):")
    print(f"  Air:              {thresholds.AIR_HU:+.0f} HU")
    print(f"  Water:            {thresholds.WATER_HU:+.0f} HU")
    print(f"  Fat:              {thresholds.FAT_HU:+.0f} HU")
    print(f"  Muscle:           {thresholds.MUSCLE_REF_HU:+.0f} HU")
    print(f"  Bone:             {thresholds.BONE_HU_MIN:+.0f} to {thresholds.BONE_HU_MAX:+.0f} HU")
    
    # Demonstrate with synthetic data
    print("\n" + "=" * 60)
    print("Example Analysis:")
    
    # Create synthetic L3 slice with different tissues
    synthetic_slice = np.zeros((512, 512), dtype=np.int16)
    
    # Add tissues
    synthetic_slice[100:200, 100:200] = -80    # VAT
    synthetic_slice[200:300, 200:300] = 40     # Muscle
    synthetic_slice[300:400, 100:200] = -120   # SAT
    synthetic_slice[250:260, 250:260] = 500    # Bone
    
    # Analyze
    analyzer = HUAnalyzer()
    results = analyzer.analyze_slice(synthetic_slice)
    stats = analyzer.get_tissue_statistics(synthetic_slice)
    
    print(f"\nPMA (Muscle):     {results['pma_mm2']:>10.1f} mm²")
    print(f"VAT (Visc. Fat):  {results['vat_mm2']:>10.1f} mm²")
    print(f"SAT (Subcut. Fat):{results['sat_mm2']:>10.1f} mm²")
    print(f"Total Fat:        {results['total_fat_mm2']:>10.1f} mm²")
    
    print("\nHU Statistics:")
    print(f"  Min: {stats['hu_min']:+.0f} HU")
    print(f"  Max: {stats['hu_max']:+.0f} HU")
    print(f"  Mean: {stats['hu_mean']:+.0f} HU")
    print(f"  Std Dev: {stats['hu_std']:+.0f} HU")

#!/usr/bin/env python3
"""
GUI için DL Inference Pipeline
- Rule-based preprocessing + DL prediction
- Post-processing ve threshold optimization
- Hybrid mode (rule-based + DL blend)
"""
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional

try:
    from .model_loader import load_vfapma_model
except ImportError:
    from model_loader import load_vfapma_model


class DLInferencePipeline:
    """GUI için DL model inference wrapper"""
    
    def __init__(self, model_checkpoint: Path, device: str = "auto"):
        """
        Args:
            model_checkpoint: best_unet.pt yolu
            device: 'auto', 'cuda', 'cpu'
        """
        self.model_loader = load_vfapma_model(model_checkpoint, device)
        self.target_size = (512, 512)  # Eğitim sırasında kullanılan boyut
    
    def prepare_input(self, 
                      hu_slice: np.ndarray,
                      ts_seg: Optional[np.ndarray] = None,
                      vertebra_mask: Optional[np.ndarray] = None) -> torch.Tensor:
        """
        3-channel input hazırla
        
        Args:
            hu_slice: (H, W) HU değerleri
            ts_seg: (H, W) TotalSegmentator segmentasyon (opsiyonel)
            vertebra_mask: (H, W) Vertebra maskesi (opsiyonel)
        
        Returns:
            input_tensor: (1, 3, 512, 512)
        """
        # Resize to 512x512
        hu_resized = cv2.resize(hu_slice.astype(np.float32), self.target_size, 
                                interpolation=cv2.INTER_LINEAR)
        
        # TS seg (yoksa zeros)
        if ts_seg is not None:
            ts_resized = cv2.resize(ts_seg.astype(np.float32), self.target_size,
                                    interpolation=cv2.INTER_NEAREST)
        else:
            ts_resized = np.zeros(self.target_size, dtype=np.float32)
        
        # Vertebra mask (yoksa zeros)
        if vertebra_mask is not None:
            vb_resized = cv2.resize(vertebra_mask.astype(np.float32), self.target_size,
                                    interpolation=cv2.INTER_NEAREST)
        else:
            vb_resized = np.zeros(self.target_size, dtype=np.float32)
        
        # Normalize HU (-150 to 250 window)
        hu_normalized = np.clip(hu_resized, -150, 250)
        hu_normalized = (hu_normalized + 150) / 400.0  # [0, 1]
        
        # Normalize TS (binary)
        ts_normalized = (ts_resized > 0).astype(np.float32)
        
        # Normalize vertebra (binary)
        vb_normalized = (vb_resized > 0).astype(np.float32)
        
        # Stack channels: (3, 512, 512)
        input_3ch = np.stack([hu_normalized, ts_normalized, vb_normalized], axis=0)
        
        # Add batch dim: (1, 3, 512, 512)
        input_tensor = torch.from_numpy(input_3ch).unsqueeze(0).float()
        
        return input_tensor
    
    def predict_masks(self, 
                      hu_slice: np.ndarray,
                      ts_seg: Optional[np.ndarray] = None,
                      vertebra_mask: Optional[np.ndarray] = None,
                      threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        DL model ile VFA/PMA/inner_abdomen maskelerini tahmin et
        
        Args:
            hu_slice: (H, W) HU değerleri
            ts_seg: (H, W) TS segmentasyon
            vertebra_mask: (H, W) Vertebra maskesi
            threshold: Binary threshold (0.5 default)
        
        Returns:
            vfa_mask: (H, W) VFA maskesi
            pma_mask: (H, W) PMA maskesi
            inner_abdomen_mask: (H, W) Inner abdomen maskesi
        """
        original_shape = hu_slice.shape
        
        # Input hazırla
        input_tensor = self.prepare_input(hu_slice, ts_seg, vertebra_mask)
        
        # Predict
        output_logits = self.model_loader.predict(input_tensor)  # (1, 3, 512, 512)
        
        # Sigmoid + threshold
        output_probs = torch.sigmoid(output_logits).cpu().numpy()[0]  # (3, 512, 512)
        
        vfa_prob = output_probs[0]
        pma_prob = output_probs[1]
        inner_prob = output_probs[2]
        
        # Binary masks
        vfa_mask_512 = (vfa_prob > threshold).astype(np.uint8)
        pma_mask_512 = (pma_prob > threshold).astype(np.uint8)
        inner_mask_512 = (inner_prob > threshold).astype(np.uint8)
        
        # Resize back to original size
        vfa_mask = cv2.resize(vfa_mask_512, (original_shape[1], original_shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        pma_mask = cv2.resize(pma_mask_512, (original_shape[1], original_shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        inner_mask = cv2.resize(inner_mask_512, (original_shape[1], original_shape[0]),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
        
        return vfa_mask, pma_mask, inner_mask
    
    def hybrid_predict(self,
                       hu_slice: np.ndarray,
                       rule_based_vfa: np.ndarray,
                       rule_based_pma: np.ndarray,
                       rule_based_inner: np.ndarray,
                       ts_seg: Optional[np.ndarray] = None,
                       vertebra_mask: Optional[np.ndarray] = None,
                       blend_alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Hybrid mode: Rule-based + DL blend
        
        Args:
            hu_slice: (H, W) HU değerleri
            rule_based_vfa: (H, W) Rule-based VFA maskesi
            rule_based_pma: (H, W) Rule-based PMA maskesi
            rule_based_inner: (H, W) Rule-based inner abdomen
            ts_seg: (H, W) TS seg
            vertebra_mask: (H, W) Vertebra mask
            blend_alpha: DL ağırlığı (0=rule-based, 1=DL only, 0.5=50/50)
        
        Returns:
            blended_vfa: (H, W) Hybrid VFA
            blended_pma: (H, W) Hybrid PMA
            blended_inner: (H, W) Hybrid inner abdomen
        """
        # DL predictions
        dl_vfa, dl_pma, dl_inner = self.predict_masks(hu_slice, ts_seg, vertebra_mask)
        
        # Blend (weighted OR/AND operation)
        if blend_alpha == 0:
            return rule_based_vfa, rule_based_pma, rule_based_inner
        elif blend_alpha == 1:
            return dl_vfa, dl_pma, dl_inner
        else:
            # Probability blending
            rule_prob_vfa = rule_based_vfa.astype(np.float32)
            rule_prob_pma = rule_based_pma.astype(np.float32)
            rule_prob_inner = rule_based_inner.astype(np.float32)
            
            dl_prob_vfa = dl_vfa.astype(np.float32)
            dl_prob_pma = dl_pma.astype(np.float32)
            dl_prob_inner = dl_inner.astype(np.float32)
            
            blended_vfa = ((1 - blend_alpha) * rule_prob_vfa + blend_alpha * dl_prob_vfa) > 0.5
            blended_pma = ((1 - blend_alpha) * rule_prob_pma + blend_alpha * dl_prob_pma) > 0.5
            blended_inner = ((1 - blend_alpha) * rule_prob_inner + blend_alpha * dl_prob_inner) > 0.5
            
            return blended_vfa.astype(bool), blended_pma.astype(bool), blended_inner.astype(bool)


def create_dl_pipeline(model_path: Path, device: str = "auto") -> Optional[DLInferencePipeline]:
    """
    DL pipeline oluştur (hata durumunda None döner)
    
    Usage:
        >>> pipeline = create_dl_pipeline("psoas_ml/ckpts/best_unet.pt")
        >>> if pipeline:
        >>>     vfa, pma, inner = pipeline.predict_masks(hu_slice, ts_seg, vb_mask)
    """
    try:
        return DLInferencePipeline(model_path, device)
    except Exception as e:
        print(f"❌ DL model yüklenemedi: {e}")
        return None


if __name__ == "__main__":
    # Test inference
    import sys
    
    if len(sys.argv) > 1:
        ckpt = Path(sys.argv[1])
        pipeline = create_dl_pipeline(ckpt)
        
        if pipeline:
            # Dummy test
            dummy_hu = np.random.randn(512, 512) * 50  # HU-like random
            dummy_ts = np.random.randint(0, 2, (512, 512)).astype(np.uint8)
            dummy_vb = np.random.randint(0, 2, (512, 512)).astype(np.uint8)
            
            vfa, pma, inner = pipeline.predict_masks(dummy_hu, dummy_ts, dummy_vb)
            
            print(f"✅ Inference test OK:")
            print(f"   VFA: {vfa.sum()} pixels")
            print(f"   PMA: {pma.sum()} pixels")
            print(f"   Inner: {inner.sum()} pixels")
    else:
        print("Usage: python infer_gui.py <checkpoint_path>")

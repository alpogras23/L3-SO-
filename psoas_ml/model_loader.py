#!/usr/bin/env python3
"""
DL Model Yükleme Modülü
- U-Net checkpoint loading
- Device management (CPU/CUDA)
- Model validation
"""
import torch
from pathlib import Path
from monai.networks.nets import UNet


class VFAPMAModelLoader:
    """AMOS22 eğitilmiş U-Net model yükleyici"""
    
    def __init__(self, checkpoint_path: Path, device: str = "auto"):
        """
        Args:
            checkpoint_path: best_unet.pt checkpoint dosya yolu
            device: 'auto', 'cuda', 'cpu'
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = self._get_device(device)
        self.model = None
        self._load_model()
    
    def _get_device(self, device_str: str) -> torch.device:
        """Device seçimi"""
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)
    
    def _load_model(self):
        """Model mimari + checkpoint yükleme"""
        # MONAI U-Net (Colab notebook ile aynı config)
        self.model = UNet(
            spatial_dims=2,
            in_channels=3,      # HU + TS + vertebra
            out_channels=3,     # VFA + PMA + inner_abdomen
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout=0.1
        ).to(self.device)
        
        # Checkpoint yükle
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint bulunamadı: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # State dict yükle (eğitim checkpoint'i ise 'model_state_dict' key'i var)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model yüklendi (epoch {checkpoint.get('epoch', 'N/A')}, "
                  f"val_loss: {checkpoint.get('val_loss', 'N/A'):.4f})")
        else:
            self.model.load_state_dict(checkpoint)
            print(f"✅ Model yüklendi: {self.checkpoint_path.name}")
        
        self.model.eval()  # Inference mode
        print(f"🔧 Device: {self.device}")
        print(f"📊 Model params: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Inference
        
        Args:
            input_tensor: (B, 3, H, W) - HU, TS, vertebra channels
        
        Returns:
            output_tensor: (B, 3, H, W) - VFA, PMA, inner_abdomen logits
        """
        with torch.no_grad():
            output = self.model(input_tensor.to(self.device))
        return output


def load_vfapma_model(checkpoint_path: Path, device: str = "auto") -> VFAPMAModelLoader:
    """
    Kullanım kolaylığı için wrapper fonksiyon
    
    Example:
        >>> model_loader = load_vfapma_model("psoas_ml/ckpts/best_unet.pt")
        >>> predictions = model_loader.predict(input_tensor)
    """
    return VFAPMAModelLoader(checkpoint_path, device)


if __name__ == "__main__":
    # Test loading
    import sys
    if len(sys.argv) > 1:
        ckpt = Path(sys.argv[1])
        model = load_vfapma_model(ckpt)
        
        # Dummy test
        dummy_input = torch.randn(1, 3, 512, 512)
        output = model.predict(dummy_input)
        print(f"✅ Test OK: Input {dummy_input.shape} → Output {output.shape}")
    else:
        print("Usage: python model_loader.py <checkpoint_path>")

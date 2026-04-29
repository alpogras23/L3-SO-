#!/usr/bin/env python3
"""
ML-Based VFA/PMA Inference Module
Eğitilmiş U-Net modeli ile VFA ve PMA hesaplama.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import cv2

# Model path
MODEL_PATH = Path(__file__).parent.parent / 'models' / 'psoas_vfa_best.pth'
IMG_SIZE = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) * 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net for multi-class segmentation"""
    
    def __init__(self, in_channels=1, out_channels=4, features=[32, 64, 128, 256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        # Encoder
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # Decoder
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, 2, 2))
            self.ups.append(DoubleConv(feature * 2, feature))
        
        # Final conv
        self.final = nn.Conv2d(features[0], out_channels, 1)
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skip_connections[i // 2]
            
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)
        
        return self.final(x)


class VFAPMAPredictor:
    """VFA ve PMA hesaplama için ML-based predictor"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path or MODEL_PATH
        self.model = None
        self.device = DEVICE
        self._load_model()
    
    def _load_model(self):
        """Modeli yükle"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model bulunamadı: {self.model_path}")
        
        self.model = UNet(in_channels=1, out_channels=4)
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model yüklendi: {self.model_path.name}")
        print(f"Device: {self.device}")
        print(f"Best Dice: {checkpoint.get('mean_dice', 'N/A'):.4f}")
    
    def detect_orientation(self, hu_array):
        """
        AMOS NIfTI eğitim verisi ile klinik DICOM arasındaki orientasyon farkını düzelt.
        
        Test sonuçları: 90° döndürme tüm klinik DICOM'larda en iyi PMA sonucunu veriyor.
        
        Returns:
            rotation_needed: Her zaman 90° (AMOS orientasyonuna uyum için)
        """
        # Test sonuçlarına göre 90° döndürme en iyi performansı veriyor
        return 90
    
    def preprocess(self, hu_array, pixel_spacing):
        """HU array'i model inputuna dönüştür (AMOS orientasyonuna çevir)"""
        original_shape = hu_array.shape
        
        # Orientasyon tespiti ve düzeltmesi
        self.rotation_needed = self.detect_orientation(hu_array)
        
        if self.rotation_needed == 90:
            hu_array = np.rot90(hu_array, k=1)  # 90 derece saat yönünde
        elif self.rotation_needed == 180:
            hu_array = np.rot90(hu_array, k=2)
        elif self.rotation_needed == 270:
            hu_array = np.rot90(hu_array, k=3)
        
        # HU normalizasyonu
        ct = np.clip(hu_array, -1024, 3071).astype(np.float32)
        ct = (ct + 1024) / 4095.0
        
        # Resize
        ct_resized = cv2.resize(ct, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        
        # Add batch and channel dimensions
        ct_tensor = torch.from_numpy(ct_resized).unsqueeze(0).unsqueeze(0)
        
        return ct_tensor, original_shape
    
    def postprocess(self, pred_tensor, original_shape, pixel_spacing):
        """Model çıktısını orijinal boyuta döndür ve maskeleri çıkar"""
        # Sigmoid ve threshold
        pred = torch.sigmoid(pred_tensor).squeeze(0).cpu().numpy()
        pred_binary = (pred > 0.5).astype(np.uint8)
        
        # Resize to original shape
        masks = {}
        channel_names = ['vfa', 'psoas_left', 'psoas_right', 'l3']
        
        for i, name in enumerate(channel_names):
            mask = cv2.resize(pred_binary[i], 
                            (original_shape[1], original_shape[0]), 
                            interpolation=cv2.INTER_NEAREST)
            
            # Orientasyon düzeltmesini geri al
            if hasattr(self, 'rotation_needed') and self.rotation_needed != 0:
                if self.rotation_needed == 180:
                    mask = np.rot90(mask, k=2)  # 180 derece geri
                elif self.rotation_needed == 90:
                    mask = np.rot90(mask, k=3)  # 90 derece geri (= 270)
                elif self.rotation_needed == 270:
                    mask = np.rot90(mask, k=1)  # 270 derece geri (= 90)
            
            masks[name] = mask
        
        return masks
    
    def predict(self, hu_array, pixel_spacing=(1.0, 1.0)):
        """
        VFA ve PMA tahmin et.
        
        Args:
            hu_array: 2D numpy array (HU değerleri)
            pixel_spacing: (row_spacing, col_spacing) mm cinsinden
        
        Returns:
            dict: VFA/PMA değerleri ve maskeler
        """
        # Preprocess
        ct_tensor, original_shape = self.preprocess(hu_array, pixel_spacing)
        ct_tensor = ct_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            pred = self.model(ct_tensor)
        
        # Postprocess
        masks = self.postprocess(pred, original_shape, pixel_spacing)
        
        # NOT: Hibrit VFA hesaplaması kaldırıldı - model'in orijinal VFA segmentasyonu
        # fasya içini doğru hesaplıyor. Sadece 90° orientasyon düzeltmesi yeterli.
        
        # Alanları hesapla
        pixel_area = pixel_spacing[0] * pixel_spacing[1]
        
        vfa_mm2 = np.sum(masks['vfa']) * pixel_area
        pma_left_mm2 = np.sum(masks['psoas_left']) * pixel_area
        pma_right_mm2 = np.sum(masks['psoas_right']) * pixel_area
        pma_total_mm2 = pma_left_mm2 + pma_right_mm2
        l3_mm2 = np.sum(masks['l3']) * pixel_area
        
        results = {
            'vfa_mm2': vfa_mm2,
            'vfa_cm2': vfa_mm2 / 100,
            'pma_left_mm2': pma_left_mm2,
            'pma_right_mm2': pma_right_mm2,
            'pma_total_mm2': pma_total_mm2,
            'pma_cm2': pma_total_mm2 / 100,
            'l3_mm2': l3_mm2,
            'masks': masks,
            'confidence': self._calculate_confidence(masks)
        }
        
        return results
    
    def _calculate_confidence(self, masks):
        """Sonuç güvenilirliğini hesapla"""
        # L3 vertebra tespit edildi mi?
        l3_area = np.sum(masks['l3'])
        # Psoas var mı?
        psoas_total = np.sum(masks['psoas_left']) + np.sum(masks['psoas_right'])
        # VFA var mı?
        vfa_area = np.sum(masks['vfa'])
        
        # Basit confidence hesabı
        conf = 0.0
        if l3_area > 100:  # L3 tespit edildi
            conf += 0.4
        if psoas_total > 500:  # Psoas tespit edildi
            conf += 0.3
        if vfa_area > 1000:  # VFA tespit edildi
            conf += 0.3
        
        return min(conf, 1.0)


def create_overlay(hu_array, masks, alpha=0.5):
    """Segmentasyon overlay görüntüsü oluştur"""
    # HU to grayscale
    hu_norm = np.clip(hu_array, -150, 250)
    hu_norm = ((hu_norm + 150) / 400 * 255).astype(np.uint8)
    
    # RGB'ye çevir
    overlay = cv2.cvtColor(hu_norm, cv2.COLOR_GRAY2RGB)
    
    # Renkler: VFA=kırmızı, Psoas=mavi, L3=sarı
    colors = {
        'vfa': (0, 0, 255),        # Kırmızı (BGR)
        'psoas_left': (255, 0, 0),  # Mavi
        'psoas_right': (255, 0, 0), # Mavi
        'l3': (0, 255, 255)         # Sarı
    }
    
    for name, mask in masks.items():
        if mask is not None and np.any(mask):
            color = colors.get(name, (0, 255, 0))
            overlay[mask > 0] = (
                overlay[mask > 0] * (1 - alpha) + 
                np.array(color) * alpha
            ).astype(np.uint8)
    
    return overlay


# Test fonksiyonu
def test_model():
    """Model testini çalıştır"""
    print("VFA/PMA ML Predictor Test")
    print("=" * 50)
    
    try:
        predictor = VFAPMAPredictor()
        
        # Test için dummy veri
        dummy_hu = np.random.randn(512, 512).astype(np.float32) * 100
        pixel_spacing = (0.8, 0.8)
        
        results = predictor.predict(dummy_hu, pixel_spacing)
        
        print(f"\nTest Sonuçları:")
        print(f"  VFA: {results['vfa_cm2']:.1f} cm²")
        print(f"  PMA: {results['pma_cm2']:.1f} cm²")
        print(f"  Confidence: {results['confidence']:.2f}")
        print("\nModel çalışıyor!")
        
    except FileNotFoundError as e:
        print(f"Hata: {e}")
        print("Model dosyası bulunamadı. Önce eğitim yapın.")


if __name__ == '__main__':
    test_model()

#!/usr/bin/env python3
"""
Export trained model for inference and production deployment.
Supports ONNX, TorchScript, and checkpoint conversion.
Usage: python scripts/export_model.py --checkpoint run_mt_full_60e/best.ckpt --format onnx --out models/psoas_l3_model.onnx
"""
import argparse
import os
import torch
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.train_nifti_mini import LitModel


def export_onnx(model, out_path, input_shape=(1, 1, 256, 256)):
    """Export to ONNX format."""
    model.eval()
    dummy_input = torch.randn(input_shape)
    
    torch.onnx.export(
        model.net,  # Use the UNet directly
        dummy_input,
        out_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    print(f"[EXPORT] ONNX model saved: {out_path}")


def export_torchscript(model, out_path, input_shape=(1, 1, 256, 256)):
    """Export to TorchScript format."""
    model.eval()
    dummy_input = torch.randn(input_shape)
    
    with torch.no_grad():
        traced_model = torch.jit.trace(model.net, dummy_input)
        traced_model.save(out_path)
    
    print(f"[EXPORT] TorchScript model saved: {out_path}")


def export_state_dict(model, out_path):
    """Export raw state dict (for PyTorch loading)."""
    torch.save(model.net.state_dict(), out_path)
    print(f"[EXPORT] State dict saved: {out_path}")


def test_inference(model, format_type, model_path):
    """Quick inference test."""
    import numpy as np
    
    print("\n[TEST] Running inference test...")
    dummy_img = np.random.randn(256, 256).astype(np.float32)
    dummy_img = (dummy_img - dummy_img.min()) / (dummy_img.max() - dummy_img.min())
    
    if format_type == 'onnx':
        import onnxruntime as ort
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        pred = session.run([output_name], {input_name: dummy_img[None, None, :, :]})[0]
        print(f"  ONNX inference: input={dummy_img.shape} → output={pred.shape}")
        
    elif format_type == 'torchscript':
        loaded_model = torch.jit.load(model_path)
        loaded_model.eval()
        with torch.no_grad():
            input_t = torch.from_numpy(dummy_img).unsqueeze(0).unsqueeze(0)
            pred = loaded_model(input_t).numpy()
        print(f"  TorchScript inference: input={dummy_img.shape} → output={pred.shape}")
        
    elif format_type == 'state_dict':
        from monai.networks.nets import UNet
        net = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64),
            strides=(2, 2),
        )
        net.load_state_dict(torch.load(model_path))
        net.eval()
        with torch.no_grad():
            input_t = torch.from_numpy(dummy_img).unsqueeze(0).unsqueeze(0)
            pred = net(input_t).numpy()
        print(f"  State dict inference: input={dummy_img.shape} → output={pred.shape}")
    
    print("[TEST] Inference successful!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to .ckpt file')
    parser.add_argument('--format', choices=['onnx', 'torchscript', 'state_dict', 'all'], 
                        default='onnx', help='Export format')
    parser.add_argument('--out', help='Output path (auto-generated if not provided)')
    parser.add_argument('--test', action='store_true', help='Run inference test after export')
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        return
    
    # Load model
    print(f"[LOAD] Loading checkpoint: {args.checkpoint}")
    model = LitModel.load_from_checkpoint(args.checkpoint)
    model.eval()
    print("[OK] Model loaded")
    
    # Auto-generate output paths
    ckpt_base = os.path.splitext(os.path.basename(args.checkpoint))[0]
    out_dir = os.path.dirname(args.checkpoint) if not args.out else os.path.dirname(args.out)
    
    formats_to_export = ['onnx', 'torchscript', 'state_dict'] if args.format == 'all' else [args.format]
    
    for fmt in formats_to_export:
        if args.out and len(formats_to_export) == 1:
            out_path = args.out
        else:
            ext = {'onnx': '.onnx', 'torchscript': '.pt', 'state_dict': '_weights.pth'}[fmt]
            out_path = os.path.join(out_dir, f'{ckpt_base}_{fmt}{ext}')
        
        if fmt == 'onnx':
            export_onnx(model, out_path)
        elif fmt == 'torchscript':
            export_torchscript(model, out_path)
        elif fmt == 'state_dict':
            export_state_dict(model, out_path)
        
        # Test if requested
        if args.test:
            test_inference(model, fmt, out_path)
    
    print("\n[DONE] Model export complete")


if __name__ == "__main__":
    main()

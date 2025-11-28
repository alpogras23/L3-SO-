import os
import sys

import cv2
import numpy as np
import torch
from monai.networks.nets import UNet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(BASE_DIR, "ckpts", "best.ckpt")


def load_net():
    net = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64),
        strides=(2, 2),
        num_res_units=2,
    )
    if os.path.isfile(CKPT):
        ckpt = torch.load(CKPT, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        state = {k.replace("net.", ""): v for k, v in state.items() if k.startswith("net.")}
        net.load_state_dict(state, strict=False)
    net.eval()
    return net


def run_one(png_path: str) -> int:
    base = os.path.splitext(os.path.basename(png_path))[0]
    img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"cannot read {png_path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscale = img.astype(np.float32)
    grayscale = (grayscale - grayscale.min()) / (np.ptp(grayscale) + 1e-6)
    with torch.no_grad():
        tensor = torch.from_numpy(grayscale[None, None, ...])
        pred = torch.sigmoid(load_net()(tensor)).squeeze().numpy()
    conf = float(pred.mean())
    mask = (pred > 0.5).astype(np.uint8) * 255
    outs_dir = os.path.join(BASE_DIR, "outs")
    os.makedirs(outs_dir, exist_ok=True)
    out_png = os.path.join(outs_dir, f"{base}_mask.png")
    out_txt = os.path.join(outs_dir, f"{base}_conf.txt")
    cv2.imwrite(out_png, mask)
    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write(f"{conf:.4f}\n")
    print("[INFER] wrote", out_png, "conf", conf)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python psoas_ml/infer.py /path/to/image.png")
        sys.exit(1)
    sys.exit(run_one(sys.argv[1]))

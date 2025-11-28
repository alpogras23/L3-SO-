import csv
import os
import time
from typing import Dict, Optional

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from torch.utils.data import DataLoader, Dataset


def _read_gray_norm(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    rng = float(img.max() - img.min())
    return (img - img.min()) / (rng + 1e-6)


def _read_mask01(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return (m > 127).astype(np.float32)


class MultiTeacherSet(Dataset):
    """Dataset backed by a CSV manifest with columns:
    image, gt_mask, ts_mask, c2c_mask, w_gt, w_ts, w_c2c
    """

    def __init__(self, csv_manifest: str):
        self.rows = []
        with open(csv_manifest, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                self.rows.append(r)
        if not self.rows:
            raise RuntimeError(f"No rows in manifest: {csv_manifest}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        g = _read_gray_norm(r["image"])  # HxW float32 in [0,1]
        y_gt = _read_mask01(r.get("gt_mask"))
        y_ts = _read_mask01(r.get("ts_mask"))
        y_c2c = _read_mask01(r.get("c2c_mask"))

        # Stack available masks and keep weights
        masks: Dict[str, torch.Tensor] = {}
        weights: Dict[str, float] = {}
        if y_gt is not None:
            masks["gt"] = torch.from_numpy(y_gt[None, ...])
            weights["gt"] = float(r.get("w_gt", 1.0))
        if y_ts is not None:
            masks["ts"] = torch.from_numpy(y_ts[None, ...])
            weights["ts"] = float(r.get("w_ts", 0.3))
        if y_c2c is not None:
            masks["c2c"] = torch.from_numpy(y_c2c[None, ...])
            weights["c2c"] = float(r.get("w_c2c", 0.3))

        return torch.from_numpy(g[None, ...]), masks, weights


class LitMT(pl.LightningModule):
    def __init__(self, lr=1e-3, throttle_ms: int = 0):
        super().__init__()
        self.save_hyperparameters()
        self.net = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64),
            strides=(2, 2),
            num_res_units=2,
        )
        self.loss = DiceCELoss(
            sigmoid=True, lambda_dice=0.7, lambda_ce=0.3, squared_pred=True
        )
        self.dice = DiceMetric(include_background=False, reduction="mean")

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", patience=6, factor=0.5
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}

    def _compute_weighted_loss(self, yhat, masks: Dict[str, torch.Tensor], weights: Dict[str, torch.Tensor]):
        """masks[k]: (B,1,H,W), weights[k]: (B,)"""
        total = torch.zeros((), device=yhat.device)
        logs = {}
        for k in ("gt", "ts", "c2c"):
            if k in masks and k in weights:
                y_true = masks[k].to(yhat.dtype)
                w_vec = weights[k].to(yhat.device).float()
                B = y_true.shape[0]
                sum_w = torch.clamp(w_vec.sum(), min=1e-6)
                loss_k = torch.zeros((), device=yhat.device)
                for i in range(B):
                    wi = w_vec[i]
                    if wi > 0:
                        loss_k = loss_k + wi * self.loss(yhat[i : i + 1], y_true[i : i + 1])
                total = total + loss_k / sum_w
                with torch.no_grad():
                    p = torch.sigmoid(yhat)
                    self.dice.reset()
                    self.dice(p, y_true)
                    logs[f"dice_{k}"] = self.dice.aggregate().item()
        return total, logs

    def training_step(self, batch, idx):
        x, masks, weights = batch
        yhat = self(x)
        loss, logs = self._compute_weighted_loss(yhat, masks, weights)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        for k, v in logs.items():
            self.log(f"train_{k}", v, prog_bar=False, on_epoch=True)
        # İsteğe bağlı CPU throttling
        throttle_ms = int(self.hparams.get("throttle_ms", 0)) if isinstance(self.hparams, dict) else int(self.hparams.throttle_ms)
        if throttle_ms > 0:
            time.sleep(throttle_ms / 1000.0)
        return loss

    def validation_step(self, batch, idx):
        x, masks, weights = batch
        yhat = self(x)
        loss, logs = self._compute_weighted_loss(yhat, masks, weights)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        for k, v in logs.items():
            self.log(f"val_{k}", v, prog_bar=True, on_epoch=True)
        return loss


def _collate_multiteacher(batch):
    """Özel collate: mask ve weight dict'lerini anahtar bazında stack eder.
    Çıkış: x (B,1,H,W), masks{key:(B,1,H,W)}, weights{key:(B,)}
    Eksik anahtarlar için sıfırlarla doldurulur.
    """
    xs, masks_list, weights_list = zip(*batch)
    x = torch.stack(xs, dim=0)
    keys = ["gt", "ts", "c2c"]
    masks_out: Dict[str, torch.Tensor] = {}
    weights_out: Dict[str, torch.Tensor] = {}
    B, _, H, W = x.shape
    for k in keys:
        ms = []
        ws = []
        for i in range(B):
            m = masks_list[i].get(k)
            if m is None:
                ms.append(torch.zeros((1, H, W), dtype=torch.float32))
                ws.append(torch.zeros(()))
            else:
                # ensure float tensor
                ms.append(m.to(dtype=torch.float32))
                w = weights_list[i].get(k, 0.0)
                ws.append(torch.as_tensor(float(w)))
        masks_out[k] = torch.stack(ms, dim=0)
        weights_out[k] = torch.stack(ws, dim=0)
    return x, masks_out, weights_out


def _build_dataloaders(csv_manifest: str, batch_size: int = 2):
    ds = MultiTeacherSet(csv_manifest)
    # Basit sabit bölme: son %20 validasyon
    n = len(ds)
    nv = max(1, int(0.2 * n))
    nt = max(1, n - nv)
    gen = torch.Generator().manual_seed(42)
    idx = torch.randperm(n, generator=gen).tolist()
    tr_idx, va_idx = idx[:nt], idx[nt:]
    tr = torch.utils.data.Subset(ds, tr_idx)
    va = torch.utils.data.Subset(ds, va_idx)
    dl_tr = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, collate_fn=_collate_multiteacher)
    dl_va = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=_collate_multiteacher)
    return dl_tr, dl_va


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-teacher training (CPU friendly)")
    parser.add_argument("--manifest", required=True, help="CSV with columns: image,gt_mask,ts_mask,c2c_mask,w_gt,w_ts,w_c2c")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=2, help="torch.set_num_threads")
    parser.add_argument("--throttle-ms", type=int, default=0, help="sleep per step to reduce heat")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "ckpts"))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # CPU ısısını azaltma: thread sayısını kısıtla
    try:
        torch.set_num_threads(max(1, int(args.threads)))
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    dl_tr, dl_va = _build_dataloaders(args.manifest, batch_size=args.batch_size)
    model = LitMT(lr=args.lr, throttle_ms=args.throttle_ms)

    ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=args.out, save_top_k=1, monitor="val_loss", mode="min", filename="best_mt"
    )
    logger = pl.loggers.CSVLogger(save_dir=args.out, name="logs_mt")

    trainer = pl.Trainer(
        max_epochs=int(args.epochs),
        accelerator="auto",
        deterministic=True,
        logger=logger,
        callbacks=[ckpt],
        log_every_n_steps=10,
        accumulate_grad_batches=2,  # ek yükü düşürmek için
    )
    trainer.fit(model, dl_tr, dl_va)
    print("[MT-TRAIN] best ckpt:", ckpt.best_model_path)


if __name__ == "__main__":
    main()

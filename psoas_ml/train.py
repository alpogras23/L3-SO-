import glob
import os

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from torch.utils.data import DataLoader, Dataset, random_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_IMG = os.path.join(BASE_DIR, "data", "images")
DATA_MSK = os.path.join(BASE_DIR, "data", "masks")
CKPT_DIR = os.path.join(BASE_DIR, "ckpts")


class PSOASSet(Dataset):
    def __init__(self, img_dir, msk_dir):
        self.imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.msk_dir = msk_dir

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        p = self.imgs[i]
        g = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if g is None:
            g = np.zeros((512, 512), np.uint8)
        if g.ndim == 3:
            g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
        g = g.astype(np.float32)
        rng = float(g.max() - g.min())
        g = (g - g.min()) / (rng + 1e-6)

        base = os.path.splitext(os.path.basename(p))[0]
        m = cv2.imread(os.path.join(self.msk_dir, f"{base}.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            m = np.zeros_like(g, np.uint8)
        m = (m > 127).astype(np.float32)
        return torch.from_numpy(g[None, ...]), torch.from_numpy(m[None, ...])


def _safe_getitem(self, i):
    p = self.imgs[i]
    g = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if g is None:
        g = np.zeros((512, 512), np.uint8)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    g = g.astype(np.float32)
    rng = float(g.max() - g.min())
    g = (g - g.min()) / (rng + 1e-6)

    base = os.path.splitext(os.path.basename(p))[0]
    m = cv2.imread(os.path.join(self.msk_dir, f"{base}.png"), cv2.IMREAD_GRAYSCALE)
    if m is None:
        m = np.zeros_like(g, np.uint8)
    m = (m > 127).astype(np.float32)
    return torch.from_numpy(g[None, ...]), torch.from_numpy(m[None, ...])


PSOASSet.__getitem__ = _safe_getitem


class LitPSOAS(pl.LightningModule):
    def __init__(self, lr=1e-3):
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
            sigmoid=True,
            lambda_dice=0.7,
            lambda_ce=0.3,
            squared_pred=True,
            smooth_nr=1e-5,
            smooth_dr=1e-5,
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

    def _step(self, batch, stage):
        x, y = batch
        yhat = self(x)
        loss = self.loss(yhat, y)
        with torch.no_grad():
            p = torch.sigmoid(yhat)
            self.dice.reset()
            self.dice(p, y)
            dice_val = self.dice.aggregate().item()
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_dice", dice_val, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, idx):
        return self._step(batch, "train")

    def validation_step(self, batch, idx):
        return self._step(batch, "val")


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    full = PSOASSet(DATA_IMG, DATA_MSK)
    n = len(full)
    if n == 0:
        print(f"[TRAIN] No training images found ({DATA_IMG}). Çıkılıyor.")
        return
    nv = max(1, int(0.2 * n))
    nt = max(1, n - nv)
    tr, va = random_split(full, [nt, nv], generator=torch.Generator().manual_seed(42))
    dl_tr = DataLoader(tr, batch_size=4, shuffle=True, num_workers=0, pin_memory=False)
    dl_va = DataLoader(va, batch_size=4, shuffle=False, num_workers=0, pin_memory=False)
    m = LitPSOAS(lr=1e-3)
    ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=CKPT_DIR, save_top_k=1, monitor="val_dice", mode="max", filename="best"
    )
    es = pl.callbacks.EarlyStopping(monitor="val_dice", patience=10, mode="max")
    logger = pl.loggers.CSVLogger(save_dir=CKPT_DIR, name="logs")
    trainer = pl.Trainer(
        max_epochs=60,
        accelerator="cpu",
        deterministic=True,
        logger=logger,
        callbacks=[ckpt, es],
        log_every_n_steps=10,
    )
    trainer.fit(m, dl_tr, dl_va)
    print("[TRAIN] best ckpt:", ckpt.best_model_path)


if __name__ == "__main__":
    main()

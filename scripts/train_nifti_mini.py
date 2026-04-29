#!/usr/bin/env python3
"""
Quick NIfTI-based multi-teacher training for AMOS mini test.
Bypasses DICOM/PNG export; reads NIfTI directly.
Usage: python scripts/train_nifti_mini.py --nifti-root DATA_ROOT --ts-merged TS_MERGED --cases amos_0001 amos_0004 amos_0005 --epochs 5
"""
import argparse
import os
import numpy as np
import pytorch_lightning as pl
import SimpleITK as sitk
import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from torch.utils.data import DataLoader, Dataset


class NIfTISliceDataset(Dataset):
    """Loads 2D slices from 3D NIfTI volumes. Applies L3-region heuristic (middle third)."""
    def __init__(self, cases_info, target_ids=[29, 55, 56], resize_to=(256, 256)):
        """
        cases_info: list of dict {nifti_img: path, ts_label: path}
        target_ids: Label IDs to extract from multi-label mask (29=vertebra_L3, 55/56=psoas)
        resize_to: Target size (H, W) for all slices to enable batching
        """
        self.slices = []
        self.target_ids = target_ids
        self.resize_to = resize_to
        for info in cases_info:
            img_sitk = sitk.ReadImage(info['nifti_img'])
            label_sitk = sitk.ReadImage(info['ts_label'])
            img_arr = sitk.GetArrayFromImage(img_sitk).astype(np.float32)  # (Z, H, W)
            label_arr = sitk.GetArrayFromImage(label_sitk).astype(np.int16)

            # L3 heuristic: middle 1/3 of volume
            nz = img_arr.shape[0]
            start_z = nz // 3
            end_z = 2 * nz // 3
            for z in range(start_z, end_z):
                slice_img = img_arr[z]  # (H, W)
                slice_label = label_arr[z]  # (H, W)
                # Check if any target label present
                if not any(np.any(slice_label == tid) for tid in target_ids):
                    continue
                self.slices.append((slice_img, slice_label))
        if not self.slices:
            raise RuntimeError("No valid slices found with target labels")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        import cv2
        img_slice, label_slice = self.slices[idx]
        # Resize to target shape
        img_slice = cv2.resize(img_slice, self.resize_to, interpolation=cv2.INTER_LINEAR)
        label_slice = cv2.resize(label_slice.astype(np.float32), self.resize_to, interpolation=cv2.INTER_NEAREST).astype(np.int16)
        # Normalize image
        img_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-6)
        # Build binary mask: union of target labels
        mask = np.zeros_like(label_slice, dtype=np.float32)
        for tid in self.target_ids:
            mask = np.maximum(mask, (label_slice == tid).astype(np.float32))
        # Add channel dim: (1, H, W)
        img_t = torch.from_numpy(img_norm).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        return img_t, mask_t


class LitModel(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.net = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64),
            strides=(2, 2),
        )
        self.loss_fn = DiceCELoss(sigmoid=True, squared_pred=True)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nifti-root", required=True, help="Path to AMOS imagesTr")
    parser.add_argument("--ts-merged", required=True, help="Path to merged TS labels")
    parser.add_argument("--cases", nargs="+", default=["amos_0001", "amos_0004", "amos_0005"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="run_mt_mini_nifti")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(args.threads)

    # Build case info list
    cases_info = []
    for case_id in args.cases:
        nifti_img = os.path.join(args.nifti_root, f"{case_id}.nii.gz")
        ts_label = os.path.join(args.ts_merged, f"{case_id}_labels.nii.gz")
        if not os.path.exists(nifti_img) or not os.path.exists(ts_label):
            print(f"[WARN] Missing files for {case_id}, skipping")
            continue
        cases_info.append({"nifti_img": nifti_img, "ts_label": ts_label})

    if len(cases_info) < 1:
        raise RuntimeError("No valid cases found")

    # Build dataset and loaders
    dataset = NIfTISliceDataset(cases_info, target_ids=[29, 55, 56])
    print(f"[INFO] Dataset: {len(dataset)} slices from {len(cases_info)} cases")

    # Split 80/20 train/val
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = LitModel(lr=args.lr)
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=args.out, monitor="val_loss", mode="min", save_top_k=1, filename="best"
    )
    logger = pl.loggers.CSVLogger(save_dir=args.out, name="logs")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="cpu",
        logger=logger,
        callbacks=[ckpt_cb],
        log_every_n_steps=5,
    )
    trainer.fit(model, train_dl, val_dl)
    print(f"[DONE] Best ckpt: {ckpt_cb.best_model_path}")


if __name__ == "__main__":
    main()

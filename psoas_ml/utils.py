import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np

PSOAS_LABELS = {"psoas_left", "psoas_right"}


def _ensure_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def cvat_to_dataset(cvat_xml: str, img_dir: str, out_img: str, out_mask: str):
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_mask, exist_ok=True)

    root = ET.parse(cvat_xml).getroot()
    images = root.findall(".//image")
    hit = miss = 0

    for im in images:
        fname = im.get("name") or im.get("file_name") or ""
        w = int(float(im.get("width")))
        h = int(float(im.get("height")))

        in_path = os.path.join(img_dir, os.path.basename(fname))
        if not os.path.isfile(in_path):
            miss += 1
            continue

        img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            miss += 1
            continue

        img = _ensure_gray(img)
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

        mask = np.zeros((h, w), np.uint8)

        for poly in im.findall(".//polygon"):
            label = (poly.get("label") or "").strip()
            if label not in PSOAS_LABELS:
                continue
            pts = []
            for xy in poly.get("points").split(";"):
                x, y = xy.split(",")
                pts.append([float(x), float(y)])
            if len(pts) >= 3:
                cv2.fillPoly(mask, [np.array(pts, np.int32)], 1)

        for shp in im.findall(".//shape"):
            label = (shp.get("label") or "").strip()
            if label not in PSOAS_LABELS:
                continue
            pts = []
            for xy in shp.get("points").split(";"):
                x, y = xy.split(",")
                pts.append([float(x), float(y)])
            if len(pts) >= 3:
                cv2.fillPoly(mask, [np.array(pts, np.int32)], 1)

        base = os.path.splitext(os.path.basename(fname))[0]
        cv2.imwrite(os.path.join(out_img, f"{base}.png"), img)
        cv2.imwrite(os.path.join(out_mask, f"{base}.png"), mask * 255)
        hit += 1

    return {"converted": hit, "missing": miss}

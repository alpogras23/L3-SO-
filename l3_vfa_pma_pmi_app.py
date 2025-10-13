from l3_vfa_pma_gui import App

if __name__ == "__main__":
    App().mainloop()  #!/usr/bin/env python3
# -*- coding: utf-8 -*-
# L3 VFA / PMA / PMI / SO – v3.0 (stabil)
# - Fasya içi VFA/PMA: ray-casting fasya teyidi + lokal onarım
# - QA paket export: overlay PNG, TXT/JSON, DICOM Secondary Capture (SC)
# - L3 düzey doğrulama, PNG↔DICOM registration, outlier-dilim dışlama, belirsizlik (HU±)

import json
import math
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import pydicom
from pydicom.uid import SecondaryCaptureImageStorage, generate_uid

VERSION = "v3.0"

# ===== Parametreler (settings.json ile override) =====
HU_FAT = (-190, -30)
HU_MUSCLE_BASE = (-29, 150)
HU_MUSCLE_FALLBACK = (-60, 200)
HU_BONE_SEED = (300, 3000)
HU_BONE_CAND = (150, 2000)
HU_BONE_EXCLUDE = (200, 4000)

PMI_CUTOFF = {"M": 6.0, "F": 3.9}
VFA_PSOAS_RATIO_CUTOFF = 2.0
VFA_OBESITY_CUTOFF_CM2 = 100

RING_INNER_MM_BASE = 6.0
RING_OUTER_MM_BASE = 20.0
VERTEBRA_BUFFER_MM_BASE = 3.0
MIN_PSOAS_AREA_MM2_BASE = 150.0
MAX_PSOAS_AREA_MM2_BASE = 2000.0
PNG_BAND_MM_BASE = 25
HEIGHT_BAND_MM_BASE = 80
GRAD_STRENGTH_BASE = 0.85

FASCIA_EDGE_Q = 0.90
FASCIA_MAX_GAP_MM = 6.0
PSOAS_SHAPE_MIN_ECC = 0.65
PSOAS_SHAPE_MIN_CONVEXITY = 0.85
PSOAS_AXIS_MAX_DEG = 25.0
PSOAS_SHAPE_WT = 1.0

PNG_REG_MAX_FEAT = 2000
PNG_REG_GOOD_MATCH = 80
PNG_REG_MIN_INLIERS = 12
L3_VERTEBRA_AREA_FRAC_RANGE = (0.003, 0.06)
L3_VERTEBRA_CIRC_MIN = 0.25
L3_RIB_EDGE_Q = 0.93
UNCERTAINTY_HU_SHIFTS = [(-10, -10), (-5, -5), (0, 0), (5, 5), (10, 10)]

# v3.0 ray-casting
RAY_COUNT = 360
RAY_MIN_HITS_RATIO = 0.80
RAY_MAX_GAP_MM = 8.0
RAY_SEARCH_MM = (5.0, 80.0)
RAY_LOCAL_REPAIR_MM = 6.0

# UI
WINDOW_SCALE = 0.85
FONT_SCALE = 0.55
LINE_THICK = 1
BOX_ALPHA = 0.65
SHOW_METRICS_ON_IMAGE = True


# ===== Yardımcılar =====
def load_settings(settings_path: Path):
    if settings_path.exists():
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))

            def tup(name, default):
                return tuple(data.get(name, default))

            globals().update(
                {
                    "HU_FAT": tup("HU_FAT", HU_FAT),
                    "HU_MUSCLE_BASE": tup("HU_MUSCLE_BASE", HU_MUSCLE_BASE),
                    "HU_MUSCLE_FALLBACK": tup("HU_MUSCLE_FALLBACK", HU_MUSCLE_FALLBACK),
                    "HU_BONE_SEED": tup("HU_BONE_SEED", HU_BONE_SEED),
                    "HU_BONE_CAND": tup("HU_BONE_CAND", HU_BONE_CAND),
                    "HU_BONE_EXCLUDE": tup("HU_BONE_EXCLUDE", HU_BONE_EXCLUDE),
                    "PMI_CUTOFF": data.get("PMI_CUTOFF", PMI_CUTOFF),
                    "VFA_PSOAS_RATIO_CUTOFF": data.get(
                        "VFA_PSOAS_RATIO_CUTOFF", VFA_PSOAS_RATIO_CUTOFF
                    ),
                    "VFA_OBESITY_CUTOFF_CM2": data.get(
                        "VFA_OBESITY_CUTOFF_CM2", VFA_OBESITY_CUTOFF_CM2
                    ),
                    "RING_INNER_MM_BASE": data.get(
                        "RING_INNER_MM_BASE", RING_INNER_MM_BASE
                    ),
                    "RING_OUTER_MM_BASE": data.get(
                        "RING_OUTER_MM_BASE", RING_OUTER_MM_BASE
                    ),
                    "VERTEBRA_BUFFER_MM_BASE": data.get(
                        "VERTEBRA_BUFFER_MM_BASE", VERTEBRA_BUFFER_MM_BASE
                    ),
                    "MIN_PSOAS_AREA_MM2_BASE": data.get(
                        "MIN_PSOAS_AREA_MM2_BASE", MIN_PSOAS_AREA_MM2_BASE
                    ),
                    "MAX_PSOAS_AREA_MM2_BASE": data.get(
                        "MAX_PSOAS_AREA_MM2_BASE", MAX_PSOAS_AREA_MM2_BASE
                    ),
                    "PNG_BAND_MM_BASE": data.get("PNG_BAND_MM_BASE", PNG_BAND_MM_BASE),
                    "HEIGHT_BAND_MM_BASE": data.get(
                        "HEIGHT_BAND_MM_BASE", HEIGHT_BAND_MM_BASE
                    ),
                    "GRAD_STRENGTH_BASE": data.get(
                        "GRAD_STRENGTH_BASE", GRAD_STRENGTH_BASE
                    ),
                    "FASCIA_EDGE_Q": data.get("FASCIA_EDGE_Q", FASCIA_EDGE_Q),
                    "FASCIA_MAX_GAP_MM": data.get(
                        "FASCIA_MAX_GAP_MM", FASCIA_MAX_GAP_MM
                    ),
                    "PSOAS_SHAPE_MIN_ECC": data.get(
                        "PSOAS_SHAPE_MIN_ECC", PSOAS_SHAPE_MIN_ECC
                    ),
                    "PSOAS_SHAPE_MIN_CONVEXITY": data.get(
                        "PSOAS_SHAPE_MIN_CONVEXITY", PSOAS_SHAPE_MIN_CONVEXITY
                    ),
                    "PSOAS_AXIS_MAX_DEG": data.get(
                        "PSOAS_AXIS_MAX_DEG", PSOAS_AXIS_MAX_DEG
                    ),
                    "PSOAS_SHAPE_WT": data.get("PSOAS_SHAPE_WT", PSOAS_SHAPE_WT),
                    "PNG_REG_MAX_FEAT": data.get("PNG_REG_MAX_FEAT", PNG_REG_MAX_FEAT),
                    "PNG_REG_GOOD_MATCH": data.get(
                        "PNG_REG_GOOD_MATCH", PNG_REG_GOOD_MATCH
                    ),
                    "PNG_REG_MIN_INLIERS": data.get(
                        "PNG_REG_MIN_INLIERS", PNG_REG_MIN_INLIERS
                    ),
                    "L3_VERTEBRA_AREA_FRAC_RANGE": data.get(
                        "L3_VERTEBRA_AREA_FRAC_RANGE", L3_VERTEBRA_AREA_FRAC_RANGE
                    ),
                    "L3_VERTEBRA_CIRC_MIN": data.get(
                        "L3_VERTEBRA_CIRC_MIN", L3_VERTEBRA_CIRC_MIN
                    ),
                    "L3_RIB_EDGE_Q": data.get("L3_RIB_EDGE_Q", L3_RIB_EDGE_Q),
                    "RAY_COUNT": data.get("RAY_COUNT", RAY_COUNT),
                    "RAY_MIN_HITS_RATIO": data.get(
                        "RAY_MIN_HITS_RATIO", RAY_MIN_HITS_RATIO
                    ),
                    "RAY_MAX_GAP_MM": data.get("RAY_MAX_GAP_MM", RAY_MAX_GAP_MM),
                    "RAY_SEARCH_MM": tuple(
                        data.get("RAY_SEARCH_MM", list(RAY_SEARCH_MM))
                    ),
                    "RAY_LOCAL_REPAIR_MM": data.get(
                        "RAY_LOCAL_REPAIR_MM", RAY_LOCAL_REPAIR_MM
                    ),
                }
            )
            print(f"[INFO] settings.json yüklendi: {settings_path}")
        except Exception as e:
            print(f"[WARN] settings.json okunamadı: {e}")


def load_hu(ds):
    try:
        from pydicom import config

        handlers = getattr(config, "image_handlers", [])
        if not any(("pylibjpeg" in str(h)) or ("gdcm" in str(h)) for h in handlers):
            print("Uyarı: JPEG2000 için pylibjpeg/gdcm handler gerekli olabilir.")
    except Exception:
        pass
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    inter = float(getattr(ds, "RescaleIntercept", 0.0))
    return arr * slope + inter


def pixel_spacing(ds):
    sp = getattr(ds, "PixelSpacing", None)
    if sp is None or len(sp) < 2:
        raise ValueError("PixelSpacing eksik.")
    return float(sp[0]), float(sp[1])


def px_area_mm2(ds):
    sy, sx = pixel_spacing(ds)
    return sy * sx


def mm_to_px(ds, mm):
    sy, sx = pixel_spacing(ds)
    s = (sy + sx) / 2.0
    return max(1, int(round(mm / s)))


def area_px_from_mm2(ds, mm2):
    sy, sx = pixel_spacing(ds)
    return int(round(mm2 / (sy * sx)))


def hu_mask(hu, lo, hi):
    return ((hu >= lo) & (hu <= hi)).astype(np.uint8) * 255


def normalize8(hu, lo=-200, hi=300):
    img = np.clip(hu, lo, hi).astype(np.float32)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def centroid(mask):
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return None
    M = cv2.moments(m)
    if M["m00"] == 0:
        return None
    return (M["m10"] / M["m00"], M["m01"] / M["m00"])


# ===== PNG→DICOM registration =====
def register_png_to_dicom(png_gray, dicom_gray):
    try:
        orb = cv2.ORB_create(nfeatures=PNG_REG_MAX_FEAT, fastThreshold=5)
        k1, d1 = orb.detectAndCompute(png_gray, None)
        k2, d2 = orb.detectAndCompute(dicom_gray, None)
        if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
            return None
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(d1, d2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < PNG_REG_GOOD_MATCH:
            return None
        src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, inliers = cv2.estimateAffine2D(
            src, dst, ransacReprojThreshold=3.0, confidence=0.99
        )
        if M is None or (inliers is not None and inliers.sum() < PNG_REG_MIN_INLIERS):
            return None
        return M
    except Exception:
        return None


def warp_png_with_affine(png_gray, shape_hw, M):
    h, w = shape_hw
    return cv2.warpAffine(
        png_gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


# ===== Segmentasyon adımları =====
def body_mask_from_hu(hu):
    img8 = normalize8(hu, -300, 300)
    img8 = cv2.GaussianBlur(img8, (7, 7), 0)
    _, th = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(img8[th > 0]) < np.mean(img8[th == 0]):
        th = 255 - th
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), 2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), 1)
    return th


def vertebra_body_mask(hu):
    h, w = hu.shape
    cx, cy = w // 2, h // 2
    seed = hu_mask(hu, *HU_BONE_SEED)
    cand = hu_mask(hu, *HU_BONE_CAND)
    num, L, S, C = cv2.connectedComponentsWithStats(seed, 8)
    if num <= 1:
        return np.zeros_like(seed)
    minA, maxA = 0.001 * (h * w), 0.06 * (h * w)
    best, sc = 0, -1e9
    for lbl in range(1, num):
        x, y, ww, hh, a = S[lbl]
        if not (minA <= a <= maxA):
            continue
        mx, my = C[lbl]
        dist = np.hypot(mx - cx, my - cy)
        comp = (L == lbl).astype(np.uint8) * 255
        edges = cv2.Canny(comp, 0, 1)
        perim = float(np.count_nonzero(edges)) + 1e-6
        circ = 4.0 * np.pi * a / (perim * perim)
        s = a - 3.0 * dist + 5000.0 * circ
        if s > sc:
            best, sc = lbl, s
    if best == 0:
        return np.zeros_like(seed)
    seed_sel = (L == best).astype(np.uint8) * 255
    grown = cv2.bitwise_and(cand, cv2.dilate(seed_sel, np.ones((15, 15), np.uint8), 1))
    grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), 2)
    grown = cv2.morphologyEx(grown, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), 1)
    num2, L2, S2, C2 = cv2.connectedComponentsWithStats(grown, 8)
    if num2 <= 1:
        return np.zeros_like(grown)
    best2, sc2 = 0, -1e9
    for lbl in range(1, num2):
        x, y, ww, hh, a = S2[lbl]
        mx, my = C2[lbl]
        dist = np.hypot(mx - cx, my - cy)
        s = a - 2.0 * dist
        if s > sc2:
            best2, sc2 = lbl, s
    vb = (L2 == best2).astype(np.uint8) * 255
    vb = cv2.morphologyEx(vb, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), 1)
    return vb


def gradient_mask_u8(img8, ring, strength=0.85):
    gx = cv2.Sobel(img8, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(img8, cv2.CV_32F, 0, 1, 3)
    mag = cv2.magnitude(gx, gy)
    vals = mag[ring > 0]
    thr = (
        np.percentile(vals, 100 * strength)
        if vals.size >= 50
        else np.percentile(mag, 90)
    )
    g = (mag >= thr).astype(np.uint8) * 255
    return cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), 1)


def refine_inner_with_active_contour(hu, inner, body_cc, vb, ds, png_edges=None):
    h, w = hu.shape
    cen = centroid(vb)
    if cen is None:
        cen = (w // 2, h // 2)
    cx, cy = int(cen[0]), int(cen[1])

    img8 = normalize8(hu, -200, 300)
    gx = cv2.Sobel(img8, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(img8, cv2.CV_32F, 0, 1, 3)
    mag = cv2.magnitude(gx, gy)
    thr = np.percentile(mag, FASCIA_EDGE_Q * 100.0)
    wall_edges = (mag >= thr).astype(np.uint8) * 255
    if png_edges is not None:
        wall_edges = cv2.bitwise_or(wall_edges, png_edges)
    wall_edges = cv2.bitwise_and(wall_edges, body_cc)

    gap = mm_to_px(ds, FASCIA_MAX_GAP_MM)
    wall_edges = cv2.morphologyEx(
        wall_edges, cv2.MORPH_CLOSE, np.ones((gap, gap), np.uint8), 1
    )

    wall_barrier = cv2.dilate(wall_edges, np.ones((3, 3), np.uint8), 1)
    barrier = cv2.bitwise_or(wall_barrier, cv2.bitwise_not(body_cc))
    mask = np.zeros((h + 2, w + 2), np.uint8)
    tmp = (barrier == 0).astype(np.uint8) * 255
    tmp2 = tmp.copy()
    cv2.floodFill(tmp2, mask, (cx, cy), 128)
    inner2 = (tmp2 == 128).astype(np.uint8) * 255
    inner2 = cv2.bitwise_and(inner2, body_cc)
    inner2 = cv2.morphologyEx(inner2, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), 1)
    return inner2, wall_edges


# --- v3.0 Ray-casting fasya teyidi + lokal onarım
def ray_cast_fascia_repair(inner, wall_edges, vb, ds):
    h, w = inner.shape
    cen = centroid(vb)
    if cen is None:
        return inner, {
            "fascia_continuity_score": 0.0,
            "ray_gap_ratio": 1.0,
            "repairs": 0,
            "flags": ["no_vertebra"],
        }
    cx, cy = cen
    min_px = mm_to_px(ds, RAY_SEARCH_MM[0])
    max_px = mm_to_px(ds, RAY_SEARCH_MM[1])
    gap_tol = mm_to_px(ds, RAY_MAX_GAP_MM)
    repair_k = mm_to_px(ds, RAY_LOCAL_REPAIR_MM)

    hits, gaps, repairs = 0, 0, 0
    flags = []
    ring_edges = wall_edges.copy()
    for a in np.linspace(0, 2 * np.pi, RAY_COUNT, endpoint=False):
        dx = math.cos(a)
        dy = math.sin(a)
        found = False
        last_edge = None
        for r in range(min_px, max_px):
            x = int(round(cx + dx * r))
            y = int(round(cy + dy * r))
            if x <= 1 or y <= 1 or x >= w - 2 or y >= h - 2:
                break
            if inner[y, x] == 0:
                break
            if ring_edges[y, x] > 0:
                if last_edge is None or (r - last_edge) > gap_tol:
                    found = True
                    last_edge = r
                    break
        if found:
            hits += 1
        else:
            gaps += 1
            # lokal onarım
            x2 = int(round(cx + dx * (min_px + (max_px - min_px) // 2)))
            y2 = int(round(cy + dy * (min_px + (max_px - min_px) // 2)))
            rr = max(3, repair_k)
            patch = np.zeros_like(inner)
            cv2.circle(patch, (x2, y2), rr, 255, -1)
            new_edges = cv2.bitwise_or(ring_edges, patch)
            new_edges = cv2.morphologyEx(
                new_edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), 1
            )
            barrier = cv2.dilate(new_edges, np.ones((3, 3), np.uint8), 1)
            barrier = cv2.bitwise_or(barrier, cv2.bitwise_not(inner))
            mask = np.zeros((h + 2, w + 2), np.uint8)
            tmp = (barrier == 0).astype(np.uint8) * 255
            tmp2 = tmp.copy()
            cv2.floodFill(tmp2, mask, (int(cx), int(cy)), 128)
            inner2 = (tmp2 == 128).astype(np.uint8) * 255
            if cv2.countNonZero(inner2) > 0:
                inner = inner2
                ring_edges = new_edges
                repairs += 1

    cont_score = hits / float(RAY_COUNT)
    gap_ratio = gaps / float(RAY_COUNT)
    if cont_score < RAY_MIN_HITS_RATIO:
        flags.append("fascia_continuity_low")
    return inner, {
        "fascia_continuity_score": round(cont_score, 3),
        "ray_gap_ratio": round(gap_ratio, 3),
        "repairs": repairs,
        "flags": flags,
    }


# ===== İç maske + PNG registration =====
def inner_abdomen_via_wall(hu, ds, png_gray, band25_mm=25, height_mm=80):
    body = body_mask_from_hu(hu)
    h, w = hu.shape
    vb = vertebra_body_mask(hu)
    cen = centroid(vb)
    if cen is None:
        cen = (w // 2, h // 2)
    cx, cy = int(cen[0]), int(cen[1])

    # body_cc
    num_b, Lb, _, _ = cv2.connectedComponentsWithStats((body > 0).astype(np.uint8), 8)
    body_cc = np.zeros_like(body)
    if num_b > 1:
        lbl = Lb[cy, cx]
        if lbl != 0:
            body_cc[Lb == lbl] = 255
    if body_cc.sum() == 0:
        body_cc = body

    # PNG kenarı + registration
    png_edges = np.zeros_like(body_cc)
    if png_gray is not None:
        dicom8 = normalize8(hu, -200, 300)
        M = register_png_to_dicom(png_gray, dicom8)
        if M is not None:
            png_gray = warp_png_with_affine(png_gray, hu.shape, M)
        g = cv2.GaussianBlur(png_gray, (3, 3), 0)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
        mag = cv2.magnitude(gx, gy)
        thr = np.percentile(mag, 88)
        full = (mag >= thr).astype(np.uint8) * 255
        band25 = mm_to_px(ds, band25_mm)
        y1, y2 = max(0, cy - band25), min(h, cy + band25)
        png_edges[y1:y2, :] = full[y1:y2, :]
        png_edges = cv2.bitwise_and(png_edges, body_cc)

    # HU + sobel duvar
    img8 = normalize8(hu, -200, 300)
    sob = gradient_mask_u8(img8, body_cc, GRAD_STRENGTH_BASE)
    muscle = hu_mask(hu, *HU_MUSCLE_BASE)
    wall0 = cv2.bitwise_or(muscle, sob)
    wall0 = cv2.bitwise_or(wall0, png_edges)
    wall0 = cv2.bitwise_and(wall0, body_cc)

    # kaba inner
    wall = cv2.morphologyEx(
        wall0, cv2.MORPH_CLOSE, np.ones((2 * mm_to_px(ds, 5) + 1,) * 2, np.uint8), 1
    )
    wall = cv2.dilate(wall, np.ones((2 * mm_to_px(ds, 2) + 1,) * 2, np.uint8), 1)
    barrier = cv2.bitwise_or(wall, cv2.bitwise_not(body_cc))
    mask = np.zeros((h + 2, w + 2), np.uint8)
    tmp = (barrier == 0).astype(np.uint8) * 255
    tmp2 = tmp.copy()
    cv2.floodFill(tmp2, mask, (cx, cy), 128)
    inner = (tmp2 == 128).astype(np.uint8) * 255
    inner = cv2.bitwise_and(inner, body_cc)

    # aktif kontur rafinesi
    inner, wall_ref = refine_inner_with_active_contour(
        hu, inner, body_cc, vb, ds, png_edges=png_edges
    )

    # ray-casting QC + lokal onarım
    inner, ray_qc = ray_cast_fascia_repair(inner, wall_ref, vb, ds)

    # ±height bandı
    bandH = mm_to_px(ds, height_mm)
    y1, y2 = max(0, cy - bandH), min(h, cy + bandH)
    height_mask = np.zeros_like(inner)
    height_mask[y1:y2, :] = 255
    inner = cv2.bitwise_and(inner, height_mask)
    inner = cv2.morphologyEx(inner, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), 1)
    return inner, wall_ref, vb, (cx, cy), ray_qc


def adaptive_muscle_mask(hu, ring, base_range, fallback_range):
    vals = hu[ring > 0]
    vals = vals[(vals > fallback_range[0]) & (vals < fallback_range[1])]
    if vals.size >= 200:
        p10, p95 = np.percentile(vals, [10, 95])
        lo = max(fallback_range[0], min(base_range[0], p10 - 15))
        hi = min(fallback_range[1], max(base_range[1], p95 + 15))
        lo2 = max(fallback_range[0], min(lo, hi - 5))
        hi2 = min(fallback_range[1], max(hi, lo2 + 5))
        return hu_mask(hu, lo2, hi2), (float(lo2), float(hi2))
    return hu_mask(hu, *base_range), base_range


def shape_orientation_scores(comp_mask):
    m = (comp_mask > 0).astype(np.uint8)
    if cv2.countNonZero(m) == 0:
        return 0.0, 0.0, 0.0, 0.0
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0, 0.0, 0.0, 0.0
    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull)) + 1e-6
    convexity = max(0.0, min(1.0, area / hull_area))
    if len(cnt) >= 5:
        (_cx, _cy), (MA, ma), angle = cv2.fitEllipse(cnt)
        a = max(MA, ma) / 2.0
        b = max(1e-6, min(MA, ma) / 2.0)
        ecc = math.sqrt(max(0.0, 1.0 - (b * b) / (a * a)))
        axis_deg = float(angle)
    else:
        M = cv2.moments(cnt)
        if M["mu20"] + M["mu02"] == 0:
            ecc = 0.0
            axis_deg = 0.0
        else:
            ang = 0.5 * math.atan2(2 * M["mu11"], (M["mu20"] - M["mu02"]))
            axis_deg = abs(np.degrees(ang))
            lam1 = (M["mu20"] + M["mu02"]) / 2 + math.sqrt(
                4 * M["mu11"] ** 2 + (M["mu20"] - M["mu02"]) ** 2
            ) / 2
            lam2 = (M["mu20"] + M["mu02"]) / 2 - math.sqrt(
                4 * M["mu11"] ** 2 + (M["mu20"] - M["mu02"]) ** 2
            ) / 2
            ecc = math.sqrt(max(0.0, 1.0 - (lam2 + 1e-6) / (lam1 + 1e-6)))
    delta = min(abs(axis_deg - 0.0), abs(axis_deg - 180.0))
    axis_ok = (
        1.0
        if delta <= PSOAS_AXIS_MAX_DEG
        else max(0.0, 1.0 - (delta - PSOAS_AXIS_MAX_DEG) / PSOAS_AXIS_MAX_DEG)
    )
    ecc_ok = min(
        1.0, max(0.0, (ecc - PSOAS_SHAPE_MIN_ECC) / (1.0 - PSOAS_SHAPE_MIN_ECC + 1e-6))
    )
    conv_ok = min(
        1.0,
        max(
            0.0,
            (convexity - PSOAS_SHAPE_MIN_CONVEXITY)
            / (1.0 - PSOAS_SHAPE_MIN_CONVEXITY + 1e-6),
        ),
    )
    shape_score = 0.45 * ecc_ok + 0.25 * conv_ok + 0.30 * axis_ok
    return float(ecc), float(convexity), float(axis_ok * 100.0), float(shape_score)


def find_psoas_inside(
    hu,
    ds,
    inner,
    vertebra,
    ring_inner_mm,
    ring_outer_mm,
    grad_strength,
    min_area_mm2,
    max_area_mm2,
):
    h, w = hu.shape
    cen = centroid(vertebra)
    if cen is None:
        return np.zeros_like(vertebra), {"reason": "vertebra_not_found"}
    cx, cy = cen

    inner_px = mm_to_px(ds, ring_inner_mm)
    outer_px = mm_to_px(ds, ring_outer_mm)
    inv_v = cv2.bitwise_not(vertebra)
    dist = cv2.distanceTransform(inv_v, cv2.DIST_L2, 3)
    bone_ex = hu_mask(hu, *HU_BONE_EXCLUDE)
    vbuf = cv2.dilate(
        vertebra,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * mm_to_px(ds, VERTEBRA_BUFFER_MM_BASE) + 1,) * 2
        ),
        1,
    )

    def ring(i, o):
        dil_in = cv2.dilate(
            vertebra,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * i + 1, 2 * i + 1)),
            1,
        )
        dil_out = cv2.dilate(
            vertebra,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * o + 1, 2 * o + 1)),
            1,
        )
        return cv2.bitwise_and(cv2.subtract(dil_out, dil_in), inner)

    rng = ring(inner_px, outer_px)
    img8 = normalize8(hu, -200, 300)
    muscle_adapt, _ = adaptive_muscle_mask(hu, rng, HU_MUSCLE_BASE, HU_MUSCLE_FALLBACK)
    gmask = gradient_mask_u8(img8, rng, grad_strength)
    base = cv2.bitwise_and(muscle_adapt, rng)
    base = cv2.bitwise_and(base, gmask)
    base = cv2.bitwise_and(base, cv2.bitwise_not(bone_ex))
    base = cv2.bitwise_and(base, cv2.bitwise_not(vbuf))

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - cx, cy - yy
    ang = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
    Rs = ((ang >= 0) & (ang <= 90)).astype(np.uint8) * 255
    Ls = ((ang >= 90) & (ang <= 180)).astype(np.uint8) * 255
    Rs = cv2.bitwise_and(Rs, inner)
    Ls = cv2.bitwise_and(Ls, inner)

    R_c = cv2.bitwise_and(base, Rs)
    L_c = cv2.bitwise_and(base, Ls)
    min_px = area_px_from_mm2(ds, min_area_mm2)
    max_px = area_px_from_mm2(ds, max_area_mm2)
    pxA = px_area_mm2(ds)

    def pick(cand):
        m = (cand > 0).astype(np.uint8)
        num, L, S, _ = cv2.connectedComponentsWithStats(m, 8)
        best = {
            "mask": None,
            "md": 1e9,
            "area": 0,
            "stats": None,
            "bbox": (0, 0),
            "shape": -1.0,
            "shape_dbg": (0, 0, 0, 0),
        }
        if num <= 1:
            return best
        for lbl in range(1, num):
            a = S[lbl, cv2.CC_STAT_AREA]
            if a < min_px or a > max_px:
                continue
            comp = L == lbl
            vals = hu[comp]
            mean_hu, std_hu = float(np.mean(vals)), float(np.std(vals))
            if not (-10 <= mean_hu <= 80 and std_hu >= 8):
                continue
            md = float(dist[comp].mean())
            if md < (2.0 / np.sqrt(pxA)):
                continue
            ecc, conv, axis_pct, shape_sc = shape_orientation_scores(
                comp.astype(np.uint8)
            )
            md_norm_inv = min(1.0, 1.0 / (md + 1e-3))
            comb = 0.6 * md_norm_inv + PSOAS_SHAPE_WT * 0.4 * shape_sc
            if (best["mask"] is None) or (comb > best["shape"]):
                best.update(
                    {
                        "mask": comp.astype(np.uint8) * 255,
                        "md": md,
                        "area": int(a),
                        "stats": (mean_hu, std_hu),
                        "bbox": (S[lbl, cv2.CC_STAT_WIDTH], S[lbl, cv2.CC_STAT_HEIGHT]),
                        "shape": comb,
                        "shape_dbg": (ecc, conv, axis_pct, shape_sc),
                    }
                )
        return best

    R = pick(R_c)
    L = pick(L_c)
    psoas = np.zeros_like(vertebra)
    if R["mask"] is not None:
        psoas = cv2.bitwise_or(psoas, R["mask"])
    if L["mask"] is not None:
        psoas = cv2.bitwise_or(psoas, L["mask"])
    psoas = cv2.bitwise_and(psoas, inner)
    psoas = cv2.bitwise_and(psoas, cv2.bitwise_not(vbuf))

    dbg = {
        "R_area_px": R["area"],
        "L_area_px": L["area"],
        "R_mean_dist": R["md"],
        "L_mean_dist": L["md"],
        "R_stats": R["stats"],
        "L_stats": L["stats"],
        "R_box": R["bbox"],
        "L_box": L["bbox"],
        "Rshape": R["shape"],
        "Lshape": L["shape"],
        "Rshape_dbg": R["shape_dbg"],
        "Lshape_dbg": L["shape_dbg"],
    }
    return psoas, dbg


def hu_adaptation(hu, inner, vertebra, ds):
    H, W = hu.shape
    pad = max(5, int(0.05 * min(H, W)))
    _ = np.median(
        np.concatenate(
            [
                hu[0:pad, 0:pad].ravel(),
                hu[0:pad, W - pad : W].ravel(),
                hu[H - pad : H, 0:pad].ravel(),
                hu[H - pad : H, W - pad : W].ravel(),
            ]
        )
    )

    inner_vals = hu[inner > 0]
    if inner_vals.size < 100:
        return (HU_FAT, HU_MUSCLE_BASE)
    fat_low = np.percentile(inner_vals, 10)
    fat_hi = np.percentile(inner_vals, 35)
    fat_lo = max(-250, fat_low - 15)
    fat_hi = min(-20, fat_hi + 15)
    fat_range = (max(HU_FAT[0], fat_lo), min(HU_FAT[1], fat_hi))

    nonfat = inner.copy()
    nonfat = cv2.bitwise_and(
        nonfat, cv2.bitwise_not(hu_mask(hu, fat_range[0], fat_range[1]))
    )
    nonfat = cv2.bitwise_and(nonfat, cv2.bitwise_not(hu_mask(hu, *HU_BONE_EXCLUDE)))
    mvals = hu[nonfat > 0]
    if mvals.size >= 200:
        p10, p95 = np.percentile(mvals, [10, 95])
        lo = max(HU_MUSCLE_FALLBACK[0], min(HU_MUSCLE_BASE[0], p10 - 15))
        hi = min(HU_MUSCLE_FALLBACK[1], max(HU_MUSCLE_BASE[1], p95 + 15))
        lo2 = max(HU_MUSCLE_FALLBACK[0], min(lo, hi - 5))
        hi2 = min(HU_MUSCLE_FALLBACK[1], max(hi, lo2 + 5))
        muscle_range = (float(lo2), float(hi2))
    else:
        muscle_range = HU_MUSCLE_BASE
    return (fat_range, muscle_range)


def vfa_pma_from_masks(hu, ds, inner, psoas_mask, fat_range=None):
    pxA = px_area_mm2(ds)
    fat_mask = hu_mask(hu, *(fat_range if fat_range else HU_FAT))
    VFA_px = int(np.count_nonzero(cv2.bitwise_and(fat_mask, inner)))
    PMA_px = int(np.count_nonzero(psoas_mask))
    VFA_mm2 = VFA_px * pxA
    PMA_mm2 = PMA_px * pxA
    return VFA_mm2, PMA_mm2


def build_overlay(hu, inner, vertebra, vfa_mask, psoas_mask):
    base = normalize8(hu, -200, 300)
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    base[inner == 0] = (base[inner == 0] * 0.25).astype(np.uint8)
    for m, col, al in (
        (vfa_mask, (0, 255, 0), 0.45),
        (psoas_mask, (0, 255, 255), 0.60),
        (vertebra, (255, 0, 0), 0.35),
    ):
        if m is None:
            continue
        m_in = cv2.bitwise_and(m, inner)
        a = ((m_in > 0).astype(np.float32)) * al
        colimg = np.full_like(base, col, np.uint8)
        base = (base * (1 - a[..., None]) + colimg * a[..., None]).astype(np.uint8)
    for m, c in ((vfa_mask, (0, 255, 0)), (psoas_mask, (0, 255, 255))):
        m_in = cv2.bitwise_and(m, inner)
        cnts, _ = cv2.findContours(
            (m_in > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(base, cnts, -1, c, 2, lineType=cv2.LINE_AA)
    return base


def confidence_breakdown(hu, ds, inner, vertebra, psoas_dbg, VFA_mm2, PMA_mm2):
    Ra, La = psoas_dbg.get("R_area_px", 0), psoas_dbg.get("L_area_px", 0)
    bilateral = 1.0 if (Ra > 0 and La > 0) else 0.0
    sym = 0.0
    if Ra > 0 and La > 0:
        sym = 1.0 - abs(Ra - La) / (Ra + La + 1e-6)
        sym = max(0.0, min(1.0, sym))

    hu_ok = 0.0
    for st in (psoas_dbg.get("R_stats"), psoas_dbg.get("L_stats")):
        if st is None:
            continue
        mean_hu, std_hu = st
        part = 0.0
        if -10 <= mean_hu <= 80:
            part += 0.6
        if std_hu >= 8:
            part += 0.4
        hu_ok += part
    hu_ok = hu_ok / 2.0

    inv_v = cv2.bitwise_not(vertebra)
    dist = cv2.distanceTransform(inv_v, cv2.DIST_L2, 3)
    md = 0.0
    cnt = 0
    for comp in ("R", "L"):
        mv = psoas_dbg.get(f"{comp}_mean_dist", None)
        if mv is not None and mv > 0:
            md += mv
            cnt += 1
    md = (md / cnt) if cnt > 0 else 0.0
    md_norm = min(0.2, (1.0 / (md + 1e-3))) / 0.2 if md > 0 else 0.0

    penalty = 0.0
    if PMA_mm2 < 80 or PMA_mm2 > 8000:
        penalty += 1.0
    if VFA_mm2 < 3000 or VFA_mm2 > 220000:
        penalty += 1.0

    shape_sc = 0.0
    for comp in ("Rshape", "Lshape"):
        val = psoas_dbg.get(comp, -1.0)
        if val >= 0:
            shape_sc = max(shape_sc, val)
    shape_sc = max(0.0, min(1.0, shape_sc))

    conf = (
        0.32 * bilateral
        + 0.22 * sym
        + 0.18 * hu_ok
        + 0.18 * md_norm
        + 0.10 * shape_sc
        - 0.30 * min(1.0, penalty)
    )
    conf = int(round(max(0.0, min(1.0, conf)) * 100))

    parts = {
        "bilateral": round(bilateral, 2),
        "symmetry": round(sym, 2),
        "hu_ok": round(hu_ok, 2),
        "bone_dist_norm": round(md_norm, 2),
        "shape": round(shape_sc, 2),
        "penalty": round(min(1.0, penalty), 2),
        "confidence": conf,
    }
    return conf, parts


def threshold_sensitivity(VFA_cm2, PMI, ratio, sex):
    txt = []
    vfa_thr = VFA_OBESITY_CUTOFF_CM2
    vfa_5 = (int(VFA_cm2 >= vfa_thr * 0.95), int(VFA_cm2 >= vfa_thr * 1.05))
    vfa_10 = (int(VFA_cm2 >= vfa_thr * 0.90), int(VFA_cm2 >= vfa_thr * 1.10))
    txt.append(
        f"VFA@100 → -5%:{vfa_5[0]} +5%:{vfa_5[1]} | -10%:{vfa_10[0]} +10%:{vfa_10[1]}"
    )
    if PMI is not None:
        pmi_thr = PMI_CUTOFF[sex]
        p5 = (int(PMI < pmi_thr * 0.95), int(PMI < pmi_thr * 1.05))
        p10 = (int(PMI < pmi_thr * 0.90), int(PMI < pmi_thr * 1.10))
        txt.append(
            f"PMI@{pmi_thr:.2f} → -5%:{p5[0]} +5%:{p5[1]} | -10%:{p10[0]} +10%:{p10[1]}"
        )
    if ratio is not None:
        r_thr = VFA_PSOAS_RATIO_CUTOFF
        r5 = (int(ratio >= r_thr * 0.95), int(ratio >= r_thr * 1.05))
        r10 = (int(ratio >= r_thr * 0.90), int(ratio >= r_thr * 1.10))
        txt.append(
            f"VFA/PMA@2.0 → -5%:{r5[0]} +5%:{r5[1]} | -10%:{r10[0]} +10%:{r10[1]}"
        )
    return " | ".join(txt)


def l3_level_check(hu, vertebra_mask):
    H, W = hu.shape
    area = float(cv2.countNonZero(vertebra_mask))
    frac = area / (H * W + 1e-6)
    cnts, _ = cv2.findContours(
        (vertebra_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    circ = 0.0
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        A = cv2.contourArea(cnt)
        P = cv2.arcLength(cnt, True) + 1e-6
        circ = 4.0 * np.pi * A / (P * P)
    img8 = normalize8(hu, -300, 300)
    e = cv2.Canny(img8, 60, 120)
    left = e[:, : W // 4]
    right = e[:, 3 * W // 4 :]
    rib_edge_density = (left.mean() + right.mean()) / 510.0
    flags = []
    lo, hi = L3_VERTEBRA_AREA_FRAC_RANGE
    if not (lo <= frac <= hi):
        flags.append("vertebra_area_out_of_range")
    if circ < L3_VERTEBRA_CIRC_MIN:
        flags.append("vertebra_low_roundness")
    if rib_edge_density > 0.10:
        flags.append("rib_edge_high")
    ok = len(flags) == 0
    return ok, {
        "area_frac": round(frac, 4),
        "roundness": round(circ, 3),
        "rib_edge": round(rib_edge_density, 3),
        "flags": flags,
    }


def uncertainty_band(hu, ds, inner, psoas_mask, fat_base, mus_base):
    results = []
    for fs, ms in UNCERTAINTY_HU_SHIFTS:
        fat = (fat_base[0] + fs, fat_base[1] + fs)
        VFA_mm2, PMA_mm2 = vfa_pma_from_masks(hu, ds, inner, psoas_mask, fat_range=fat)
        results.append((fs, ms, VFA_mm2, PMA_mm2))
    VFA_vals = [v for _, _, v, _ in results]
    PMA_vals = [p for _, _, _, p in results]
    return {
        "VFA_mm2_min": min(VFA_vals),
        "VFA_mm2_max": max(VFA_vals),
        "PMA_mm2_min": min(PMA_vals),
        "PMA_mm2_max": max(PMA_vals),
        "grid": results,
    }


# ===== Tek dilim çekirdeği =====
def process_slice_core(hu, ds, png_gray, use_png, band25_mm, height_mm):
    inner, wall, vb, (_cx, _cy), ray_qc = inner_abdomen_via_wall(
        hu, ds, png_gray if use_png else None, band25_mm, height_mm
    )
    (fat_range, muscle_range) = hu_adaptation(hu, inner, vb, ds)
    global HU_MUSCLE_BASE, HU_FAT
    old_muscle, old_fat = HU_MUSCLE_BASE, HU_FAT
    HU_MUSCLE_BASE, HU_FAT = muscle_range, fat_range

    psoas_mask, dbg = find_psoas_inside(
        hu,
        ds,
        inner,
        vb,
        ring_inner_mm=RING_INNER_MM_BASE,
        ring_outer_mm=RING_OUTER_MM_BASE,
        grad_strength=GRAD_STRENGTH_BASE,
        min_area_mm2=MIN_PSOAS_AREA_MM2_BASE,
        max_area_mm2=MAX_PSOAS_AREA_MM2_BASE,
    )
    vfa_mask = cv2.bitwise_and(hu_mask(hu, *HU_FAT), inner)
    HU_MUSCLE_BASE, HU_FAT = old_muscle, old_fat

    VFA_mm2, PMA_mm2 = vfa_pma_from_masks(hu, ds, inner, psoas_mask)
    return (
        inner,
        vb,
        vfa_mask,
        psoas_mask,
        VFA_mm2,
        PMA_mm2,
        dbg,
        (fat_range, muscle_range),
        ray_qc,
    )


# ===== Komşu dilimler =====
def neighbor_slices(dcm_path: Path):
    try:
        ds0 = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
        series_uid = ds0.get("SeriesInstanceUID", None)
        inst_num0 = int(ds0.get("InstanceNumber", 0))
        if series_uid is None:
            return []
        folder = dcm_path.parent
        cand = []
        for p in folder.glob("*.dcm"):
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=True)
                if ds.get("SeriesInstanceUID", None) == series_uid:
                    i = int(ds.get("InstanceNumber", 0))
                    cand.append((i, p))
            except Exception:
                continue
        cand = sorted(cand, key=lambda x: x[0])
        idx = [i for i, (n, _) in enumerate(cand) if cand[i][0] == inst_num0]
        if not idx:
            return []
        k = idx[0]
        neigh = []
        if k - 1 >= 0:
            neigh.append(cand[k - 1][1])
        if k + 1 < len(cand):
            neigh.append(cand[k + 1][1])
        return neigh
    except Exception:
        return []


def tukey_filter(values):
    vals = np.array(values, dtype=float)
    if len(vals) <= 2:
        return vals.tolist()
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    keep = (vals >= lo) & (vals <= hi)
    kept = vals[keep]
    return kept.tolist() if kept.size > 0 else vals.tolist()


# ===== Ana iş akışı =====
def process_one(
    dicom_path: Path,
    png_path: Path,
    height_m: float,
    sex: str,
    use_png=True,
    fuse_neighbors=True,
):
    ds = pydicom.dcmread(str(dicom_path))
    hu = load_hu(ds)
    png_gray = None
    if use_png and png_path and Path(png_path).exists():
        png_gray = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)

    (
        inner,
        vb,
        vfa_mask,
        psoas_mask,
        VFA_mm2,
        PMA_mm2,
        dbg,
        (fat_range, muscle_range),
        ray_qc,
    ) = process_slice_core(
        hu, ds, png_gray, use_png, PNG_BAND_MM_BASE, HEIGHT_BAND_MM_BASE
    )

    l3_ok, l3_dbg = l3_level_check(hu, vb)

    VFA_list, PMA_list = [VFA_mm2], [PMA_mm2]
    if fuse_neighbors:
        for npath in neighbor_slices(Path(dicom_path)):
            try:
                dsn = pydicom.dcmread(str(npath))
                hun = load_hu(dsn)
                inner_n, vb_n, vfa_n, psoas_n, VFA_n, PMA_n, dbg_n, _, _ = (
                    process_slice_core(
                        hun, dsn, None, False, PNG_BAND_MM_BASE, HEIGHT_BAND_MM_BASE
                    )
                )
                VFA_list.append(VFA_n)
                PMA_list.append(PMA_n)
            except Exception:
                continue

    VFA_keep = tukey_filter(VFA_list)
    PMA_keep = tukey_filter(PMA_list)
    VFA_mm2_f = float(np.median(VFA_keep))
    PMA_mm2_f = float(np.median(PMA_keep))

    uncert = uncertainty_band(hu, ds, inner, psoas_mask, fat_range, muscle_range)

    VFA_cm2 = VFA_mm2_f / 100.0
    ratio = (VFA_mm2_f / PMA_mm2_f) if PMA_mm2_f > 0 else None
    PMI = (PMA_mm2_f / 100.0) / (height_m**2) if height_m > 0 else None
    vfa_high = VFA_cm2 >= VFA_OBESITY_CUTOFF_CM2
    sarcopenia = (PMI is not None) and (PMI < PMI_CUTOFF[sex])
    so_flag = int(
        sarcopenia
        and ((ratio is not None and ratio >= VFA_PSOAS_RATIO_CUTOFF) or vfa_high)
    )

    conf, parts = confidence_breakdown(hu, ds, inner, vb, dbg, VFA_mm2_f, PMA_mm2_f)
    sens_txt = threshold_sensitivity(VFA_cm2, PMI, ratio, sex)
    ov = build_overlay(hu, inner, vb, vfa_mask, psoas_mask)

    warnings = []
    if not l3_ok:
        warnings.append("L3_check_warn")
    if ray_qc.get("fascia_continuity_score", 0) < RAY_MIN_HITS_RATIO:
        warnings.append("fascia_continuity_warn")

    res = {
        "ID": Path(dicom_path).stem,
        "Sex": sex,
        "Height_m": round(height_m, 3),
        "PMA_mm2": round(PMA_mm2_f, 1),
        "VFA_mm2": round(VFA_mm2_f, 1),
        "VFA_cm2": round(VFA_cm2, 1),
        "VFA_PMA_ratio": (round(ratio, 2) if ratio is not None else ""),
        "PMI_cm2_per_m2": (round(PMI, 2) if PMI is not None else ""),
        "Sarcopenia(PMI<cutoff)": int(bool(sarcopenia)),
        f"VFA≥{VFA_OBESITY_CUTOFF_CM2}cm2": int(bool(vfa_high)),
        "Obesity_VFA_based": int(bool(vfa_high)),
        "SO_Flag(1=yes)": so_flag,
        "Confidence": conf,
        "ConfidenceParts": parts,
        "Sensitivity": sens_txt,
        "FusionSlices": len(VFA_list),
        "FusionKept": {"VFA": VFA_keep, "PMA": PMA_keep},
        "Uncertainty": uncert,
        "L3_Check": {"ok": int(l3_ok), **l3_dbg},
        "HU_ranges": {"fat": fat_range, "muscle": muscle_range},
        "FasciaQC": ray_qc,
        "Warnings": warnings,
    }
    return (
        res,
        ov,
        {
            "hu": hu,
            "inner": inner,
            "vb": vb,
            "vfa_mask": cv2.bitwise_and(hu_mask(hu, *fat_range), inner),
            "psoas_mask": psoas_mask,
            "ds": ds,
        },
    )


# ===== QA Export =====
def save_overlay_png(img_bgr, out_dir, case_id):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{case_id}_overlay.png"
    cv2.imwrite(str(p), img_bgr)
    return p


def save_results_txt_json(res, out_dir, case_id):
    out_dir.mkdir(parents=True, exist_ok=True)
    t = out_dir / f"{case_id}_results.txt"
    j = out_dir / f"{case_id}_results.json"

    vfa_key = f"VFA≥{VFA_OBESITY_CUTOFF_CM2}cm2"
    ratio_val = res.get("VFA_PMA_ratio", "")
    pmi_val = res.get("PMI_cm2_per_m2", "")
    ratio_str = "NA" if ratio_val == "" else f"{ratio_val:.2f}"
    pmi_str = "NA" if pmi_val == "" else f"{pmi_val:.2f}"

    lines = [
        f"=== RESULT @ {VERSION} ===",
        f"ID: {res['ID']} | Sex: {res['Sex']} | Height(m): {res['Height_m']:.2f}",
        f"PMA: {res['PMA_mm2']:.1f} mm²   VFA: {res['VFA_mm2']:.1f} mm² ({res['VFA_cm2']:.1f} cm²)",
        f"VFA/PMA: {ratio_str}   PMI: {pmi_str} cm²/m²",
        f"Sarcopenia(PMI<cutoff): {res['Sarcopenia(PMI<cutoff)']}  {vfa_key}: {res[vfa_key]}  SO: {res['SO_Flag(1=yes)']}",
        f"Confidence: {res.get('Confidence', 0)} | parts: {res.get('ConfidenceParts', {})}",
        f"L3-check: {res.get('L3_Check', {})}",
        f"FasciaQC: {res.get('FasciaQC', {})} | Warnings: {res.get('Warnings', [])}",
        f"Uncertainty(HU±): {res.get('Uncertainty', {})}",
        f"Sensitivity: {res.get('Sensitivity', '')}",
        f"Fusion slices used: {res.get('FusionSlices', 1)} | kept: VFA{len(res['FusionKept']['VFA'])} PMA{len(res['FusionKept']['PMA'])}",
        f"HU ranges (fat/muscle): {res.get('HU_ranges', {})}",
    ]
    with open(t, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(j, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    return t, j


def export_sc_dicom_with_overlay(ds, overlay_bgr, out_dir, case_id):
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = overlay_bgr.shape[:2]
    sc = pydicom.dataset.Dataset()
    sc.file_meta = pydicom.dataset.FileMetaDataset()
    sc.file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    sc.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    sc.file_meta.ImplementationClassUID = generate_uid()
    sc.SOPClassUID = sc.file_meta.MediaStorageSOPClassUID
    sc.SOPInstanceUID = sc.file_meta.MediaStorageSOPInstanceUID
    sc.Modality = "OT"
    sc.PatientName = getattr(ds, "PatientName", "Anon")
    sc.PatientID = getattr(ds, "PatientID", "")
    sc.StudyInstanceUID = getattr(ds, "StudyInstanceUID", generate_uid())
    sc.SeriesInstanceUID = generate_uid()
    sc.FrameOfReferenceUID = getattr(ds, "FrameOfReferenceUID", generate_uid())
    sc.StudyDate = datetime.now().strftime("%Y%m%d")
    sc.StudyTime = datetime.now().strftime("%H%M%S")
    sc.SeriesNumber = "999"
    sc.InstanceNumber = "1"
    sc.SamplesPerPixel = 3
    sc.PhotometricInterpretation = "RGB"
    sc.Rows = h
    sc.Columns = w
    sc.BitsAllocated = 8
    sc.BitsStored = 8
    sc.HighBit = 7
    sc.PixelRepresentation = 0
    sc.PatientOrientation = ""
    sc.ImageType = ["DERIVED", "SECONDARY", "SCOVERLAY"]
    sc.SecondaryCaptureDeviceManufacturer = "AutoSO"
    sc.PixelData = overlay_bgr.tobytes()
    p = out_dir / f"{case_id}_SC_overlay.dcm"
    pydicom.dcmwrite(str(p), sc, write_like_original=False)
    return p


# ===== UI =====
def annotate_metrics(img, res):
    pad = 8
    gap = 6
    font = cv2.FONT_HERSHEY_SIMPLEX
    ratio_val = res.get("VFA_PMA_ratio", "")
    pmi_val = res.get("PMI_cm2_per_m2", "")
    ratio_str = "NA" if ratio_val == "" else f"{ratio_val:.2f}"
    pmi_str = "NA" if pmi_val == "" else f"{pmi_val:.2f}"
    vfa_key = f"VFA≥{VFA_OBESITY_CUTOFF_CM2}cm2"
    parts = res.get("ConfidenceParts", {})
    l3 = res.get("L3_Check", {})
    fqc = res.get("FasciaQC", {})
    lines = [
        f"ID:{res['ID']}  Sex:{res['Sex']}  H:{res['Height_m']:.2f} m   v{VERSION}",
        f"PMA: {res['PMA_mm2']:.1f} mm²  |  VFA: {res['VFA_mm2']:.1f} mm² ({res['VFA_cm2']:.1f} cm²)",
        f"VFA/PMA: {ratio_str}   PMI: {pmi_str} cm²/m²   SO: {res['SO_Flag(1=yes)']}",
        f"Sarcopenia(PMI<cutoff): {res['Sarcopenia(PMI<cutoff)']}   {vfa_key}: {res[vfa_key]}",
        f"Confidence: {res.get('Confidence',0)}  (bil:{parts.get('bilateral','-')}, sym:{parts.get('symmetry','-')}, HU:{parts.get('hu_ok','-')}, bone:{parts.get('bone_dist_norm','-')}, shape:{parts.get('shape','-')}, pen:{parts.get('penalty','-')})",
        f"L3-check: ok={res['L3_Check']['ok']} | area_frac={l3.get('area_frac','-')} | round={l3.get('roundness','-')} | rib={l3.get('rib_edge','-')} | flags={l3.get('flags',[])}",
        f"FasciaQC: cont={fqc.get('fascia_continuity_score','-')} gap={fqc.get('ray_gap_ratio','-')} repairs={fqc.get('repairs','-')} flags={fqc.get('flags',[])}",
        f"Unc.(HU±): VFA [{res['Uncertainty']['VFA_mm2_min']:.0f}-{res['Uncertainty']['VFA_mm2_max']:.0f}]  PMA [{res['Uncertainty']['PMA_mm2_min']:.0f}-{res['Uncertainty']['PMA_mm2_max']:.0f}] mm²",
        f"Sensitivity: {res.get('Sensitivity','')}",
        f"Fusion slices used: {res.get('FusionSlices',1)}  kept(VFA/PMA): {len(res['FusionKept']['VFA'])}/{len(res['FusionKept']['PMA'])}",
        f"Warnings: {res.get('Warnings',[])}",
    ]
    sizes = [cv2.getTextSize(t, font, FONT_SCALE, LINE_THICK)[0] for t in lines]
    box_w = max(s[0] for s in sizes) + 2 * pad
    box_h = sum(s[1] for s in sizes) + (len(sizes) - 1) * gap + 2 * pad
    x0, y0 = 12, 12
    x1, y1 = x0 + box_w, y0 + box_h
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, BOX_ALPHA, img, 1 - BOX_ALPHA, 0)
    y = y0 + pad + sizes[0][1]
    for t in lines:
        cv2.putText(
            img,
            t,
            (x0 + pad, y),
            font,
            FONT_SCALE,
            (255, 255, 255),
            LINE_THICK,
            cv2.LINE_AA,
        )
        y += cv2.getTextSize(t, font, FONT_SCALE, LINE_THICK)[0][1] + gap
    return img


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"L3 SO Analizi – {VERSION}")
        self.geometry("900x670")
        self.resizable(False, False)
        load_settings(Path(__file__).with_name("settings.json"))

        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="x")
        ttk.Label(frm, text="DICOM (.dcm):").grid(row=0, column=0, sticky="w")
        self.dcm_var = tk.StringVar(value="")
        ttk.Entry(frm, textvariable=self.dcm_var, width=64).grid(
            row=0, column=1, padx=6, pady=4, sticky="w"
        )
        ttk.Button(frm, text="Seç...", command=self.pick_dcm).grid(
            row=0, column=2, padx=4
        )

        ttk.Label(frm, text="PNG (aynı kesit):").grid(row=1, column=0, sticky="w")
        self.png_var = tk.StringVar(value="")
        ttk.Entry(frm, textvariable=self.png_var, width=64).grid(
            row=1, column=1, padx=6, pady=4, sticky="w"
        )
        ttk.Button(frm, text="Seç...", command=self.pick_png).grid(
            row=1, column=2, padx=4
        )

        self.use_png_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            frm, text="PNG kullan (kapat: DICOM-only)", variable=self.use_png_var
        ).grid(row=2, column=1, sticky="w")

        ttk.Label(frm, text="Cinsiyet:").grid(row=3, column=0, sticky="w")
        self.sex_var = tk.StringVar(value="M")
        ttk.Combobox(
            frm, textvariable=self.sex_var, values=["M", "F"], state="readonly", width=5
        ).grid(row=3, column=1, sticky="w")

        ttk.Label(frm, text="Boy (cm):").grid(row=4, column=0, sticky="w")
        self.h_var = tk.StringVar(value="180")
        ttk.Entry(frm, textvariable=self.h_var, width=10).grid(
            row=4, column=1, sticky="w"
        )

        self.fuse_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            frm, text="Çok-dilim füzyon (L3±1)", variable=self.fuse_var
        ).grid(row=5, column=1, sticky="w")

        rowb = ttk.Frame(self)
        rowb.pack(fill="x", padx=10, pady=(6, 2))
        ttk.Button(rowb, text="Analiz et", command=self.run).pack(side="left")
        ttk.Button(rowb, text="QA paketini dışa aktar", command=self.export_now).pack(
            side="left", padx=8
        )

        ttk.Label(self, text="Sonuçlar:").pack(anchor="w", padx=10)
        self.txt = tk.Text(self, height=18, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=10, pady=6)
        ttk.Label(
            self,
            text="Not: Fasya İÇİ ölçüm; registration, L3-check, ray-casting onarım ve outlier filtresi aktiftir.",
        ).pack(anchor="w", padx=10, pady=(0, 8))
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self._last = None

    def pick_dcm(self):
        f = filedialog.askopenfilename(
            title="DICOM seç", filetypes=[("DICOM", "*.dcm"), ("Tümü", "*.*")]
        )
        if f:
            self.dcm_var.set(f)

    def pick_png(self):
        f = filedialog.askopenfilename(
            title="PNG seç", filetypes=[("PNG", "*.png"), ("Tümü", "*.*")]
        )
        if f:
            self.png_var.set(f)

    def run(self):
        try:
            dcm = Path(self.dcm_var.get().strip())
            png = (
                Path(self.png_var.get().strip()) if self.png_var.get().strip() else None
            )
            if not dcm.exists():
                messagebox.showerror("Hata", "Geçerli DICOM seçin.")
                return
            sex = self.sex_var.get().upper()
            if sex not in ("M", "F"):
                messagebox.showerror("Hata", "Cinsiyet M/F olmalı.")
                return
            try:
                h_cm = float(self.h_var.get().replace(",", "."))
            except Exception:
                messagebox.showerror("Hata", "Boy (cm) sayısal olmalı.")
                return
            if h_cm <= 0:
                messagebox.showerror("Hata", "Boy pozitif olmalı.")
                return

            use_png = bool(self.use_png_var.get())
            fuse_neighbors = bool(self.fuse_var.get())

            res, ov, parts = process_one(
                dcm,
                png,
                h_cm / 100.0,
                sex,
                use_png=use_png,
                fuse_neighbors=fuse_neighbors,
            )
            if SHOW_METRICS_ON_IMAGE:
                ov = annotate_metrics(ov, res)
            vis = cv2.resize(
                ov, (int(ov.shape[1] * WINDOW_SCALE), int(ov.shape[0] * WINDOW_SCALE))
            )
            cv2.imshow(f"{res['ID']} overlay", vis)
            cv2.waitKey(1)
            self.show_res(res)
            self._last = {
                "res": res,
                "overlay": ov,
                "ds": parts["ds"],
                "img_parts": parts,
                "dcm_path": str(dcm),
            }
        except Exception as e:
            messagebox.showerror("Hata", f"İşleme hatası: {e}")

    def show_res(self, res):
        self.txt.delete("1.0", "end")
        vfa_key = f"VFA≥{VFA_OBESITY_CUTOFF_CM2}cm2"
        ratio_val = res.get("VFA_PMA_ratio", "")
        pmi_val = res.get("PMI_cm2_per_m2", "")
        ratio_str = "NA" if ratio_val == "" else f"{ratio_val:.2f}"
        pmi_str = "NA" if pmi_val == "" else f"{pmi_val:.2f}"

        def w(s=""):
            self.txt.insert("end", s + "\n")

        w(f"=== RESULT @ {VERSION} ===")
        w(f"ID: {res['ID']} | Sex: {res['Sex']} | Height(m): {res['Height_m']:.2f}")
        w(
            f"PMA: {res['PMA_mm2']:.1f} mm²   VFA: {res['VFA_mm2']:.1f} mm² ({res['VFA_cm2']:.1f} cm²)"
        )
        w(f"VFA/PMA: {ratio_str}   PMI: {pmi_str} cm²/m²")
        w(
            f"Sarcopenia(PMI<cutoff): {res['Sarcopenia(PMI<cutoff)']}  {vfa_key}: {res[vfa_key]}  SO: {res['SO_Flag(1=yes)']}"
        )
        w(
            f"Confidence: {res.get('Confidence', 0)} | parts: {res.get('ConfidenceParts', {})}"
        )
        w(f"L3-check: {res.get('L3_Check', {})}")
        w(f"FasciaQC: {res.get('FasciaQC', {})} | Warnings: {res.get('Warnings', [])}")
        u = res.get("Uncertainty", {})
        w(
            f"Uncertainty(HU±): VFA[{u.get('VFA_mm2_min','-')}-{u.get('VFA_mm2_max','-')}]  PMA[{u.get('PMA_mm2_min','-')}-{u.get('PMA_mm2_max','-')}] (mm²)"
        )
        w(f"Sensitivity: {res.get('Sensitivity', '')}")
        w(
            f"Fusion slices used: {res.get('FusionSlices', 1)} | kept: VFA{len(res['FusionKept']['VFA'])} PMA{len(res['FusionKept']['PMA'])}"
        )
        w(f"HU ranges (fat/muscle): {res.get('HU_ranges', {})}")

    def export_now(self):
        if not self._last:
            messagebox.showwarning("Uyarı", "Önce bir analiz çalıştırın.")
            return
        out_dir = filedialog.askdirectory(title="QA paket çıkış klasörü seçin")
        if not out_dir:
            return
        out_dir = (
            Path(out_dir)
            / f"{self._last['res']['ID']}_QA_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        try:
            _ = save_overlay_png(
                self._last["overlay"], out_dir, self._last["res"]["ID"]
            )
            _ = save_results_txt_json(
                self._last["res"], out_dir, self._last["res"]["ID"]
            )
            _ = export_sc_dicom_with_overlay(
                self._last["ds"],
                self._last["overlay"],
                out_dir,
                self._last["res"]["ID"],
            )
            messagebox.showinfo("Tamam", f"QA paket hazır: {out_dir}")
        except Exception as e:
            messagebox.showerror("Hata", f"Dışa aktarım başarısız: {e}")

    def on_close(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    App().mainloop()

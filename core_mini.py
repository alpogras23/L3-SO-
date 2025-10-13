#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basit ve sağlam bir çekirdek (mini):
- DICOM yükler, HU hesaplar
- İç abdominal alanı (fasya içi) kaba olarak tahmin eder
- Vertebra gövdesini bulur
- VFA: HU_FAT aralığında, inner içinde alan (mm^2)
- PMA: Vertebra çevresinde halka içinde, HU_MUSCLE_BASE'e göre kas alanı (mm^2)
- Overlay üretir
- settings.json ile konfigürasyonu override eder
Bu dosya, bozuk/karmaşık çekirdeklerde yaşanan hatalara karşı güvenli bir alternatif sağlar.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pydicom

# --- Konfigürasyon (varsayılanlar) ---
CFG = dict(
    HU_FAT=(-190, -30),
    HU_MUSCLE_BASE=(-29, 150),
    HU_BONE_SEED=(300, 3000),
    HU_BONE_CAND=(150, 2000),
    HU_BONE_EXCLUDE=(200, 4000),
    PMI_CUTOFF={"M": 6.0, "F": 3.9},
    VFA_PSOAS_RATIO_CUTOFF=2.0,
    VFA_OBESITY_CUTOFF_CM2=100.0,
    FASCIA_EDGE_Q=0.90,
    GRAD_STRENGTH_BASE=0.90,
    PSOAS_RING_INNER_MM=6.0,
    PSOAS_RING_OUTER_MM=22.0,
)


# --- CFG yardımcıları ---
def _as_float(x, dflt):
    try:
        return float(x)
    except Exception:
        return float(dflt)


def cfg_float(name: str, default: float) -> float:
    v = CFG.get(name, default)
    return _as_float(v, default)


def cfg_range(name: str, default: Tuple[float, float]) -> Tuple[float, float]:
    v = CFG.get(name, default)
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        try:
            return (_as_float(v[0], default[0]), _as_float(v[1], default[1]))
        except Exception:
            return default
    return default


def cfg_dict(name: str, default: dict) -> dict:
    v = CFG.get(name, default)
    return v if isinstance(v, dict) else default


def load_settings(root_dir: Path) -> None:
    sp = root_dir / "settings.json"
    if not sp.exists():
        return
    try:
        data = json.loads(sp.read_text(encoding="utf-8"))
        for k, v in data.items():
            if isinstance(v, list):
                CFG[k] = tuple(v)
            else:
                CFG[k] = v
        print(f"[INFO] settings.json yüklendi: {sp}")
    except Exception as e:
        print(f"[WARN] settings.json okunamadı: {e}")


# --- Yardımcılar ---


def load_hu(ds) -> np.ndarray:
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    inter = float(getattr(ds, "RescaleIntercept", 0.0))
    return arr * slope + inter


def pixel_spacing(ds) -> Tuple[float, float]:
    sp = getattr(ds, "PixelSpacing", None)
    if sp is None or len(sp) < 2:
        raise ValueError("PixelSpacing eksik.")
    return float(sp[0]), float(sp[1])


def px_area_mm2(ds) -> float:
    sy, sx = pixel_spacing(ds)
    return sy * sx


def mm_to_px(ds, mm: float) -> int:
    sy, sx = pixel_spacing(ds)
    s = (sy + sx) / 2.0
    return max(1, int(round(mm / max(1e-6, s))))


def normalize8(hu: np.ndarray, lo: float = -300, hi: float = 300) -> np.ndarray:
    img = np.clip(hu, lo, hi).astype(np.float32)
    norm = cv2.normalize(img, img.copy(), 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def hu_mask(hu: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return ((hu >= lo) & (hu <= hi)).astype(np.uint8) * 255


def centroid(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return None
    M = cv2.moments(m)
    if M["m00"] == 0:
        return None
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


# --- Basit segmentasyon ---


def body_mask_from_hu(hu: np.ndarray) -> np.ndarray:
    img8 = normalize8(hu, -300, 300)
    img8 = cv2.GaussianBlur(img8, (7, 7), 0)
    _, th = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Kanal ortalamalarını OpenCV ile hesaplamak, statik tip şikayetlerini önler
    m1 = cv2.mean(img8, mask=(th > 0).astype(np.uint8))[0]
    m0 = cv2.mean(img8, mask=(th == 0).astype(np.uint8))[0]
    if float(m1) < float(m0):
        th = 255 - th
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    return th


def vertebra_body_mask(hu: np.ndarray) -> np.ndarray:
    h, w = hu.shape
    cx, cy = w // 2, h // 2
    lo_s, hi_s = cfg_range("HU_BONE_SEED", (300, 3000))
    lo_c, hi_c = cfg_range("HU_BONE_CAND", (150, 2000))
    seed = hu_mask(hu, lo_s, hi_s)
    cand = hu_mask(hu, lo_c, hi_c)
    num, L, S, C = cv2.connectedComponentsWithStats(seed, connectivity=8)
    if num <= 1:
        return np.zeros_like(seed)
    minA, maxA = 0.001 * (h * w), 0.06 * (h * w)
    best, sc = 0, -1e9
    for lbl in range(1, num):
        x, y, ww, hh, a = S[lbl]
        if not (minA <= a <= maxA):
            continue
        mx, my = C[lbl]
        dist = math.hypot(mx - cx, my - cy)
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
    grown = cv2.bitwise_and(
        cand, cv2.dilate(seed_sel, np.ones((15, 15), np.uint8), iterations=1)
    )
    grown = cv2.morphologyEx(
        grown, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2
    )
    grown = cv2.morphologyEx(
        grown, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1
    )
    num2, L2, S2, _ = cv2.connectedComponentsWithStats(grown, connectivity=8)
    if num2 <= 1:
        return np.zeros_like(grown)
    best2, sc2 = 0, -1e9
    for lbl in range(1, num2):
        x, y, ww, hh, a = S2[lbl]
        comp2 = (L2 == lbl).astype(np.uint8)
        Mm = cv2.moments(comp2)
        if Mm["m00"] == 0:
            continue
        mx, my = Mm["m10"] / Mm["m00"], Mm["m01"] / Mm["m00"]
        dist = math.hypot(mx - cx, my - cy)
        s = a - 2.0 * dist
        if s > sc2:
            best2, sc2 = lbl, s
    vb = (L2 == best2).astype(np.uint8) * 255
    vb = cv2.morphologyEx(vb, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    return vb


def inner_abdomen_via_wall(
    hu: np.ndarray, ds, fascia_q: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    h, w = hu.shape
    body = body_mask_from_hu(hu)
    vb = vertebra_body_mask(hu)
    cen = centroid(vb) or (w // 2, h // 2)
    cx, cy = int(cen[0]), int(cen[1])

    # Sadece DICOM gradientine dayalı duvar kenarı
    img8 = normalize8(hu, -200, 300)
    gx = cv2.Sobel(img8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img8, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    q = float(cfg_float("FASCIA_EDGE_Q", 0.90))
    mag_flat = mag.reshape(-1).astype(np.float32)
    thr = float(np.percentile(mag_flat, q * 100.0))
    wall_edges = (mag >= thr).astype(np.uint8) * 255
    wall_edges = cv2.bitwise_and(wall_edges, body)

    gap = mm_to_px(ds, 6.0)
    wall_edges = cv2.morphologyEx(
        wall_edges, cv2.MORPH_CLOSE, np.ones((gap, gap), np.uint8), iterations=1
    )

    wall_barrier = cv2.dilate(wall_edges, np.ones((3, 3), np.uint8), iterations=1)
    barrier = cv2.bitwise_or(wall_barrier, cv2.bitwise_not(body))
    mask = np.zeros((h + 2, w + 2), np.uint8)
    tmp = (barrier == 0).astype(np.uint8) * 255
    tmp2 = tmp.copy()
    cv2.floodFill(tmp2, mask, (cx, cy), (128,))
    inner = (tmp2 == 128).astype(np.uint8) * 255
    inner = cv2.bitwise_and(inner, body)
    inner = cv2.morphologyEx(
        inner, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1
    )
    return inner, wall_edges, vb, (cx, cy)


def psoas_ring_mask(inner: np.ndarray, vb: np.ndarray, ds) -> np.ndarray:
    r_in = mm_to_px(ds, cfg_float("PSOAS_RING_INNER_MM", 6.0))
    r_out = mm_to_px(ds, cfg_float("PSOAS_RING_OUTER_MM", 22.0))
    dil_in = cv2.dilate(
        vb,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_in + 1, 2 * r_in + 1)),
        iterations=1,
    )
    dil_out = cv2.dilate(
        vb,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_out + 1, 2 * r_out + 1)),
        iterations=1,
    )
    ring = cv2.bitwise_and(cv2.subtract(dil_out, dil_in), inner)
    return ring


def find_psoas_simple(
    hu: np.ndarray, inner: np.ndarray, vb: np.ndarray, ds
) -> np.ndarray:
    ring = psoas_ring_mask(inner, vb, ds)
    lo_m, hi_m = cfg_range("HU_MUSCLE_BASE", (-29, 150))
    muscle = hu_mask(hu, lo_m, hi_m)
    cand = cv2.bitwise_and(muscle, ring)
    # Büyük komponentleri seçelim (en fazla iki taraf)
    num, L, S, _ = cv2.connectedComponentsWithStats(
        (cand > 0).astype(np.uint8), connectivity=8
    )
    out = np.zeros_like(cand)
    if num <= 1:
        return out
    stats = []
    h, w = hu.shape
    cx = w // 2
    for lbl in range(1, num):
        a = int(S[lbl, cv2.CC_STAT_AREA])
        if a < 50:
            continue
        comp = (L == lbl).astype(np.uint8) * 255
        # Sağ/sol ayrı en büyükleri al
        xs, ys = np.where(comp > 0)
        if xs.size == 0:
            continue
        xm = int(xs.mean())
        side = "R" if xm > cx else "L"
        stats.append((a, side, comp))
    # En büyük R ve L
    best_R = max([t for t in stats if t[1] == "R"], default=None, key=lambda t: t[0])
    best_L = max([t for t in stats if t[1] == "L"], default=None, key=lambda t: t[0])
    for item in (best_R, best_L):
        if item is not None:
            out = cv2.bitwise_or(out, item[2])
    # Küçük aç-kapa
    k = np.ones((3, 3), np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k, iterations=1)
    return out


def vfa_pma_from_masks(hu, ds, inner, psoas_mask, fat_range=None):
    pxA = px_area_mm2(ds)
    if fat_range is None:
        lo, hi = cfg_range("HU_FAT", (-190, -30))
    else:
        lo, hi = float(fat_range[0]), float(fat_range[1])
    fat_mask = hu_mask(hu, lo, hi)
    vfa_px = int(np.count_nonzero(cv2.bitwise_and(fat_mask, inner)))
    pma_px = int(np.count_nonzero(psoas_mask))
    VFA_mm2 = float(vfa_px * pxA)
    PMA_mm2 = float(pma_px * pxA)
    return VFA_mm2, PMA_mm2


def build_overlay(hu, inner, vb, vfa_mask, psoas_mask):
    base = normalize8(hu, -200, 300)
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    base[inner == 0] = (base[inner == 0] * 0.25).astype(np.uint8)
    for m, col, al in (
        (vfa_mask, (0, 255, 0), 0.35),
        (psoas_mask, (0, 255, 255), 0.55),
        (vb, (255, 0, 0), 0.35),
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


# --- Sonuç veri sınıfı ---
@dataclass
class Metrics:
    case_id: str
    sex: str
    height_m: float
    PMA_mm2: float
    VFA_mm2: float
    ratio_VFA_PMA: Optional[float]
    PMI_cm2_per_m2: Optional[float]
    vfa_high: int
    sarcopenia_pmi: int
    so_flag: int


# --- Ana API ---


def run_case(
    dcm_path: Path, height_cm: float, sex: str, png_path: Optional[Path] = None
) -> Tuple[Metrics, Optional[np.ndarray]]:
    dcm_path = Path(dcm_path)
    ds = pydicom.dcmread(str(dcm_path))
    hu = load_hu(ds)

    inner, wall, vb, _ = inner_abdomen_via_wall(hu, ds)
    psoas = find_psoas_simple(hu, inner, vb, ds)

    VFA_mm2, PMA_mm2 = vfa_pma_from_masks(hu, ds, inner, psoas, fat_range=CFG["HU_FAT"])
    VFA_cm2 = VFA_mm2 / 100.0
    ratio = (VFA_mm2 / PMA_mm2) if PMA_mm2 > 0 else None
    height_m = height_cm / 100.0
    PMI = (PMA_mm2 / 100.0) / (height_m**2) if height_m > 0 else None

    vfa_high = int(VFA_cm2 >= cfg_float("VFA_OBESITY_CUTOFF_CM2", 100.0))
    pmi_cut = float(cfg_dict("PMI_CUTOFF", {"M": 6.0, "F": 3.9}).get(sex.upper(), 6.0))
    sarcopenia = int((PMI is not None) and (PMI < pmi_cut))
    so_flag = int(
        bool(sarcopenia)
        and (
            (ratio is not None and ratio >= cfg_float("VFA_PSOAS_RATIO_CUTOFF", 2.0))
            or vfa_high
        )
    )

    lo_f2, hi_f2 = cfg_range("HU_FAT", (-190, -30))
    vfa_mask = cv2.bitwise_and(hu_mask(hu, lo_f2, hi_f2), inner)
    overlay = build_overlay(hu, inner, vb, vfa_mask, psoas)

    res = Metrics(
        case_id=dcm_path.stem,
        sex=sex.upper(),
        height_m=height_m,
        PMA_mm2=float(round(PMA_mm2, 1)),
        VFA_mm2=float(round(VFA_mm2, 1)),
        ratio_VFA_PMA=(float(round(ratio, 2)) if ratio is not None else None),
        PMI_cm2_per_m2=(float(round(PMI, 2)) if PMI is not None else None),
        vfa_high=int(vfa_high),
        sarcopenia_pmi=int(sarcopenia),
        so_flag=int(so_flag),
    )
    return res, overlay


def process_one(
    dicom_path, png_path=None, height_m=1.70, sex="M", use_png=True, fuse_neighbors=True
):
    """
    Esnek çağrı sözleşmesi:
    - Yol formu: process_one(dcm_path: Path|str, png_path: Optional[Path|str], height_m: float, sex: str,...)
    - Ham veri formu: process_one(hu: np.ndarray, ds: pydicom.Dataset, sex: str, height_m: float,...)
    """
    try:
        load_settings(Path(__file__).resolve().parent)
        # Yol formu mu, HU+DS formu mu?
        if isinstance(dicom_path, (str, Path)):
            dcm_p = Path(dicom_path)
            res_obj, overlay = run_case(dcm_p, float(height_m) * 100.0, str(sex))
        else:
            # quick_test: (hu, ds) ve keyword argümanları ile gelmiş
            hu = np.asarray(dicom_path)
            ds = png_path
            inner, wall, vb, _ = inner_abdomen_via_wall(hu, ds)
            psoas = find_psoas_simple(hu, inner, vb, ds)
            VFA_mm2, PMA_mm2 = vfa_pma_from_masks(
                hu, ds, inner, psoas, fat_range=CFG["HU_FAT"]
            )
            VFA_cm2 = VFA_mm2 / 100.0
            ratio = (VFA_mm2 / PMA_mm2) if PMA_mm2 > 0 else None
            PMI = (
                (PMA_mm2 / 100.0) / (float(height_m) ** 2)
                if float(height_m) > 0
                else None
            )
            vfa_high = int(VFA_cm2 >= cfg_float("VFA_OBESITY_CUTOFF_CM2", 100.0))
            pmi_cut = float(
                cfg_dict("PMI_CUTOFF", {"M": 6.0, "F": 3.9}).get(str(sex).upper(), 6.0)
            )
            sarcopenia = int((PMI is not None) and (PMI < pmi_cut))
            so_flag = int(
                bool(sarcopenia)
                and (
                    (
                        ratio is not None
                        and ratio >= cfg_float("VFA_PSOAS_RATIO_CUTOFF", 2.0)
                    )
                    or vfa_high
                )
            )
            lo_f2, hi_f2 = cfg_range("HU_FAT", (-190, -30))
            vfa_mask = cv2.bitwise_and(hu_mask(hu, lo_f2, hi_f2), inner)
            overlay = build_overlay(hu, inner, vb, vfa_mask, psoas)
            res_obj = Metrics(
                case_id="inline",
                sex=str(sex).upper(),
                height_m=float(height_m),
                PMA_mm2=float(round(PMA_mm2, 1)),
                VFA_mm2=float(round(VFA_mm2, 1)),
                ratio_VFA_PMA=(float(round(ratio, 2)) if ratio is not None else None),
                PMI_cm2_per_m2=(float(round(PMI, 2)) if PMI is not None else None),
                vfa_high=int(vfa_high),
                sarcopenia_pmi=int(sarcopenia),
                so_flag=int(so_flag),
            )
        vfa_cm2 = res_obj.VFA_mm2 / 100.0
        res = {
            "ID": res_obj.case_id,
            "Sex": res_obj.sex,
            "Height_m": float(res_obj.height_m),
            "PMA_mm2": float(res_obj.PMA_mm2),
            "VFA_mm2": float(res_obj.VFA_mm2),
            "VFA_cm2": float(round(vfa_cm2, 1)),
            "VFA_PMA_ratio": (
                float(res_obj.ratio_VFA_PMA)
                if res_obj.ratio_VFA_PMA is not None
                else None
            ),
            "PMI_cm2_per_m2": (
                float(res_obj.PMI_cm2_per_m2)
                if res_obj.PMI_cm2_per_m2 is not None
                else None
            ),
            "Sarcopenia(PMI<cutoff)": int(res_obj.sarcopenia_pmi),
            f"VFA≥{int(cfg_float('VFA_OBESITY_CUTOFF_CM2', 100.0))}cm2": int(
                res_obj.vfa_high
            ),
            "Obesity_VFA_based": int(res_obj.vfa_high),
            "SO_Flag(1=yes)": int(res_obj.so_flag),
            "VFA_OBESITY_CUTOFF": float(cfg_float("VFA_OBESITY_CUTOFF_CM2", 100.0)),
            "PMI_CUTOFF_USED": float(
                cfg_dict("PMI_CUTOFF", {"M": 6.0, "F": 3.9}).get(res_obj.sex, 6.0)
            ),
            "VFA_PSOAS_RATIO_CUTOFF_USED": float(
                cfg_float("VFA_PSOAS_RATIO_CUTOFF", 2.0)
            ),
        }
        return res, overlay
    except Exception:
        raise

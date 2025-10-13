#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AO_observer.py — Uygulamaya dokunmadan dışarıdan kanıt toplar.
- Belirttiğin klasörü periyodik tarar (varsayılan 1 sn)
- settings.json, çekirdek .py, DICOM/PNG (INPUT) ile
  results.txt/json, overlay.png, SC.dcm (OUTPUT) değişimlerini izler
- “INPUT değişti ama OUTPUT değişmedi” durumunu net raporlar
- Tüm log: terminal + AO_observer_log.txt
Kullanım:
  python3 AO_observer.py --dir ~/Desktop/L3_SO_ANALYSIS --interval 1 --fast
"""

import argparse
import fnmatch
import hashlib
import os
import sys
import time
from datetime import datetime
from pathlib import Path

INPUT_NAMES = {
    "settings.json",
    "l3_vfa_pma_core.py",
    "l3_vfa_pma_core_v3_7.py",
    "l3_vfa_pma_gui.py",
}
INPUT_EXTS = {".dcm", ".dicom", ".png", ".py", ".json"}
OUTPUT_HINT_EXT = {".txt", ".json", ".png", ".csv", ".dcm"}
OUTPUT_NAME_HINTS = ("overlay", "results", "QA", "SC_overlay")
LOGFILE = "AO_observer_log.txt"


def sha256sum(p: Path, fast=False, fast_bytes=1024 * 1024):
    h = hashlib.sha256()
    try:
        with p.open("rb") as f:
            if fast:
                h.update(f.read(fast_bytes))
            else:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        return f"ERR:{e}"


def classify(p: Path):
    name = p.name.lower()
    ext = p.suffix.lower()
    if name in INPUT_NAMES:
        return "INPUT"
    if ext in INPUT_EXTS:
        if name.endswith("_results.json") or "results" in name or "overlay" in name:
            return "OUTPUT"
        return "INPUT"
    if ext in OUTPUT_HINT_EXT or any(k in name for k in OUTPUT_NAME_HINTS):
        return "OUTPUT"
    return "UNKNOWN"


def matches_any(name, patterns):
    for pat in patterns:
        if fnmatch.fnmatch(name, pat):
            return True
    return False


def scan(root: Path, include, exclude, fast=False):
    entries = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root))
        if include and not matches_any(rel, include):
            continue
        if exclude and matches_any(rel, exclude):
            continue
        try:
            st = p.stat()
            entries[rel] = (
                st.st_size,
                int(st.st_mtime),
                sha256sum(p, fast=fast),
                classify(p),
            )
        except Exception:
            pass
    return entries


def log(msg, quiet=False):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    if not quiet:
        print(line, flush=True)
    try:
        with open(LOGFILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--include", default="*.dcm,*.dicom,*.png,*.py,*.json,*.txt,*.csv")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    root = Path(os.path.expanduser(args.dir)).resolve()
    if not root.exists():
        print(f"HATA: Klasör yok: {root}")
        sys.exit(1)

    include = [s.strip() for s in args.include.split(",") if s.strip()]
    exclude = [s.strip() for s in args.exclude.split(",") if s.strip()]

    log(
        f"OBSERVE START dir={root} interval={args.interval}s fast={args.fast} include={include} exclude={exclude}",
        args.quiet,
    )
    prev = scan(root, include, exclude, fast=args.fast)
    log(f"BASELINE count={len(prev)}", args.quiet)

    if args.once:
        log("Tek tarama tamam.", args.quiet)
        return

    while True:
        time.sleep(args.interval)
        now = scan(root, include, exclude, fast=args.fast)
        new = sorted(set(now) - set(prev))
        gone = sorted(set(prev) - set(now))
        common = sorted(set(now) & set(prev))

        for nf in new:
            s, m, h, c = now[nf]
            log(f"[NEW] {c:7s} {nf} size={s} mtime={m} sha={h[:12]}", args.quiet)
        for gf in gone:
            s, m, h, c = prev[gf]
            log(f"[DEL] {c:7s} {gf} (was size={s} sha={h[:12]})", args.quiet)

        chg_inp, chg_out = [], []
        for cf in common:
            s0, m0, h0, c0 = prev[cf]
            s1, m1, h1, c1 = now[cf]
            if (s0, m0, h0) != (s1, m1, h1):
                if c1 == "INPUT":
                    chg_inp.append(cf)
                if c1 == "OUTPUT":
                    chg_out.append(cf)
                log(
                    f"[CHG] {c1:7s} {cf} size {s0}->{s1} mtime {m0}->{m1} sha {h0[:12]}->{h1[:12]}",
                    args.quiet,
                )

        if chg_inp and not chg_out:
            log(
                "[NOTE] INPUT değişti ama OUTPUT değişmedi → çekirdek/ayar kullanılmıyor olabilir."
            )
        if chg_out and not chg_inp:
            log(
                "[NOTE] OUTPUT değişti ama INPUT değişmedi → eski sonucu tekrar yazma/önbellek olabilir."
            )
        if chg_inp and chg_out:
            log("[NOTE] INPUT ve OUTPUT birlikte değişti → beklenen tepki.")

        prev = now


if __name__ == "__main__":
    main()

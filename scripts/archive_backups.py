#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basit arşiv aracı: Kök klasördeki yedek/kopya dosyaları güvenle
`_archive_all_copies/<tarih-saat>/` altına taşır.

Kapsam (varsayılan örnek desenler):
- *.bak, *.bak*, *.save, *~
- .bak uzantılı varyantlar (örn. .bak_debugfix, .bak.20251012-121837)

Hariç tutulan klasörler:
- .git, .venv, __pycache__, .vscode
- build, L3 Analyzer.app, _archive_all_copies, DEBUG_OUT, _debug_out*

Kullanım:
  python scripts/archive_backups.py --dry-run        # yalnızca ne yapacağını gösterir
  python scripts/archive_backups.py                  # gerçekten taşır
  python scripts/archive_backups.py --patterns "*.bak,*.save,*~"  # özel desenler
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Set

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = ROOT / "_archive_all_copies"

DEFAULT_PATTERNS = [
    "*.bak",
    "*.bak2",
    "*.bak_*",
    "*.bak.*",
    "*.save",
    "*~",
]

EXCLUDE_DIRS: Set[str] = {
    ".git",
    ".venv",
    "__pycache__",
    ".vscode",
    "build",
    "L3 Analyzer.app",
    "_archive_all_copies",
    "DEBUG_OUT",
    "_debug_out",
    "_debug_out_FORCE",
}


def should_exclude(path: Path) -> bool:
    parts = set(p for p in path.parts)
    return any(ex in parts for ex in EXCLUDE_DIRS)


def iter_matches(patterns: Iterable[str]) -> Iterable[Path]:
    seen: Set[Path] = set()
    for pat in patterns:
        for p in ROOT.rglob(pat):
            if p.is_dir():
                continue
            if should_exclude(p):
                continue
            # Kendimizi ve core dosyalarını yanlışlıkla taşıma
            if p.name in {
                "core_mini.py",
                "l3_vfa_pma_core_v3_7.py",
                "l3_vfa_pma_v3_5.py",
            }:
                continue
            if p not in seen:
                seen.add(p)
                yield p


def unique_dest(base: Path) -> Path:
    if not base.exists():
        return base
    stem = base.stem
    suf = base.suffix
    parent = base.parent
    i = 1
    while True:
        cand = parent / f"{stem}__{i}{suf}"
        if not cand.exists():
            return cand
        i += 1


def archive_files(files: List[Path], dry_run: bool = True) -> int:
    if not files:
        print("[INFO] Arşivlenecek dosya bulunamadı.")
        return 0
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    batch_root = ARCHIVE_ROOT / ts
    count = 0
    for src in files:
        rel = src.relative_to(ROOT)
        dest = batch_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest = unique_dest(dest)
        if dry_run:
            print(f"[DRY] move: {src} -> {dest}")
        else:
            print(f"[DO ] move: {src} -> {dest}")
            shutil.move(str(src), str(dest))
        count += 1
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--patterns",
        default=",".join(DEFAULT_PATTERNS),
        help="Virgülle ayrılmış glob desenleri",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Sadece göster, taşıma yapma"
    )
    args = ap.parse_args()

    patterns = [p.strip() for p in args.patterns.split(",") if p.strip()]
    files = list(iter_matches(patterns))
    print(f"[INFO] Kökte: {ROOT}")
    print(f"[INFO] Desenler: {patterns}")
    print(f"[INFO] Bulunan: {len(files)} dosya")
    n = archive_files(files, dry_run=bool(args.dry_run))
    print(
        f"[OK] İşlem tamamlandı, {n} dosya {'listelendi' if args.dry_run else 'taşındı'}."
    )


if __name__ == "__main__":
    main()

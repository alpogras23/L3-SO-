#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L3 Core v3.5 (proxy) – Bu dosya, kararlı mini çekirdeğe (core_mini.py) yönlendiren ince bir sarmalayıcıdır.
Amaç: Tek bir sağlam motoru kullanarak GUI ve CLI yollarının tutarlı ve hatasız çalışmasını sağlamak.

Kamu API:
- Metrics (dataclass)
- load_settings(root_dir: Path) -> None
- run_case(dcm_path: Path, height_cm: float, sex: str, png_path: Optional[Path] = None)
- process_one(...): Yol formu veya (HU+DS) formu ile esnek kullanım

Not: Eski v3.5 içeriği (CNN, PDF vb.) bu proxy’de yoktur. Gerekirse core_mini üzerine bağımsız yardımcı
modüllerde yeniden sağlanabilir.
"""

from __future__ import annotations

from pathlib import Path  # noqa: F401  (API belgesinde atıf için tutuluyor)

# Bu dosya yalnızca core_mini’nin sabitlenmiş API’sini yeniden dışa aktarır.
try:
    from core_mini import Metrics, load_settings, process_one, run_case  # type: ignore
except Exception as _e:  # pragma: no cover - yükleme zamanı hatası
    # Yükleme hatasını daha okunur hale getirelim
    raise ImportError(
        "core_mini.py bulunamadı veya yüklenemedi; lütfen projenin kökünde core_mini.py olduğundan emin olun."
    ) from _e

__all__ = [
    "Metrics",
    "load_settings",
    "process_one",
    "run_case",
]

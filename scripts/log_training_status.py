#!/usr/bin/env python3
"""Log current training & segmentation status as structured JSON lines.

Bir defalık çalıştırılabilir veya --interval ile periyodik izleme yapabilir.

Örnek:
  python scripts/log_training_status.py                          # Tek snapshot
  python scripts/log_training_status.py --interval 1800          # 30 dakikada bir log ekle

Log formatı (JSON satırı):
{
  "ts": "2025-11-25T19:55:12Z",
  "segmentation": {"abd": 4, "total": 3},
  "merged": 3,
  "runs": [
     {"name": "run_mt_mini_nifti", "epoch": 4, "train_loss": 1.0365, "val_loss": 0.99345}
  ]
}

Notlar:
 - Eğitim run dizinleri: run_mt_* (içinde logs/version_*/metrics.csv aranır)
 - metrics.csv son satırlar çift kolon: val_loss satırı train_loss boş, train_loss satırı val_loss boş.
 - Segmentasyon durumunu .done dosyalarını sayarak çıkarır.
"""

from __future__ import annotations
import argparse
import csv
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

BASE = Path(__file__).resolve().parent.parent  # proje kökü

SEG_ROOT = BASE / "data" / "ts_labels_amos"
MERGE_ROOT = BASE / "data" / "ts_merged_amos"

@dataclass
class RunMetrics:
    name: str
    epoch: Optional[int]
    train_loss: Optional[float]
    val_loss: Optional[float]

    def as_dict(self):
        return {
            "name": self.name,
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
        }


def find_run_dirs(pattern: str = "run_mt_*") -> List[Path]:
    return [p for p in BASE.glob(pattern) if p.is_dir()]


def parse_metrics(run_dir: Path) -> Optional[RunMetrics]:
    # Lightning log yolları (gözlenen yapı: logs/version_1/metrics.csv)
    logs_dir = run_dir / "logs"
    if not logs_dir.is_dir():
        return None
    version_dirs = sorted(logs_dir.glob("version_*"))
    if not version_dirs:
        return None
    metrics_path = version_dirs[-1] / "metrics.csv"  # son versiyonu al
    if not metrics_path.is_file():
        return None

    last_train = None
    last_val = None
    last_epoch_train = None
    last_epoch_val = None

    with metrics_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Her epoch'ta iki satır (train + val) gözleniyor.
            epoch_str = row.get("epoch")
            epoch = int(epoch_str) if epoch_str not in (None, "") else None
            tl = row.get("train_loss")
            vl = row.get("val_loss")
            if tl:
                try:
                    last_train = float(tl)
                    last_epoch_train = epoch
                except ValueError:
                    pass
            if vl:
                try:
                    last_val = float(vl)
                    last_epoch_val = epoch
                except ValueError:
                    pass

    # Epoch tutarsız olabilir; öncelik val_epoch -> train_epoch
    epoch_final = last_epoch_val if last_epoch_val is not None else last_epoch_train
    if last_train is None and last_val is None:
        return None
    return RunMetrics(name=run_dir.name, epoch=epoch_final, train_loss=last_train, val_loss=last_val)


def count_segmentation_done() -> dict:
    abd = len(list(SEG_ROOT.glob("*_abdominal_muscles/.done"))) if SEG_ROOT.exists() else 0
    total = len(list(SEG_ROOT.glob("*_total/.done"))) if SEG_ROOT.exists() else 0
    return {"abd": abd, "total": total}


def count_merged() -> int:
    if not MERGE_ROOT.exists():
        return 0
    return len(list(MERGE_ROOT.glob("*_labels.nii.gz")))


def _get_current_case_from_log() -> Optional[str]:
    log_file = BASE / "run_ts_batch_all.log"
    if not log_file.is_file():
        return None
    try:
        lines = log_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None
    # Önce son [RUN] satırını ara
    for line in reversed(lines):
        if line.startswith("[RUN]") and "task" in line:
            parts = line.split()
            # format: [RUN] amos_0006 task total -> ...
            if len(parts) >= 2:
                return parts[1]
        if line.startswith("=== Case"):
            parts = line.split()
            if len(parts) >= 3:
                return parts[2]
    return None


def _process_info(pattern: str) -> List[Dict[str, Any]]:
    """ps çıktısından süreç CPU/MEM bilgilerini al.

    pattern örnekleri:
      - "TotalSegmentator"
      - "train_nifti_mini.py"
    """
    try:
        # macOS BSD ps: -axo pid,%cpu,%mem,command
        out = subprocess.check_output(["ps", "-axo", "pid,%cpu,%mem,command"], text=True)
    except Exception:
        return []
    rows = out.splitlines()[1:]
    info = []
    for r in rows:
        if pattern in r:
            parts = r.strip().split(None, 3)
            if len(parts) == 4:
                pid, cpu, mem, cmd = parts
                try:
                    info.append({"pid": int(pid), "cpu": float(cpu), "mem": float(mem), "cmd": cmd[:200]})
                except ValueError:
                    continue
    return info


def snapshot(pattern: str, include_proc: bool) -> dict:
    runs_data = []
    for rd in find_run_dirs(pattern):
        m = parse_metrics(rd)
        if m:
            runs_data.append(m.as_dict())
    runs_data.sort(key=lambda x: (x.get("epoch") if x.get("epoch") is not None else -1, x["name"]))

    snap = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "segmentation": count_segmentation_done(),
        "merged": count_merged(),
        "current_case": _get_current_case_from_log(),
        "runs": runs_data,
    }
    if include_proc:
        snap["proc"] = {
            "segmentator": _process_info("TotalSegmentator"),
            "trainers": _process_info("train_nifti_mini.py"),
            "auto_checkpoint": _process_info("auto_train_checkpoints.sh"),
        }
    return snap


def main():
    ap = argparse.ArgumentParser(description="Eğitim & segmentasyon durumunu logla (JSON satırları).")
    ap.add_argument("--log-file", default="training_monitor.log", help="Çıkış log dosyası (append)")
    ap.add_argument("--runs-pattern", default="run_mt_*", help="Eğitim dizin glob deseni")
    ap.add_argument("--interval", type=int, default=0, help=">0 ise saniye cinsinden periyodik loglama (örn: 1800 = 30 dk)")
    ap.add_argument("--stdout", action="store_true", help="Ayrıca stdout'a yaz")
    ap.add_argument("--include-proc", action="store_true", help="Süreç CPU/MEM bilgilerini ekle")
    args = ap.parse_args()

    log_path = BASE / args.log_file
    interval = args.interval

    def write_once():
        snap = snapshot(args.runs_pattern, args.include_proc)
        line = json.dumps(snap, ensure_ascii=False)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        if args.stdout:
            print(line)

    if interval <= 0:
        write_once()
        return

    try:
        while True:
            write_once()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Çıkılıyor (CTRL+C)")


if __name__ == "__main__":
    main()

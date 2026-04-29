import json
import os
import sys

import cv2

ROOT = os.getcwd()
PROJECT_DIR = os.path.join(ROOT, "desktop_project")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if PROJECT_DIR not in sys.path and os.path.isdir(PROJECT_DIR):
    sys.path.insert(0, PROJECT_DIR)

try:
    from l3_vfa_pma_core_v3_7 import process_one
except ModuleNotFoundError:
    from desktop_project.l3_vfa_pma_core_v3_7 import process_one  # type: ignore


def main() -> int:
    if len(sys.argv) < 5:
        print(
            "usage: python tools/eval_one.py /path/case.dcm /path/case.png HEIGHT_M SEX(M|F)"
        )
        return 1

    dcm_path = sys.argv[1]
    png_path = sys.argv[2]
    height_m = float(sys.argv[3])
    sex = sys.argv[4]

    fallback = png_path or dcm_path or ""
    case_id = os.path.splitext(os.path.basename(fallback))[0] or "case"

    result, overlay = process_one(dcm_path or None, png_path or None, height_m, sex)

    os.makedirs("out_eval", exist_ok=True)

    if overlay is not None:
        cv2.imwrite(f"out_eval/{case_id}_overlay.png", overlay)

    with open(f"out_eval/{case_id}_results.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(
        "[OK]",
        f"out_eval/{case_id}_overlay.png",
        f"out_eval/{case_id}_results.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

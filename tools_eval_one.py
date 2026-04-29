import json
import os
import sys

import cv2

sys.path.insert(0, os.getcwd())

from l3_vfa_pma_core_v3_7 import process_one


def main() -> None:
    if len(sys.argv) < 5:
        print(
            "usage: python tools_eval_one.py /path/to/case.dcm /path/to/case.png HEIGHT_M SEX(M|F)"
        )
        sys.exit(1)

    dcm, png, height_m, sex = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
    case_id = os.path.splitext(os.path.basename(png or dcm))[0]

    results, overlay = process_one(
        dcm, png, height_m, sex, case_id=case_id
    )

    os.makedirs("out_eval", exist_ok=True)
    overlay_path = f"out_eval/{case_id}_overlay.png"
    results_path = f"out_eval/{case_id}_results.json"

    cv2.imwrite(overlay_path, overlay)
    with open(results_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    print("[OK]", overlay_path, results_path)


if __name__ == "__main__":
    main()

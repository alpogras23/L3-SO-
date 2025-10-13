import sys

import numpy as np
import pydicom

import core_mini as core

if len(sys.argv) < 4:
    print("Kullanım: python quick_test.py <dcm_path> <Sex:M/F> <Height_m>")
    sys.exit(1)

dcm_path, sex, h = sys.argv[1], sys.argv[2].upper(), float(sys.argv[3])
ds = pydicom.dcmread(dcm_path)
slope = float(getattr(ds, "RescaleSlope", 1.0))
inter = float(getattr(ds, "RescaleIntercept", 0.0))
hu = ds.pixel_array.astype(np.float32) * slope + inter

res, overlay = core.process_one(hu, ds, sex=sex, height_m=h)
print("[TEST] RESULT:", res)
print(
    "[TEST] FORCE OUT DIR: /Users/alperenogras/Desktop/L3_SO_ANALYSIS/_debug_out_FORCE"
)

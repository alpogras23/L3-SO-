#!/usr/bin/env bash
# Quick manifest builder for AMOS mini test (3 cases).
# Creates CSV for multiteacher_train.py from NIfTI and merged TS labels.
# Usage: bash scripts/build_amos_mini_manifest.sh
set -euo pipefail

DATA_ROOT="$HOME/Desktop/amos22/imagesTr"
TS_MERGED="data/ts_merged_amos"
OUT_CSV="data/amos_mini_manifest.csv"
PY="${PYTHON_BIN:-$(pwd)/.venv/bin/python}"

mkdir -p "$(dirname "$OUT_CSV")"

# Write CSV header
echo "image,gt_mask,ts_mask,c2c_mask,w_gt,w_ts,w_c2c" > "$OUT_CSV"

for case_id in amos_0001 amos_0004 amos_0005; do
  nifti_img="$DATA_ROOT/${case_id}.nii.gz"
  ts_label="$TS_MERGED/${case_id}_labels.nii.gz"
  
  if [[ ! -f "$nifti_img" || ! -f "$ts_label" ]]; then
    echo "[WARN] Missing files for $case_id; skipping" >&2
    continue
  fi

  # For mini test: no AMOS GT psoas, no C2C yet (TS as main teacher)
  # w_gt=0 (AMOS GT has organs only), w_ts=1.0 (primary teacher), w_c2c=0 (not run yet)
  echo "$nifti_img,,$ts_label,,0.0,1.0,0.0" >> "$OUT_CSV"
done

line_count=$(wc -l < "$OUT_CSV")
if [[ $line_count -lt 2 ]]; then
  echo "[ERR] Manifest has no data rows" >&2; exit 1
fi

echo "[OK] Manifest written: $OUT_CSV ($((line_count - 1)) rows)"

#!/usr/bin/env bash
# Merge per-task TotalSegmentator outputs into multi-label mask for each case.
# Requires: abdominal_muscles + total tasks already run. Optional: tissue_types.
# Usage: bash scripts/ts_merge_all.sh --ts-root data/ts_labels_amos --out-dir data/ts_merged_amos [--cases amos_0001 amos_0002] [--include-fat]
# Output: out-dir/{case}_labels.nii.gz
set -euo pipefail

TS_ROOT="data/ts_labels_amos"
OUT_DIR="data/ts_merged_amos"
INCLUDE_FAT=0
CASES=()
PY="${PYTHON_BIN:-$(pwd)/.venv/bin/python}"
MERGER="scripts/merge_nifti_labels.py"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ts-root) TS_ROOT="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --include-fat) INCLUDE_FAT=1; shift 1 ;;
    --cases) shift; while [[ $# -gt 0 && $1 != --* ]]; do CASES+=("$1"); shift; done ;;
    --help) grep '^#' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$OUT_DIR"

if [[ ${#CASES[@]} -eq 0 ]]; then
  # derive from ts root (portable, no mapfile)
  while IFS= read -r line; do
    cid="$(echo "$line" | cut -d '_' -f1-2)"
    [[ -n "$cid" ]] && CAES_TMP+=("$cid") || true
  done < <(ls -1 "$TS_ROOT")
  # unique sort
  if [[ ${#CAES_TMP[@]} -gt 0 ]]; then
    CASES=($(printf '%s
' "${CAES_TMP[@]}" | sort -u))
  fi
fi

if [[ ! -f "$MERGER" ]]; then
  echo "Merger script not found: $MERGER" >&2; exit 1
fi

for case_id in "${CASES[@]}"; do
  # Expect directories: {case}_abdominal_muscles and {case}_total (and maybe {case}_tissue_types)
  abd_dir="$TS_ROOT/${case_id}_abdominal_muscles"
  total_dir="$TS_ROOT/${case_id}_total"
  tissue_dir="$TS_ROOT/${case_id}_tissue_types"

  out_path="$OUT_DIR/${case_id}_labels.nii.gz"
  if [[ -f "$out_path" ]]; then
    echo "[SKIP] $case_id merged already exists"
    continue
  fi

  if [[ ! -d "$abd_dir" || ! -f "$abd_dir/psoas_major_left.nii.gz" ]]; then
    echo "[WARN] Missing abdominal_muscles outputs for $case_id; skipping" >&2
    continue
  fi
  if [[ ! -d "$total_dir" || ! -f "$total_dir/vertebrae_L3.nii.gz" ]]; then
    echo "[WARN] Missing total outputs for $case_id; skipping" >&2
    continue
  fi

  cmd=("$PY" "$MERGER" --out "$out_path" \
    --label 55:"$abd_dir/psoas_major_left.nii.gz" \
    --label 56:"$abd_dir/psoas_major_right.nii.gz" \
    --label 29:"$total_dir/vertebrae_L3.nii.gz")

  if [[ $INCLUDE_FAT -eq 1 ]]; then
    if [[ -f "$tissue_dir/subcutaneous_fat.nii.gz" && -f "$tissue_dir/torso_fat.nii.gz" ]]; then
      cmd+=(--label 100:"$tissue_dir/torso_fat.nii.gz" --label 101:"$tissue_dir/subcutaneous_fat.nii.gz")
    else
      echo "[WARN] Fat labels requested but tissue_types outputs missing for $case_id" >&2
    fi
  fi

  echo "[MERGE] $case_id -> $out_path"
  "${cmd[@]}"
  echo "[OK] $case_id merged"
  echo
 done

echo "All merges attempted."

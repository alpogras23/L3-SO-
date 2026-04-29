#!/usr/bin/env bash
# Batch TotalSegmentator inference for AMOS cases.
# Default tasks: abdominal_muscles total
# Usage: bash scripts/ts_batch_segment.sh --input-root /path/imagesTr --out-root data/ts_labels_amos [--tasks "abdominal_muscles total"] [--max-cases N]
# Resumable: skips case tasks with .done marker.
# CPU friendly: sequential execution. Adjust TOTALSEG_ARGS for throttling.
set -euo pipefail

INPUT_ROOT=""
OUT_ROOT="data/ts_labels_amos"
TASKS=("abdominal_muscles" "total")
MAX_CASES=0
TOTALSEG_BIN="${TOTALSEG_BIN:-$(pwd)/.venv/bin/TotalSegmentator}"
TOTALSEG_ARGS="${TOTALSEG_ARGS:-}" # extra args e.g. --fast

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-root) INPUT_ROOT="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --tasks) IFS=' ' read -r -a TASKS <<< "$2"; shift 2 ;;
    --max-cases) MAX_CASES="$2"; shift 2 ;;
    --help) grep '^#' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$INPUT_ROOT" ]]; then
  echo "--input-root required" >&2; exit 1
fi
mkdir -p "$OUT_ROOT"

CASE_FILES=()
while IFS= read -r f; do
  CASE_FILES+=("$f")
done < <(find "$INPUT_ROOT" -maxdepth 1 -name 'amos_*.nii.gz' | sort)
if [[ ${#CASE_FILES[@]} -eq 0 ]]; then
  echo "No AMOS .nii.gz files found in $INPUT_ROOT" >&2; exit 1
fi

COUNT=0
for f in "${CASE_FILES[@]}"; do
  case_id=$(basename "$f" .nii.gz)
  ((COUNT++))
  if [[ $MAX_CASES -gt 0 && $COUNT -gt $MAX_CASES ]]; then
    echo "Reached max cases ($MAX_CASES). Stopping."; break
  fi
  echo "=== Case $case_id ($COUNT) ==="
  for task in "${TASKS[@]}"; do
    out_dir="$OUT_ROOT/${case_id}_${task}"
    done_marker="$out_dir/.done"
    if [[ -f "$done_marker" ]]; then
      echo "[SKIP] $case_id task $task already done"
      continue
    fi
    mkdir -p "$out_dir"
    echo "[RUN] $case_id task $task -> $out_dir"
    set +e
    "$TOTALSEG_BIN" -i "$f" -o "$out_dir" -ta "$task" $TOTALSEG_ARGS
    rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
      echo "[ERR] Task $task failed for $case_id (rc=$rc)" >&2
      continue
    fi
    touch "$done_marker"
    echo "[OK] $case_id $task"
  done
  echo
done

echo "All requested cases processed (or max limit reached)."

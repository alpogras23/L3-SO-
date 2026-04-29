#!/usr/bin/env bash
# Parallel TotalSegmentator batch processing (2-3 concurrent jobs)
# UYARI: CPU thermal load artar! Sadece güçlü sistemlerde kullan.
# Usage: bash scripts/ts_batch_parallel.sh --input-root DATA --out-root OUT [--parallel 2]
set -euo pipefail

INPUT_ROOT=""
OUT_ROOT="data/ts_labels_amos"
PARALLEL=2
TASKS=("abdominal_muscles" "total")
TOTALSEG_BIN="${TOTALSEG_BIN:-$(pwd)/.venv/bin/TotalSegmentator}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-root) INPUT_ROOT="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --parallel) PARALLEL="$2"; shift 2 ;;
    --tasks) IFS=' ' read -r -a TASKS <<< "$2"; shift 2 ;;
    --help) grep '^#' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$INPUT_ROOT" ]]; then
  echo "--input-root required" >&2; exit 1
fi
mkdir -p "$OUT_ROOT"

# Collect unprocessed cases
PENDING_CASES=()
while IFS= read -r f; do
  case_id=$(basename "$f" .nii.gz)
  # Check if both tasks done
  all_done=1
  for task in "${TASKS[@]}"; do
    done_marker="$OUT_ROOT/${case_id}_${task}/.done"
    if [[ ! -f "$done_marker" ]]; then
      all_done=0
      break
    fi
  done
  if [[ $all_done -eq 0 ]]; then
    PENDING_CASES+=("$case_id:$f")
  fi
done < <(find "$INPUT_ROOT" -maxdepth 1 -name 'amos_*.nii.gz' | sort)

echo "[INFO] Total pending cases: ${#PENDING_CASES[@]}"
echo "[INFO] Parallel jobs: $PARALLEL"
echo

# Process function for single case
process_case() {
  local case_id="$1"
  local input_file="$2"
  local log_file="$OUT_ROOT/${case_id}.log"
  
  echo "[START] $case_id (PID $$)" | tee -a "$log_file"
  
  for task in "${TASKS[@]}"; do
    out_dir="$OUT_ROOT/${case_id}_${task}"
    done_marker="$out_dir/.done"
    
    if [[ -f "$done_marker" ]]; then
      echo "  [SKIP] $task already done" | tee -a "$log_file"
      continue
    fi
    
    mkdir -p "$out_dir"
    echo "  [RUN] $task" | tee -a "$log_file"
    
    if "$TOTALSEG_BIN" -i "$input_file" -o "$out_dir" -ta "$task" >> "$log_file" 2>&1; then
      touch "$done_marker"
      echo "  [OK] $task" | tee -a "$log_file"
    else
      echo "  [ERR] $task failed (see $log_file)" | tee -a "$log_file"
    fi
  done
  
  echo "[DONE] $case_id" | tee -a "$log_file"
}

export -f process_case
export OUT_ROOT TASKS TOTALSEG_BIN

# Parallel execution using xargs
printf '%s\n' "${PENDING_CASES[@]}" | xargs -P "$PARALLEL" -I {} bash -c 'IFS=: read -r cid fpath <<< "{}"; process_case "$cid" "$fpath"'

echo
echo "All pending cases processed in parallel."

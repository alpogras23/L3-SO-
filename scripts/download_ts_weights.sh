#!/usr/bin/env bash
set -euo pipefail

# TotalSegmentator weight downloader
# Usage: bash scripts/download_ts_weights.sh [dataset_ids]
# If no IDs passed, defaults to 952 299
# Existing directories will be skipped unless --force provided.

FORCE=0
IDS=()
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    [0-9]*) IDS+=("$arg") ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

if [ ${#IDS[@]} -eq 0 ]; then
  IDS=(952 299)
fi

CACHE_DIR="$HOME/.totalsegmentator/nnunet/results"
mkdir -p "$CACHE_DIR"

# Map dataset ID -> URL
get_url() {
  local id="$1"
  case "$id" in
    297) echo "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset297_TotalSegmentator_total_3mm_1559subj.zip" ;;
    299) echo "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset299_body_1559subj.zip" ;;
    952) echo "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset952_abdominal_muscles_167subj.zip" ;;
    *) echo "Unsupported dataset id: $id" >&2; return 1 ;;
  esac
}

for id in "${IDS[@]}"; do
  url="$(get_url "$id")" || exit 1
  zip_name="$(basename "$url")"
  target_dir_name="${zip_name%.zip}" # remove .zip
  target_dir="$CACHE_DIR/$target_dir_name"

  if [ -d "$target_dir" ] && [ $FORCE -eq 0 ]; then
    echo "[SKIP] $id already present: $target_dir (use --force to re-download)"
    continue
  fi

  tmp_zip="/tmp/$zip_name"
  echo "[DL] Dataset $id -> $tmp_zip"
  # Resume download if partial
  curl -L -C - -o "$tmp_zip" "$url"

  echo "[UNZIP] $tmp_zip -> $CACHE_DIR"
  rm -rf "$target_dir" # ensure clean
  unzip -q "$tmp_zip" -d "$CACHE_DIR"

  # Basic sanity check: expect plans.json or dataset.json inside maybe
  files_count=$(find "$target_dir" -type f | wc -l | tr -d ' ')
  if [ "$files_count" -lt 5 ]; then
    echo "[WARN] Low file count ($files_count) in $target_dir. Check if download completed." >&2
  else
    echo "[OK] $target_dir contains $files_count files"
  fi

  # Remove zip to save space
  rm -f "$tmp_zip"
  echo "[DONE] Dataset $id installed."
  echo
 done

echo "All requested datasets processed."

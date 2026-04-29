#!/usr/bin/env bash
# 2 paralel iş ile belirli vakaları işle
# Kullanım: bash scripts/run_parallel_custom.sh

set -euo pipefail

NIFTI_ROOT="${HOME}/Desktop/amos22/imagesTr"
TS_ROOT="data/ts_labels_amos"
# Paralel iş sayısı ortam değişkeniyle override edilebilir (PARALLEL_JOBS)
PARALLEL=${PARALLEL_JOBS:-2}

# Adaptif paralel ayarı: macOS'ta top ile CPU kullanımını örnekle
adapt_parallel() {
  # CPU usage: "CPU usage: 10.00% user, 5.00% sys, 85.00% idle"
  local idle
  idle=$(top -l 1 | grep "CPU usage" | awk -F',' '{print $3}' | awk '{print int($1)}' 2>/dev/null || echo 50)
  # Konservatif: idle > 75 -> 3 iş; idle 55-75 -> 2 iş; idle < 55 -> 1 iş
  if [[ "$idle" -ge 75 ]]; then
    PARALLEL=3
  elif [[ "$idle" -ge 55 ]]; then
    PARALLEL=2
  else
    PARALLEL=1
  fi
}

# macOS'ta nice/renice tutarsız olduğundan kaldırıldı. Termal güvenlik için
# paralel iş sayısını sınırlıyoruz (PARALLEL_JOBS). Ayrıca Total için hızlandırma argümanları.
ABD_ARGS="-ta abdominal_muscles"      # abdominal_muscles için --fast kullanmıyoruz
TOTAL_ARGS="-ta total --fast --roi_subset torso"  # total için hızlandırma

# Mevcut NIfTI dosyalarından eksik vakaları otomatik bul
get_pending_cases() {
  local found=()
  for nii in "${NIFTI_ROOT}"/amos_*.nii.gz; do
    [[ -f "$nii" ]] || continue
    local case_id=$(basename "$nii" .nii.gz)
    local abd="${TS_ROOT}/${case_id}_abdominal_muscles/.done"
    local tot="${TS_ROOT}/${case_id}_total/.done"
    if [[ ! -f "$abd" ]] || [[ ! -f "$tot" ]]; then
      found+=("$case_id")
    fi
  done
  printf '%s\n' "${found[@]}" | head -30
}

# İşlenecek vakalar (dinamik) - macOS bash 3.2 uyumlu
CASES=()
while IFS= read -r case_id; do
  CASES+=("$case_id")
done < <(get_pending_cases)

process_case() {
  local case_id=$1
  local nifti="${NIFTI_ROOT}/${case_id}.nii.gz"
  
  if [[ ! -f "$nifti" ]]; then
    echo "[SKIP] $case_id - nifti not found"
    return 0
  fi
  
  # Abdominal muscles (no --fast flag, not compatible)
  local abd_dir="${TS_ROOT}/${case_id}_abdominal_muscles"
  if [[ ! -f "${abd_dir}/.done" ]]; then
    echo "[RUN] $case_id abdominal_muscles"
    mkdir -p "$abd_dir"
      .venv/bin/TotalSegmentator -i "$nifti" -o "$abd_dir" ${ABD_ARGS} > "${abd_dir}.log" 2>&1
    touch "${abd_dir}/.done"
    echo "[OK] $case_id abdominal_muscles"
  else
    echo "[SKIP] $case_id abdominal_muscles (already done)"
  fi
  
  # Total
  local total_dir="${TS_ROOT}/${case_id}_total"
  if [[ ! -f "${total_dir}/.done" ]]; then
    echo "[RUN] $case_id total"
    mkdir -p "$total_dir"
      .venv/bin/TotalSegmentator -i "$nifti" -o "$total_dir" ${TOTAL_ARGS} > "${total_dir}.log" 2>&1
    touch "${total_dir}/.done"
    echo "[OK] $case_id total"
  else
    echo "[SKIP] $case_id total (already done)"
  fi

  # Küçük bekleme ile sistemin soğumasına izin ver (varsayılan 3 sn)
  sleep "${COOLDOWN_SECS:-3}"
}

export -f process_case
export NIFTI_ROOT TS_ROOT

adapt_parallel
echo "=== Processing ${#CASES[@]} cases with $PARALLEL parallel jobs (adaptive) ==="
printf '%s\n' "${CASES[@]}" | xargs -P "$PARALLEL" -I {} bash -c 'process_case "$@"' _ {}

echo "=== All cases processed ==="

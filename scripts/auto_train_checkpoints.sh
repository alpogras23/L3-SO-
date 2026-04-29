#!/usr/bin/env bash
# Automatic checkpoint & incremental training for AMOS dataset.
# As cases complete, automatically merge and train incrementally.
# Usage: bash scripts/auto_train_checkpoints.sh [--watch-interval 3600]
set -euo pipefail

WATCH_INTERVAL=3600  # 1 hour
TS_ROOT="data/ts_labels_amos"
MERGED_ROOT="data/ts_merged_amos"
NIFTI_ROOT="$HOME/Desktop/amos22/imagesTr"
CHECKPOINT_LOG="auto_train_checkpoints.log"
LAST_TRAINED_COUNT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --watch-interval) WATCH_INTERVAL="$2"; shift 2 ;;
    --help) grep '^#' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$CHECKPOINT_LOG"
}

log "Auto-training checkpoints started (interval: ${WATCH_INTERVAL}s)"

while true; do
  # Count completed cases
  completed_abd=$(ls "$TS_ROOT"/*_abdominal_muscles/.done 2>/dev/null | wc -l | tr -d ' ')
  completed_total=$(ls "$TS_ROOT"/*_total/.done 2>/dev/null | wc -l | tr -d ' ')
  completed_min=$((completed_abd < completed_total ? completed_abd : completed_total))
  
  log "Completed cases: $completed_min (abd=$completed_abd, total=$completed_total)"
  
  # Check for training checkpoints
  if [[ $completed_min -ge 10 && $LAST_TRAINED_COUNT -lt 10 ]]; then
    log "🎯 CHECKPOINT: 10 cases reached. Starting merge + 10-epoch training..."
    
    # Merge first 10
    bash scripts/ts_merge_all.sh --ts-root "$TS_ROOT" --out-dir "$MERGED_ROOT" 2>&1 | tee -a "$CHECKPOINT_LOG"
    
    # Train
    python scripts/train_nifti_mini.py \
      --nifti-root "$NIFTI_ROOT" \
      --ts-merged "$MERGED_ROOT" \
      --epochs 10 \
      --batch-size 2 \
      --threads 1 \
      --out run_mt_10cases_10ep \
      2>&1 | tee -a "$CHECKPOINT_LOG"
    
    LAST_TRAINED_COUNT=10
    log "✅ 10-case checkpoint training complete"
    
  elif [[ $completed_min -ge 25 && $LAST_TRAINED_COUNT -lt 25 ]]; then
    log "🎯 CHECKPOINT: 25 cases reached. Starting merge + 20-epoch training..."
    
    bash scripts/ts_merge_all.sh --ts-root "$TS_ROOT" --out-dir "$MERGED_ROOT" 2>&1 | tee -a "$CHECKPOINT_LOG"
    
    python scripts/train_nifti_mini.py \
      --nifti-root "$NIFTI_ROOT" \
      --ts-merged "$MERGED_ROOT" \
      --epochs 20 \
      --batch-size 2 \
      --threads 1 \
      --out run_mt_25cases_20ep \
      2>&1 | tee -a "$CHECKPOINT_LOG"
    
    LAST_TRAINED_COUNT=25
    log "✅ 25-case checkpoint training complete"
    
  elif [[ $completed_min -ge 50 && $LAST_TRAINED_COUNT -lt 50 ]]; then
    log "🎯 CHECKPOINT: 50 cases reached. Starting merge + 30-epoch training..."
    
    bash scripts/ts_merge_all.sh --ts-root "$TS_ROOT" --out-dir "$MERGED_ROOT" 2>&1 | tee -a "$CHECKPOINT_LOG"
    
    python scripts/train_nifti_mini.py \
      --nifti-root "$NIFTI_ROOT" \
      --ts-merged "$MERGED_ROOT" \
      --epochs 30 \
      --batch-size 1 \
      --threads 1 \
      --out run_mt_50cases_30ep \
      2>&1 | tee -a "$CHECKPOINT_LOG"
    
    LAST_TRAINED_COUNT=50
    log "✅ 50-case checkpoint training complete"
    
  elif [[ $completed_min -ge 100 && $LAST_TRAINED_COUNT -lt 100 ]]; then
    log "🎯 CHECKPOINT: 100 cases reached. Starting merge + 40-epoch training..."
    
    bash scripts/ts_merge_all.sh --ts-root "$TS_ROOT" --out-dir "$MERGED_ROOT" 2>&1 | tee -a "$CHECKPOINT_LOG"
    
    python scripts/train_nifti_mini.py \
      --nifti-root "$NIFTI_ROOT" \
      --ts-merged "$MERGED_ROOT" \
      --epochs 40 \
      --batch-size 1 \
      --threads 1 \
      --out run_mt_100cases_40ep \
      2>&1 | tee -a "$CHECKPOINT_LOG"
    
    LAST_TRAINED_COUNT=100
    log "✅ 100-case checkpoint training complete"
    
  elif [[ $completed_min -ge 200 && $LAST_TRAINED_COUNT -lt 200 ]]; then
    log "🎯 CHECKPOINT: 200 cases reached. Starting merge + 50-epoch training..."
    
    bash scripts/ts_merge_all.sh --ts-root "$TS_ROOT" --out-dir "$MERGED_ROOT" 2>&1 | tee -a "$CHECKPOINT_LOG"
    
    python scripts/train_nifti_mini.py \
      --nifti-root "$NIFTI_ROOT" \
      --ts-merged "$MERGED_ROOT" \
      --epochs 50 \
      --batch-size 1 \
      --threads 1 \
      --out run_mt_200cases_50ep \
      2>&1 | tee -a "$CHECKPOINT_LOG"
    
    LAST_TRAINED_COUNT=200
    log "✅ 200-case checkpoint training complete"
    
  elif [[ $completed_min -ge 240 && $LAST_TRAINED_COUNT -lt 240 ]]; then
    log "🎯 FINAL CHECKPOINT: All 240 cases complete! Starting merge + 60-epoch final training..."
    
    bash scripts/ts_merge_all.sh --ts-root "$TS_ROOT" --out-dir "$MERGED_ROOT" 2>&1 | tee -a "$CHECKPOINT_LOG"
    
    python scripts/train_nifti_mini.py \
      --nifti-root "$NIFTI_ROOT" \
      --ts-merged "$MERGED_ROOT" \
      --epochs 60 \
      --batch-size 1 \
      --threads 1 \
      --out run_mt_full_240cases_60ep \
      2>&1 | tee -a "$CHECKPOINT_LOG"
    
    LAST_TRAINED_COUNT=240
    log "🎉 FINAL TRAINING COMPLETE! Best model: run_mt_full_240cases_60ep/best.ckpt"
    break
  fi
  
  # Sleep until next check
  log "Next check in ${WATCH_INTERVAL}s..."
  sleep "$WATCH_INTERVAL"
done

log "Auto-training checkpoints completed."

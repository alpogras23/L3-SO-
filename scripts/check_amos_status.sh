#!/usr/bin/env bash
# Quick status checker for AMOS multi-teacher training pipeline.
# Usage: bash scripts/check_amos_status.sh

set -euo pipefail

echo "=== AMOS Multi-Teacher Training Status ==="
echo

# Background process check
echo "📊 Background Processes:"
if pgrep -f "ts_batch_segment" > /dev/null; then
  echo "  ✅ Batch segmentation RUNNING ($(pgrep -f ts_batch_segment))"
  
  # Find TotalSegmentator process
  if pgrep -f "TotalSegmentator" > /dev/null; then
    ts_pid=$(pgrep -f "TotalSegmentator" | head -1)
    ts_cpu=$(ps -p "$ts_pid" -o %cpu= 2>/dev/null | tr -d ' ' || echo "N/A")
    ts_mem=$(ps -p "$ts_pid" -o %mem= 2>/dev/null | tr -d ' ' || echo "N/A")
    echo "    └─ TotalSegmentator: PID=$ts_pid CPU=${ts_cpu}% MEM=${ts_mem}%"
  fi
else
  echo "  ⏸️  No batch segmentation running"
fi
echo

# Segmentation progress
echo "🔬 Segmentation Progress:"
abd_done=$(ls data/ts_labels_amos/*_abdominal_muscles/.done 2>/dev/null | wc -l | tr -d ' ')
total_done=$(ls data/ts_labels_amos/*_total/.done 2>/dev/null | wc -l | tr -d ' ')
echo "  Abdominal muscles: $abd_done / 240"
echo "  Total (vertebrae): $total_done / 240"

if [[ -f "run_ts_batch_all.log" ]]; then
  current_case=$(grep -E "^=== Case" run_ts_batch_all.log | tail -1 | awk '{print $3}' || echo "unknown")
  echo "  Current case: $current_case"
fi
echo

# Merged labels
echo "🔗 Merged Labels:"
merged_count=$(ls data/ts_merged_amos/*_labels.nii.gz 2>/dev/null | wc -l | tr -d ' ')
echo "  Merged files: $merged_count / 240"
echo

# Training checkpoints
echo "🎓 Training Checkpoints:"
if [[ -d "run_mt_mini_nifti" ]]; then
  ckpt_count=$(ls run_mt_mini_nifti/*.ckpt 2>/dev/null | wc -l | tr -d ' ')
  echo "  Mini test (5 epoch): $ckpt_count checkpoint(s)"
fi
if [[ -d "run_mt_full_60e" ]]; then
  ckpt_count=$(ls run_mt_full_60e/*.ckpt 2>/dev/null | wc -l | tr -d ' ')
  echo "  Full training (60 epoch): $ckpt_count checkpoint(s)"
fi
echo

# Disk usage
echo "💾 Disk Usage:"
if [[ -d "data/ts_labels_amos" ]]; then
  echo "  TS labels: $(du -sh data/ts_labels_amos 2>/dev/null | awk '{print $1}')"
fi
if [[ -d "data/ts_merged_amos" ]]; then
  echo "  TS merged: $(du -sh data/ts_merged_amos 2>/dev/null | awk '{print $1}')"
fi
echo

# Recent log lines
echo "📝 Recent Log (last 5 lines):"
if [[ -f "run_ts_batch_all.log" ]]; then
  tail -5 run_ts_batch_all.log | sed 's/^/  /'
else
  echo "  (No log file)"
fi
echo

# Next steps
echo "🎯 Next Steps:"
if [[ $abd_done -lt 10 || $total_done -lt 10 ]]; then
  echo "  1. Wait for first 10 cases to complete"
  echo "  2. Monitor: tail -f run_ts_batch_all.log"
elif [[ $merged_count -lt $abd_done ]]; then
  echo "  1. Merge completed segmentations:"
  echo "     bash scripts/ts_merge_all.sh --ts-root data/ts_labels_amos --out-dir data/ts_merged_amos"
elif [[ $merged_count -ge 10 && $merged_count -lt 50 ]]; then
  echo "  1. Test training with first 10-20 cases"
  echo "  2. Continue monitoring segmentation progress"
elif [[ $merged_count -ge 50 ]]; then
  echo "  1. Consider intermediate training (30 epoch) with 50+ cases"
  echo "  2. Plan full 240-case training setup"
else
  echo "  Monitoring segmentation progress..."
fi
echo

echo "For detailed monitoring: tail -f run_ts_batch_all.log"

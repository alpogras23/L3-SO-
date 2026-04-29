#!/usr/bin/env bash
# Canlı yüzde ilerleme göstergesi.
# Kullanım:
#   bash scripts/live_percent_monitor.sh            # varsayılan interval=5s
#   bash scripts/live_percent_monitor.sh --interval 2
#   bash scripts/live_percent_monitor.sh --no-color
# Çıkmak için CTRL+C

set -euo pipefail

INTERVAL=5
COLOR=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval)
      shift
      INTERVAL=${1:-5}
      ;;
    --no-color)
      COLOR=0
      ;;
    *)
      echo "Bilinmeyen argüman: $1" >&2
      exit 1
      ;;
  esac
  shift || true
done

total_cases=240

color() {
  if [[ $COLOR -eq 1 ]]; then
    printf "$1"
  fi
}

draw_bar() {
  local pct=$1
  local width=${2:-40}
  local filled=$(( pct * width / 100 ))
  local bar=$(printf '%*s' "$filled" | tr ' ' '#')
  local empty=$(printf '%*s' "$((width-filled))" | tr ' ' '-')
  printf '[%s%s] %5.1f%%' "$bar" "$empty" "$pct"
}

get_case_progress() {
  # TQDM satırını yakala (örn: " 15%|█▍        | 4/27 [..]")
  if [[ -f run_ts_batch_all.log ]]; then
    local line
    line=$(grep -E '[0-9]+%\|' run_ts_batch_all.log | tail -1 || true)
    if [[ -n "$line" ]]; then
      local pct
      pct=$(echo "$line" | sed -E 's/^ *([0-9]+)%\|.*/\1/')
      local step total
      step=$(echo "$line" | awk -F '|' '{print $3}' | awk '{print $1}' | cut -d'/' -f1)
      total=$(echo "$line" | awk -F '|' '{print $3}' | awk '{print $1}' | cut -d'/' -f2)
      echo "$pct;$step;$total"
      return 0
    fi
  fi
  echo ""  # boş
}

trap 'echo; echo "Çıkış"; exit 0' INT

while true; do
  abd_done=$(ls data/ts_labels_amos/*_abdominal_muscles/.done 2>/dev/null | wc -l | tr -d ' ')
  total_done=$(ls data/ts_labels_amos/*_total/.done 2>/dev/null | wc -l | tr -d ' ')
  merged_done=$(ls data/ts_merged_amos/*_labels.nii.gz 2>/dev/null | wc -l | tr -d ' ')

  abd_pct=$(awk -v d=$abd_done -v t=$total_cases 'BEGIN{if(t>0)printf("%.2f",d*100/t);else print 0}')
  total_pct=$(awk -v d=$total_done -v t=$total_cases 'BEGIN{if(t>0)printf("%.2f",d*100/t);else print 0}')
  merged_pct=$(awk -v d=$merged_done -v t=$total_cases 'BEGIN{if(t>0)printf("%.2f",d*100/t);else print 0}')

  current_case=""
  if [[ -f run_ts_batch_all.log ]]; then
    current_case=$(grep -E '^=== Case' run_ts_batch_all.log | tail -1 | awk '{print $3}' || true)
    if [[ -z "$current_case" ]]; then
      current_case=$(grep -E '\[RUN\] amos_' run_ts_batch_all.log | tail -1 | awk '{print $2}' || true)
    fi
  fi

  case_prog=$(get_case_progress)  # pct;step;total
  case_pct=""; case_step=""; case_total=""
  if [[ -n "$case_prog" ]]; then
    case_pct=$(echo "$case_prog" | cut -d';' -f1)
    case_step=$(echo "$case_prog" | cut -d';' -f2)
    case_total=$(echo "$case_prog" | cut -d';' -f3)
  fi

  printf "\r"  # satır başına dön
  color "\033[1;36m"; printf "SEG:"; color "\033[0m"; printf " Abd:"; draw_bar "${abd_pct%.*}" 50
  printf "  Total:"; draw_bar "${total_pct%.*}" 50
  printf "  Merged:"; draw_bar "${merged_pct%.*}" 50

  if [[ -n "$current_case" ]]; then
    printf "  | Case %s" "$current_case"
    if [[ -n "$case_pct" ]]; then
      printf " (slice %s/%s, %s%%)" "$case_step" "$case_total" "$case_pct"
    fi
  fi

  # ETA kaba tahmini (lineer): tamamlanan total_done vaka üzerinden hız yakalanamazsa boş bırak
  if [[ $total_done -gt 0 ]]; then
    # Basit kalan gün tahmini (her vaka ~4 saat varsayımı)
    remaining=$((total_cases - total_done))
    est_hours=$(awk -v r=$remaining 'BEGIN{printf("%.1f", r*4)}')
    printf "  ETA≈%sh" "$est_hours"
  fi

  sleep "$INTERVAL"
done

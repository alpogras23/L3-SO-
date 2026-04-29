#!/bin/zsh
# Workspace -> Desktop kopyasını eşitle (tek kaynak olsun)
# Uyarı: Mevcut Desktop kopyasındaki değişiklikleri ezebilir.
# Güvenli kullanım için önce DRY_RUN=1 ile deneyin veya BACKUP=1 ile otomatik yedek alın.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SRC="$REPO_ROOT"
DST="$HOME/Desktop/L3_SO_ANALYSIS"

# Opsiyonlar
# DRY_RUN=1 -> rsync --dry-run
# BACKUP=1  -> senkron öncesi hedef klasörün timestamp'li bir kopyasını al
DRY_RUN_FLAG="${DRY_RUN:-0}"
BACKUP_FLAG="${BACKUP:-0}"

print() { builtin print -r -- "$*"; }

print "Kaynak:  $SRC"
print "Hedef:   $DST"
if [[ "$DRY_RUN_FLAG" == "1" ]]; then
  print "Mod:     DRY RUN (sadece neler değişecek listelenir)"
fi
if [[ "$BACKUP_FLAG" == "1" ]]; then
  print "Yedek:   Aktif (senkron öncesi $DST yedeklenecek)"
fi

mkdir -p "$DST"

# İsteğe bağlı yedek
if [[ "$BACKUP_FLAG" == "1" ]]; then
  TS="$(date +%Y%m%d-%H%M%S)"
  BACKUP_DIR="${DST%/}_backup_${TS}"
  print "→ Yedek alınıyor: $BACKUP_DIR"
  cp -a "$DST" "$BACKUP_DIR"
fi

# rsync opsiyonları
RSYNC_OPTS=(
  -a
  --delete
  --itemize-changes
  --exclude ".git/"
  --exclude ".DS_Store"
  --exclude "*.pyc"
  --exclude "__pycache__/"
  --exclude "desktop_project"   # workspace'taki hatalı symlink'i taşımayı engelle
  --exclude ".venv*/"          # .venv ve .venv_old_* dahil
  --exclude "psoas_ml/ckpts/"
  --exclude "psoas_ml/data/"
)

if [[ "$DRY_RUN_FLAG" == "1" ]]; then
  RSYNC_OPTS+=(--dry-run)
fi

print "→ Senkron başlıyor..."
rsync "${RSYNC_OPTS[@]}" "$SRC/" "$DST/"

if [[ "$DRY_RUN_FLAG" == "1" ]]; then
  print "\nℹ️  DRY RUN tamamlandı. Gerçek eşitleme için DRY_RUN=0 ile yeniden çalıştırın."
else
  print "\n✅ Eşitleme bitti. Desktop kopyası güncellendi."
  print "Artık Desktop kısayolları da workspace ile birebir aynı kodu çalıştırır."
fi

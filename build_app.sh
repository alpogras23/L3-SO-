#!/bin/zsh
APP_NAME="L3 Analyzer"
APP_DIR="$APP_NAME.app"
ICON_FILE="icon.icns"
SCRIPT="l3_vfa_pma_pmi_app.py"

# .app yapısını oluştur
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"

# Bilgileri yaz
cat > "$APP_DIR/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key> <string>$APP_NAME</string>
    <key>CFBundleDisplayName</key> <string>$APP_NAME</string>
    <key>CFBundleExecutable</key> <string>run.sh</string>
    <key>CFBundleIconFile</key> <string>icon.icns</string>
    <key>CFBundleIdentifier</key> <string>com.alperen.l3analyzer</string>
    <key>CFBundlePackageType</key> <string>APPL</string>
    <key>CFBundleVersion</key> <string>1.0</string>
    <key>CFBundleShortVersionString</key> <string>1.0</string>
</dict>
</plist>
EOF

# Çalıştırma dosyası
cat > "$APP_DIR/Contents/MacOS/run.sh" <<EOF
#!/bin/zsh
cd "\$(dirname "\$0")"/../../
source .venv/bin/activate
python "$SCRIPT"
EOF

chmod +x "$APP_DIR/Contents/MacOS/run.sh"

# İkonu kopyala
if [ -f "$ICON_FILE" ]; then
  cp "$ICON_FILE" "$APP_DIR/Contents/Resources/"
fi

echo "✅ $APP_NAME.app oluşturuldu."

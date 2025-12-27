# RunPod VS Code Extension Setup

## 1️⃣ RunPod API Key Alın

1. **RunPod Console'a gidin**: https://www.runpod.io/console/user/settings
2. Sol menü → **"API Keys"**
3. **"+ API Key"** → Name: `vscode-extension` → **Read & Write** → **Create**
4. API key'i kopyalayın (örnek: `ABC123XYZ...`)

## 2️⃣ VS Code'da API Key'i Kaydedin

### Yöntem A: Command Palette (⌘+Shift+P)
1. `⌘+Shift+P` → `RunPod: Set API Key` yazın
2. API key'inizi yapıştırın → Enter

### Yöntem B: Settings UI
1. `⌘+,` (Settings açılır)
2. Arama kutusuna `runpod` yazın
3. **"Runpod: Api Key"** alanını bulun
4. API key'inizi yapıştırın

### Yöntem C: settings.json
1. `⌘+Shift+P` → `Preferences: Open User Settings (JSON)`
2. Ekleyin:
```json
{
  "runpod.apiKey": "YOUR_API_KEY_HERE"
}
```

## 3️⃣ RunPod Template Deploy Et

### Command Palette'ten (⌘+Shift+P):

1. **`RunPod: Deploy Pod`** yazın → Enter
2. Pod konfigürasyonu:
   - **Name**: `l3-teacher-generation`
   - **GPU Type**: RTX 4090 seçin
   - **Docker Image**: `alpogras23/l3-teacher-generation:latest` 
     (veya GitHub build: `ghcr.io/alpogras23/l3-so-:psoas-improvement`)
   - **Volume**: 150GB
   - **Environment Variables**:
     ```
     INPUT_DIR=/runpod-volume/amos22
     OUTPUT_DIR=/runpod-volume/teacher_labels
     OMP_NUM_THREADS=8
     ```

3. **Deploy** butonuna tıklayın

## 4️⃣ Pod'a Bağlan

Pod deploy olduktan sonra:

1. **Activity Bar** → **RunPod** icon (yeni eklendi)
2. **Pods** listesinde yeni pod'unuzu görün
3. Sağ tık → **"Connect to Pod"**
4. VS Code yeni pencere açar (pod içinde terminal/editor)

## 5️⃣ AMOS22 Upload

Pod'a bağlandıktan sonra:

### Integrated Terminal'den:
```bash
# Pod içinde
cd /runpod-volume
ls -lh
```

### Local'den Pod'a dosya transfer:
1. `⌘+Shift+P` → `RunPod: Upload Files to Pod`
2. Local AMOS22 folder'ı seçin
3. Remote path: `/runpod-volume/amos22`

Veya VS Code Explorer'da:
1. Sol sidebar → RunPod Pods
2. Pod'a sağ tık → "Browse Files"
3. Dosyaları drag&drop ile upload

## 6️⃣ Script Çalıştır

Pod terminal'de:
```bash
cd /workspace
./run_teacher_generation.sh
```

Veya VS Code'dan:
1. `⌘+Shift+P` → `RunPod: Run Command on Pod`
2. Komut: `/workspace/run_teacher_generation.sh`

## 7️⃣ İzle

Pod terminal'de:
```bash
tail -f /tmp/*.log
watch -n 5 'ls /runpod-volume/teacher_labels/ | wc -l'
```

## 8️⃣ Sonuçları İndir

1. Pod'a bağlıyken Explorer'da `/runpod-volume/teacher_labels/` aç
2. Folder'a sağ tık → **"Download"**
3. Local path seçin: `~/Desktop/teacher_labels_output/`

## 9️⃣ Pod'u Durdur

1. RunPod sidebar → Pod'a sağ tık → **"Stop Pod"**
2. Veya **"Terminate Pod"** (volume'u da silmek için)

---

## 🚨 Sorun Giderme

### "API Key geçersiz" hatası
- API key'i tekrar kopyalayın, başında/sonunda boşluk olmasın
- RunPod console'da key'in "Read & Write" yetkisi olduğunu kontrol edin

### "GPU yok" hatası
- Pod deploy ederken GPU tipi seçtiğinizden emin olun
- RTX 4090 yoksa RTX 3090 veya A100 deneyin

### "Volume mount edilmedi"
- Pod oluştururken "Network Volume" ekleyin
- Mount path: `/runpod-volume`

### Docker image build edilmiyor
- GitHub repo public olmalı
- Branch: `psoas-improvement` doğru yazıldı mı?
- Dockerfile path: `.github/runpod/Dockerfile` doğru mu?

---

## ⚡ Hızlı Komutlar

| Komut | Kısayol |
|-------|---------|
| Deploy Pod | `⌘+Shift+P` → `RunPod: Deploy` |
| Connect | RunPod sidebar → Pod → Sağ tık → Connect |
| Terminal | `⌃+`` (pod bağlıyken) |
| Upload | `⌘+Shift+P` → `RunPod: Upload` |
| Stop Pod | RunPod sidebar → Stop |

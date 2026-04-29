#!/bin/bash
# TotalSegmentator Model Manuel İndirme Script'i
# Kullanım: bash scripts/download_ts_models.sh

set -e

# Renk tanımları
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

CACHE_DIR="$HOME/.totalsegmentator/nnunet/results"

echo -e "${GREEN}=== TotalSegmentator Model İndirme ===${NC}"
echo "Model dizini: $CACHE_DIR"
echo ""

# Model dizinini oluştur
mkdir -p "$CACHE_DIR"

# Fonksiyon: Model indir ve extract et
download_model() {
    local task_id=$1
    local url=$2
    local filename=$(basename "$url")
    local extract_dir="$CACHE_DIR"
    
    echo -e "${YELLOW}[Task $task_id] İndiriliyor: $filename${NC}"
    
    # Geçici indirme dizini
    local temp_file="$CACHE_DIR/$filename"
    
    # İndir
    if curl -L -o "$temp_file" "$url"; then
        echo -e "${GREEN}[Task $task_id] İndirme tamamlandı${NC}"
        
        # Extract
        echo -e "${YELLOW}[Task $task_id] Extract ediliyor...${NC}"
        unzip -q "$temp_file" -d "$extract_dir"
        
        # Temizle
        rm "$temp_file"
        echo -e "${GREEN}[Task $task_id] Kurulum tamamlandı ✅${NC}"
        echo ""
    else
        echo -e "${RED}[Task $task_id] İndirme HATASI ❌${NC}"
        return 1
    fi
}

echo "Hangi modelleri indirmek istiyorsunuz?"
echo "1) Sadece gerekli modeller (Task 952: Abdominal Muscles, Task 299: Body)"
echo "2) Tüm ana CT modelleri (291-300)"
echo "3) Sadece Task 952 (Abdominal Muscles - PSOAS)"
echo "4) İptal"
read -p "Seçim (1-4): " choice

case $choice in
    1)
        echo -e "${GREEN}Gerekli modeller indiriliyor...${NC}"
        download_model 952 "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset952_abdominal_muscles_167subj.zip"
        download_model 299 "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset299_body_1559subj.zip"
        ;;
    2)
        echo -e "${GREEN}Tüm ana CT modelleri indiriliyor...${NC}"
        download_model 291 "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset291_TotalSegmentator_part1_organs_1559subj.zip"
        download_model 292 "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset292_TotalSegmentator_part2_vertebrae_1532subj.zip"
        download_model 293 "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset293_TotalSegmentator_part3_cardiac_1559subj.zip"
        download_model 294 "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset294_TotalSegmentator_part4_muscles_1559subj.zip"
        download_model 295 "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset295_TotalSegmentator_part5_ribs_1559subj.zip"
        download_model 299 "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset299_body_1559subj.zip"
        download_model 300 "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset300_body_6mm_1559subj.zip"
        download_model 952 "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset952_abdominal_muscles_167subj.zip"
        ;;
    3)
        echo -e "${GREEN}Task 952 (Abdominal Muscles) indiriliyor...${NC}"
        download_model 952 "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset952_abdominal_muscles_167subj.zip"
        ;;
    4)
        echo "İptal edildi."
        exit 0
        ;;
    *)
        echo -e "${RED}Geçersiz seçim!${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}=== Kurulum Tamamlandı ===${NC}"
echo ""
echo "Yüklü modeller:"
ls -lh "$CACHE_DIR" | grep "Dataset"

echo ""
echo -e "${YELLOW}Test için çalıştırın:${NC}"
echo ".venv/bin/TotalSegmentator -i test.nii.gz -o output -ta abdominal_muscles"

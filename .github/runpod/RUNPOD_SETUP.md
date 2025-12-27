# RunPod GPU Setup for L3 Teacher Generation

## 📦 Prerequisites
- RunPod account: https://runpod.io
- AMOS22 dataset (.nii.gz files)
- Docker installed locally (for building image)

## 🔨 Step 1: Build and Push Docker Image

```bash
# From project root
cd /Users/alperenogras/Desktop/L3_SO_ANALYSIS

# Build Docker image
docker build -f .github/runpod/Dockerfile -t l3-teacher-generation:latest .

# Tag for Docker Hub (replace YOUR_USERNAME)
docker tag l3-teacher-generation:latest YOUR_USERNAME/l3-teacher-generation:latest

# Push to Docker Hub
docker login
docker push YOUR_USERNAME/l3-teacher-generation:latest
```

**Alternative**: RunPod supports GitHub Container Registry (ghcr.io) as well.

## 🚀 Step 2: Create RunPod Pod

1. **Go to RunPod Console**: https://www.runpod.io/console/pods
2. **Click "Deploy"** → **"GPU Pod"**
3. **Select GPU**:
   - **Recommended**: RTX 4090 ($0.34/hr) or A100 40GB ($1.39/hr)
   - **Budget**: RTX 3090 ($0.24/hr)
   - **High-end**: H100 ($2.89/hr)

4. **Configure Pod**:
   - **Template**: Select "Custom"
   - **Container Image**: `YOUR_USERNAME/l3-teacher-generation:latest`
   - **Container Disk**: 50 GB (minimum)
   - **Volume Mount**: 
     - Enable "Add Network Volume" or "Add Pod Volume"
     - Size: 100-200 GB (for AMOS22 dataset + outputs)
     - Mount path: `/runpod-volume`

5. **Environment Variables** (Optional):
   ```
   INPUT_DIR=/runpod-volume/amos22
   OUTPUT_DIR=/runpod-volume/teacher_labels
   FAST_MODE=false
   OMP_NUM_THREADS=8
   ```

6. **Click "Deploy"**

## 📤 Step 3: Upload AMOS22 Dataset

### Option A: Upload via RunPod Web Interface
1. Open your Pod
2. Click "Connect" → "Web Terminal" or "Jupyter"
3. Upload files to `/runpod-volume/amos22/`

### Option B: Upload via SSH (Faster for large datasets)
```bash
# Get pod SSH info from RunPod console
ssh root@<pod-ip> -p <ssh-port> -i ~/.ssh/id_ed25519

# On your local machine, rsync data
rsync -avz -e "ssh -p <ssh-port> -i ~/.ssh/id_ed25519" \
  /path/to/amos22/ \
  root@<pod-ip>:/runpod-volume/amos22/
```

### Option C: Download directly on pod
```bash
# SSH into pod
ssh root@<pod-ip> -p <ssh-port>

# Download AMOS22 (if you have a direct link)
cd /runpod-volume
wget <amos22-dataset-url>
unzip amos22.zip -d amos22/
```

## ▶️ Step 4: Run Teacher Generation

### Auto-start (if pod was configured correctly)
The container will automatically run `/workspace/run_teacher_generation.sh` on startup.

### Manual start (via SSH)
```bash
# SSH into pod
ssh root@<pod-ip> -p <ssh-port>

# Check GPU
nvidia-smi

# Run script
cd /workspace
./run_teacher_generation.sh
```

### Monitor progress
```bash
# Real-time logs
tail -f /workspace/teacher_generation.log

# Check output
ls -lh /runpod-volume/teacher_labels/
```

## 📥 Step 5: Download Results

### Option A: Download via RunPod interface
1. Open pod → File browser
2. Navigate to `/runpod-volume/teacher_labels/`
3. Download folder as ZIP

### Option B: Download via SSH/rsync (Recommended)
```bash
# From your local machine
rsync -avz -e "ssh -p <ssh-port> -i ~/.ssh/id_ed25519" \
  root@<pod-ip>:/runpod-volume/teacher_labels/ \
  ./teacher_labels_output/
```

### Option C: Upload to cloud storage from pod
```bash
# Install rclone on pod
apt-get update && apt-get install -y rclone

# Configure Azure/GCP/AWS
rclone config

# Sync to Azure Blob
rclone sync /runpod-volume/teacher_labels/ azure:your-container/teacher_labels/
```

## 💰 Cost Estimation

| GPU | Price/hr | Est. Time (240 cases) | Total Cost |
|-----|----------|----------------------|------------|
| RTX 3090 | $0.24 | 6-8 hours | $1.44-1.92 |
| RTX 4090 | $0.34 | 4-6 hours | $1.36-2.04 |
| A100 40GB | $1.39 | 2-3 hours | $2.78-4.17 |
| H100 | $2.89 | 1-2 hours | $2.89-5.78 |

**Recommended**: RTX 4090 for best price/performance (~$2 total)

## 🛠️ Troubleshooting

### Issue: Container fails to start
```bash
# Check logs
docker logs <container-id>

# Verify image exists
docker images | grep l3-teacher
```

### Issue: GPU not detected
```bash
# Inside pod
nvidia-smi

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: Out of memory
- Use smaller GPU batch or switch to A100 (40GB VRAM)
- Reduce `OMP_NUM_THREADS` to 4

### Issue: Data not found
```bash
# Verify mount
ls -lh /runpod-volume/amos22/

# Check file count
find /runpod-volume/amos22/ -name "*.nii.gz" | wc -l
```

## 🔄 Stopping and Cleanup

```bash
# From RunPod console:
1. Stop pod (to pause billing)
2. Download results first!
3. Delete pod when done
```

**Important**: RunPod charges by the minute. Stop your pod immediately after results are downloaded.

## 📊 Expected Output Structure

```
/runpod-volume/teacher_labels/
├── amos_0001/
│   ├── hu_slice.npy          # HU values at L3
│   ├── psoas_left.npy        # Binary mask
│   ├── psoas_right.npy       # Binary mask
│   ├── vat.npy               # Visceral adipose tissue mask
│   ├── inner_abdomen.npy     # Inner abdominal wall mask
│   └── metadata.json         # Slice info, spacing, etc.
├── amos_0002/
│   └── ...
└── amos_0240/
    └── ...
```

## 🎯 Next Steps After Generation

1. Download `teacher_labels/` to your local machine
2. Upload to Azure ML workspace as dataset:
   ```bash
   az ml data create --name amos22-teacher-labels \
     --version 1 \
     --path ./teacher_labels/ \
     --resource-group l3-rg \
     --workspace-name l3-vfa-pma-ws
   ```
3. Continue with Step 2 training (U-Net psoas segmentation)

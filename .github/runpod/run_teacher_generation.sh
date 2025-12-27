#!/bin/bash
set -e

echo "========================================================================"
echo "🚀 RunPod L3 Teacher Generation - Starting"
echo "========================================================================"

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  No GPU detected - will run on CPU (slow)"
fi

# Default paths (can be overridden via environment variables)
INPUT_DIR="${INPUT_DIR:-/runpod-volume/amos22}"
OUTPUT_DIR="${OUTPUT_DIR:-/runpod-volume/teacher_labels}"
FAST_MODE="${FAST_MODE:-false}"
ROI_SUBSET="${ROI_SUBSET:-body}"

echo ""
echo "📁 Configuration:"
echo "   Input:  $INPUT_DIR"
echo "   Output: $OUTPUT_DIR"
echo "   Fast:   $FAST_MODE"
echo "   ROI:    $ROI_SUBSET"
echo ""

# Verify input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ ERROR: Input directory not found: $INPUT_DIR"
    echo "   Please mount your AMOS22 dataset to /runpod-volume/amos22"
    exit 1
fi

# Count input files
FILE_COUNT=$(find "$INPUT_DIR" -name "*.nii.gz" | wc -l)
echo "📊 Found $FILE_COUNT NII.GZ files to process"

if [ "$FILE_COUNT" -eq 0 ]; then
    echo "❌ ERROR: No .nii.gz files found in $INPUT_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run teacher generation
echo ""
echo "========================================================================"
echo "🔥 Starting TotalSegmentator Teacher Generation"
echo "========================================================================"

cd /workspace
python step1_real_teachers_all.py \
    --input-data "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Teacher generation completed successfully!"
    echo "📁 Results saved to: $OUTPUT_DIR"
    
    # Count output files
    OUTPUT_COUNT=$(find "$OUTPUT_DIR" -type d -mindepth 1 -maxdepth 1 | wc -l)
    echo "📊 Generated labels for $OUTPUT_COUNT cases"
else
    echo "❌ Teacher generation failed with exit code: $EXIT_CODE"
fi
echo "========================================================================"

exit $EXIT_CODE

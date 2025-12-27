#!/bin/bash
# Quick build and test script for RunPod container

set -e

echo "🔨 Building RunPod Docker container..."
docker build -f .github/runpod/Dockerfile -t l3-teacher-generation:latest .

echo ""
echo "✅ Build complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Tag for your Docker Hub account:"
echo "      docker tag l3-teacher-generation:latest YOUR_USERNAME/l3-teacher-generation:latest"
echo ""
echo "   2. Push to Docker Hub:"
echo "      docker login"
echo "      docker push YOUR_USERNAME/l3-teacher-generation:latest"
echo ""
echo "   3. Follow instructions in .github/runpod/RUNPOD_SETUP.md"
echo ""
echo "🧪 To test locally (if you have AMOS22 data):"
echo "   docker run --rm --gpus all \\"
echo "     -v /path/to/amos22:/runpod-volume/amos22 \\"
echo "     -v /path/to/output:/runpod-volume/teacher_labels \\"
echo "     l3-teacher-generation:latest"

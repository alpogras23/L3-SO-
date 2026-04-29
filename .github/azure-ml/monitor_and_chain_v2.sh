#!/bin/bash
set -e

JOB1="willing_tree_xjsd3mlpwt"
echo "🔍 Monitoring Step 1: $JOB1"
echo "Portal: https://ml.azure.com/runs/$JOB1"
echo ""

# Wait for Step 1
while true; do
  STATUS=$(az ml job show --name $JOB1 -g l3-rg -w l3-vfa-pma-ws --query status -o tsv 2>/dev/null || echo "Unknown")
  echo "[$(date +%H:%M:%S)] Step 1 Status: $STATUS"
  
  if [ "$STATUS" = "Completed" ]; then
    echo "✅ Step 1 COMPLETED!"
    break
  elif [ "$STATUS" = "Failed" ] || [ "$STATUS" = "Canceled" ]; then
    echo "❌ Step 1 FAILED: $STATUS"
    exit 1
  fi
  
  sleep 60  # Check every minute (mock generation is fast)
done

# Submit Step 2
echo ""
echo "🚀 Starting Step 2: Training..."
JOB2=$(az ml job create --file job_step2_training.yml -g l3-rg -w l3-vfa-pma-ws \
  --set inputs.teacher_labels=azureml://jobs/$JOB1/outputs/teacher_labels \
  --query name -o tsv)

echo "✅ Step 2 Job ID: $JOB2"
echo "Portal: https://ml.azure.com/runs/$JOB2"
echo ""

# Wait for Step 2
while true; do
  STATUS=$(az ml job show --name $JOB2 -g l3-rg -w l3-vfa-pma-ws --query status -o tsv 2>/dev/null || echo "Unknown")
  echo "[$(date +%H:%M:%S)] Step 2 Status: $STATUS"
  
  if [ "$STATUS" = "Completed" ]; then
    echo "✅ Step 2 COMPLETED!"
    break
  elif [ "$STATUS" = "Failed" ] || [ "$STATUS" = "Canceled" ]; then
    echo "❌ Step 2 FAILED: $STATUS"
    exit 1
  fi
  
  sleep 300  # Check every 5 minutes (training is long)
done

# Submit Step 3
echo ""
echo "🚀 Starting Step 3: Inference..."
JOB3=$(az ml job create --file job_step3_inference.yml -g l3-rg -w l3-vfa-pma-ws \
  --set inputs.trained_model=azureml://jobs/$JOB2/outputs/trained_model \
  --query name -o tsv)

echo "✅ Step 3 Job ID: $JOB3"
echo "Portal: https://ml.azure.com/runs/$JOB3"
echo ""

# Wait for Step 3
while true; do
  STATUS=$(az ml job show --name $JOB3 -g l3-rg -w l3-vfa-pma-ws --query status -o tsv 2>/dev/null || echo "Unknown")
  echo "[$(date +%H:%M:%S)] Step 3 Status: $STATUS"
  
  if [ "$STATUS" = "Completed" ]; then
    echo "✅ Step 3 COMPLETED!"
    echo "🎉 FULL PIPELINE COMPLETED!"
    echo "Results available at: https://ml.azure.com/runs/$JOB3"
    break
  elif [ "$STATUS" = "Failed" ] || [ "$STATUS" = "Canceled" ]; then
    echo "❌ Step 3 FAILED: $STATUS"
    exit 1
  fi
  
  sleep 60  # Check every minute
done

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║              🎉 PIPELINE TAMAMLANDI! 🎉                              ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Job IDs:"
echo "  Step 1 (Teachers): $JOB1"
echo "  Step 2 (Training): $JOB2"
echo "  Step 3 (Inference): $JOB3"
echo ""
echo "Download results:"
echo "  az ml job download --name $JOB3 -g l3-rg -w l3-vfa-pma-ws --all"


#!/bin/bash
#
# Auto-Pipeline Orchestrator
# Monitors Step 1, auto-chains Step 2, then Step 3
# Runs in background, provides real-time status updates
#

set -e

# Configuration
JOB_STEP1="shy_roti_nkl78tcjgr"
WORKSPACE="l3-vfa-pma-ws"
RESOURCE_GROUP="l3-rg"
POLL_INTERVAL=30  # Check status every 30 seconds
MAX_WAIT_HOURS=10  # Max wait time for Step 1

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Activate venv
source "$PROJECT_ROOT/.venv/bin/activate"

# Helper functions
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

get_job_status() {
    local job_id=$1
    az ml job show --name "$job_id" --query "status" -o tsv 2>/dev/null || echo "Unknown"
}

# ============================================================================
# STEP 1: Monitor Teacher Generation
# ============================================================================

log "🚀 Starting pipeline orchestrator..."
log "📌 Step 1 Job: $JOB_STEP1"
log "⏱️  Poll interval: ${POLL_INTERVAL}s, Max wait: ${MAX_WAIT_HOURS}h"

start_time=$(date +%s)
max_seconds=$((MAX_WAIT_HOURS * 3600))

log "⏳ Monitoring Step 1 (Teacher Generation)..."
echo ""

while true; do
    status=$(get_job_status "$JOB_STEP1")
    elapsed=$(($(date +%s) - start_time))
    elapsed_min=$((elapsed / 60))
    
    case $status in
        Completed)
            log "✅ Step 1 COMPLETED after ${elapsed_min}min"
            break
            ;;
        Running)
            pct=$((elapsed * 100 / max_seconds))
            [ $pct -gt 100 ] && pct=100
            log "⏳ Step 1 running... (${elapsed_min}min, ${pct}% timeout)"
            ;;
        Failed)
            log "❌ Step 1 FAILED after ${elapsed_min}min"
            log "Check logs: https://ml.azure.com/runs/$JOB_STEP1"
            exit 1
            ;;
        Queued|Starting)
            log "🔄 Step 1 status: $status (${elapsed_min}min elapsed)"
            ;;
        *)
            log "⚠️  Unknown status: $status"
            ;;
    esac
    
    if [ $elapsed -gt $max_seconds ]; then
        log "❌ TIMEOUT: Step 1 exceeded ${MAX_WAIT_HOURS}h"
        exit 1
    fi
    
    sleep $POLL_INTERVAL
done

echo ""
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "✅ STEP 1 COMPLETE - Teacher labels ready"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ============================================================================
# STEP 2: Submit U-Net Training
# ============================================================================

log "📌 Step 2: Starting U-Net Training (60 epochs)..."

JOB_STEP2=$(az ml job create \
    --file "$SCRIPT_DIR/job_step2_training.yml" \
    --set inputs.teacher_labels="azureml://jobs/$JOB_STEP1/outputs/teacher_labels" \
    --query "name" -o tsv 2>&1 | tail -1)

if [ -z "$JOB_STEP2" ] || [ ${#JOB_STEP2} -lt 5 ]; then
    log "❌ Failed to create Step 2 job"
    exit 1
fi

log "✅ Step 2 Job created: $JOB_STEP2"
log "🔗 Portal: https://ml.azure.com/runs/$JOB_STEP2"

echo ""
log "⏳ Monitoring Step 2 (U-Net Training, ~2-4h)..."
echo ""

start_time2=$(date +%s)
max_seconds2=$((6 * 3600))  # Max 6 hours for training

while true; do
    status=$(get_job_status "$JOB_STEP2")
    elapsed=$(($(date +%s) - start_time2))
    elapsed_min=$((elapsed / 60))
    elapsed_hr=$((elapsed_min / 60))
    
    case $status in
        Completed)
            log "✅ Step 2 COMPLETED after ${elapsed_hr}h ${$((elapsed_min % 60))}min"
            break
            ;;
        Running)
            pct=$((elapsed * 100 / max_seconds2))
            [ $pct -gt 100 ] && pct=100
            log "⏳ Step 2 running... (${elapsed_hr}h${$((elapsed_min % 60))}m, ${pct}% timeout)"
            ;;
        Failed)
            log "❌ Step 2 FAILED after ${elapsed_hr}h ${$((elapsed_min % 60))}min"
            log "Check logs: https://ml.azure.com/runs/$JOB_STEP2"
            exit 1
            ;;
        Queued|Starting)
            log "🔄 Step 2 status: $status (${elapsed_min}min elapsed)"
            ;;
        *)
            log "⚠️  Unknown status: $status"
            ;;
    esac
    
    if [ $elapsed -gt $max_seconds2 ]; then
        log "❌ TIMEOUT: Step 2 exceeded 6h"
        exit 1
    fi
    
    sleep $POLL_INTERVAL
done

echo ""
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "✅ STEP 2 COMPLETE - Model trained"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ============================================================================
# STEP 3: Submit VFA/PMA Inference
# ============================================================================

log "📌 Step 3: Starting Production Inference..."

JOB_STEP3=$(az ml job create \
    --file "$SCRIPT_DIR/job_step3_inference.yml" \
    --set inputs.model_checkpoint="azureml://jobs/$JOB_STEP2/outputs/model" \
    --query "name" -o tsv 2>&1 | tail -1)

if [ -z "$JOB_STEP3" ] || [ ${#JOB_STEP3} -lt 5 ]; then
    log "❌ Failed to create Step 3 job"
    exit 1
fi

log "✅ Step 3 Job created: $JOB_STEP3"
log "🔗 Portal: https://ml.azure.com/runs/$JOB_STEP3"

echo ""
log "⏳ Monitoring Step 3 (Inference, ~10-15min)..."
echo ""

start_time3=$(date +%s)
max_seconds3=$((30 * 60))  # Max 30 minutes for inference

while true; do
    status=$(get_job_status "$JOB_STEP3")
    elapsed=$(($(date +%s) - start_time3))
    elapsed_min=$((elapsed / 60))
    
    case $status in
        Completed)
            log "✅ Step 3 COMPLETED after ${elapsed_min}min"
            break
            ;;
        Running)
            pct=$((elapsed * 100 / max_seconds3))
            [ $pct -gt 100 ] && pct=100
            log "⏳ Step 3 running... (${elapsed_min}min, ${pct}% timeout)"
            ;;
        Failed)
            log "❌ Step 3 FAILED after ${elapsed_min}min"
            log "Check logs: https://ml.azure.com/runs/$JOB_STEP3"
            exit 1
            ;;
        Queued|Starting)
            log "🔄 Step 3 status: $status (${elapsed_min}min elapsed)"
            ;;
        *)
            log "⚠️  Unknown status: $status"
            ;;
    esac
    
    if [ $elapsed -gt $max_seconds3 ]; then
        log "❌ TIMEOUT: Step 3 exceeded 30min"
        exit 1
    fi
    
    sleep $POLL_INTERVAL
done

echo ""
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "✅✅✅ FULL PIPELINE COMPLETE ✅✅✅"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

log "📊 SUMMARY"
log "Step 1 (Teachers): ${JOB_STEP1}"
log "Step 2 (Training): ${JOB_STEP2}"
log "Step 3 (Inference): ${JOB_STEP3}"
log ""
log "📁 Results available in:"
log "  - Teachers: azureml://jobs/$JOB_STEP1/outputs/teacher_labels"
log "  - Model: azureml://jobs/$JOB_STEP2/outputs/model"
log "  - Results: azureml://jobs/$JOB_STEP3/outputs/results"
log ""
log "🔗 View all jobs: https://ml.azure.com/experiments/azure-ml"

exit 0

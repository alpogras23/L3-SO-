#!/bin/bash
#
# Auto-Orchestration Pipeline v3
# Chains all 3 steps with HU-calibrated inference
# Monitors Step 1 → auto-submits Step 2 → auto-submits Step 3
#

set -e

LOG_FILE="/tmp/pipeline_log.txt"
WORKSPACE="l3-vfa-pma-ws"
RESOURCE_GROUP="l3-vfa-pma-rg"
PROJECT_DIR="/Users/alperenogras/Desktop/L3_SO_ANALYSIS"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1${NC}" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $1${NC}" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] ℹ️  $1${NC}" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}" | tee -a "$LOG_FILE"
}

# Initialize log
log "=========================================="
log "L3 VFA/PMA Auto-Pipeline v3 (HU-Calibrated)"
log "=========================================="
log "Project: $PROJECT_DIR"
log "Workspace: $WORKSPACE"
log "Log: $LOG_FILE"

# Step 1: Submit teacher generation job
submit_step1() {
    log_info "STEP 1: Submitting teacher label generation"
    
    cd "$PROJECT_DIR"
    
    JOB_NAME="step1_teachers_$(date +%s)"
    log "Job name: $JOB_NAME"
    
    STEP1_JOB=$(az ml job create \
        --file .github/azure-ml/job_step1_teachers.yml \
        --set display_name="Step 1: Teacher Gen ($JOB_NAME)" \
        --query name -o tsv 2>&1)
    
    if [ -z "$STEP1_JOB" ]; then
        log_error "Failed to submit Step 1 job"
        return 1
    fi
    
    log_success "Step 1 submitted: $STEP1_JOB"
    echo "$STEP1_JOB" > /tmp/step1_job_id.txt
    return 0
}

# Monitor step 1 completion
monitor_step1() {
    STEP1_JOB="$1"
    MAX_WAIT=$((10 * 3600))  # 10 hours
    POLL_INTERVAL=30  # 30 seconds
    ELAPSED=0
    
    log_info "Monitoring Step 1: $STEP1_JOB"
    
    while [ $ELAPSED -lt $MAX_WAIT ]; do
        STATUS=$(az ml job show --name "$STEP1_JOB" --query status -o tsv 2>&1)
        
        log "Step 1 status: $STATUS (elapsed: $((ELAPSED/60))m)"
        
        if [ "$STATUS" = "Completed" ]; then
            log_success "Step 1 COMPLETED"
            return 0
        elif [ "$STATUS" = "Failed" ] || [ "$STATUS" = "Cancelled" ]; then
            log_error "Step 1 $STATUS"
            return 1
        fi
        
        sleep $POLL_INTERVAL
        ELAPSED=$((ELAPSED + POLL_INTERVAL))
    done
    
    log_error "Step 1 timeout after $((MAX_WAIT/3600)) hours"
    return 1
}

# Step 2: Submit U-Net training job
submit_step2() {
    STEP1_JOB="$1"
    
    log_info "STEP 2: Submitting U-Net training (60 epochs)"
    
    cd "$PROJECT_DIR"
    
    JOB_NAME="step2_train_$(date +%s)"
    log "Job name: $JOB_NAME"
    
    # Get Step 1 output location
    STEP1_OUTPUT=$(az ml job show --name "$STEP1_JOB" \
        --query "outputs.teacher_labels.path" -o tsv 2>&1)
    
    log "Step 1 output: $STEP1_OUTPUT"
    
    STEP2_JOB=$(az ml job create \
        --file .github/azure-ml/job_step2_training.yml \
        --set \
          display_name="Step 2: U-Net Training ($JOB_NAME)" \
          inputs.teacher_labels=$STEP1_OUTPUT \
        --query name -o tsv 2>&1)
    
    if [ -z "$STEP2_JOB" ]; then
        log_error "Failed to submit Step 2 job"
        return 1
    fi
    
    log_success "Step 2 submitted: $STEP2_JOB"
    echo "$STEP2_JOB" > /tmp/step2_job_id.txt
    return 0
}

# Monitor step 2 completion
monitor_step2() {
    STEP2_JOB="$1"
    MAX_WAIT=$((6 * 3600))  # 6 hours
    POLL_INTERVAL=60  # 60 seconds
    ELAPSED=0
    
    log_info "Monitoring Step 2: $STEP2_JOB"
    
    while [ $ELAPSED -lt $MAX_WAIT ]; do
        STATUS=$(az ml job show --name "$STEP2_JOB" --query status -o tsv 2>&1)
        
        ELAPSED_MIN=$((ELAPSED/60))
        log "Step 2 status: $STATUS (elapsed: ${ELAPSED_MIN}m)"
        
        if [ "$STATUS" = "Completed" ]; then
            log_success "Step 2 COMPLETED"
            return 0
        elif [ "$STATUS" = "Failed" ] || [ "$STATUS" = "Cancelled" ]; then
            log_error "Step 2 $STATUS"
            return 1
        fi
        
        sleep $POLL_INTERVAL
        ELAPSED=$((ELAPSED + POLL_INTERVAL))
    done
    
    log_error "Step 2 timeout after $((MAX_WAIT/3600)) hours"
    return 1
}

# Step 3: Submit inference job (HU-calibrated)
submit_step3() {
    STEP2_JOB="$1"
    
    log_info "STEP 3: Submitting VFA/PMA inference (HU-Calibrated)"
    log_info "HU Thresholds: Muscle [-29, +150], VAT [-150, -50], SAT [-190, -30]"
    
    cd "$PROJECT_DIR"
    
    JOB_NAME="step3_inference_$(date +%s)"
    log "Job name: $JOB_NAME"
    
    # Get Step 2 output location
    STEP2_OUTPUT=$(az ml job show --name "$STEP2_JOB" \
        --query "outputs.trained_checkpoint.path" -o tsv 2>&1)
    
    log "Step 2 output: $STEP2_OUTPUT"
    
    STEP3_JOB=$(az ml job create \
        --file .github/azure-ml/job_step3_inference.yml \
        --set \
          display_name="Step 3: VFA/PMA Inference ($JOB_NAME)" \
          inputs.checkpoint=$STEP2_OUTPUT/best_unet.pth \
          inputs.teacher_labels=$STEP2_OUTPUT/teacher_labels \
        --query name -o tsv 2>&1)
    
    if [ -z "$STEP3_JOB" ]; then
        log_error "Failed to submit Step 3 job"
        return 1
    fi
    
    log_success "Step 3 submitted: $STEP3_JOB"
    echo "$STEP3_JOB" > /tmp/step3_job_id.txt
    return 0
}

# Monitor step 3 completion
monitor_step3() {
    STEP3_JOB="$1"
    MAX_WAIT=$((30 * 60))  # 30 minutes
    POLL_INTERVAL=30  # 30 seconds
    ELAPSED=0
    
    log_info "Monitoring Step 3: $STEP3_JOB"
    
    while [ $ELAPSED -lt $MAX_WAIT ]; do
        STATUS=$(az ml job show --name "$STEP3_JOB" --query status -o tsv 2>&1)
        
        log "Step 3 status: $STATUS (elapsed: $((ELAPSED/60))m)"
        
        if [ "$STATUS" = "Completed" ]; then
            log_success "Step 3 COMPLETED"
            return 0
        elif [ "$STATUS" = "Failed" ] || [ "$STATUS" = "Cancelled" ]; then
            log_error "Step 3 $STATUS"
            return 1
        fi
        
        sleep $POLL_INTERVAL
        ELAPSED=$((ELAPSED + POLL_INTERVAL))
    done
    
    log_error "Step 3 timeout after $((MAX_WAIT/60)) minutes"
    return 1
}

# Get final results
get_results() {
    STEP3_JOB="$1"
    
    log_info "Retrieving final results"
    
    RESULTS_PATH=$(az ml job show --name "$STEP3_JOB" \
        --query "outputs.inference_results.path" -o tsv 2>&1)
    
    log_success "Results location: $RESULTS_PATH"
    
    log_info "Copying results locally..."
    mkdir -p "$PROJECT_DIR/inference_results_final"
    
    # Note: In real execution, would copy from Azure storage
    # For now, just document the location
    echo "Results available at: $RESULTS_PATH" > "$PROJECT_DIR/inference_results_final/README.txt"
    
    return 0
}

# Main orchestration loop
main() {
    log_info "Starting auto-orchestration pipeline"
    
    # Step 1
    if ! submit_step1; then
        log_error "Failed to submit Step 1"
        exit 1
    fi
    
    STEP1_JOB=$(cat /tmp/step1_job_id.txt)
    
    if ! monitor_step1 "$STEP1_JOB"; then
        log_error "Step 1 failed or timed out"
        exit 1
    fi
    
    # Step 2
    if ! submit_step2 "$STEP1_JOB"; then
        log_error "Failed to submit Step 2"
        exit 1
    fi
    
    STEP2_JOB=$(cat /tmp/step2_job_id.txt)
    
    if ! monitor_step2 "$STEP2_JOB"; then
        log_error "Step 2 failed or timed out"
        exit 1
    fi
    
    # Step 3
    if ! submit_step3 "$STEP2_JOB"; then
        log_error "Failed to submit Step 3"
        exit 1
    fi
    
    STEP3_JOB=$(cat /tmp/step3_job_id.txt)
    
    if ! monitor_step3 "$STEP3_JOB"; then
        log_error "Step 3 failed or timed out"
        exit 1
    fi
    
    # Get results
    if ! get_results "$STEP3_JOB"; then
        log_warning "Could not retrieve results"
    fi
    
    log_success "=========================================="
    log_success "🎉 PIPELINE COMPLETE!"
    log_success "=========================================="
    log "Step 1: $STEP1_JOB"
    log "Step 2: $STEP2_JOB"
    log "Step 3: $STEP3_JOB"
    log "=========================================="
}

# Run
main

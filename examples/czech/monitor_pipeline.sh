#!/bin/bash
# Czech CosyVoice Pipeline Monitor
# Checks pipeline status every 60 minutes and auto-restarts on failure
#
# Run with: nohup bash monitor_pipeline.sh > /tmp/czech_pipeline_monitor.log 2>&1 &
# Stop with: kill $(cat /tmp/czech_pipeline_monitor.pid)

PIPELINE_SCRIPT="/mnt/8TB/AUDIO/TEXT_TO_SPEECH/CosyVoice/examples/czech/run_all_stages.sh"
PIPELINE_LOG="/tmp/czech_cosyvoice_pipeline.log"
PIPELINE_PID_FILE="/tmp/czech_cosyvoice_pipeline.pid"
MONITOR_LOG="/tmp/czech_pipeline_monitor.log"
CHECK_INTERVAL=3600  # 60 minutes in seconds

# Save monitor PID
echo $$ > /tmp/czech_pipeline_monitor.pid

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

check_for_errors() {
    # Check for common errors in log
    if strings "$PIPELINE_LOG" 2>/dev/null | grep -qi "error\|exception\|traceback\|failed\|killed"; then
        return 1
    fi
    return 0
}

get_current_stage() {
    # Get the current stage from log
    strings "$PIPELINE_LOG" 2>/dev/null | grep -E "^=+ Stage" | tail -1
}

get_progress() {
    # Get latest progress
    strings "$PIPELINE_LOG" 2>/dev/null | grep -E "\[.*\].*%.*ETA:" | tail -1
}

is_pipeline_running() {
    # Check if main pipeline process is running
    if [ -f "$PIPELINE_PID_FILE" ]; then
        PID=$(cat "$PIPELINE_PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0
        fi
    fi

    # Also check for any extraction/training processes
    if pgrep -f "extract_embedding\|extract_speech_token\|train_czech" > /dev/null 2>&1; then
        return 0
    fi

    return 1
}

is_pipeline_complete() {
    # Check if pipeline finished successfully
    if strings "$PIPELINE_LOG" 2>/dev/null | grep -q "Pipeline Complete!"; then
        return 0
    fi
    return 1
}

restart_pipeline() {
    log "Restarting pipeline..."

    # Kill any existing processes
    if [ -f "$PIPELINE_PID_FILE" ]; then
        PID=$(cat "$PIPELINE_PID_FILE")
        kill "$PID" 2>/dev/null
    fi
    pkill -f "extract_embedding\|extract_speech_token\|train_czech" 2>/dev/null

    sleep 5

    # Restart pipeline
    cd /mnt/8TB/AUDIO/TEXT_TO_SPEECH/CosyVoice/examples/czech
    nohup bash "$PIPELINE_SCRIPT" >> "$PIPELINE_LOG" 2>&1 &
    echo $! > "$PIPELINE_PID_FILE"

    log "Pipeline restarted with PID: $(cat $PIPELINE_PID_FILE)"
}

log "=========================================="
log "Pipeline Monitor Started"
log "Check interval: ${CHECK_INTERVAL}s (60 min)"
log "=========================================="

# Initial check
if ! is_pipeline_running; then
    log "Pipeline not running. Starting..."
    restart_pipeline
fi

# Main monitoring loop
while true; do
    log "--- Performing check ---"

    # Check if pipeline completed
    if is_pipeline_complete; then
        log "SUCCESS: Pipeline completed successfully!"
        log "Training outputs: /mnt/8TB/TRAINING_OUTPUTS/Fun-CosyVoice3-0.5B-2512_CZECH_30s_200hours_lr1e-5_$(date +%Y-%m-%d)"
        log "Monitor exiting."
        exit 0
    fi

    # Check if pipeline is running
    if is_pipeline_running; then
        STAGE=$(get_current_stage)
        PROGRESS=$(get_progress)
        log "Pipeline running"
        log "  Stage: $STAGE"
        log "  Progress: $PROGRESS"

        # Check for errors even if running
        if ! check_for_errors; then
            log "WARNING: Errors detected in log, but process still running. Monitoring..."
        fi
    else
        log "Pipeline stopped unexpectedly!"

        # Check if there were errors
        if ! check_for_errors; then
            log "Errors detected in log:"
            strings "$PIPELINE_LOG" | grep -i "error\|exception\|traceback" | tail -5
        fi

        log "Auto-restarting pipeline (will resume from checkpoint)..."
        restart_pipeline
    fi

    log "Next check in 60 minutes..."
    log ""

    # Sleep for check interval
    sleep $CHECK_INTERVAL
done

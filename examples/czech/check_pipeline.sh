#!/bin/bash
# Quick pipeline status check
# Usage: bash check_pipeline.sh

LOG="/tmp/czech_cosyvoice_pipeline.log"

echo "=== Czech CosyVoice Pipeline Status ==="
echo ""

# Current stage (try new format first, then old format)
STAGE=$(strings "$LOG" 2>/dev/null | grep -E "^\[STAGE:" | tail -1)
if [ -z "$STAGE" ]; then
    STAGE=$(strings "$LOG" 2>/dev/null | grep -E "^=+ Stage [0-9]" | tail -1)
fi
echo "Current stage: $STAGE"

# Latest progress (if any)
PROGRESS=$(strings "$LOG" 2>/dev/null | grep -E "\[.*\].*%.*ETA:" | tail -1)
if [ -n "$PROGRESS" ]; then
    echo "Progress: $PROGRESS"
fi

# Check if process is running
if pgrep -f "run_all_stages|extract_embedding|extract_speech|train_czech" > /dev/null 2>&1; then
    echo "Status: RUNNING"
else
    if strings "$LOG" 2>/dev/null | grep -qE "\[STAGE: DONE\]|Pipeline Complete"; then
        echo "Status: COMPLETED"
    else
        echo "Status: STOPPED (may need restart)"
    fi
fi

echo ""

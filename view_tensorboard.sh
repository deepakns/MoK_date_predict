#!/bin/bash
# TensorBoard Viewer Script
# Usage: ./view_tensorboard.sh [optional_logdir] [optional_port]

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate MoK_date_predict

# Default values
LOGDIR="${1:-logs}"
PORT="${2:-6006}"

echo "=========================================="
echo "Starting TensorBoard"
echo "=========================================="
echo "Log directory: $LOGDIR"
echo "Port: $PORT"
echo ""
echo "Open in browser: http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop TensorBoard"
echo "=========================================="
echo ""

# Check if logdir exists
if [ ! -d "$LOGDIR" ]; then
    echo "Error: Log directory '$LOGDIR' not found!"
    exit 1
fi

# Check if there are any event files
EVENT_FILES=$(find "$LOGDIR" -name "events.out.tfevents.*" 2>/dev/null | wc -l)
if [ "$EVENT_FILES" -eq 0 ]; then
    echo "Warning: No TensorBoard event files found in '$LOGDIR'"
    echo "Make sure you've run training with tensorboard: true in config"
    echo ""
fi

# Launch TensorBoard
tensorboard --logdir="$LOGDIR" --port="$PORT" --bind_all

#!/bin/bash

# Test script for visualize_intermediate_layers.py with data loader integration
# This demonstrates the new year-based data loading feature

echo "=========================================================================="
echo "Testing Intermediate Layer Visualization with Data Loader"
echo "=========================================================================="
echo ""

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate MoK_date_predict

# Configuration
CONFIG_FILE="config/model_config.yml"
CHECKPOINT_FILE="path/to/your/checkpoint.pt"  # Update this path
OUTPUT_DIR="./visualizations_dataloader_test"
YEAR=2000  # Change this to any year you want to visualize

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Checkpoint file: $CHECKPOINT_FILE"
echo "  Year to load: $YEAR"
echo "  Output directory: $OUTPUT_DIR"
echo ""
echo "Note: Data directory and normalization settings will be read from config file"
echo "      Normalization will be applied based on normalize_strategy in config"
echo ""

# Run visualization with data loader
echo "Running visualization with data from year $YEAR..."
echo ""

python src/models/visualize_intermediate_layers.py \
    --checkpoint "$CHECKPOINT_FILE" \
    --config "$CONFIG_FILE" \
    --year $YEAR \
    --output "$OUTPUT_DIR" \
    --num-features 16 \
    --device cpu

echo ""
echo "=========================================================================="
echo "Test complete! Check the output directory for results:"
echo "  $OUTPUT_DIR"
echo "=========================================================================="
# Intermediate Layer Visualization for MoK_CNN_Predictor_V2

This tool allows you to visualize the intermediate representations (feature maps) from all convolution and pooling blocks in the MoK_CNN_Predictor_V2 model.

## Overview

The visualization script captures outputs from:
- **Coarsening Pool Layer**: 4x4 average pooling that reduces input from 1440×481 to 360×120
- **Conv Block 1**: 32 channels (360×120 → 180×60)
- **Conv Block 2**: 64 channels (180×60 → 90×30)
- **Conv Block 3**: 128 channels (90×30 → 45×15)
- **Conv Block 4**: 256 channels (45×15 → 23×8)
- **Conv Block 5**: 512 channels (23×8 → 12×4)
- **Conv Block 6**: 1024 channels (12×4 → 6×2)
- **Global Average Pool**: 1024 channels (6×2 → 1×1)

## Installation

The script requires the following packages (already in your environment):
```bash
torch
matplotlib
numpy
pyyaml
```

## Usage

### Basic Usage with Random Input (for testing)

```bash
conda activate MoK_date_predict
python src/models/visualize_intermediate_layers.py \
    --random-input \
    --in-channels 16 \
    --output ./visualizations \
    --num-features 16
```

### Using a Trained Model Checkpoint

```bash
conda activate MoK_date_predict
python src/models/visualize_intermediate_layers.py \
    --checkpoint /path/to/model_checkpoint.pt \
    --config /path/to/config.yml \
    --random-input \
    --output ./visualizations \
    --num-features 16
```

### Using Real Input Data from Data Loader (Recommended)

Load data directly from NetCDF files for a specific year:

```bash
conda activate MoK_date_predict
python src/models/visualize_intermediate_layers.py \
    --checkpoint /path/to/model_checkpoint.pt \
    --config config/model_config.yml \
    --data-dir /gdata2/ERA5/monthly \
    --year 2000 \
    --output ./visualizations \
    --num-features 16
```

### Using Pre-loaded Input Data (.pt file)

```bash
conda activate MoK_date_predict
python src/models/visualize_intermediate_layers.py \
    --checkpoint /path/to/model_checkpoint.pt \
    --data /path/to/input_tensor.pt \
    --output ./visualizations \
    --num-features 16
```

### Using with GPU

```bash
conda activate MoK_date_predict
python src/models/visualize_intermediate_layers.py \
    --checkpoint /path/to/model_checkpoint.pt \
    --random-input \
    --device cuda \
    --output ./visualizations
```

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--checkpoint` | Path to model checkpoint (.pt or .pth file) | None |
| `--config` | Path to config file (optional) | None |
| `--data` | Path to input data tensor (.pt file) | None |
| `--data-dir` | Directory containing NetCDF files (e.g., /gdata2/ERA5/monthly) | None |
| `--year` | Year to load from data loader (requires --data-dir) | None |
| `--output` | Output directory for visualizations | `./visualizations` |
| `--num-features` | Number of feature maps to display per layer | 16 |
| `--device` | Device to run on (cpu or cuda) | cpu |
| `--random-input` | Use random input data (for testing) | False |
| `--in-channels` | Number of input channels for random data | 36 |

## Output Files

The script generates the following visualizations:

### 1. Summary Plot (`summary_all_layers.png`)
- Overview showing one representative feature map from each layer
- Displays spatial dimensions and channel counts
- Good for quick understanding of the model's feature hierarchy

### 2. Detailed Layer Visualizations
Individual PNG files for each layer showing multiple feature maps:
- `Coarsening_Pool_4x4_AvgPool.png` - Initial pooling layer
- `Conv_Block_1_32_channels.png` - First convolutional block
- `Conv_Block_2_64_channels.png` - Second convolutional block
- `Conv_Block_3_128_channels.png` - Third convolutional block
- `Conv_Block_4_256_channels.png` - Fourth convolutional block
- `Conv_Block_5_512_channels.png` - Fifth convolutional block
- `Conv_Block_6_1024_channels.png` - Sixth convolutional block
- `Global_Average_Pool.png` - Final global average pooling

Each detailed visualization shows:
- Multiple feature maps (controlled by `--num-features`)
- Individual colorbars for each feature map
- Channel indices for easy reference

## Examples

### Example 1: Quick Test with Default Model
```bash
python src/models/visualize_intermediate_layers.py \
    --random-input \
    --in-channels 16 \
    --output ./test_viz \
    --num-features 8
```

### Example 2: Visualize Trained Model Performance
```bash
python src/models/visualize_intermediate_layers.py \
    --checkpoint results/checkpoints/best_model.pt \
    --config config/model_config_with_wandb.yml \
    --random-input \
    --output results/visualizations/intermediate_layers \
    --num-features 32
```

### Example 3: Analyze Specific Input Sample
First, save a sample from your dataset:
```python
import torch
# Assuming you have a dataloader
for batch in dataloader:
    input_sample = batch['input'][0:1]  # Take first sample
    torch.save(input_sample, 'sample_input.pt')
    break
```

Then visualize:
```bash
python src/models/visualize_intermediate_layers.py \
    --checkpoint results/checkpoints/best_model.pt \
    --data sample_input.pt \
    --output results/visualizations/sample_analysis
```

### Example 4: Analyze Model Performance on Specific Year (NEW)

Visualize how your trained model processes data from a specific year using the data loader:

```bash
# Analyze year 2020 data
python src/models/visualize_intermediate_layers.py \
    --checkpoint results/checkpoints/best_model.pt \
    --config config/model_config.yml \
    --data-dir /gdata2/ERA5/monthly \
    --year 2020 \
    --output ./visualizations/year_2020 \
    --num-features 24

# Compare multiple years by running separately
for year in 2015 2016 2017 2018 2019 2020; do
    python src/models/visualize_intermediate_layers.py \
        --checkpoint results/checkpoints/best_model.pt \
        --config config/model_config.yml \
        --data-dir /gdata2/ERA5/monthly \
        --year $year \
        --output ./visualizations/year_$year \
        --num-features 16
done
```

This approach:
- Loads data directly from NetCDF files using your configured dataset
- Shows the actual target value and model prediction
- Displays prediction error for analysis
- Uses the exact same preprocessing as during training

## Understanding the Visualizations

### Feature Map Interpretation

1. **Early Layers (Conv Blocks 1-2)**:
   - Detect low-level features (edges, gradients, textures)
   - Larger spatial dimensions preserve spatial information
   - Useful for understanding what patterns the model detects

2. **Middle Layers (Conv Blocks 3-4)**:
   - Capture intermediate features (patterns, structures)
   - More abstract representations
   - Important for understanding feature composition

3. **Deep Layers (Conv Blocks 5-6)**:
   - High-level semantic features
   - Very abstract representations
   - Directly related to prediction task

4. **Global Average Pool**:
   - Single value per channel
   - Represents the "strength" of each high-level feature
   - These values feed into the final prediction layer

### Statistics Output

The script also prints statistics for each layer:
- **Shape**: Tensor dimensions (batch, channels, height, width)
- **Mean**: Average activation value (indicates overall activation level)
- **Std**: Standard deviation (indicates activation variability)
- **Min/Max**: Range of activations (helps identify saturated units)

## Programmatic Usage

You can also use the `IntermediateLayerVisualizer` class in your own scripts:

```python
import torch
from src.models.architectures.cnn_v2 import MoK_CNN_Predictor_V2
from src.models.visualize_intermediate_layers import IntermediateLayerVisualizer

# Create or load model
model = MoK_CNN_Predictor_V2(in_channels=16)

# Create visualizer
visualizer = IntermediateLayerVisualizer(model)

# Run forward pass with your data
input_data = torch.randn(1, 16, 1440, 481)
output = visualizer.forward(input_data)

# Print statistics
visualizer.print_layer_statistics()

# Create summary plot
fig = visualizer.create_summary_plot(save_path='summary.png')

# Visualize specific layer
fig = visualizer.visualize_feature_maps(
    layer_name='Conv Block 3 (128 channels)',
    num_features=32,
    save_path='conv3_features.png'
)

# Generate all visualizations
visualizer.visualize_all_layers(
    output_dir='./visualizations',
    num_features_per_layer=16
)

# Clean up
visualizer.remove_hooks()
```

## Tips for Analysis

1. **Compare Different Training Stages**: Visualize checkpoints from different epochs to see how features evolve during training

2. **Analyze Failure Cases**: When the model makes incorrect predictions, visualize the intermediate representations to understand why

3. **Channel Pruning**: Identify channels with consistently low activations - these might be candidates for pruning

4. **Transfer Learning**: Compare feature maps when using pre-trained weights vs. random initialization

5. **Data Distribution Shift**: Compare features for training vs. test data to detect distribution shifts

## Troubleshooting

### Out of Memory Error
- Reduce `--num-features` to visualize fewer channels per layer
- Use `--device cpu` instead of cuda
- Process one layer at a time programmatically

### Import Errors
- Make sure you've activated the correct conda environment: `conda activate MoK_date_predict`
- Verify all dependencies are installed

### Checkpoint Loading Issues
- Ensure the checkpoint matches the model architecture
- Provide the config file if the checkpoint doesn't contain model configuration
- Check that the checkpoint file is not corrupted

## Additional Resources

- Model architecture: [src/models/architectures/cnn_v2.py](architectures/cnn_v2.py)
- Training script: [src/training/scripts/train_model.py](../training/scripts/train_model.py)
- Configuration files: [config/](../../config/)

## Citation

If you use this visualization tool in your research, please cite the MoK_date_predict project.

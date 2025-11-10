"""
Visualization tool for intermediate representations in MoK_CNN_Predictor_V2.

This script allows you to visualize the intermediate feature maps from all
convolution and pooling blocks in the MoK_CNN_Predictor_V2 model.

The script reads data configuration (data_dir, variables, time steps, etc.)
from the config file to ensure consistency with training configuration.
Normalization is automatically applied based on the normalize_strategy in the config.

Usage:
    # Load data from dataset for a specific year (with normalization)
    python visualize_intermediate_layers.py --checkpoint <path_to_checkpoint> --config <path_to_config> --year <year> --output <output_dir>

    # Use a saved tensor file
    python visualize_intermediate_layers.py --checkpoint <path_to_checkpoint> --data <path_to_data.pt> --output <output_dir>

    # Use random input (for testing)
    python visualize_intermediate_layers.py --checkpoint <path_to_checkpoint> --random-input --output <output_dir>

Example:
    python visualize_intermediate_layers.py --checkpoint models/best_model.pt --config config/model_config.yml --year 2015 --output visualizations/

Note:
    When using --year, the data will be normalized according to the normalize_strategy in the config:
    - normalize_strategy: 0 = No normalization
    - normalize_strategy: 1 = Apply training data statistics (requires saved stats for the model)
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.architectures.cnn_v2 import MoK_CNN_Predictor_V2


class IntermediateLayerVisualizer:
    """
    Class to extract and visualize intermediate representations from MoK_CNN_Predictor_V2.

    This class hooks into the model's forward pass to capture outputs from:
    - Coarsening pooling layer
    - All 6 convolution blocks (after activation)
    - Global average pooling layer
    """

    def __init__(self, model: nn.Module):
        """
        Initialize the visualizer with a model.

        Args:
            model: MoK_CNN_Predictor_V2 instance
        """
        self.model = model
        self.model.eval()  # Set to evaluation mode

        # Dictionary to store intermediate outputs
        self.intermediate_outputs: Dict[str, torch.Tensor] = {}

        # Register forward hooks to capture intermediate outputs
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on layers we want to visualize."""

        # Define layers to hook with friendly names
        layers_to_hook = [
            ('coarsen_pool', 'Coarsening Pool (4x4 AvgPool)'),
            ('act1', 'Conv Block 1 (32 channels)'),
            ('act2', 'Conv Block 2 (64 channels)'),
            ('act3', 'Conv Block 3 (128 channels)'),
            ('act4', 'Conv Block 4 (256 channels)'),
            ('act5', 'Conv Block 5 (512 channels)'),
            ('act6', 'Conv Block 6 (1024 channels)'),
            ('global_avg_pool', 'Global Average Pool'),
        ]

        def create_hook(name):
            """Create a hook function that captures the output."""
            def hook(module, input, output):
                self.intermediate_outputs[name] = output.detach().cpu()
            return hook

        # Register hooks
        for layer_name, friendly_name in layers_to_hook:
            layer = getattr(self.model, layer_name)
            hook = layer.register_forward_hook(create_hook(friendly_name))
            self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run forward pass and capture intermediate outputs.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Model output
        """
        self.intermediate_outputs.clear()

        with torch.no_grad():
            output = self.model(x)

        return output

    def visualize_feature_maps(
        self,
        layer_name: str,
        num_features: int = 16,
        figsize: Tuple[int, int] = (20, 12),
        cmap: str = 'RdBu_r',
        save_path: Optional[str] = None
    ):
        """
        Visualize feature maps from a specific layer.

        Args:
            layer_name: Name of the layer to visualize
            num_features: Number of feature maps to display (max, ignored for Global Average Pool)
            figsize: Figure size for the plot
            cmap: Colormap for visualization
            save_path: Path to save the figure (optional)
        """
        if layer_name not in self.intermediate_outputs:
            raise ValueError(f"Layer '{layer_name}' not found in captured outputs. "
                           f"Available layers: {list(self.intermediate_outputs.keys())}")

        # Get the feature maps (batch, channels, height, width)
        feature_maps = self.intermediate_outputs[layer_name]

        # Use first sample in batch
        if len(feature_maps.shape) == 4:
            feature_maps = feature_maps[0]  # (channels, height, width)
        elif len(feature_maps.shape) == 3:
            feature_maps = feature_maps[0]  # (channels, 1, 1) for global pool

        num_channels = feature_maps.shape[0]

        # Check if this is Global Average Pool layer (1x1 spatial dimensions)
        is_global_pool = 'Global Average Pool' in layer_name or (
            len(feature_maps.shape) == 3 and
            feature_maps.shape[1] == 1 and
            feature_maps.shape[2] == 1
        )

        if is_global_pool:
            # Special visualization for Global Average Pool
            # Show all channels as a square grid
            return self._visualize_global_pool(layer_name, feature_maps, cmap, save_path, figsize)

        # Regular feature map visualization
        num_features = min(num_features, num_channels)

        # Calculate grid size
        cols = 4
        rows = (num_features + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        fig.suptitle(f'{layer_name}\nShowing {num_features} of {num_channels} feature maps',
                     fontsize=16, fontweight='bold')

        # Flatten axes for easier indexing
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes_flat = axes.flatten()

        # Plot feature maps
        for idx in range(num_features):
            ax = axes_flat[idx]

            # Get feature map
            feature_map = feature_maps[idx].numpy()

            # Plot with symmetric colormap centered at 0
            vmax = np.abs(feature_map).max()
            im = ax.imshow(feature_map, cmap=cmap, aspect='auto',
                          vmin=-vmax, vmax=vmax)
            ax.set_title(f'Channel {idx}', fontsize=10)
            ax.axis('off')

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide unused subplots
        for idx in range(num_features, len(axes_flat)):
            axes_flat[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")

        return fig

    def _visualize_global_pool(
        self,
        layer_name: str,
        feature_maps: torch.Tensor,
        cmap: str,
        save_path: Optional[str],
        figsize: Tuple[int, int]
    ):
        """
        Special visualization for Global Average Pool layer.
        Shows all channels as a sqrt(channels) x sqrt(channels) grid.

        Args:
            layer_name: Name of the layer
            feature_maps: Feature maps tensor (channels, 1, 1) or (channels,)
            cmap: Colormap to use
            save_path: Path to save the figure
            figsize: Figure size
        """
        # Flatten if needed
        if len(feature_maps.shape) == 3:
            values = feature_maps[:, 0, 0].numpy()  # (channels,)
        else:
            values = feature_maps.numpy()

        num_channels = len(values)

        # Calculate grid size (as close to square as possible)
        grid_size = int(np.ceil(np.sqrt(num_channels)))

        # Pad values to fill the grid
        padded_size = grid_size * grid_size
        padded_values = np.zeros(padded_size)
        padded_values[:num_channels] = values

        # Reshape to grid
        grid = padded_values.reshape(grid_size, grid_size)

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig.suptitle(f'{layer_name}\nAll {num_channels} channels visualized as {grid_size}×{grid_size} grid',
                     fontsize=16, fontweight='bold')

        # Plot with symmetric colormap centered at 0
        vmax = np.abs(values).max()
        im = ax.imshow(grid, cmap=cmap, aspect='equal',
                      vmin=-vmax, vmax=vmax, interpolation='nearest')

        # Add grid lines to separate channels
        for i in range(grid_size + 1):
            ax.axhline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
            ax.axvline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Channel Value', rotation=270, labelpad=15)

        # Add text annotations for each cell
        for i in range(grid_size):
            for j in range(grid_size):
                channel_idx = i * grid_size + j
                if channel_idx < num_channels:
                    value = values[channel_idx]
                    # Choose text color based on background
                    text_color = 'white' if abs(value) > vmax * 0.5 else 'black'
                    ax.text(j, i, f'{channel_idx}\n{value:.2f}',
                           ha='center', va='center',
                           fontsize=8, color=text_color, weight='bold')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f'Channel values after global average pooling', fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")

        return fig

    def visualize_all_layers(
        self,
        output_dir: str,
        num_features_per_layer: int = 16,
        cmap: str = 'RdBu_r'
    ):
        """
        Visualize feature maps from all captured layers.

        Args:
            output_dir: Directory to save visualizations
            num_features_per_layer: Number of feature maps to display per layer
            cmap: Colormap for visualization (default: 'RdBu_r' - Red-Blue diverging)
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nGenerating visualizations for {len(self.intermediate_outputs)} layers...")
        print(f"Output directory: {output_dir}\n")

        for layer_name in self.intermediate_outputs.keys():
            print(f"Processing: {layer_name}")

            # Create safe filename
            safe_name = layer_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
            save_path = os.path.join(output_dir, f'{safe_name}.png')

            # Visualize and save
            try:
                self.visualize_feature_maps(
                    layer_name=layer_name,
                    num_features=num_features_per_layer,
                    cmap=cmap,
                    save_path=save_path
                )
                plt.close()  # Close figure to free memory
            except Exception as e:
                print(f"  Error visualizing {layer_name}: {e}")

        print(f"\nAll visualizations saved to: {output_dir}")

    def create_summary_plot(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (20, 16),
        cmap: str = 'RdBu_r'
    ):
        """
        Create a summary plot showing one representative feature map from each layer.

        Args:
            save_path: Path to save the figure (optional)
            figsize: Figure size for the plot
            cmap: Colormap for visualization (default: 'RdBu_r' - Red-Blue diverging)
        """
        num_layers = len(self.intermediate_outputs)

        fig, axes = plt.subplots(2, 4, figsize=figsize)
        fig.suptitle('MoK_CNN_Predictor_V2: Intermediate Representations Summary',
                     fontsize=18, fontweight='bold')

        axes_flat = axes.flatten()

        for idx, (layer_name, feature_maps) in enumerate(self.intermediate_outputs.items()):
            if idx >= len(axes_flat):
                break

            ax = axes_flat[idx]

            # Get first feature map from first sample
            if len(feature_maps.shape) == 4:
                feature_map = feature_maps[0, 0].numpy()  # First channel of first sample
            else:
                feature_map = feature_maps[0, 0].numpy()

            # Get shape info
            shape_info = f"Shape: {tuple(feature_maps.shape[1:])}"

            # Plot with symmetric colormap centered at 0
            vmax = np.abs(feature_map).max()
            im = ax.imshow(feature_map, cmap=cmap, aspect='auto',
                          vmin=-vmax, vmax=vmax)
            ax.set_title(f'{layer_name}\n{shape_info}', fontsize=9)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide unused subplots
        for idx in range(len(self.intermediate_outputs), len(axes_flat)):
            axes_flat[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved summary plot to: {save_path}")

        return fig

    def print_layer_statistics(self):
        """Print statistics about each layer's output."""
        print("\n" + "="*80)
        print("INTERMEDIATE LAYER STATISTICS")
        print("="*80)

        for layer_name, feature_maps in self.intermediate_outputs.items():
            shape = tuple(feature_maps.shape)
            mean = feature_maps.mean().item()
            std = feature_maps.std().item()
            min_val = feature_maps.min().item()
            max_val = feature_maps.max().item()

            print(f"\n{layer_name}:")
            print(f"  Shape: {shape}")
            print(f"  Mean: {mean:.4f}")
            print(f"  Std:  {std:.4f}")
            print(f"  Min:  {min_val:.4f}")
            print(f"  Max:  {max_val:.4f}")

        print("\n" + "="*80)


def calculate_in_channels_from_config(config: dict) -> int:
    """
    Calculate the number of input channels based on data configuration.

    Args:
        config: Configuration dictionary loaded from YAML

    Returns:
        Number of input channels
    """
    data_config = config.get('data', {})

    # Get variable lists
    surf_vars = data_config.get('input_geo_var_surf', ['ttr', 'msl', 't2m', 'sst', 'tcc'])
    press_vars = data_config.get('input_geo_var_press', ['u', 'v', 'z'])
    time_steps = data_config.get('time_steps', [1, 2])
    pressure_levels = data_config.get('pressure_levels', [0, 1])

    # Calculate channels
    num_surf_channels = len(surf_vars) * len(time_steps) if surf_vars else 0
    num_press_channels = len(press_vars) * len(pressure_levels) * len(time_steps) if press_vars else 0

    # Add static channels
    num_static_channels = 0
    if data_config.get('include_landsea', True):
        num_static_channels += 1
    if data_config.get('include_lat', True):
        num_static_channels += 1
    if data_config.get('include_lon', True):
        num_static_channels += 1

    total_channels = num_surf_channels + num_press_channels + num_static_channels

    print(f"  Calculated input channels from config:")
    print(f"    Surface variables: {surf_vars} × {len(time_steps)} time steps = {num_surf_channels} channels")
    print(f"    Pressure variables: {press_vars} × {len(pressure_levels)} levels × {len(time_steps)} time steps = {num_press_channels} channels")
    print(f"    Static channels: {num_static_channels} (landsea: {data_config.get('include_landsea', True)}, lat: {data_config.get('include_lat', True)}, lon: {data_config.get('include_lon', True)})")
    print(f"    Total: {total_channels} channels")

    return total_channels


def load_model_from_checkpoint(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    device: str = 'cpu'
) -> MoK_CNN_Predictor_V2:
    """
    Load model from checkpoint.
    If config_path is provided, automatically calculates in_channels from data configuration.

    Args:
        checkpoint_path: Path to model checkpoint (.pt or .pth file)
        config_path: Optional path to config file
        device: Device to load model on

    Returns:
        Loaded model
    """
    print(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Try to get model config from checkpoint or config file
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            model_config = config.get('model', {})

            # Calculate in_channels from data config if not explicitly set
            if 'in_channels' not in model_config:
                in_channels = calculate_in_channels_from_config(config)
                model_config['in_channels'] = in_channels
    elif 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
    else:
        # Use defaults
        model_config = {
            'in_channels': 36,
            'num_classes': 1,
            'dropout_rate': 0.5,
            'activation': {
                'type': 'relu',
                'prelu_mode': 'shared',
                'leaky_relu_slope': 0.01
            },
            'l2_reg': 0.0001
        }
        print("Warning: Using default model configuration")

    # Create model
    activation_config = model_config.get('activation', {})
    model = MoK_CNN_Predictor_V2(
        in_channels=model_config.get('in_channels', 36),
        dropout_rate=model_config.get('dropout_rate', 0.5),
        num_classes=model_config.get('num_classes', 1),
        activation_type=activation_config.get('type', 'relu'),
        prelu_mode=activation_config.get('prelu_mode', 'shared'),
        leaky_relu_slope=activation_config.get('leaky_relu_slope', 0.01),
        l2_reg=model_config.get('l2_reg', 0.0001)
    )

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"  Input channels: {model_config.get('in_channels', 36)}")
    print(f"  Activation: {activation_config.get('type', 'relu')}")

    return model


def create_sample_input(
    batch_size: int = 1,
    in_channels: int = 36,
    height: int = 1440,
    width: int = 481,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Create a random sample input tensor.

    Args:
        batch_size: Batch size
        in_channels: Number of input channels
        height: Input height
        width: Input width
        device: Device to create tensor on

    Returns:
        Random input tensor
    """
    return torch.randn(batch_size, in_channels, height, width, device=device)


def load_data_from_dataloader(
    year: int,
    config_path: str,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, float, dict]:
    """
    Load data for a specific year using the MonthlyERA5Dataset.
    Reads data directory and all data configuration from the config file.
    Applies normalization based on normalize_strategy in config.

    Args:
        year: Year to load data for
        config_path: Path to config file containing data_dir and data settings
        device: Device to load data on

    Returns:
        Tuple of (input_tensor, target_value, metadata)
    """
    try:
        from src.data_pipeline.loaders.dataset_classes.monthly_dataset import MonthlyERA5Dataset
        from src.data_pipeline.preprocessing.transformers.data_transforms import NormalizeWithPrecomputedStats
        from src.data_pipeline.preprocessing.normstats import load_normalization_stats, stats_exist
    except ImportError:
        try:
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            from src.data_pipeline.loaders.dataset_classes.monthly_dataset import MonthlyERA5Dataset
            from src.data_pipeline.preprocessing.transformers.data_transforms import NormalizeWithPrecomputedStats
            from src.data_pipeline.preprocessing.normstats import load_normalization_stats, stats_exist
        except ImportError:
            raise ImportError("Could not import required modules. Make sure the data_pipeline module is available.")

    # Load config - this is required now
    if not config_path or not os.path.exists(config_path):
        raise ValueError(f"Config file is required and must exist. Provided path: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        data_config = config.get('data', {})

        # Extract data directory from config
        data_dir = data_config.get('data_dir')
        if not data_dir:
            raise ValueError("'data_dir' must be specified in the config file under 'data' section")

        # Get normalize_strategy from config
        normalize_strategy = data_config.get('normalize_strategy', 1)
        model_name = config['model']['name']

        # Extract data configuration with defaults matching config file
        dataset_kwargs = {
            'time_steps': data_config.get('time_steps', [1, 2]),
            'pressure_levels': data_config.get('pressure_levels', [0, 1]),
            'input_geo_var_surf': data_config.get('input_geo_var_surf', None),
            'input_geo_var_press': data_config.get('input_geo_var_press', None),
            'include_lat': data_config.get('include_lat', True),
            'include_lon': data_config.get('include_lon', True),
            'include_landsea': data_config.get('include_landsea', True),
            'target_file': data_config.get('target_file', None),
        }

        print(f"\nLoading data from config file: {config_path}")
        print(f"Data directory: {data_dir}")
        print(f"Surface variables: {dataset_kwargs['input_geo_var_surf']}")
        print(f"Pressure variables: {dataset_kwargs['input_geo_var_press']}")
        print(f"Time steps: {dataset_kwargs['time_steps']}")
        print(f"Pressure levels: {dataset_kwargs['pressure_levels']}")
        print(f"Include lat: {dataset_kwargs['include_lat']}")
        print(f"Include lon: {dataset_kwargs['include_lon']}")
        print(f"Include land-sea mask: {dataset_kwargs['include_landsea']}")
        print(f"Normalize strategy: {normalize_strategy}")

    # Handle normalization based on strategy
    normalize_transform = None
    if normalize_strategy == 1:
        # Strategy 1: Normalize using training data statistics
        print("\nApplying normalization using training data statistics...")
        if not stats_exist(model_name):
            raise FileNotFoundError(
                f"Normalization statistics not found for model '{model_name}'. "
                f"Please run training first to compute and save statistics, or set normalize_strategy: 0 in config."
            )

        norm_stats = load_normalization_stats(model_name, device=torch.device(device), verbose=True)
        normalize_transform = NormalizeWithPrecomputedStats(
            mean=norm_stats.mean,
            std=norm_stats.std,
            static_channel_indices=norm_stats.static_channel_indices
        )
        print("✓ Normalization transform created and will be applied to data")
    elif normalize_strategy == 0:
        print("\nNo normalization will be applied (normalize_strategy: 0)")
        normalize_transform = None
    else:
        raise ValueError(
            f"Unsupported normalize_strategy: {normalize_strategy}. "
            f"Supported values: 0 (no normalization), 1 (training data stats)"
        )

    # Create dataset for specific year with normalization transform
    dataset = MonthlyERA5Dataset(
        data_dir=data_dir,
        years=[year],  # Only load specified year
        transform=normalize_transform,
        **dataset_kwargs
    )

    if len(dataset) == 0:
        raise ValueError(f"No data found for year {year} in {data_dir}")

    # Get data for the year (should be index 0 since we only loaded one year)
    data, target = dataset[0]
    metadata = dataset.get_metadata(0)

    # Get channel info
    channel_info = dataset.get_channel_info()

    print(f"\nData loaded successfully:")
    print(f"  Year: {metadata['year']}")
    print(f"  Data shape: {data.shape}")
    print(f"  Target value: {target.item():.2f}")
    print(f"  Number of channels: {channel_info['num_channels']}")
    print(f"  Channel names: {channel_info['channel_names']}")

    # Convert to batch format and move to device
    input_tensor = data.unsqueeze(0).to(device)  # Add batch dimension

    return input_tensor, target.item(), metadata


def main():
    parser = argparse.ArgumentParser(
        description='Visualize intermediate representations in MoK_CNN_Predictor_V2'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to model checkpoint (.pt or .pth file)'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config file (optional, will try to infer from checkpoint)'
    )
    parser.add_argument(
        '--data',
        type=str,
        help='Path to input data tensor (.pt file) or use random data if not provided'
    )
    parser.add_argument(
        '--year',
        type=int,
        help='Year to load from data loader (requires --config with data_dir specified)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./visualizations',
        help='Output directory for visualizations (default: ./visualizations)'
    )
    parser.add_argument(
        '--num-features',
        type=int,
        default=16,
        help='Number of feature maps to display per layer (default: 16)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to run on (default: cpu)'
    )
    parser.add_argument(
        '--random-input',
        action='store_true',
        help='Use random input data (for testing without real data)'
    )
    parser.add_argument(
        '--in-channels',
        type=int,
        default=36,
        help='Number of input channels for random data (default: 36)'
    )

    args = parser.parse_args()

    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU instead")
        args.device = 'cpu'

    # Load model
    if args.checkpoint:
        model = load_model_from_checkpoint(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            device=args.device
        )
    else:
        print("No checkpoint provided, creating model with default configuration...")
        model = MoK_CNN_Predictor_V2(
            in_channels=args.in_channels,
            dropout_rate=0.5,
            num_classes=1
        )
        model.to(args.device)
        print(f"Created model with {args.in_channels} input channels")

    # Load or create input data
    target_value = None
    year_info = None

    if args.year:
        # Load from data loader using config file
        if not args.config:
            raise ValueError("--config is required when using --year to load data from dataset")

        input_tensor, target_value, metadata = load_data_from_dataloader(
            year=args.year,
            config_path=args.config,
            device=args.device
        )
        year_info = f"Year {args.year} (target={target_value:.2f})"
    elif args.data and not args.random_input:
        # Load from .pt file
        print(f"\nLoading input data from: {args.data}")
        input_tensor = torch.load(args.data, map_location=args.device, weights_only=True)
        print(f"Input shape: {input_tensor.shape}")
    else:
        # Generate random data
        print("\nGenerating random input data...")
        # Get in_channels from model
        in_channels = args.in_channels
        if hasattr(model, 'conv1'):
            in_channels = model.conv1.in_channels

        input_tensor = create_sample_input(
            batch_size=1,
            in_channels=in_channels,
            device=args.device
        )
        print(f"Created random input with shape: {input_tensor.shape}")

    # Create visualizer
    print("\nInitializing visualizer...")
    visualizer = IntermediateLayerVisualizer(model)

    # Run forward pass
    print("Running forward pass...")
    output = visualizer.forward(input_tensor)
    print(f"Model output shape: {output.shape}")

    if year_info:
        print(f"\nInput data: {year_info}")

    print(f"Model prediction: {output.item():.4f}")
    if target_value is not None:
        error = output.item() - target_value
        print(f"Prediction error: {error:.4f}")

    # Print statistics
    visualizer.print_layer_statistics()

    # Create output directory
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # Generate summary plot
    print("\nGenerating summary plot...")
    summary_path = os.path.join(output_dir, 'summary_all_layers.png')
    visualizer.create_summary_plot(save_path=summary_path)
    plt.close()

    # Generate detailed visualizations for all layers
    print("\nGenerating detailed visualizations...")
    visualizer.visualize_all_layers(
        output_dir=output_dir,
        num_features_per_layer=args.num_features
    )

    # Clean up
    visualizer.remove_hooks()

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - summary_all_layers.png (overview of all layers)")
    for layer_name in visualizer.intermediate_outputs.keys():
        safe_name = layer_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        print(f"  - {safe_name}.png")
    print("\n")


if __name__ == "__main__":
    main()

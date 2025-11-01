"""
Utility script to generate predictions from a trained model checkpoint.

This script loads a trained model from a checkpoint and generates predictions
for train, validation, and test sets without requiring retraining.

Usage:
    # Generate predictions using best checkpoint from default location
    # (will look for checkpoints/{model_name}_best_model.pth)
    python generate_predictions.py --config config/model_config.yml

    # Specify a custom checkpoint
    python generate_predictions.py --config config/model_config.yml --checkpoint checkpoints/MoK_CNN_02_best_model.pth

    # Override model name for output files
    python generate_predictions.py --config config/model_config.yml --model-name MoK_CNN_03
"""

import sys
import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import pandas as pd
from tqdm import tqdm

# Add src directory to Python path
current_file = Path(__file__)
scripts_dir = current_file.parent
training_dir = scripts_dir.parent
src_dir = training_dir.parent
sys.path.insert(0, str(src_dir))

# Import project modules
from models.architectures.cnn import MoK_CNN_Predictor
from data_pipeline.loaders.utils import load_config_and_create_dataloaders


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    split_name: str,
    output_path: Path
) -> None:
    """
    Generate predictions and save to CSV file.

    Args:
        model: PyTorch model
        data_loader: DataLoader for the split
        device: Device to use
        split_name: Name of the split (train/val/test)
        output_path: Path to save CSV file
    """
    model.eval()

    predictions = []
    targets = []
    years = []

    print(f"\nGenerating predictions for {split_name} set...")
    pbar = tqdm(data_loader, desc=f"Predicting {split_name}", ncols=100)

    with torch.no_grad():
        for data, metadata in pbar:
            # Move data to device
            data = data.to(device)

            # Extract target from metadata
            target = torch.stack([t for t in metadata['target']]).to(device)

            # Forward pass
            output = model(data)

            # Store predictions, targets, and years
            predictions.extend(output.cpu().numpy().flatten().tolist())
            targets.extend(target.cpu().numpy().flatten().tolist())
            years.extend(metadata['year'])

    # Create DataFrame
    df = pd.DataFrame({
        'Year': years,
        'Target': targets,
        'Prediction': predictions,
        'Error': [p - t for p, t in zip(predictions, targets)],
        'Absolute_Error': [abs(p - t) for p, t in zip(predictions, targets)]
    })

    # Sort by year
    df = df.sort_values('Year').reset_index(drop=True)

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Saved predictions to: {output_path}")

    # Print summary statistics
    mae = df['Absolute_Error'].mean()
    rmse = (df['Error'] ** 2).mean() ** 0.5
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")


def generate_predictions_from_checkpoint(
    config_path: str,
    checkpoint_path: str = None,
    model_name_override: str = None,
    device: str = 'cuda'
) -> None:
    """
    Load a trained model from checkpoint and generate predictions.

    Args:
        config_path: Path to YAML configuration file
        checkpoint_path: Path to model checkpoint (if None, uses default location)
        model_name_override: Optional model name override for output files
        device: Device to use ('cuda' or 'cpu')
    """
    print("=" * 80)
    print("Loading Configuration")
    print("=" * 80)
    print(f"Config file: {config_path}")

    # Load configuration
    config = load_config(config_path)

    # Override model name if provided
    if model_name_override:
        config['model']['name'] = model_name_override
        print(f"Model name overridden to: {model_name_override}")

    model_name = config['model']['name']

    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("\n" + "=" * 80)
    print("Loading Data")
    print("=" * 80)

    train_loader, val_loader, test_loader = load_config_and_create_dataloaders(
        config_path=config_path
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    print("\n" + "=" * 80)
    print("Creating Model")
    print("=" * 80)

    model = MoK_CNN_Predictor(in_channels=36)
    model = model.to(device)

    # Initialize lazy modules
    print("Initializing lazy modules...")
    dummy_input = torch.randn(1, 36, 1440, 481).to(device)
    with torch.no_grad():
        _ = model(dummy_input)
    print("Lazy modules initialized.")

    # Load checkpoint
    print("\n" + "=" * 80)
    print("Loading Model Checkpoint")
    print("=" * 80)

    # Determine checkpoint path
    if checkpoint_path is None:
        # Use default checkpoint location with model name
        project_root = src_dir.parent
        checkpoint_path = project_root / 'checkpoints' / f'{model_name}_best_model.pth'

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    print(f"  Validation loss: {checkpoint['metrics']['val_loss']:.6f}")
    if 'val_rmse' in checkpoint['metrics']:
        print(f"  Validation RMSE: {checkpoint['metrics']['val_rmse']:.6f}")

    # Generate predictions
    print("\n" + "=" * 80)
    print("Generating Predictions")
    print("=" * 80)

    # Create results directory
    project_root = src_dir.parent
    results_dir = project_root / 'results' / 'tables'

    # Save predictions for all splits
    save_predictions(
        model=model,
        data_loader=train_loader,
        device=device,
        split_name='train',
        output_path=results_dir / f'{model_name}_train_predictions.csv'
    )

    save_predictions(
        model=model,
        data_loader=val_loader,
        device=device,
        split_name='val',
        output_path=results_dir / f'{model_name}_val_predictions.csv'
    )

    save_predictions(
        model=model,
        data_loader=test_loader,
        device=device,
        split_name='test',
        output_path=results_dir / f'{model_name}_test_predictions.csv'
    )

    print("\n" + "=" * 80)
    print("All predictions saved successfully!")
    print("=" * 80)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Generate predictions from a trained model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/model_config.yml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (default: checkpoints/{model_name}_best_model.pth)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Override model name from config (used for output file names)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for inference (cpu or cuda)'
    )

    args = parser.parse_args()

    generate_predictions_from_checkpoint(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        model_name_override=args.model_name,
        device=args.device
    )


if __name__ == "__main__":
    main()

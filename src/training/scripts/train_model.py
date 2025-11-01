"""
Main training script for the MoK_CNN_Predictor model.

This script loads data from the configuration file, initializes the model,
and trains it using the specified hyperparameters.

Usage:
    # Basic training
    python train_model.py --config config/model_config.yml

    # Resume from checkpoint
    python train_model.py --config config/model_config.yml --resume checkpoints/best_model.pth

    # Override model name (useful for experiments)
    python train_model.py --config config/model_config.yml --model-name MoK_CNN_03

    # Custom seed and device
    python train_model.py --config config/model_config.yml --seed 123 --device cuda
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import pandas as pd
from tqdm import tqdm
import numpy as np
import random

# Add src directory to Python path so we can import project modules
# Current file: .../MoK_date_predict/src/training/scripts/train_model.py
# We want:     .../MoK_date_predict/src
current_file = Path(__file__)  # train_model.py
scripts_dir = current_file.parent  # scripts/
training_dir = scripts_dir.parent  # training/
src_dir = training_dir.parent  # src/
sys.path.insert(0, str(src_dir))

# Import project modules
from models.architectures.cnn import MoK_CNN_Predictor, create_model
from data_pipeline.loaders.utils import load_config_and_create_dataloaders
from training.callbacks import ModelCheckpoint, EarlyStopping, TensorBoardLogger


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Make CUDA operations deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to: {seed}")


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


def get_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """
    Create optimizer from configuration.

    Args:
        model: PyTorch model
        config: Training configuration dictionary

    Returns:
        PyTorch optimizer
    """
    optimizer_name = config['training']['optimizer'].lower()
    lr = config['training']['learning_rate']

    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    return optimizer


def get_loss_function(config: dict) -> nn.Module:
    """
    Create loss function from configuration.

    Args:
        config: Training configuration dictionary

    Returns:
        PyTorch loss function
    """
    loss_name = config['training']['loss'].lower()

    if loss_name == 'mse':
        loss_fn = nn.MSELoss()
    elif loss_name == 'mae' or loss_name == 'l1':
        loss_fn = nn.L1Loss()
    elif loss_name == 'huber':
        loss_fn = nn.HuberLoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")

    return loss_fn


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """
    Train the model for one epoch.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to use (cpu or cuda)
        epoch: Current epoch number

    Returns:
        Dictionary containing training metrics
    """
    model.train()

    total_loss = 0.0
    num_batches = len(train_loader)

    # Create progress bar
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", ncols=100, leave=False)

    for batch_idx, (data, metadata) in enumerate(pbar):
        # Move data to device
        data = data.to(device)

        # Extract target from metadata
        # metadata['target'] is a list of tensors, one per sample in batch
        target = torch.stack([t for t in metadata['target']]).to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Compute loss
        loss = loss_fn(output, target)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Compute average metrics
    avg_loss = total_loss / num_batches
    rmse = avg_loss ** 0.5  # Square root of MSE

    return {'loss': avg_loss, 'rmse': rmse}


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate the model.

    Args:
        model: PyTorch model
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device to use (cpu or cuda)

    Returns:
        Dictionary containing validation metrics
    """
    model.eval()

    total_loss = 0.0
    num_batches = len(val_loader)

    # Create progress bar
    pbar = tqdm(val_loader, desc="Validating", ncols=100, leave=False)

    with torch.no_grad():
        for data, metadata in pbar:
            # Move data to device
            data = data.to(device)

            # Extract target from metadata
            target = torch.stack([t for t in metadata['target']]).to(device)

            # Forward pass
            output = model(data)

            # Compute loss
            loss = loss_fn(output, target)

            # Accumulate loss
            total_loss += loss.item()

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Compute average metrics
    avg_loss = total_loss / num_batches
    rmse = avg_loss ** 0.5  # Square root of MSE

    return {'loss': avg_loss, 'rmse': rmse}


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


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: dict,
    device: torch.device,
    resume_checkpoint: str = None
) -> None:
    """
    Main training loop.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        config: Configuration dictionary
        device: Device to use (cpu or cuda)
        resume_checkpoint: Optional path to checkpoint to resume from
    """
    # Get training configuration
    num_epochs = config['training']['epochs']

    # Create optimizer and loss function
    optimizer = get_optimizer(model, config)
    loss_fn = get_loss_function(config)

    # Create callbacks
    model_name = config['model']['name']
    checkpoint_callback = ModelCheckpoint(
        checkpoint_dir='checkpoints',
        monitor=config['training']['checkpoint']['monitor'],
        mode='min' if 'loss' in config['training']['checkpoint']['monitor'] else 'max',
        save_best_only=config['training']['checkpoint']['save_best_only'],
        verbose=True,
        model_name=model_name
    )

    early_stopping = EarlyStopping(
        monitor=config['training']['early_stopping']['monitor'],
        patience=config['training']['early_stopping']['patience'],
        mode='min' if 'loss' in config['training']['early_stopping']['monitor'] else 'max',
        verbose=True
    ) if config['training']['early_stopping']['enabled'] else None

    tb_logger = TensorBoardLogger(
        log_dir=config['logging']['log_dir'],
        log_histograms=True
    ) if config['logging']['tensorboard'] else None

    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_checkpoint:
        checkpoint_info = checkpoint_callback.load_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_path=resume_checkpoint
        )
        start_epoch = checkpoint_info['epoch'] + 1
        print(f"Resumed training from epoch {start_epoch}")

    # Training loop
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        print("-" * 80)

        # Train for one epoch
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch
        )

        # Validate
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_fn,
            device=device
        )

        # Print metrics
        print(f"\nEpoch [{epoch + 1}/{num_epochs}] Results:")
        print(f"  Train Loss: {train_metrics['loss']:.6f}  |  Train RMSE: {train_metrics['rmse']:.6f}")
        print(f"  Val Loss:   {val_metrics['loss']:.6f}  |  Val RMSE:   {val_metrics['rmse']:.6f}")

        # Combine metrics for callbacks
        all_metrics = {
            'train_loss': train_metrics['loss'],
            'train_rmse': train_metrics['rmse'],
            'val_loss': val_metrics['loss'],
            'val_rmse': val_metrics['rmse']
        }

        # Log to TensorBoard
        if tb_logger:
            tb_logger.log_epoch_metrics(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                model=model,
                optimizer=optimizer
            )

        # Save checkpoint
        checkpoint_callback.on_epoch_end(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=all_metrics
        )

        # Check early stopping
        if early_stopping and early_stopping.on_epoch_end(epoch, all_metrics):
            print("\nEarly stopping triggered. Training stopped.")
            break

    print("\n" + "=" * 80)
    print("Training Completed!")
    print("=" * 80)

    # Close TensorBoard logger
    if tb_logger:
        tb_logger.close()

    # Load best model checkpoint for predictions
    print("\n" + "=" * 80)
    print("Loading Best Model for Predictions")
    print("=" * 80)

    best_checkpoint_path = checkpoint_callback.best_model_path
    if best_checkpoint_path and Path(best_checkpoint_path).exists():
        print(f"Loading best model from: {best_checkpoint_path}")
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded best model from epoch {checkpoint['epoch']}")
        print(f"  Best val_loss: {checkpoint['metrics']['val_loss']:.6f}")
    else:
        print("Warning: No best checkpoint found, using final model state")

    # Save predictions for all splits
    print("\n" + "=" * 80)
    print("Generating Predictions")
    print("=" * 80)

    # Get model name and create results directory
    model_name = config['model']['name']
    project_root = src_dir.parent
    results_dir = project_root / 'results' / 'tables'

    # Save predictions for train, val, and test sets
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
    """Main function to run training."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train MoK_CNN_Predictor model')
    parser.add_argument(
        '--config',
        type=str,
        default='config/model_config.yml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training (cpu or cuda)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Override model name from config (used for saving predictions and checkpoints)'
    )

    args = parser.parse_args()

    # Load configuration first to check for seed in config
    print("=" * 80)
    print("Loading Configuration")
    print("=" * 80)
    print(f"Config file: {args.config}")

    config = load_config(args.config)

    # Override model name if provided via command line
    if args.model_name:
        config['model']['name'] = args.model_name
        print(f"Model name overridden to: {args.model_name}")

    # Set random seed for reproducibility
    # Use seed from config if available and not overridden by command line
    seed = args.seed if args.seed != 42 else config.get('training', {}).get('random_seed', 42)
    print("\n" + "=" * 80)
    print("Setting Random Seed")
    print("=" * 80)
    set_random_seed(seed)

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load data
    print("\n" + "=" * 80)
    print("Loading Data")
    print("=" * 80)

    train_loader, val_loader, test_loader = load_config_and_create_dataloaders(
        config_path=args.config
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    print("\n" + "=" * 80)
    print("Creating Model")
    print("=" * 80)

    model = create_model()
    model = model.to(device)

    # Initialize lazy modules with a dummy forward pass
    print("Initializing lazy modules...")
    dummy_input = torch.randn(1, 36, 1440, 481).to(device)
    with torch.no_grad():
        _ = model(dummy_input)
    print("Lazy modules initialized.")

    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")

    # Train model
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        resume_checkpoint=args.resume
    )


if __name__ == "__main__":
    main()

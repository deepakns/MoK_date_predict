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

    # Force recompute normalization statistics (useful when changing data preprocessing)
    python train_model.py --config config/model_config.yml --force-recompute-stats

    # Combined example: Resume training with custom seed and force recompute stats
    python train_model.py --config config/model_config.yml --resume checkpoints/epoch_50.pth --seed 456 --force-recompute-stats
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

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
from data_pipeline.loaders.utils import load_config_and_create_dataloaders, parse_year_spec
from data_pipeline.preprocessing.normstats import (
    compute_normalization_stats,
    save_normalization_stats,
    load_normalization_stats,
    stats_exist
)
from data_pipeline.preprocessing.transformers import NormalizeWithPrecomputedStats
from data_pipeline.loaders.dataset_classes.monthly_dataset import MonthlyERA5Dataset
from training.callbacks import ModelCheckpoint, EarlyStopping, TensorBoardLogger, WandBLogger
from training.utils.lr_scheduler import ReduceLROnPlateauWithRestore
from training.utils.weight_init import initialize_weights


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

    # Get weight_decay for L2 regularization (default: 0.0 for no regularization)
    weight_decay = config['training'].get('weight_decay', 0.0)

    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        # AdamW has built-in decoupled weight decay, which is more effective
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    print(f"Optimizer: {optimizer_name}, Learning rate: {lr}, Weight decay (L2 reg): {weight_decay}")

    return optimizer


def get_scheduler(optimizer: torch.optim.Optimizer, config: dict, num_epochs: int = None, steps_per_epoch: int = None):
    """
    Create learning rate scheduler from configuration.

    Args:
        optimizer: PyTorch optimizer
        config: Training configuration dictionary
        num_epochs: Total number of training epochs (required for OneCycleLR)
        steps_per_epoch: Number of batches per epoch (required for OneCycleLR)

    Returns:
        Learning rate scheduler or None if not configured
    """
    scheduler_config = config['training'].get('lr_scheduler', None)

    if scheduler_config is None or not scheduler_config.get('enabled', False):
        print("Learning rate scheduler: Disabled")
        return None

    scheduler_type = scheduler_config.get('type', 'ReduceLROnPlateau').lower()
    restore_on_reduce = scheduler_config.get('restore_best_on_reduce', False)

    if scheduler_type == 'reduce_on_plateau' or scheduler_type == 'reducelronplateau':
        mode = scheduler_config.get('mode', 'min')
        factor = scheduler_config.get('factor', 0.5)
        patience = scheduler_config.get('patience', 5)
        min_lr = scheduler_config.get('min_lr', 1e-6)
        threshold = scheduler_config.get('threshold', 1e-4)

        if restore_on_reduce:
            # Use custom scheduler that restores model to best epoch when reducing LR
            scheduler = ReduceLROnPlateauWithRestore(
                optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
                min_lr=min_lr,
                threshold=threshold,
                verbose=True
            )
            print(f"Learning rate scheduler: ReduceLROnPlateauWithRestore")
            print(f"  Mode: {mode}, Factor: {factor}, Patience: {patience}")
            print(f"  Min LR: {min_lr}, Threshold: {threshold}")
            print(f"  ⚠ Model will restore to best epoch when LR is reduced")
        else:
            # Use standard PyTorch scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
                min_lr=min_lr,
                threshold=threshold
            )
            print(f"Learning rate scheduler: ReduceLROnPlateau")
            print(f"  Mode: {mode}, Factor: {factor}, Patience: {patience}")
            print(f"  Min LR: {min_lr}, Threshold: {threshold}")

        return scheduler

    elif scheduler_type == 'steplr':
        step_size = scheduler_config.get('step_size', 30)
        gamma = scheduler_config.get('gamma', 0.1)

        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )

        print(f"Learning rate scheduler: StepLR")
        print(f"  Step size: {step_size} epochs")
        print(f"  Gamma: {gamma} (LR *= {gamma} every {step_size} epochs)")

        return scheduler

    elif scheduler_type == 'onecyclelr':
        if num_epochs is None or steps_per_epoch is None:
            raise ValueError("OneCycleLR requires num_epochs and steps_per_epoch")

        max_lr = float(scheduler_config.get('max_lr', config['training']['learning_rate'] * 10))
        pct_start = float(scheduler_config.get('pct_start', 0.3))
        anneal_strategy = scheduler_config.get('anneal_strategy', 'cos')
        div_factor = float(scheduler_config.get('div_factor', 25.0))
        final_div_factor = float(scheduler_config.get('final_div_factor', 1e4))

        total_steps = num_epochs * steps_per_epoch

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            div_factor=div_factor,
            final_div_factor=final_div_factor
        )

        # OneCycleLR sets the initial LR immediately upon creation
        initial_lr = optimizer.param_groups[0]['lr']

        print(f"Learning rate scheduler: OneCycleLR")
        print(f"  Max LR: {max_lr}")
        print(f"  Initial LR: {initial_lr:.2e} (max_lr / div_factor)")
        print(f"  Final LR: {max_lr / (div_factor * final_div_factor):.2e}")
        print(f"  Total steps: {total_steps} ({num_epochs} epochs × {steps_per_epoch} steps/epoch)")
        print(f"  Warmup: {pct_start * 100:.0f}% of training")
        print(f"  Annealing: {anneal_strategy}")
        print(f"  ⚠ LR will be updated every batch (not every epoch)")

        return scheduler

    elif scheduler_type == 'exponentiallr':
        gamma = scheduler_config.get('gamma', 0.95)

        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma
        )

        print(f"Learning rate scheduler: ExponentialLR")
        print(f"  Gamma: {gamma} (LR *= {gamma} every epoch)")
        print(f"  LR decay formula: lr_epoch = lr_initial × {gamma}^epoch")

        return scheduler

    else:
        raise ValueError(
            f"Unsupported scheduler type: {scheduler_type}. "
            f"Supported: 'ReduceLROnPlateau', 'StepLR', 'OneCycleLR', 'ExponentialLR'"
        )


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
    epoch: int,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
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
        scheduler: Optional scheduler for batch-level stepping (e.g., OneCycleLR)

    Returns:
        Dictionary containing training metrics
    """
    model.train()

    total_loss = 0.0
    num_batches = len(train_loader)

    # Create progress bar
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", ncols=100, leave=False)

    for batch_idx, (data, target) in enumerate(pbar):
        # Move data and target to device
        data = data.to(device)
        target = target.to(device)

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

        # Step scheduler if it requires batch-level stepping (OneCycleLR)
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

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
        for data, target in pbar:
            # Move data and target to device
            data = data.to(device)
            target = target.to(device)

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
) -> Dict[str, float]:
    """
    Generate predictions and save to CSV file.

    Args:
        model: PyTorch model
        data_loader: DataLoader for the split
        device: Device to use
        split_name: Name of the split (train/val/test)
        output_path: Path to save CSV file

    Returns:
        Dictionary containing evaluation metrics (target_mean, target_std, mae, rmse, r2)
    """
    model.eval()

    predictions = []
    targets = []
    years = []

    print(f"\nGenerating predictions for {split_name} set...")
    pbar = tqdm(data_loader, desc=f"Predicting {split_name}", ncols=100)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(pbar):
            # Move data to device for model inference
            data = data.to(device)

            # Forward pass
            output = model(data)

            # Get metadata for years (need to access dataset directly)
            # Calculate which samples are in this batch
            batch_size = data.size(0)
            start_idx = batch_idx * data_loader.batch_size
            end_idx = start_idx + batch_size
            batch_years = [data_loader.dataset.get_metadata(i)['year']
                          for i in range(start_idx, end_idx)]

            # Store predictions, targets, and years
            predictions.extend(output.cpu().numpy().flatten().tolist())
            targets.extend(target.numpy().flatten().tolist())
            years.extend(batch_years)

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

    # Calculate statistics
    target_mean = df['Target'].mean()
    target_std = df['Target'].std()
    mae = df['Absolute_Error'].mean()
    rmse = (df['Error'] ** 2).mean() ** 0.5

    # Calculate R² score
    # R² = 1 - (SS_res / SS_tot)
    # where SS_res = sum of squared residuals, SS_tot = total sum of squares
    ss_res = (df['Error'] ** 2).sum()
    ss_tot = ((df['Target'] - target_mean) ** 2).sum()
    r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Saved predictions to: {output_path}")

    # Print summary statistics
    print(f"  Target Mean: {target_mean:.4f}")
    print(f"  Target Std:  {target_std:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2_score:.4f}")

    # Return statistics
    return {
        'target_mean': target_mean,
        'target_std': target_std,
        'mae': mae,
        'rmse': rmse,
        'r2': r2_score,
        'num_samples': len(df)
    }


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

    # Create learning rate scheduler
    # Note: For OneCycleLR, we need to pass num_epochs and steps_per_epoch
    scheduler = get_scheduler(
        optimizer,
        config,
        num_epochs=num_epochs,
        steps_per_epoch=len(train_loader)
    )

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

    # Initialize Weights & Biases logger if enabled
    wandb_logger = None
    if config['logging'].get('wandb', {}).get('enabled', False):
        wandb_config = config['logging']['wandb']
        wandb_logger = WandBLogger(
            project=wandb_config.get('project', 'MoK_date_predict'),
            entity=wandb_config.get('entity', None),
            name=wandb_config.get('name', model_name),
            config=config,
            log_gradients=wandb_config.get('log_gradients', False),
            log_model=wandb_config.get('log_model', True),
            tags=wandb_config.get('tags', []),
            notes=wandb_config.get('notes', None)
        )
        # Watch model for gradient tracking if enabled
        if wandb_config.get('watch_model', False):
            wandb_logger.watch_model(
                model,
                log=wandb_config.get('watch_log', 'gradients'),
                log_freq=wandb_config.get('watch_freq', 100)
            )

    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_checkpoint:
        checkpoint_info = checkpoint_callback.load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
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
        # Pass scheduler for OneCycleLR which requires batch-level stepping
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            scheduler=scheduler if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR) else None
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

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning Rate: {current_lr:.2e}")

        # Combine metrics for callbacks
        all_metrics = {
            'train_loss': train_metrics['loss'],
            'train_rmse': train_metrics['rmse'],
            'val_loss': val_metrics['loss'],
            'val_rmse': val_metrics['rmse']
        }

        # Step the learning rate scheduler (if enabled)
        if scheduler is not None:
            # Check if using custom scheduler with restore capability
            if isinstance(scheduler, ReduceLROnPlateauWithRestore):
                # Custom scheduler needs model to restore state when LR is reduced
                restored = scheduler.step(val_metrics['loss'], model, epoch=epoch)
                if restored:
                    print(f"  ⚠ Model restored to best epoch - continuing training from there")
                    # Reset early stopping counter to give model a fresh chance with new LR
                    if early_stopping:
                        early_stopping.reset()
                        print(f"  ✓ Early stopping counter reset")
            elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Standard PyTorch ReduceLROnPlateau (metric-based)
                scheduler.step(val_metrics['loss'])
            elif isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                # OneCycleLR steps per batch, not per epoch (handled in train_one_epoch)
                pass
            else:
                # Other schedulers (StepLR, ExponentialLR, etc.) step per epoch
                scheduler.step()

        # Log to TensorBoard
        if tb_logger:
            tb_logger.log_epoch_metrics(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                model=model,
                optimizer=optimizer
            )

        # Log to Weights & Biases
        if wandb_logger:
            wandb_logger.log_epoch_metrics(
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
            metrics=all_metrics,
            scheduler=scheduler
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

    # Note: Don't close wandb_logger yet - we need it for final metrics logging

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

    # Save predictions for train, val, and test sets and collect statistics
    train_stats = save_predictions(
        model=model,
        data_loader=train_loader,
        device=device,
        split_name='train',
        output_path=results_dir / f'{model_name}_train_predictions.csv'
    )

    val_stats = save_predictions(
        model=model,
        data_loader=val_loader,
        device=device,
        split_name='val',
        output_path=results_dir / f'{model_name}_val_predictions.csv'
    )

    test_stats = save_predictions(
        model=model,
        data_loader=test_loader,
        device=device,
        split_name='test',
        output_path=results_dir / f'{model_name}_test_predictions.csv'
    )

    # Create summary statistics file
    summary_df = pd.DataFrame({
        'Split': ['Train', 'Validation', 'Test'],
        'Num_Samples': [train_stats['num_samples'], val_stats['num_samples'], test_stats['num_samples']],
        'Target_Mean': [train_stats['target_mean'], val_stats['target_mean'], test_stats['target_mean']],
        'Target_Std': [train_stats['target_std'], val_stats['target_std'], test_stats['target_std']],
        'MAE': [train_stats['mae'], val_stats['mae'], test_stats['mae']],
        'RMSE': [train_stats['rmse'], val_stats['rmse'], test_stats['rmse']],
        'R2': [train_stats['r2'], val_stats['r2'], test_stats['r2']]
    })

    summary_path = results_dir / f'{model_name}_metrics_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Saved metrics summary to: {summary_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("Metrics Summary")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    # Log final summary metrics to W&B
    if wandb_logger:
        # Log summary metrics
        wandb_logger.log_summary({
            'final/train_mae': train_stats['mae'],
            'final/train_rmse': train_stats['rmse'],
            'final/train_r2': train_stats['r2'],
            'final/val_mae': val_stats['mae'],
            'final/val_rmse': val_stats['rmse'],
            'final/val_r2': val_stats['r2'],
            'final/test_mae': test_stats['mae'],
            'final/test_rmse': test_stats['rmse'],
            'final/test_r2': test_stats['r2']
        })

        # Log predictions tables if enabled
        if config['logging']['wandb'].get('log_predictions', False):
            train_pred_df = pd.read_csv(results_dir / f'{model_name}_train_predictions.csv')
            val_pred_df = pd.read_csv(results_dir / f'{model_name}_val_predictions.csv')
            test_pred_df = pd.read_csv(results_dir / f'{model_name}_test_predictions.csv')

            wandb_logger.log_predictions(train_pred_df, 'train')
            wandb_logger.log_predictions(val_pred_df, 'val')
            wandb_logger.log_predictions(test_pred_df, 'test')

        # Log best model checkpoint
        if best_checkpoint_path and Path(best_checkpoint_path).exists():
            wandb_logger.log_model_checkpoint(
                checkpoint_path=str(best_checkpoint_path),
                best_metric=checkpoint['metrics']['val_loss']
            )

    print("\n" + "=" * 80)
    print("All predictions saved successfully!")
    print("=" * 80)

    # Close Weights & Biases logger (after all logging is complete)
    if wandb_logger:
        wandb_logger.finish()


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
    parser.add_argument(
        '--force-recompute-stats',
        action='store_true',
        help='Force recomputation of normalization statistics even if they exist'
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

    # First, load data WITHOUT normalization to compute statistics
    train_loader_no_norm, val_loader, test_loader = load_config_and_create_dataloaders(
        config_path=args.config
    )

    print(f"Train batches: {len(train_loader_no_norm)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Check normalization strategy from config
    data_config = config.get('data', {})
    normalize_strategy = data_config.get('normalize_strategy', 1)  # Default to 1 for backward compatibility

    print("\n" + "=" * 80)
    print("Normalization Strategy")
    print("=" * 80)
    print(f"normalize_strategy: {normalize_strategy}")

    normalize_transform = None
    model_name = config['model']['name']

    if normalize_strategy == 1:
        # Strategy 1: Normalize using training data statistics
        print("Strategy: Normalize using training data statistics (spatially-varying)")

        # Check if we should force recomputation
        force_recompute = args.force_recompute_stats
        stats_already_exist = stats_exist(model_name)

        if force_recompute and stats_already_exist:
            print(f"⚠ Forcing recomputation of normalization statistics (--force-recompute-stats flag set)")
            print(f"   Existing statistics for '{model_name}' will be overwritten.")

        if stats_already_exist and not force_recompute:
            print(f"Loading existing normalization statistics for '{model_name}'...")
            norm_stats = load_normalization_stats(model_name, verbose=True)
        else:
            if not stats_already_exist:
                print(f"Computing normalization statistics from training data for '{model_name}'...")
            else:
                print(f"Recomputing normalization statistics from training data for '{model_name}'...")
            norm_stats = compute_normalization_stats(
                train_loader=train_loader_no_norm,
                model_name=model_name,
                device=device,
                verbose=True
            )
            # Save statistics for future use
            save_normalization_stats(norm_stats, verbose=True)

        # Create normalization transform
        normalize_transform = NormalizeWithPrecomputedStats(
            mean=norm_stats.mean,
            std=norm_stats.std,
            static_channel_indices=norm_stats.static_channel_indices
        )
        print("✓ Normalization transform created")

    elif normalize_strategy == 0:
        # Strategy 0: No normalization
        print("Strategy: No normalization (using raw data)")
        normalize_transform = None

    else:
        # Future strategies can be added here
        raise ValueError(
            f"Unsupported normalize_strategy: {normalize_strategy}. "
            f"Supported values: 0 (no normalization), 1 (training data stats)"
        )

    # Recreate dataloaders with or without normalization
    print("\nRecreating dataloaders...")
    train_years = parse_year_spec(data_config.get('train_years', []))
    val_years = parse_year_spec(data_config.get('val_years', []))
    test_years = parse_year_spec(data_config.get('test_years', []))

    # Get time_steps configuration (with backward compatibility)
    time_steps = data_config.get('time_steps', None)
    num_time_steps = data_config.get('num_time_steps', None)

    # Get input variable lists
    input_geo_var_surf = data_config.get('input_geo_var_surf', None)
    input_geo_var_press = data_config.get('input_geo_var_press', None)

    # Get optional static channel flags
    include_lat = data_config.get('include_lat', True)
    include_lon = data_config.get('include_lon', True)
    include_landsea = data_config.get('include_landsea', True)

    # Create datasets with optional normalization
    train_dataset = MonthlyERA5Dataset(
        data_dir=data_config.get('data_dir', '/gdata2/ERA5/monthly'),
        years=train_years,
        time_steps=time_steps,
        num_time_steps=num_time_steps,  # Fallback for backward compatibility
        pressure_levels=data_config.get('pressure_levels', [0, 1]),
        target_file=data_config.get('target_file', None),
        transform=normalize_transform,  # None if strategy=0
        input_geo_var_surf=input_geo_var_surf,
        input_geo_var_press=input_geo_var_press,
        include_lat=include_lat,
        include_lon=include_lon,
        include_landsea=include_landsea
    )

    val_dataset = MonthlyERA5Dataset(
        data_dir=data_config.get('data_dir', '/gdata2/ERA5/monthly'),
        years=val_years,
        time_steps=time_steps,
        num_time_steps=num_time_steps,  # Fallback for backward compatibility
        pressure_levels=data_config.get('pressure_levels', [0, 1]),
        target_file=data_config.get('target_file', None),
        transform=normalize_transform,  # None if strategy=0
        input_geo_var_surf=input_geo_var_surf,
        input_geo_var_press=input_geo_var_press,
        include_lat=include_lat,
        include_lon=include_lon,
        include_landsea=include_landsea
    )

    test_dataset = MonthlyERA5Dataset(
        data_dir=data_config.get('data_dir', '/gdata2/ERA5/monthly'),
        years=test_years,
        time_steps=time_steps,
        num_time_steps=num_time_steps,  # Fallback for backward compatibility
        pressure_levels=data_config.get('pressure_levels', [0, 1]),
        target_file=data_config.get('target_file', None),
        transform=normalize_transform,  # None if strategy=0
        input_geo_var_surf=input_geo_var_surf,
        input_geo_var_press=input_geo_var_press,
        include_lat=include_lat,
        include_lon=include_lon,
        include_landsea=include_landsea
    )

    # Create dataloaders
    batch_size = data_config.get('batch_size', 32)
    num_workers = data_config.get('num_workers', 4)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    normalization_status = "with normalization" if normalize_strategy == 1 else "without normalization"
    print(f"✓ Dataloaders created {normalization_status}")
    print(f"  Train: {len(train_loader)} batches, {len(train_dataset)} samples")
    print(f"  Val:   {len(val_loader)} batches, {len(val_dataset)} samples")
    print(f"  Test:  {len(test_loader)} batches, {len(test_dataset)} samples")

    # Get channel information from dataset to determine model input shape
    channel_info = train_dataset.get_channel_info()
    in_channels = channel_info['num_channels']
    print(f"\nDataset configuration:")
    print(f"  Total channels: {in_channels}")
    print(f"  Time steps: {channel_info['time_steps']}")
    print(f"  Pressure levels: {channel_info['pressure_levels']}")

    # Create model
    print("\n" + "=" * 80)
    print("Creating Model")
    print("=" * 80)

    # Get dropout_rate from config (default to 0.5 if not specified)
    dropout_rate = config.get('model', {}).get('dropout_rate', 0.5)
    print(f"Dropout rate: {dropout_rate}")

    model = create_model(in_channels=in_channels, dropout_rate=dropout_rate)
    model = model.to(device)

    # Get spatial dimensions from first batch
    sample_data, _ = train_dataset[0]
    _, spatial_h, spatial_w = sample_data.shape

    # Initialize lazy modules with a dummy forward pass
    print("Initializing lazy modules...")
    dummy_input = torch.randn(1, in_channels, spatial_h, spatial_w).to(device)
    with torch.no_grad():
        _ = model(dummy_input)
    print("Lazy modules initialized.")

    # Initialize weights if specified in config
    weight_init_config = config.get('model', {}).get('weight_init', {})
    if weight_init_config.get('enabled', False):
        print("\n" + "=" * 80)
        print("Initializing Weights")
        print("=" * 80)

        # Extract initialization parameters
        init_params = {
            'method': weight_init_config.get('method', 'kaiming_normal'),
            'activation': weight_init_config.get('activation', 'relu'),
            'bias_init': weight_init_config.get('bias_init', 'zeros'),
            'bias_value': weight_init_config.get('bias_value', 0.0),
            'mode': weight_init_config.get('mode', 'fan_in'),
            'gain': weight_init_config.get('gain', 1.0),
            'mean': weight_init_config.get('mean', 0.0),
            'std': weight_init_config.get('std', 0.01),
            'a': weight_init_config.get('a', 0.0),
            'b': weight_init_config.get('b', 1.0),
            'verbose': weight_init_config.get('verbose', True)
        }

        # Initialize weights
        initialize_weights(model, **init_params)
    else:
        print("\nWeight initialization: Using PyTorch defaults")

    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {num_params:,}")
    print(f"  Trainable parameters: {num_trainable:,}")

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

"""
Main training script for MoK CNN models (supports cnn, cnn_v1, and cnn_v2 architectures).

This script loads data from the configuration file, initializes the model,
and trains it using the specified hyperparameters. It supports three CNN architectures
that can be selected via the config file's 'architecture' parameter.

Supported Model Architectures:
    - cnn (current/default): Advanced CNN with residual blocks, configurable SPP,
                             and flexible normalization. Supports lat/lon as input channels.

    - cnn_v1 (old/original): Simplified CNN with 6 convolutional blocks (stride-2),
                              channel progression: input → 32 → 64 → 128 → 256 → 512 → 1024,
                              spatial pyramid pooling, selective L2 regularization.

    - cnn_v2 (with positional embeddings): Advanced CNN that uses lat/lon coordinates
                                            to create sinusoidal positional embeddings
                                            (similar to Transformers) which are merged
                                            with each input channel. Lat/lon are NOT
                                            used as input channels but for embeddings.

Normalization Options:
    - normalize_strategy: 0 (none) or 1 (precomputed statistics)
    - add_per_channel_norm: true/false (applies per-channel min-max [0,1] normalization)
    - Combined: Can use precomputed stats + per-channel min-max for enhanced normalization

Configuration:
    Set architecture in config file:
        model:
          architecture: "cnn"     # Options: "cnn", "cnn_v1", "cnn_v2"
          pos_embedding_dim: 16   # For cnn_v2 only
          num_frequencies: 16     # For cnn_v2 only

        data:
          normalize_strategy: 1           # 0: none, 1: precomputed stats
          add_per_channel_norm: true      # Add per-channel min-max normalization
          include_lat: true               # For cnn_v2: MUST be true (used for embeddings)
          include_lon: true               # For cnn/cnn_v1: optional input channels

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
from sklearn.metrics import f1_score

# Add src directory to Python path so we can import project modules
# Current file: .../MoK_date_predict/src/training/scripts/train_model.py
# We want:     .../MoK_date_predict/src
current_file = Path(__file__)  # train_model.py
scripts_dir = current_file.parent  # scripts/
training_dir = scripts_dir.parent  # training/
src_dir = training_dir.parent  # src/
sys.path.insert(0, str(src_dir))

# Import project modules
# Import CNN architectures
from models.architectures.cnn import create_model as create_model_cnn
from models.architectures.cnn_v1 import create_model as create_model_cnn_v1
from models.architectures.cnn_v2 import create_model as create_model_v2
from data_pipeline.loaders.utils import load_config_and_create_dataloaders, parse_year_spec
from data_pipeline.preprocessing.normstats import (
    compute_normalization_stats,
    save_normalization_stats,
    load_normalization_stats,
    stats_exist
)
from data_pipeline.preprocessing.transformers import NormalizeWithPrecomputedStats
from data_pipeline.loaders.dataset_classes.monthly_dataset import MonthlyERA5Dataset
from data_pipeline.loaders.dataset_classes.era5_raw_dataset import ERA5RawDataset
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

    Supported optimizers:
        - adam: Adam optimizer
        - adamw: AdamW optimizer with decoupled weight decay (recommended)
        - sgd: SGD with momentum (0.9)
        - rmsprop: RMSprop optimizer with alpha=0.99, eps=1e-8

    Note:
        For models with get_parameter_groups() method (like MoK_CNN_Predictor_V2),
        L2 regularization is applied selectively to conv layers only.
    """
    optimizer_name = config['training']['optimizer'].lower()
    lr = config['training']['learning_rate']

    # Get weight_decay for L2 regularization (default: 0.0 for no regularization)
    weight_decay = config['training'].get('weight_decay', 0.0)

    # Get RMSprop specific parameters (if applicable)
    rmsprop_alpha = config['training'].get('rmsprop_alpha', 0.99)
    rmsprop_eps = config['training'].get('rmsprop_eps', 1e-8)
    rmsprop_momentum = config['training'].get('rmsprop_momentum', 0.0)

    # Check if model has get_parameter_groups method for selective L2 regularization
    if hasattr(model, 'get_parameter_groups'):
        print(f"Using model's parameter groups for selective L2 regularization")
        param_groups = model.get_parameter_groups()

        if optimizer_name == 'adam':
            optimizer = optim.Adam(param_groups, lr=lr)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(param_groups, lr=lr, momentum=0.9)
        elif optimizer_name == 'adamw':
            # AdamW has built-in decoupled weight decay, which is more effective
            optimizer = optim.AdamW(param_groups, lr=lr)
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(param_groups, lr=lr, alpha=rmsprop_alpha,
                                     eps=rmsprop_eps, momentum=rmsprop_momentum)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}. Choose from: adam, adamw, sgd, rmsprop")

        print(f"Optimizer: {optimizer_name}, Learning rate: {lr}")
        if optimizer_name == 'rmsprop':
            print(f"  RMSprop alpha: {rmsprop_alpha}, eps: {rmsprop_eps}, momentum: {rmsprop_momentum}")
        print(f"  L2 regularization on conv layers: {weight_decay}")
        print(f"  No L2 regularization on BatchNorm/biases/output layer")
    else:
        # Fallback to standard optimizer with uniform weight decay
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            # AdamW has built-in decoupled weight decay, which is more effective
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=rmsprop_alpha,
                                     eps=rmsprop_eps, momentum=rmsprop_momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}. Choose from: adam, adamw, sgd, rmsprop")

        print(f"Optimizer: {optimizer_name}, Learning rate: {lr}, Weight decay (L2 reg): {weight_decay}")
        if optimizer_name == 'rmsprop':
            print(f"  RMSprop alpha: {rmsprop_alpha}, eps: {rmsprop_eps}, momentum: {rmsprop_momentum}")

    return optimizer


def get_scheduler(optimizer: torch.optim.Optimizer, config: dict, num_epochs: int = None, steps_per_epoch: int = None, metric_mode: str = 'min'):
    """
    Create learning rate scheduler from configuration.

    Args:
        optimizer: PyTorch optimizer
        config: Training configuration dictionary
        num_epochs: Total number of training epochs (required for OneCycleLR)
        steps_per_epoch: Number of batches per epoch (required for OneCycleLR)
        metric_mode: Mode for metric-based schedulers ('min' or 'max')
                     This should match the validation metric mode

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
        # Use metric_mode passed from validation metric configuration
        # This ensures scheduler and checkpoint callback use the same mode
        mode = metric_mode
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
    elif loss_name == 'cross_entropy' or loss_name == 'crossentropy' or loss_name == 'CrossEntropyLoss':
        # CrossEntropyLoss for multiclass classification
        # Expects raw logits (no softmax) and class indices as targets
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")

    return loss_fn


def extract_lat_lon_from_dataset(dataset) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[list]]:
    """
    Extract lat and lon grids from the dataset and return indices to remove.

    Args:
        dataset: Dataset instance with lat/lon data

    Returns:
        Tuple of (lat_tensor, lon_tensor, indices_to_remove) or (None, None, None) if not available
        indices_to_remove is a list of channel indices to exclude from the input for CNN V2
    """
    try:
        # Get a sample to extract lat/lon
        sample_data, _ = dataset[0]
        channel_info = dataset.get_channel_info()

        # Check if lat and lon are in the channels
        has_lat = dataset.include_lat
        has_lon = dataset.include_lon

        if not (has_lat and has_lon):
            return None, None, None

        # Find lat and lon channel indices
        lat_idx = None
        lon_idx = None

        # Get metadata to find exact channel positions
        metadata = dataset.get_metadata(0)
        channel_names = metadata['channel_names']

        for i, name in enumerate(channel_names):
            if name == 'lat':
                lat_idx = i
            elif name == 'lon':
                lon_idx = i

        if lat_idx is None or lon_idx is None:
            return None, None, None

        # Extract lat and lon from sample
        lat = sample_data[lat_idx]  # (H, W)
        lon = sample_data[lon_idx]  # (H, W)

        # Return indices to remove
        indices_to_remove = [lat_idx, lon_idx]

        return lat, lon, indices_to_remove
    except Exception as e:
        print(f"Warning: Could not extract lat/lon from dataset: {e}")
        return None, None, None


def remove_lat_lon_channels(data: torch.Tensor, indices_to_remove: list) -> torch.Tensor:
    """
    Remove lat and lon channels from the input data.

    Args:
        data: Input tensor of shape (B, C, H, W)
        indices_to_remove: List of channel indices to remove

    Returns:
        Tensor with lat/lon channels removed
    """
    if indices_to_remove is None or len(indices_to_remove) == 0:
        return data

    # Create a mask for channels to keep
    all_indices = set(range(data.size(1)))
    indices_to_keep = sorted(all_indices - set(indices_to_remove))

    # Select only the channels to keep
    return data[:, indices_to_keep, :, :]


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    gaussian_noise_std: float = 0.0,
    is_cnn_v2: bool = False,
    lat: Optional[torch.Tensor] = None,
    lon: Optional[torch.Tensor] = None,
    lat_lon_indices: Optional[list] = None
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
        gaussian_noise_std: Standard deviation for Gaussian noise augmentation
        is_cnn_v2: Whether using CNN V2 architecture (requires lat/lon)
        lat: Latitude tensor for CNN V2 (H, W)
        lon: Longitude tensor for CNN V2 (H, W)
        lat_lon_indices: List of channel indices for lat/lon to remove from input

    Returns:
        Dictionary containing training metrics
    """
    model.train()

    total_loss = 0.0
    num_batches = len(train_loader)

    # For classification metrics
    is_classification = isinstance(loss_fn, nn.CrossEntropyLoss)
    all_predictions = []
    all_targets = []

    # Create progress bar
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", ncols=100, leave=False)

    for batch_idx, (data, target) in enumerate(pbar):
        # Move data and target to device
        data = data.to(device)
        target = target.to(device)

        # Apply Gaussian noise augmentation (training only)
        if gaussian_noise_std > 0:
            noise = torch.randn_like(data) * gaussian_noise_std
            data = data + noise

        # For classification, targets must be integers (class indices)
        if is_classification:
            target = target.long().view(-1)  # Convert to long and flatten to 1D

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        if is_cnn_v2:
            # CNN V2 requires lat and lon
            if lat is None or lon is None:
                raise ValueError("CNN V2 requires lat and lon to be provided")
            # Remove lat/lon channels from input data
            if lat_lon_indices is not None and len(lat_lon_indices) > 0:
                data = remove_lat_lon_channels(data, lat_lon_indices)
            # Prepare lat and lon for batch: expand to (B, 1, H, W)
            batch_size = data.size(0)
            lat_batch = lat.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1).to(device)
            lon_batch = lon.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1).to(device)
            output = model(data, lat_batch, lon_batch)
        else:
            # Standard forward pass
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

        # Collect predictions and targets for classification metrics
        if is_classification:
            _, predicted = torch.max(output, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Compute average metrics
    avg_loss = total_loss / num_batches

    metrics = {'loss': avg_loss}

    # For classification: compute F1 score
    if is_classification:
        f1_macro = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
        metrics['f1_score'] = f1_macro
    else:
        # For regression: compute RMSE from predictions and targets
        # Note: We don't collect predictions/targets during training to save memory
        # So we use the average loss as a proxy (only valid if loss_fn is MSE)
        rmse = avg_loss ** 0.5
        metrics['rmse'] = rmse

    return metrics


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    train_target_mean: Optional[float] = None,
    is_cnn_v2: bool = False,
    lat: Optional[torch.Tensor] = None,
    lon: Optional[torch.Tensor] = None,
    lat_lon_indices: Optional[list] = None
) -> Dict[str, float]:
    """
    Validate the model.

    Args:
        model: PyTorch model
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device to use (cpu or cuda)
        train_target_mean: Mean of training targets (for skill score calculation in regression)
        is_cnn_v2: Whether using CNN V2 architecture (requires lat/lon)
        lat: Latitude tensor for CNN V2 (H, W)
        lon: Longitude tensor for CNN V2 (H, W)
        lat_lon_indices: List of channel indices for lat/lon to remove from input

    Returns:
        Dictionary containing validation metrics
    """
    model.eval()

    total_loss = 0.0
    num_batches = len(val_loader)

    # For classification metrics
    is_classification = isinstance(loss_fn, nn.CrossEntropyLoss)
    all_predictions = []
    all_targets = []

    # Create progress bar
    pbar = tqdm(val_loader, desc="Validating", ncols=100, leave=False)

    with torch.no_grad():
        for data, target in pbar:
            # Move data and target to device
            data = data.to(device)
            target = target.to(device)

            # For classification, targets must be integers (class indices)
            if is_classification:
                target = target.long().view(-1)  # Convert to long and flatten to 1D

            # Forward pass
            if is_cnn_v2:
                # CNN V2 requires lat and lon
                if lat is None or lon is None:
                    raise ValueError("CNN V2 requires lat and lon to be provided")
                # Remove lat/lon channels from input data
                if lat_lon_indices is not None and len(lat_lon_indices) > 0:
                    data = remove_lat_lon_channels(data, lat_lon_indices)
                # Prepare lat and lon for batch: expand to (B, 1, H, W)
                batch_size = data.size(0)
                lat_batch = lat.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1).to(device)
                lon_batch = lon.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1).to(device)
                output = model(data, lat_batch, lon_batch)
            else:
                # Standard forward pass
                output = model(data)

            # Compute loss
            loss = loss_fn(output, target)

            # Accumulate loss
            total_loss += loss.item()

            # Collect predictions and targets for metrics
            if is_classification:
                _, predicted = torch.max(output, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
            else:
                # For regression: collect raw predictions and targets
                # Handle both single and batch predictions
                pred_np = output.squeeze().cpu().numpy()
                target_np = target.squeeze().cpu().numpy()

                # Convert scalars to lists for extend
                if pred_np.ndim == 0:
                    all_predictions.append(pred_np.item())
                    all_targets.append(target_np.item())
                else:
                    all_predictions.extend(pred_np)
                    all_targets.extend(target_np)

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Compute average metrics
    avg_loss = total_loss / num_batches

    metrics = {'loss': avg_loss}

    # For classification: compute F1 score
    if is_classification:
        from sklearn.metrics import f1_score
        f1_macro = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
        metrics['f1_score'] = f1_macro
    else:
        # For regression: compute multiple metrics
        import numpy as np
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        predictions = np.array(all_predictions)
        targets = np.array(all_targets)

        # RMSE (computed from predictions and targets)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        metrics['rmse'] = rmse

        # MAE (Mean Absolute Error)
        mae = mean_absolute_error(targets, predictions)
        metrics['mae'] = mae

        # Skill Score: 1 - (MSE_model / MSE_baseline)
        # where baseline predicts the training mean for all samples
        # A score of 1 means perfect predictions, 0 means same as baseline, <0 means worse than baseline
        if train_target_mean is not None:
            mse_model = mean_squared_error(targets, predictions)
            mse_baseline = mean_squared_error(targets, np.full_like(targets, train_target_mean))
            skill_score = 1 - (mse_model / mse_baseline) if mse_baseline > 0 else 0.0
            metrics['skill_score'] = skill_score
        else:
            # Fallback: compute R² if training mean not provided
            from sklearn.metrics import r2_score
            r2 = r2_score(targets, predictions)
            metrics['r2'] = r2

    return metrics


def save_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    split_name: str,
    output_path: Path,
    is_classification: bool = False,
    num_classes: int = None,
    train_target_mean: Optional[float] = None,
    is_cnn_v2: bool = False,
    lat: Optional[torch.Tensor] = None,
    lon: Optional[torch.Tensor] = None,
    lat_lon_indices: Optional[list] = None
) -> Dict[str, float]:
    """
    Generate predictions and save to CSV file.

    Args:
        model: PyTorch model
        data_loader: DataLoader for the split
        device: Device to use
        split_name: Name of the split (train/val/test)
        output_path: Path to save CSV file
        is_classification: If True, output predicted classes; if False, output regression values
        num_classes: Number of classes for classification (if None, computed from data)
        train_target_mean: Mean of training targets (for skill score calculation in regression)
        is_cnn_v2: Whether using CNN V2 architecture (requires lat/lon)
        lat: Latitude tensor for CNN V2 (H, W)
        lon: Longitude tensor for CNN V2 (H, W)
        lat_lon_indices: List of channel indices for lat/lon to remove from input

    Returns:
        Dictionary containing evaluation metrics
        - For classification: accuracy, f1_macro, f1_per_class, confusion_matrix, num_samples, num_correct
        - For regression: target_mean, target_std, mae, rmse, skill_score (or r2), num_samples
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
            if is_cnn_v2:
                # CNN V2 requires lat and lon
                if lat is None or lon is None:
                    raise ValueError("CNN V2 requires lat and lon to be provided")
                # Remove lat/lon channels from input data
                if lat_lon_indices is not None and len(lat_lon_indices) > 0:
                    data = remove_lat_lon_channels(data, lat_lon_indices)
                # Prepare lat and lon for batch: expand to (B, 1, H, W)
                batch_size = data.size(0)
                lat_batch = lat.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1).to(device)
                lon_batch = lon.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1).to(device)
                output = model(data, lat_batch, lon_batch)
            else:
                # Standard forward pass
                output = model(data)

            # Get metadata for years (need to access dataset directly)
            # Calculate which samples are in this batch
            batch_size = data.size(0)
            start_idx = batch_idx * data_loader.batch_size
            end_idx = start_idx + batch_size
            batch_years = [data_loader.dataset.get_metadata(i)['year']
                          for i in range(start_idx, end_idx)]

            # Store predictions based on task type
            if is_classification:
                # For classification: get predicted class (argmax of logits)
                pred_classes = torch.argmax(output, dim=1).cpu().numpy().tolist()
                predictions.extend(pred_classes)
            else:
                # For regression: use raw output
                predictions.extend(output.cpu().numpy().flatten().tolist())

            targets.extend(target.numpy().flatten().tolist())
            years.extend(batch_years)

    # Create DataFrame
    if is_classification:
        # Classification: predictions and targets are class indices
        df = pd.DataFrame({
            'Year': years,
            'Target_Class': [int(t) for t in targets],
            'Predicted_Class': [int(p) for p in predictions],
            'Correct': [int(p) == int(t) for p, t in zip(predictions, targets)]
        })
    else:
        # Regression: predictions and targets are continuous values
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

    # Calculate and print statistics based on task type
    if is_classification:
        # Classification metrics
        from sklearn.metrics import confusion_matrix, f1_score, classification_report

        y_true = df['Target_Class'].values
        y_pred = df['Predicted_Class'].values

        # Use provided num_classes or compute from data
        if num_classes is None:
            all_classes = sorted(set(y_true) | set(y_pred))
            num_classes = max(all_classes) + 1 if all_classes else 0

        accuracy = df['Correct'].mean()
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0, labels=list(range(num_classes)))
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(num_classes)))
        conf_matrix = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  F1 Score (Macro): {f1_macro:.4f}")
        print(f"  Correct:  {df['Correct'].sum()}/{len(df)}")

        # Print per-class F1 scores
        print(f"\n  Per-Class F1 Scores:")
        for class_idx, f1 in enumerate(f1_per_class):
            print(f"    Class {class_idx}: {f1:.4f}")

        # Save confusion matrix to separate file
        conf_matrix_path = output_path.parent / f"{output_path.stem}_confusion_matrix.csv"
        conf_matrix_df = pd.DataFrame(
            conf_matrix,
            index=[f"True_{i}" for i in range(len(conf_matrix))],
            columns=[f"Pred_{i}" for i in range(len(conf_matrix))]
        )
        conf_matrix_df.to_csv(conf_matrix_path)
        print(f"\n  ✓ Confusion matrix saved to: {conf_matrix_path}")

        # Return statistics
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': conf_matrix.tolist(),
            'num_samples': len(df),
            'num_correct': int(df['Correct'].sum())
        }
    else:
        # Regression metrics
        target_mean = df['Target'].mean()
        target_std = df['Target'].std()
        mae = df['Absolute_Error'].mean()
        rmse = (df['Error'] ** 2).mean() ** 0.5

        # Calculate Skill Score: 1 - (MSE_model / MSE_baseline)
        # where baseline predicts the training mean for all samples
        # A score of 1 means perfect predictions, 0 means same as baseline, <0 means worse than baseline
        if train_target_mean is not None:
            mse_model = (df['Error'] ** 2).mean()
            baseline_errors = df['Target'] - train_target_mean
            mse_baseline = (baseline_errors ** 2).mean()
            skill_score = 1 - (mse_model / mse_baseline) if mse_baseline > 0 else 0.0

            print(f"  Target Mean: {target_mean:.4f}")
            print(f"  Target Std:  {target_std:.4f}")
            print(f"  MAE:  {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  Skill Score: {skill_score:.4f} (baseline: always predict {train_target_mean:.4f})")

            # Return statistics
            return {
                'target_mean': target_mean,
                'target_std': target_std,
                'mae': mae,
                'rmse': rmse,
                'skill_score': skill_score,
                'num_samples': len(df)
            }
        else:
            # Fallback: Calculate R² score if training mean not provided
            # R² = 1 - (SS_res / SS_tot)
            # where SS_res = sum of squared residuals, SS_tot = total sum of squares
            ss_res = (df['Error'] ** 2).sum()
            ss_tot = ((df['Target'] - target_mean) ** 2).sum()
            r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

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
    resume_checkpoint: str = None,
    is_cnn_v2: bool = False,
    lat: Optional[torch.Tensor] = None,
    lon: Optional[torch.Tensor] = None,
    lat_lon_indices: Optional[list] = None
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
        is_cnn_v2: Whether using CNN V2 architecture (requires lat/lon)
        lat: Latitude tensor for CNN V2 (H, W)
        lon: Longitude tensor for CNN V2 (H, W)
        lat_lon_indices: List of channel indices for lat/lon to remove from input
    """
    # Get training configuration
    num_epochs = config['training']['epochs']

    # Get Gaussian noise configuration for data augmentation
    augmentation_config = config.get('data', {}).get('augmentation', {})
    gaussian_noise_config = augmentation_config.get('gaussian_noise', {})
    gaussian_noise_enabled = gaussian_noise_config.get('enabled', False)
    gaussian_noise_std = gaussian_noise_config.get('std', 0.01) if gaussian_noise_enabled else 0.0

    if gaussian_noise_enabled:
        print(f"Gaussian noise augmentation enabled: std={gaussian_noise_std}")
    else:
        print("Gaussian noise augmentation disabled")

    # Create optimizer and loss function
    optimizer = get_optimizer(model, config)
    loss_fn = get_loss_function(config)

    # Get validation metric from config (default to 'loss')
    # This needs to be done BEFORE creating the scheduler
    val_metric = config['training'].get('val_metric', 'loss')

    # Get metric mode from config, or auto-determine based on metric type
    val_metric_mode = config['training'].get('val_metric_mode', None)

    if val_metric_mode:
        # Use user-specified mode
        metric_mode = val_metric_mode
    else:
        # Auto-determine mode based on metric name
        # Higher is better for: r2, skill_score, f1_score, accuracy
        # Lower is better for: loss, mae, rmse, mse
        if val_metric in ['r2', 'skill_score', 'f1_score', 'accuracy']:
            metric_mode = 'max'
        else:
            metric_mode = 'min'

    # Create learning rate scheduler
    # Note: For OneCycleLR, we need to pass num_epochs and steps_per_epoch
    # Pass metric_mode so ReduceLROnPlateau uses the same mode as validation metric
    scheduler = get_scheduler(
        optimizer,
        config,
        num_epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        metric_mode=metric_mode
    )

    # Create callbacks
    model_name = config['model']['name']

    print(f"Validation metric: {val_metric} (mode: {metric_mode})")
    print(f"Monitor metric name: val_{val_metric}")
    print(f"Checkpoint will save when {val_metric} {'increases' if metric_mode == 'max' else 'decreases'}")

    # Build monitor metric name for validation
    monitor_metric = f'val_{val_metric}'

    checkpoint_callback = ModelCheckpoint(
        checkpoint_dir='checkpoints',
        monitor=monitor_metric,
        mode=metric_mode,
        save_best_only=config['training']['checkpoint']['save_best_only'],
        verbose=True,
        model_name=model_name
    )

    # Verify the checkpoint callback is configured correctly
    print(f"Checkpoint callback configured:")
    print(f"  - Monitoring: {checkpoint_callback.monitor}")
    print(f"  - Mode: {checkpoint_callback.mode}")
    print(f"  - Best metric initialized to: {checkpoint_callback.best_metric}")

    early_stopping = EarlyStopping(
        monitor=monitor_metric,
        patience=config['training']['early_stopping']['patience'],
        mode=metric_mode,
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

    # Compute training target mean for skill score calculation (regression only)
    train_target_mean = None
    num_classes = config.get('model', {}).get('num_classes', 1)
    is_classification = num_classes > 1

    if not is_classification:
        # Compute mean of training targets efficiently from CSV file
        # This is much faster than iterating through the entire DataLoader
        print("\nComputing training target mean for skill score calculation...")

        target_file = config['data']['target_file']
        train_years_spec = config['data']['train_years']
        train_years = parse_year_spec(train_years_spec)

        # Read target CSV and filter for training years
        targets_df = pd.read_csv(target_file)
        train_targets_df = targets_df[targets_df['Year'].isin(train_years)]

        if len(train_targets_df) == 0:
            print(f"WARNING: No training samples found in target file for specified years")
            print(f"Available years in CSV: {sorted(targets_df['Year'].unique())}")
            print("Falling back to using entire dataset mean for skill score calculation")
            train_targets_df = targets_df

        # Get the target column - use DateRelJun01 by default, or check model config
        target_idx = config.get('model', {}).get('target', None)

        if isinstance(target_idx, int) and target_idx < len(targets_df.columns):
            # Use the specified column index (1-indexed in config maps to 0-indexed in DataFrame)
            target_col = targets_df.columns[target_idx]
        elif 'DateRelJun01' in train_targets_df.columns:
            target_col = 'DateRelJun01'
        else:
            # Fallback to second column
            target_col = train_targets_df.columns[1]

        train_target_mean = train_targets_df[target_col].mean()
        print(f"Training target mean: {train_target_mean:.4f} (from {len(train_targets_df)} samples in column '{target_col}')")

    # Training loop
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    print(f"Monitoring validation metric: {monitor_metric} (mode: {metric_mode})")
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
            scheduler=scheduler if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR) else None,
            gaussian_noise_std=gaussian_noise_std,
            is_cnn_v2=is_cnn_v2,
            lat=lat,
            lon=lon,
            lat_lon_indices=lat_lon_indices
        )

        # Validate
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            train_target_mean=train_target_mean,
            is_cnn_v2=is_cnn_v2,
            lat=lat,
            lon=lon,
            lat_lon_indices=lat_lon_indices
        )

        # Print metrics
        print(f"\nEpoch [{epoch + 1}/{num_epochs}] Results:")

        # Check if classification or regression
        if 'f1_score' in train_metrics:
            # Classification: print F1 scores
            print(f"  Train Loss: {train_metrics['loss']:.6f}  |  Train F1: {train_metrics['f1_score']:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.6f}  |  Val F1:   {val_metrics['f1_score']:.4f}")
        else:
            # Regression: print multiple metrics
            print(f"  Train Loss: {train_metrics['loss']:.6f}  |  Train RMSE: {train_metrics['rmse']:.6f}")
            print(f"  Val Loss:   {val_metrics['loss']:.6f}  |  Val RMSE:   {val_metrics['rmse']:.6f}")

            # Print skill score or R2 depending on what's available
            if 'skill_score' in val_metrics:
                print(f"  Val MAE:    {val_metrics['mae']:.6f}  |  Val Skill:  {val_metrics['skill_score']:.6f}")
            else:
                print(f"  Val MAE:    {val_metrics['mae']:.6f}  |  Val R2:     {val_metrics.get('r2', float('nan')):.6f}")

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning Rate: {current_lr:.2e}")

        # Combine metrics for callbacks
        all_metrics = {
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss']
        }

        # Add task-specific metrics
        if 'f1_score' in train_metrics:
            all_metrics['train_f1_score'] = train_metrics['f1_score']
            all_metrics['val_f1_score'] = val_metrics['f1_score']
        else:
            all_metrics['train_rmse'] = train_metrics['rmse']
            all_metrics['val_rmse'] = val_metrics['rmse']
            all_metrics['val_mae'] = val_metrics['mae']
            # Add skill_score or r2 depending on which is available
            if 'skill_score' in val_metrics:
                all_metrics['val_skill_score'] = val_metrics['skill_score']
            elif 'r2' in val_metrics:
                all_metrics['val_r2'] = val_metrics['r2']

        # Ensure the monitored metric is available in all_metrics
        # This handles cases where val_metric might use different naming conventions
        if monitor_metric not in all_metrics and val_metric in val_metrics:
            all_metrics[monitor_metric] = val_metrics[val_metric]

        # Step the learning rate scheduler (if enabled)
        if scheduler is not None:
            # Get the metric value to use for scheduler (use configured val_metric)
            scheduler_metric_value = val_metrics.get(val_metric, val_metrics['loss'])

            # Check if using custom scheduler with restore capability
            if isinstance(scheduler, ReduceLROnPlateauWithRestore):
                # Custom scheduler needs model to restore state when LR is reduced
                restored = scheduler.step(scheduler_metric_value, model, epoch=epoch)
                if restored:
                    print(f"  ⚠ Model restored to best epoch - continuing training from there")
                    # Reset early stopping counter to give model a fresh chance with new LR
                    if early_stopping:
                        early_stopping.reset()
                        print(f"  ✓ Early stopping counter reset")
            elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Standard PyTorch ReduceLROnPlateau (metric-based)
                scheduler.step(scheduler_metric_value)
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
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
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
    num_classes = config.get('model', {}).get('num_classes', 1)
    is_classification = num_classes > 1

    project_root = src_dir.parent
    results_dir = project_root / 'results' / 'tables'

    # Save predictions for train, val, and test sets and collect statistics
    train_stats = save_predictions(
        model=model,
        data_loader=train_loader,
        device=device,
        split_name='train',
        output_path=results_dir / f'{model_name}_train_predictions.csv',
        is_classification=is_classification,
        num_classes=num_classes,
        is_cnn_v2=is_cnn_v2,
        lat=lat,
        lon=lon,
        lat_lon_indices=lat_lon_indices
    )

    # For regression, get training target mean for skill score calculation
    train_target_mean = train_stats.get('target_mean', None) if not is_classification else None

    val_stats = save_predictions(
        model=model,
        data_loader=val_loader,
        device=device,
        split_name='val',
        output_path=results_dir / f'{model_name}_val_predictions.csv',
        is_classification=is_classification,
        num_classes=num_classes,
        train_target_mean=train_target_mean,
        is_cnn_v2=is_cnn_v2,
        lat=lat,
        lon=lon,
        lat_lon_indices=lat_lon_indices
    )

    test_stats = save_predictions(
        model=model,
        data_loader=test_loader,
        device=device,
        split_name='test',
        output_path=results_dir / f'{model_name}_test_predictions.csv',
        is_classification=is_classification,
        num_classes=num_classes,
        train_target_mean=train_target_mean,
        is_cnn_v2=is_cnn_v2,
        lat=lat,
        lon=lon,
        lat_lon_indices=lat_lon_indices
    )

    # Create summary statistics file based on task type
    if is_classification:
        summary_df = pd.DataFrame({
            'Split': ['Train', 'Validation', 'Test'],
            'Num_Samples': [train_stats['num_samples'], val_stats['num_samples'], test_stats['num_samples']],
            'Num_Correct': [train_stats['num_correct'], val_stats['num_correct'], test_stats['num_correct']],
            'Accuracy': [train_stats['accuracy'], val_stats['accuracy'], test_stats['accuracy']],
            'F1_Macro': [train_stats['f1_macro'], val_stats['f1_macro'], test_stats['f1_macro']]
        })

        # Save per-class F1 scores to a separate file
        # Use num_classes from config (already defined above at line 817)
        per_class_f1_df = pd.DataFrame({
            'Class': [f'Class_{i}' for i in range(num_classes)],
            'Train_F1': train_stats['f1_per_class'],
            'Val_F1': val_stats['f1_per_class'],
            'Test_F1': test_stats['f1_per_class']
        })
        per_class_f1_path = results_dir / f'{model_name}_per_class_f1.csv'
        per_class_f1_df.to_csv(per_class_f1_path, index=False)
        print(f"✓ Saved per-class F1 scores to: {per_class_f1_path}")
    else:
        # Use skill_score if available, otherwise fall back to r2
        metric_name = 'Skill_Score' if 'skill_score' in val_stats else 'R2'
        metric_key = 'skill_score' if 'skill_score' in val_stats else 'r2'

        summary_df = pd.DataFrame({
            'Split': ['Train', 'Validation', 'Test'],
            'Num_Samples': [train_stats['num_samples'], val_stats['num_samples'], test_stats['num_samples']],
            'Target_Mean': [train_stats['target_mean'], val_stats['target_mean'], test_stats['target_mean']],
            'Target_Std': [train_stats['target_std'], val_stats['target_std'], test_stats['target_std']],
            'MAE': [train_stats['mae'], val_stats['mae'], test_stats['mae']],
            'RMSE': [train_stats['rmse'], val_stats['rmse'], test_stats['rmse']],
            metric_name: [train_stats.get(metric_key, float('nan')),
                         val_stats.get(metric_key, float('nan')),
                         test_stats.get(metric_key, float('nan'))]
        })

    summary_path = results_dir / f'{model_name}_metrics_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Saved metrics summary to: {summary_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("Metrics Summary")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    # For classification, also print per-class F1 scores
    if is_classification:
        print("\n" + "=" * 80)
        print("Per-Class F1 Scores")
        print("=" * 80)
        print(per_class_f1_df.to_string(index=False))

    # Log final summary metrics to W&B
    if wandb_logger:
        # Log summary metrics based on task type
        if is_classification:
            # Build summary dict with accuracy and F1 scores
            summary_dict = {
                'final/train_accuracy': train_stats['accuracy'],
                'final/val_accuracy': val_stats['accuracy'],
                'final/test_accuracy': test_stats['accuracy'],
                'final/train_f1_macro': train_stats['f1_macro'],
                'final/val_f1_macro': val_stats['f1_macro'],
                'final/test_f1_macro': test_stats['f1_macro']
            }

            # Add per-class F1 scores
            for class_idx in range(len(train_stats['f1_per_class'])):
                summary_dict[f'final/train_f1_class_{class_idx}'] = train_stats['f1_per_class'][class_idx]
                summary_dict[f'final/val_f1_class_{class_idx}'] = val_stats['f1_per_class'][class_idx]
                summary_dict[f'final/test_f1_class_{class_idx}'] = test_stats['f1_per_class'][class_idx]

            wandb_logger.log_summary(summary_dict)
        else:
            # Build summary dict for regression with skill_score or r2
            summary_dict = {
                'final/train_mae': train_stats['mae'],
                'final/train_rmse': train_stats['rmse'],
                'final/val_mae': val_stats['mae'],
                'final/val_rmse': val_stats['rmse'],
                'final/test_mae': test_stats['mae'],
                'final/test_rmse': test_stats['rmse']
            }

            # Add skill_score or r2 depending on which is available
            if 'skill_score' in val_stats:
                summary_dict.update({
                    'final/train_skill_score': train_stats.get('skill_score', float('nan')),
                    'final/val_skill_score': val_stats['skill_score'],
                    'final/test_skill_score': test_stats['skill_score']
                })
            elif 'r2' in val_stats:
                summary_dict.update({
                    'final/train_r2': train_stats['r2'],
                    'final/val_r2': val_stats['r2'],
                    'final/test_r2': test_stats['r2']
                })

            wandb_logger.log_summary(summary_dict)

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
    add_per_channel_norm = data_config.get('add_per_channel_norm', False)  # Per-channel min-max normalization

    print("\n" + "=" * 80)
    print("Normalization Strategy")
    print("=" * 80)
    print(f"normalize_strategy: {normalize_strategy}")
    print(f"add_per_channel_norm: {add_per_channel_norm}")

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
        print("✓ Normalization transform created (precomputed stats)")

        # Add per-channel min-max normalization if requested
        if add_per_channel_norm:
            from data_pipeline.preprocessing.transformers import PerChannelMinMaxNormalize, Compose
            per_channel_transform = PerChannelMinMaxNormalize(
                static_channel_indices=norm_stats.static_channel_indices
            )
            normalize_transform = Compose([normalize_transform, per_channel_transform])
            print("✓ Added per-channel min-max normalization [0, 1]")

    elif normalize_strategy == 0:
        # Strategy 0: No normalization (or only per-channel min-max)
        if add_per_channel_norm:
            from data_pipeline.preprocessing.transformers import PerChannelMinMaxNormalize
            # Determine static channel indices based on dataset configuration
            # We'll get this from the dataset metadata
            print("Strategy: Per-channel min-max normalization only (no precomputed stats)")
            # Will set static_channel_indices after creating datasets
            normalize_transform = None  # Will be created after we know the channel info
        else:
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

    # Get dataset type
    dataset_type = data_config.get('dataset_type', 'preprocessed')

    # Create datasets based on type
    if dataset_type == 'raw':
        # Raw 6-hourly data with temporal aggregation
        print(f"Using raw ERA5 dataset with temporal_aggregation: {data_config.get('temporal_aggregation', 'daily')}")

        # Get source directory mappings
        input_geo_var_surf_src = data_config.get('input_geo_var_surf_src', None)
        input_geo_var_press_src = data_config.get('input_geo_var_press_src', None)
        temporal_aggregation = data_config.get('temporal_aggregation', 'daily')

        train_dataset = ERA5RawDataset(
            base_dir=data_config.get('data_dir', '/gdata2/ERA5'),
            years=train_years,
            time_steps=time_steps,
            temporal_aggregation=temporal_aggregation,
            pressure_levels=data_config.get('pressure_levels', [0, 1]),
            target_file=data_config.get('target_file', None),
            transform=normalize_transform,
            input_geo_var_surf=input_geo_var_surf,
            input_geo_var_press=input_geo_var_press,
            input_geo_var_surf_src=input_geo_var_surf_src,
            input_geo_var_press_src=input_geo_var_press_src,
            include_lat=include_lat,
            include_lon=include_lon,
            include_landsea=include_landsea
        )

        val_dataset = ERA5RawDataset(
            base_dir=data_config.get('data_dir', '/gdata2/ERA5'),
            years=val_years,
            time_steps=time_steps,
            temporal_aggregation=temporal_aggregation,
            pressure_levels=data_config.get('pressure_levels', [0, 1]),
            target_file=data_config.get('target_file', None),
            transform=normalize_transform,
            input_geo_var_surf=input_geo_var_surf,
            input_geo_var_press=input_geo_var_press,
            input_geo_var_surf_src=input_geo_var_surf_src,
            input_geo_var_press_src=input_geo_var_press_src,
            include_lat=include_lat,
            include_lon=include_lon,
            include_landsea=include_landsea
        )

        test_dataset = ERA5RawDataset(
            base_dir=data_config.get('data_dir', '/gdata2/ERA5'),
            years=test_years,
            time_steps=time_steps,
            temporal_aggregation=temporal_aggregation,
            pressure_levels=data_config.get('pressure_levels', [0, 1]),
            target_file=data_config.get('target_file', None),
            transform=normalize_transform,
            input_geo_var_surf=input_geo_var_surf,
            input_geo_var_press=input_geo_var_press,
            input_geo_var_surf_src=input_geo_var_surf_src,
            input_geo_var_press_src=input_geo_var_press_src,
            include_lat=include_lat,
            include_lon=include_lon,
            include_landsea=include_landsea
        )

    else:  # dataset_type == 'preprocessed' (default)
        # Preprocessed monthly or weekly data
        print("Using preprocessed ERA5 dataset (monthly or weekly)")

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

    # Handle per-channel normalization for strategy 0
    if normalize_strategy == 0 and add_per_channel_norm:
        from data_pipeline.preprocessing.transformers import PerChannelMinMaxNormalize
        # Get static channel indices from dataset
        metadata = train_dataset.get_metadata(0)
        channel_names = metadata['channel_names']
        static_channel_indices = []
        for i, name in enumerate(channel_names):
            if name in ['lat', 'lon', 'land_sea_mask']:
                static_channel_indices.append(i)

        # Create per-channel min-max transform
        normalize_transform = PerChannelMinMaxNormalize(
            static_channel_indices=static_channel_indices
        )

        # Update datasets with the transform
        train_dataset.transform = normalize_transform
        val_dataset.transform = normalize_transform
        test_dataset.transform = normalize_transform
        print(f"✓ Per-channel min-max normalization applied (excluding {len(static_channel_indices)} static channels)")

    normalization_status = "with normalization" if normalize_strategy == 1 or (normalize_strategy == 0 and add_per_channel_norm) else "without normalization"
    print(f"✓ Dataloaders created {normalization_status}")
    print(f"  Train: {len(train_loader)} batches, {len(train_dataset)} samples")
    print(f"  Val:   {len(val_loader)} batches, {len(val_dataset)} samples")
    print(f"  Test:  {len(test_loader)} batches, {len(test_dataset)} samples")

    # Get channel information from dataset to determine model input shape
    channel_info = train_dataset.get_channel_info()
    in_channels = channel_info['num_channels']
    print(f"\nDataset configuration:")
    print(f"  Total channels: {in_channels}")
    # Some datasets may not have time_steps/pressure_levels in channel_info
    if 'time_steps' in channel_info:
        print(f"  Time steps: {channel_info['time_steps']}")
    else:
        print(f"  Time steps: {time_steps}")
    if 'pressure_levels' in channel_info:
        print(f"  Pressure levels: {channel_info['pressure_levels']}")
    else:
        print(f"  Pressure levels: {data_config.get('pressure_levels', [0, 1])}")

    # Create model
    print("\n" + "=" * 80)
    print("Creating Model")
    print("=" * 80)

    # Get dropout_rate from config (default to 0.5 if not specified)
    dropout_rate = config.get('model', {}).get('dropout_rate', 0.5)
    print(f"Dropout rate: {dropout_rate}")

    # Get num_classes from config (default to 1 for regression)
    num_classes = config.get('model', {}).get('num_classes', 1)
    if num_classes > 1:
        print(f"Classification mode: {num_classes} classes")
    else:
        print(f"Regression mode: single output")

    # Get activation configuration
    activation_config = config.get('model', {}).get('activation', {})
    activation_type = activation_config.get('type', 'relu')
    prelu_mode = activation_config.get('prelu_mode', 'shared')
    leaky_relu_slope = activation_config.get('leaky_relu_slope', 0.01)

    print(f"Activation function: {activation_type}")
    if activation_type == 'prelu':
        print(f"  PReLU mode: {prelu_mode}")
    elif activation_type == 'leaky_relu':
        print(f"  LeakyReLU slope: {leaky_relu_slope}")

    # Get L2 regularization strength from config
    l2_reg = config.get('training', {}).get('weight_decay', 0.0001)
    print(f"L2 regularization: {l2_reg}")

    # Get SPP configuration
    spp_option = config.get('model', {}).get('spp_option', 2)
    spp_ops = config.get('model', {}).get('spp_ops', [1])
    print(f"SPP pooling type: {['max only', 'avg only', 'max and avg'][spp_option]}")
    print(f"SPP pyramid levels: {spp_ops}")

    # Get normalization configuration
    norm_type = config.get('model', {}).get('cnn_norm', 0)
    norm_num_groups = config.get('model', {}).get('cnn_norm_num_groups', 32)
    norm_names = {0: "BatchNorm", 1: "LayerNorm", 2: "InstanceNorm", 3: "GroupNorm"}
    if norm_type == 3:
        print(f"Normalization type: {norm_type} ({norm_names.get(norm_type, 'Unknown')}, num_groups={norm_num_groups})")
    else:
        print(f"Normalization type: {norm_type} ({norm_names.get(norm_type, 'Unknown')})")

    # Get architecture type
    architecture = config.get('model', {}).get('architecture', 'cnn').lower()
    print(f"Architecture: {architecture}")

    # Determine if using CNN V2
    is_cnn_v2 = architecture == 'cnn_v2'

    # For CNN V2, adjust in_channels to exclude lat/lon
    model_in_channels = in_channels
    if is_cnn_v2:
        # CNN V2 uses lat/lon for positional embeddings, not as input channels
        # Subtract lat and lon channels if they were included
        if include_lat:
            model_in_channels -= 1
        if include_lon:
            model_in_channels -= 1
        print(f"  CNN V2: Using lat/lon for positional embeddings (not as input channels)")
        print(f"  Model input channels: {model_in_channels} (excluding lat/lon)")

    # Create model based on architecture
    if is_cnn_v2:
        # CNN V2 with positional embeddings
        pos_embedding_dim = config.get('model', {}).get('pos_embedding_dim', 16)
        num_frequencies = config.get('model', {}).get('num_frequencies', 16)
        print(f"  Positional embedding dim: {pos_embedding_dim}")
        print(f"  Num frequencies: {num_frequencies}")

        model = create_model_v2(
            in_channels=model_in_channels,
            dropout_rate=dropout_rate,
            num_classes=num_classes,
            activation_type=activation_type,
            prelu_mode=prelu_mode,
            leaky_relu_slope=leaky_relu_slope,
            l2_reg=l2_reg,
            spp_option=spp_option,
            spp_ops=spp_ops,
            norm_type=norm_type,
            norm_num_groups=norm_num_groups,
            pos_embedding_dim=pos_embedding_dim,
            num_frequencies=num_frequencies
        )
    elif architecture == 'cnn_v1':
        # Old CNN V1 (simplified architecture)
        print("  Using old CNN V1 model (simplified architecture)")
        model = create_model_cnn_v1(
            in_channels=in_channels,
            dropout_rate=dropout_rate,
            num_classes=num_classes,
            activation_type=activation_type,
            prelu_mode=prelu_mode,
            leaky_relu_slope=leaky_relu_slope,
            l2_reg=l2_reg,
            spp_option=spp_option,
            spp_ops=spp_ops,
            norm_type=norm_type,
            norm_num_groups=norm_num_groups
        )
    else:
        # CNN (current/default architecture)
        print("  Using current CNN model (default)")
        model = create_model_cnn(
            in_channels=in_channels,
            dropout_rate=dropout_rate,
            num_classes=num_classes,
            activation_type=activation_type,
            prelu_mode=prelu_mode,
            leaky_relu_slope=leaky_relu_slope,
            l2_reg=l2_reg,
            spp_option=spp_option,
            spp_ops=spp_ops,
            norm_type=norm_type,
            norm_num_groups=norm_num_groups
        )

    model = model.to(device)

    # Get spatial dimensions from first batch
    sample_data, _ = train_dataset[0]
    _, spatial_h, spatial_w = sample_data.shape

    # Extract lat/lon for CNN V2
    lat_tensor = None
    lon_tensor = None
    lat_lon_indices = None
    if is_cnn_v2:
        print("Extracting lat/lon from dataset for CNN V2...")
        lat_tensor, lon_tensor, lat_lon_indices = extract_lat_lon_from_dataset(train_dataset)
        if lat_tensor is None or lon_tensor is None:
            raise ValueError(
                "CNN V2 requires lat and lon to be loaded from the dataset. "
                "Set include_lat=true and include_lon=true in the config. "
                "The lat/lon grids will be extracted for positional embeddings and "
                "automatically removed from the input channels."
            )
        print(f"  Lat shape: {lat_tensor.shape}")
        print(f"  Lon shape: {lon_tensor.shape}")
        print(f"  Lat/Lon channel indices to remove from input: {lat_lon_indices}")

    # Initialize lazy modules with a dummy forward pass
    print("Initializing lazy modules...")
    if is_cnn_v2:
        # CNN V2 requires lat and lon for forward pass
        dummy_input = torch.randn(1, model_in_channels, spatial_h, spatial_w).to(device)
        dummy_lat = lat_tensor.unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dimensions -> (1, 1, H, W)
        dummy_lon = lon_tensor.unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model(dummy_input, dummy_lat, dummy_lon)
    else:
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
        resume_checkpoint=args.resume,
        is_cnn_v2=is_cnn_v2,
        lat=lat_tensor,
        lon=lon_tensor,
        lat_lon_indices=lat_lon_indices
    )


if __name__ == "__main__":
    main()

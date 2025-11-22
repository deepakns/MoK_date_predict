# Learning Rate Scheduler Implementation

## Overview
A learning rate scheduler has been implemented to automatically reduce the learning rate when validation loss plateaus. This helps the model fine-tune its weights when training stagnates.

## Implementation Details

### Scheduler Type: ReduceLROnPlateau
- **Purpose**: Reduces learning rate when a monitored metric (validation loss) stops improving
- **Behavior**: When val_loss doesn't improve for a specified number of epochs (patience), the learning rate is reduced by a multiplication factor

### Configuration
The scheduler is configured in [config/model_config.yml](config/model_config.yml):

```yaml
lr_scheduler:
  enabled: true
  type: "ReduceLROnPlateau"
  # Note: Mode is automatically set to match val_metric_mode
  factor: 0.5              # Multiply LR by this factor when reducing
  patience: 5              # Number of epochs with no improvement before reducing LR
  min_lr: 1.0e-6           # Minimum learning rate
  threshold: 1.0e-4        # Threshold for measuring improvement
```

**Important**: The scheduler's mode (min/max) is automatically determined from the `val_metric_mode` configuration. This ensures the scheduler and checkpoint callback always use the same mode for metric tracking.

### Key Parameters

| Parameter | Default Value | Description |
|-----------|--------------|-------------|
| `enabled` | `true` | Enable/disable the scheduler |
| `type` | `"ReduceLROnPlateau"` | Type of scheduler to use |
| `factor` | `0.5` | Multiplicative factor for LR reduction (new_lr = lr × factor) |
| `patience` | `5` | Number of epochs to wait before reducing LR |
| `min_lr` | `1.0e-6` | Minimum learning rate threshold |
| `threshold` | `1.0e-4` | Minimum change in metric to qualify as improvement |

**Note**: The `mode` parameter is no longer configurable in the scheduler section. It is automatically inherited from `training.val_metric_mode` to ensure consistency between scheduler and checkpoint callback.

## How It Works

1. **Monitor**: The scheduler monitors validation loss after each epoch
2. **Patience**: If val_loss doesn't improve for 5 consecutive epochs, the LR is reduced
3. **Reduction**: Learning rate is multiplied by the factor (e.g., 0.001 → 0.0005)
4. **Minimum**: LR will never go below the specified minimum (1e-6)

### Example Timeline
```
Epoch 1-10: LR = 0.001, val_loss improving
Epoch 11-15: LR = 0.001, val_loss plateaus (no improvement)
Epoch 16: LR reduced to 0.0005 (after 5 epochs of no improvement)
Epoch 17-25: LR = 0.0005, val_loss improving again
Epoch 26-30: LR = 0.0005, val_loss plateaus
Epoch 31: LR reduced to 0.00025
```

## Code Changes

### 1. Scheduler Creation Function
Added `get_scheduler()` function in [train_model.py:126-168](src/training/scripts/train_model.py#L126-L168):
```python
def get_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    """Create learning rate scheduler from configuration."""
    scheduler_config = config['training'].get('lr_scheduler', None)

    if scheduler_config is None or not scheduler_config.get('enabled', False):
        return None

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_config.get('mode', 'min'),
        factor=scheduler_config.get('factor', 0.5),
        patience=scheduler_config.get('patience', 5),
        min_lr=scheduler_config.get('min_lr', 1e-6),
        threshold=scheduler_config.get('threshold', 1e-4),
        verbose=True
    )

    return scheduler
```

### 2. Training Loop Integration
The scheduler is stepped after each validation phase in [train_model.py:516-519](src/training/scripts/train_model.py#L516-L519):
```python
# Step the learning rate scheduler (if enabled)
if scheduler is not None:
    scheduler.step(val_metrics['loss'])
```

### 3. Learning Rate Logging
Current learning rate is printed after each epoch in [train_model.py:504-506](src/training/scripts/train_model.py#L504-L506):
```python
# Get current learning rate
current_lr = optimizer.param_groups[0]['lr']
print(f"  Learning Rate: {current_lr:.2e}")
```

### 4. Checkpoint Integration
The scheduler state is saved in checkpoints and restored when resuming training:
- Save: [train_model.py:532-538](src/training/scripts/train_model.py#L532-L538)
- Load: [train_model.py:464-469](src/training/scripts/train_model.py#L464-L469)

## Usage

### Training with Scheduler (Default)
```bash
python train_model.py --config config/model_config.yml
```

### Disabling the Scheduler
To disable the scheduler, set `enabled: false` in the config file:
```yaml
lr_scheduler:
  enabled: false
```

### Adjusting Scheduler Behavior
You can tune the scheduler by modifying these parameters:

**More Aggressive LR Reduction:**
```yaml
lr_scheduler:
  factor: 0.2        # Reduce LR more aggressively (80% reduction)
  patience: 3        # Reduce sooner (after 3 epochs)
```

**More Conservative:**
```yaml
lr_scheduler:
  factor: 0.8        # Reduce LR less aggressively (20% reduction)
  patience: 10       # Wait longer before reducing
```

## Benefits

1. **Automatic Fine-tuning**: No need to manually adjust learning rates during training
2. **Better Convergence**: Helps escape local minima and achieve better final performance
3. **Training Stability**: Prevents divergence when approaching optimal weights
4. **Resume-friendly**: Scheduler state is saved in checkpoints for seamless resumption

## Monitoring

When the scheduler reduces the learning rate, you'll see messages like:
```
Epoch 00016: reducing learning rate of group 0 to 5.0000e-04.
```

The current learning rate is also displayed after each epoch:
```
Epoch [16/100] Results:
  Train Loss: 12.345678  |  Train RMSE: 3.514321
  Val Loss:   13.456789  |  Val RMSE:   3.668765
  Learning Rate: 5.00e-04
```

## Notes

- The scheduler only affects the learning rate, not other training dynamics
- Early stopping (if enabled) works independently and may stop training before LR reaches minimum
- The scheduler is compatible with all optimizers (Adam, AdamW, SGD)
- Scheduler state is included in checkpoints for proper resumption

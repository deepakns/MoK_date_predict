# Weights & Biases (wandb) Integration Guide

## Overview

Weights & Biases (wandb) is a powerful experiment tracking and visualization platform that has been integrated into the training pipeline. It works alongside TensorBoard to provide cloud-based experiment tracking, model versioning, and collaboration features.

## Key Features

- **Cloud-based tracking**: Access your experiments from anywhere
- **Experiment comparison**: Compare multiple runs side-by-side
- **Model versioning**: Automatic model checkpoint management
- **Hyperparameter tracking**: Log all configuration parameters
- **Collaborative**: Share results with team members
- **Rich visualizations**: Interactive plots and tables
- **System monitoring**: Automatic GPU/CPU/memory tracking

## Getting Started

### 1. Create a W&B Account

If you don't have one already:
1. Go to https://wandb.ai
2. Sign up for a free account
3. Get your API key from https://wandb.ai/authorize

### 2. Login to W&B

First time setup:
```bash
cd /home/deepakns/Work/MoK_date_predict
source ~/miniconda3/etc/profile.d/conda.sh
conda activate MoK_date_predict
wandb login
```

When prompted, paste your API key. This only needs to be done once per machine.

### 3. Enable W&B in Configuration

Edit [config/model_config.yml](../config/model_config.yml):

```yaml
logging:
  tensorboard: true  # Keep TensorBoard enabled

  wandb:
    enabled: true  # Enable W&B logging
    project: "MoK_date_predict"  # Your project name
    entity: null  # Your username or team (optional)
    name: null  # Auto-generate from model name
    tags: ["experiment1", "baseline"]  # Add tags for organization
    notes: "Testing new learning rate scheduler"  # Experiment notes
    log_model: true  # Save model checkpoints to W&B
    log_predictions: false  # Log prediction tables (can be large)
    watch_model: false  # Track gradients (can slow training)
```

### 4. Run Training

```bash
python src/training/scripts/train_model.py --config config/model_config.yml
```

You'll see output like:
```
Weights & Biases logging initialized
  Project: MoK_date_predict
  Run name: MoK_CNN_11
  Run URL: https://wandb.ai/username/MoK_date_predict/runs/xyz123
```

Click the URL to view your experiment in real-time!

## Configuration Options

### Basic Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Enable/disable W&B logging |
| `project` | str | `"MoK_date_predict"` | W&B project name |
| `entity` | str | `null` | Username or team name (optional) |
| `name` | str | `null` | Run name (auto-generated from model name if null) |
| `tags` | list | `[]` | Tags for organizing experiments |
| `notes` | str | `null` | Notes about the experiment |

### Advanced Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_model` | bool | `true` | Upload model checkpoints to W&B |
| `log_predictions` | bool | `false` | Log prediction tables (can be large) |
| `log_gradients` | bool | `false` | Log gradient histograms |
| `watch_model` | bool | `false` | Enable `wandb.watch()` for gradient tracking |
| `watch_log` | str | `"gradients"` | What to watch: `"gradients"`, `"parameters"`, `"all"` |
| `watch_freq` | int | `100` | Watch logging frequency (batches) |

## What Gets Logged

### 1. Training Metrics (Every Epoch)
- `train/loss`: Training loss
- `train/rmse`: Training RMSE
- `val/loss`: Validation loss
- `val/rmse`: Validation RMSE
- `learning_rate`: Current learning rate

### 2. Configuration
All hyperparameters from `model_config.yml`:
- Model architecture settings
- Training parameters (LR, optimizer, loss, etc.)
- Data configuration
- Scheduler settings

### 3. System Metrics (Automatic)
- GPU utilization
- CPU usage
- Memory usage
- Network I/O

### 4. Final Results
- Best validation metrics
- Test set performance (MAE, RMSE, R²)
- Model checkpoints (if enabled)
- Prediction tables (if enabled)

## Using the W&B Dashboard

### 1. View Real-time Training

After starting training, open the run URL. You'll see:

- **Overview**: Summary of the run with key metrics
- **Charts**: Interactive plots of loss, RMSE, learning rate
- **System**: GPU/CPU/memory usage
- **Logs**: Console output from training
- **Files**: Model checkpoints and predictions

### 2. Compare Experiments

To compare multiple runs:

1. Navigate to your project page
2. Select multiple runs using checkboxes
3. Click "Compare" to see side-by-side metrics
4. Create custom charts and tables

### 3. Organize with Tags

Use tags to organize experiments:

```yaml
wandb:
  tags: ["lr-0.001", "dropout-0.2", "with-scheduler"]
```

Then filter by tags in the dashboard.

## Example Configurations

### Minimal Configuration (Default)
```yaml
wandb:
  enabled: true
  project: "MoK_date_predict"
```

### Full Logging (For Debugging)
```yaml
wandb:
  enabled: true
  project: "MoK_date_predict"
  tags: ["debug", "full-logging"]
  notes: "Full logging for debugging training issues"
  log_model: true
  log_predictions: true
  log_gradients: true
  watch_model: true
  watch_log: "all"
  watch_freq: 50
```

**⚠️ Warning**: Full logging can significantly slow down training and use more storage.

### Production Configuration
```yaml
wandb:
  enabled: true
  project: "MoK_date_predict"
  entity: "your-team"
  tags: ["production", "v1.0"]
  notes: "Production run for paper results"
  log_model: true
  log_predictions: false  # Save storage
  log_gradients: false  # Save time
  watch_model: false
```

## Experiment Organization

### Project Structure

Organize your experiments using projects and tags:

```
MoK_date_predict/           # Main project
├── Baseline runs           # Tag: "baseline"
├── Learning rate tests     # Tag: "lr-experiments"
├── Architecture variants   # Tag: "architecture"
└── Final models           # Tag: "production"
```

### Naming Convention

Use descriptive run names:

```yaml
wandb:
  name: "MoK_CNN_11_lr0.001_dropout0.2_scheduler"
```

Or let it auto-generate from model name in config:
```yaml
model:
  name: "MoK_CNN_11_lr0.001_dropout0.2"
```

## Resuming Experiments

To resume a W&B run after interruption, the run ID is automatically handled by the checkpoint system. Just resume training normally:

```bash
python train_model.py --config config/model_config.yml --resume checkpoints/best_model.pth
```

## Best Practices

### 1. Use Tags Liberally
```yaml
tags: ["baseline", "lr-0.001", "batch-10", "no-scheduler"]
```

### 2. Add Meaningful Notes
```yaml
notes: "Testing impact of ReduceLROnPlateau with patience=5"
```

### 3. Keep log_predictions=false for Large Datasets
Prediction tables can be very large. Only enable for small experiments:
```yaml
log_predictions: false  # Recommended for production
```

### 4. Disable watch_model for Fast Training
`wandb.watch()` can slow down training significantly:
```yaml
watch_model: false  # Recommended unless debugging
```

### 5. Use Separate Projects for Different Stages
- Development: `MoK_date_predict-dev`
- Production: `MoK_date_predict`
- Ablation studies: `MoK_date_predict-ablation`

## Integration with TensorBoard

Both TensorBoard and W&B can run simultaneously:

```yaml
logging:
  tensorboard: true   # Local, detailed visualizations
  wandb:
    enabled: true     # Cloud, experiment tracking
```

**When to use each:**
- **TensorBoard**: Detailed local analysis, gradient inspection, architecture graphs
- **W&B**: Experiment comparison, collaboration, cloud access, model registry

## Troubleshooting

### Problem: "wandb: ERROR Error while calling W&B API"

**Solution**: Check your internet connection or login again:
```bash
wandb login --relogin
```

### Problem: "Run already exists"

**Solution**: This shouldn't happen with our implementation, but if it does:
```bash
wandb sync --clean  # Clean up failed syncs
```

### Problem: Training is slow with W&B enabled

**Solutions:**
1. Disable gradient logging:
   ```yaml
   log_gradients: false
   watch_model: false
   ```

2. Reduce watch frequency:
   ```yaml
   watch_freq: 1000  # Higher = less frequent
   ```

3. Disable prediction logging:
   ```yaml
   log_predictions: false
   ```

### Problem: Too much disk usage

**Solution**: W&B caches data locally in `wandb/` directory. Clean up old runs:
```bash
wandb sync --clean
rm -rf wandb/  # Nuclear option: removes all cached data
```

## Advanced Features

### Custom Metrics

The code already logs comprehensive metrics, but you can extend `WandBLogger`:

```python
# In train_model.py or custom script
if wandb_logger:
    wandb_logger.log_summary({
        'custom_metric': value
    })
```

### Sweeps (Hyperparameter Tuning)

W&B supports automated hyperparameter sweeps. Create a sweep config:

```yaml
# sweep_config.yaml
program: src/training/scripts/train_model.py
method: bayes
metric:
  name: val/loss
  goal: minimize
parameters:
  learning_rate:
    min: 0.0001
    max: 0.01
  dropout_rate:
    values: [0.1, 0.2, 0.3, 0.5]
```

Run sweep:
```bash
wandb sweep sweep_config.yaml
wandb agent username/project/sweep_id
```

### Artifacts (Model Registry)

Best models are automatically saved as artifacts when `log_model: true`. Access them:

```python
import wandb

run = wandb.init()
artifact = run.use_artifact('model-MoK_CNN_11:latest')
artifact_dir = artifact.download()
```

## Comparison: W&B vs TensorBoard

| Feature | TensorBoard | Weights & Biases |
|---------|-------------|------------------|
| **Cost** | Free | Free (with limits), paid plans available |
| **Hosting** | Local | Cloud-based |
| **Collaboration** | Manual sharing | Built-in sharing |
| **Experiment Comparison** | Limited | Excellent |
| **Hyperparameter Tracking** | Manual | Automatic |
| **Model Versioning** | Manual | Automatic |
| **Setup Complexity** | Low | Medium (requires account) |
| **Privacy** | Full (local) | Data sent to cloud |
| **Speed** | Fast | Can be slower (network) |

**Recommendation**: Use both! TensorBoard for local development, W&B for experiment tracking.

## Resources

- **W&B Documentation**: https://docs.wandb.ai
- **W&B Python Library**: https://github.com/wandb/wandb
- **Examples**: https://wandb.ai/site/articles
- **Community**: https://community.wandb.ai

## Quick Reference

### Enable W&B
```yaml
logging:
  wandb:
    enabled: true
```

### View Run
Look for the URL in training output:
```
Run URL: https://wandb.ai/username/project/runs/xyz
```

### Disable W&B
```yaml
logging:
  wandb:
    enabled: false
```

### Login
```bash
wandb login
```

### Logout
```bash
wandb logout
```

## Summary

Weights & Biases provides powerful experiment tracking that complements TensorBoard:

✅ **Enabled by default**: No (set `enabled: true` in config)
✅ **Requires account**: Yes (free at https://wandb.ai)
✅ **Setup time**: 5 minutes
✅ **Works with TensorBoard**: Yes, both can run simultaneously
✅ **Recommended for**: Experiment tracking, collaboration, model versioning

Start with W&B disabled and enable it when you need cloud-based experiment tracking!

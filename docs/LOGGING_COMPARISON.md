# Logging Options Comparison

This project supports both **TensorBoard** and **Weights & Biases** for experiment tracking and visualization.

## Quick Comparison

| Feature | TensorBoard | Weights & Biases |
|---------|-------------|------------------|
| **Access** | Local only | Cloud (anywhere) |
| **Setup** | None | Requires account |
| **Cost** | Free | Free (with limits) |
| **Collaboration** | Manual | Built-in |
| **Experiment Comparison** | Basic | Advanced |
| **Model Registry** | ❌ | ✅ |
| **Hyperparameter Tracking** | Manual | Automatic |
| **System Monitoring** | Limited | Comprehensive |
| **Storage** | Local disk | Cloud |
| **Speed** | Fast | Slower (network) |
| **Privacy** | Complete | Data in cloud |

## When to Use Each

### Use TensorBoard When:
- ✅ Working locally and don't need remote access
- ✅ Want fastest performance with no network overhead
- ✅ Need complete privacy (data stays local)
- ✅ Want detailed gradient visualizations
- ✅ Analyzing model architecture graphs
- ✅ Don't want to create external accounts

### Use Weights & Biases When:
- ✅ Comparing multiple experiments
- ✅ Collaborating with team members
- ✅ Need to access results from anywhere
- ✅ Want automatic model versioning
- ✅ Running hyperparameter sweeps
- ✅ Sharing results with stakeholders
- ✅ Building a model registry

### Use Both (Recommended):
- ✅ **TensorBoard** for detailed local analysis during development
- ✅ **W&B** for experiment tracking and production runs

## Configuration

### Enable Both (Recommended)
```yaml
logging:
  tensorboard: true
  wandb:
    enabled: true
    project: "MoK_date_predict"
```

### TensorBoard Only (Default)
```yaml
logging:
  tensorboard: true
  wandb:
    enabled: false
```

### W&B Only
```yaml
logging:
  tensorboard: false
  wandb:
    enabled: true
    project: "MoK_date_predict"
```

## Quick Start

### TensorBoard
```bash
# Start training (already logging to TensorBoard)
python train_model.py --config config/model_config.yml

# View in browser
./view_tensorboard.sh
# Open: http://localhost:6006
```

### Weights & Biases
```bash
# One-time setup
wandb login  # Enter API key from wandb.ai

# Enable in config
# Edit config/model_config.yml: set wandb.enabled: true

# Start training
python train_model.py --config config/model_config.yml

# View in browser (URL printed in console)
# Open: https://wandb.ai/username/project/runs/xyz
```

## What Gets Logged

### Both Systems Log:
- Training loss and RMSE (per epoch)
- Validation loss and RMSE (per epoch)
- Learning rate over time
- Training duration
- Model hyperparameters

### TensorBoard Extras:
- Model architecture graph
- Weight histograms
- Gradient distributions
- Detailed layer analysis

### W&B Extras:
- System metrics (GPU/CPU/memory)
- Experiment comparison tables
- Hyperparameter sweeps
- Model artifacts/checkpoints
- Prediction tables (optional)
- Team collaboration features

## Storage Requirements

### TensorBoard
- **Location**: `logs/` directory
- **Size**: ~10-50 MB per training run
- **Cleanup**: `rm -rf logs/` to clear

### Weights & Biases
- **Local Cache**: `wandb/` directory (~50-100 MB)
- **Cloud Storage**: Included in free tier
- **Cleanup**: `wandb sync --clean`

## Performance Impact

### TensorBoard
- **Overhead**: Minimal (~1-2% slower)
- **Network**: None (local only)
- **Disk I/O**: Low

### W&B (Default Settings)
- **Overhead**: Low (~2-5% slower)
- **Network**: Periodic uploads (minimal impact)
- **Disk I/O**: Low

### W&B (Full Logging)
```yaml
wandb:
  log_gradients: true
  watch_model: true
  log_predictions: true
```
- **Overhead**: Significant (~10-30% slower)
- **Network**: Frequent uploads
- **Disk I/O**: High

**Recommendation**: Keep `log_gradients`, `watch_model`, and `log_predictions` disabled for production runs.

## Common Workflows

### Development Workflow
```yaml
logging:
  tensorboard: true   # Fast local analysis
  wandb:
    enabled: false    # Not needed during development
```

### Experiment Workflow
```yaml
logging:
  tensorboard: true   # Local detailed analysis
  wandb:
    enabled: true     # Track experiments
    log_model: true
    log_predictions: false
```

### Production Workflow
```yaml
logging:
  tensorboard: true
  wandb:
    enabled: true
    log_model: true
    log_predictions: false
    tags: ["production", "v1.0"]
```

### Debugging Workflow
```yaml
logging:
  tensorboard: true
  wandb:
    enabled: true
    log_gradients: true      # For debugging
    watch_model: true        # For debugging
    log_predictions: true    # For debugging
    tags: ["debug"]
```

## Troubleshooting

### TensorBoard Issues
| Issue | Solution |
|-------|----------|
| No data showing | Check `tensorboard: true` in config |
| Port in use | Use: `./view_tensorboard.sh logs 6007` |
| Old data showing | Clear: `rm -rf logs/` or use new log_dir |

### W&B Issues
| Issue | Solution |
|-------|----------|
| API error | Run: `wandb login --relogin` |
| Slow training | Disable: `watch_model`, `log_gradients`, `log_predictions` |
| Disk usage | Run: `wandb sync --clean` |
| Privacy concerns | Use TensorBoard only |

## Documentation

- **TensorBoard Guide**: [docs/TENSORBOARD_GUIDE.md](TENSORBOARD_GUIDE.md)
- **W&B Guide**: [docs/WANDB_GUIDE.md](WANDB_GUIDE.md)
- **Learning Rate Scheduler**: [docs/LEARNING_RATE_SCHEDULER.md](LEARNING_RATE_SCHEDULER.md)

## Summary

**For most users:**
```yaml
logging:
  tensorboard: true   # Always enabled
  wandb:
    enabled: false    # Enable when needed
```

**Start with TensorBoard**, then add W&B when you need:
- Experiment comparison
- Cloud access
- Team collaboration
- Model versioning

Both tools work together seamlessly!

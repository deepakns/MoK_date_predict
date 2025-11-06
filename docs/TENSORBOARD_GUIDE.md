# TensorBoard Guide

## Quick Start

### Method 1: Using the Helper Script (Easiest)

```bash
cd /home/deepakns/Work/MoK_date_predict
./view_tensorboard.sh
```

Then open in your browser: **http://localhost:6006**

### Method 2: Manual Launch

```bash
cd /home/deepakns/Work/MoK_date_predict
source ~/miniconda3/etc/profile.d/conda.sh
conda activate MoK_date_predict
tensorboard --logdir=logs
```

## Usage Options

### View All Training Runs
```bash
./view_tensorboard.sh logs
```

### View Specific Model Logs
```bash
# If you organize logs by model name:
./view_tensorboard.sh logs/MoK_CNN_11
```

### Use Custom Port
```bash
./view_tensorboard.sh logs 6007
```

### Compare Multiple Experiments
Create subdirectories in logs for different experiments:
```
logs/
├── experiment1/
│   └── events.out.tfevents.*
├── experiment2/
│   └── events.out.tfevents.*
└── experiment3/
    └── events.out.tfevents.*
```

Then run:
```bash
tensorboard --logdir=logs
```

TensorBoard will show all experiments with different colored lines for comparison.

## What You'll See

### 1. Scalars Tab (Most Important)
Displays training metrics over time:
- **train_loss**: Training loss per epoch
- **train_rmse**: Training RMSE per epoch
- **val_loss**: Validation loss per epoch
- **val_rmse**: Validation RMSE per epoch
- **learning_rate**: Learning rate over time (shows scheduler reductions!)

**What to Look For:**
- Training and validation loss should both decrease
- Gap between train and val loss indicates overfitting
- Learning rate steps down when validation plateaus
- Early stopping point (if triggered)

### 2. Histograms Tab
Shows distribution of model parameters and gradients:
- Weight distributions per layer
- Gradient distributions (helps identify vanishing/exploding gradients)

**What to Look For:**
- Weights should have reasonable distributions (not all zeros or very large)
- Gradients should flow (not vanishing to zero)

### 3. Graphs Tab
Visualizes the model architecture:
- Shows the computational graph
- Displays layer connections
- Shows tensor shapes

## Tips and Tricks

### 1. Smoothing
Use the smoothing slider in TensorBoard to reduce noise in the plots:
- Slider at top left (default: 0.6)
- Increase for smoother curves
- Decrease to see more detail

### 2. Refresh Data
Click the refresh button (top right) to load new data from ongoing training.

### 3. Download Data
Click the download icon to export plot data as CSV or JSON.

### 4. Compare Runs
Toggle runs on/off using checkboxes on the left sidebar.

### 5. Zoom and Pan
- Click and drag to zoom into specific regions
- Double-click to reset zoom
- Use mouse wheel to zoom in/out

## Remote Access

If you're training on a remote server and want to view TensorBoard locally:

### Option 1: SSH Tunnel (Recommended)

**On your local machine:**
```bash
ssh -L 6006:localhost:6006 deepakns@your-server-address
```

**On the remote server:**
```bash
cd /home/deepakns/Work/MoK_date_predict
./view_tensorboard.sh
```

**Then on your local machine**, open: http://localhost:6006

### Option 2: Bind to All Interfaces

**On remote server:**
```bash
tensorboard --logdir=logs --host=0.0.0.0 --port=6006
```

**On local machine**, open: http://your-server-ip:6006

⚠️ **Security Warning**: This exposes TensorBoard to the network. Only use on trusted networks.

## Organizing Logs

### Best Practice: Organize by Model/Experiment

Modify your training script or config to organize logs:

```yaml
# In model_config.yml
logging:
  log_dir: "logs/MoK_CNN_11_experiment1"
  tensorboard: true
```

This creates a structure like:
```
logs/
├── MoK_CNN_10_baseline/
├── MoK_CNN_11_with_scheduler/
├── MoK_CNN_11_high_dropout/
└── MoK_CNN_11_more_layers/
```

Then compare all experiments:
```bash
tensorboard --logdir=logs
```

### Current Logs

Your current logs are in: `/home/deepakns/Work/MoK_date_predict/logs/`

To view them:
```bash
./view_tensorboard.sh
```

## Monitoring Learning Rate Scheduler

With the new learning rate scheduler implementation, you can now monitor:

1. **When LR is reduced**: Look for step-downs in the learning rate plot
2. **Impact on loss**: See if loss improves after LR reduction
3. **Number of reductions**: Count how many times LR was reduced
4. **Final LR**: Check what learning rate the model converged at

**Example interpretation:**
```
Epoch 1-20:  LR = 0.001, loss decreasing
Epoch 21-25: LR = 0.001, loss plateaus
Epoch 26:    LR = 0.0005 (reduced!)
Epoch 27-40: LR = 0.0005, loss decreases again
```

## Troubleshooting

### No Data Showing
**Problem**: TensorBoard shows "No dashboards are active for the current data set"

**Solutions:**
1. Check tensorboard is enabled in config:
   ```yaml
   logging:
     tensorboard: true
   ```
2. Make sure training has started and logged at least one epoch
3. Verify event files exist:
   ```bash
   ls -la logs/events.out.tfevents.*
   ```

### Port Already in Use
**Problem**: "Address already in use"

**Solution:** Use a different port:
```bash
./view_tensorboard.sh logs 6007
```

### Old Data Showing
**Problem**: TensorBoard shows old training runs

**Solution:**
1. Clear old logs: `rm logs/events.out.tfevents.*`
2. Or use a new log directory in config
3. Or toggle runs off in the sidebar

### Can't Access from Browser
**Problem**: "Unable to connect" or "Connection refused"

**Solutions:**
1. Check TensorBoard is running: `ps aux | grep tensorboard`
2. Verify the port: default is 6006
3. Try: http://127.0.0.1:6006 instead of http://localhost:6006
4. Check firewall settings (if remote)

## Advanced Features

### Profiling (Optional)
To enable profiling for performance analysis:
```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # Training code here
    pass

prof.export_chrome_trace("trace.json")
```

### Custom Scalars
Already implemented in your codebase via `TensorBoardLogger.log_epoch_metrics()`

### HParams Tab
Track hyperparameter experiments (can be added to your training script).

## Resources

- **Official Docs**: https://www.tensorflow.org/tensorboard
- **PyTorch TensorBoard**: https://pytorch.org/docs/stable/tensorboard.html
- **GitHub**: https://github.com/tensorflow/tensorboard

## Summary

**Most Common Command:**
```bash
cd /home/deepakns/Work/MoK_date_predict
./view_tensorboard.sh
```

**Then open:** http://localhost:6006

**Watch For:**
- Validation loss decreasing
- Learning rate step-downs (from scheduler)
- Early stopping point
- Overfitting (train/val gap)

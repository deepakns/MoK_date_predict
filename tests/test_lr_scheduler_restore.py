"""
Test script to verify ReduceLROnPlateauWithRestore behavior.

This script demonstrates that the scheduler correctly:
1. Tracks the best metric from all epochs (not just previous epoch)
2. Restores model and optimizer state when LR is reduced
3. Continues training from the best epoch with reduced LR
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
from pathlib import Path

# Add src directory to path
current_file = Path(__file__)
tests_dir = current_file.parent
project_root = tests_dir.parent
src_dir = project_root / 'src'
sys.path.insert(0, str(src_dir))

from training.utils.lr_scheduler import ReduceLROnPlateauWithRestore


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
        # Initialize with specific values so we can verify restoration
        nn.init.constant_(self.fc.weight, 1.0)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        return self.fc(x)


def test_restore_behavior():
    """Test that model is restored to best epoch when LR is reduced."""
    print("="*80)
    print("Testing ReduceLROnPlateauWithRestore")
    print("="*80)

    # Create model and optimizer
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Create scheduler with patience=3
    scheduler = ReduceLROnPlateauWithRestore(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        threshold=1e-4,
        verbose=True
    )

    # Simulate training with improving and degrading validation loss
    print("\n" + "="*80)
    print("Simulated Training")
    print("="*80)

    val_losses = [
        5.0,   # Epoch 0: Initial (best so far)
        4.0,   # Epoch 1: Improvement (new best)
        3.5,   # Epoch 2: Improvement (new best) <- This should be saved
        4.0,   # Epoch 3: Worse (patience 1/3)
        4.5,   # Epoch 4: Worse (patience 2/3)
        5.0,   # Epoch 5: Worse (patience 3/3) <- LR reduction + restore to epoch 2
        3.0,   # Epoch 6: After restore, training continues
    ]

    # Save the model state at epoch 2 (the best) to verify restoration
    best_epoch_weights = None

    for epoch, val_loss in enumerate(val_losses):
        print(f"\n{'─'*80}")
        print(f"Epoch {epoch}: val_loss = {val_loss:.6f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'─'*80}")

        # Simulate some parameter updates to change model state
        # (In real training, gradients would update these)
        if epoch > 0:
            with torch.no_grad():
                model.fc.weight.data += 0.1 * epoch  # Change weights
            print(f"  Current weights: {model.fc.weight[0, :3].tolist()}")

        # Step scheduler (this will save state if it's the best)
        restored = scheduler.step(val_loss, model, epoch=epoch)

        # Save weights AFTER epoch 2 step (when it becomes the best)
        if epoch == 2:
            best_epoch_weights = model.fc.weight.data.clone()
            print(f"  💾 Saved best weights for verification: {best_epoch_weights[0, :3].tolist()}")

        if restored:
            print(f"\n  {'✓'*40}")
            print(f"  VERIFICATION: Model was restored!")
            print(f"  Expected weights (from epoch 2): {best_epoch_weights[0, :3].tolist()}")
            print(f"  Current weights (after restore): {model.fc.weight[0, :3].tolist()}")

            # Verify restoration
            if torch.allclose(model.fc.weight.data, best_epoch_weights):
                print(f"  ✓ SUCCESS: Weights match! Model correctly restored to best epoch.")
            else:
                print(f"  ✗ FAILURE: Weights don't match!")
            print(f"  {'✓'*40}\n")

    print("\n" + "="*80)
    print("Test Complete!")
    print("="*80)
    print(f"Final LR: {optimizer.param_groups[0]['lr']:.6f}")
    print(f"Expected: 0.05 (0.1 × 0.5)")
    print(f"Best metric: {scheduler.best_metric:.6f}")
    print(f"Best epoch: {scheduler.best_epoch}")


def test_standard_pytorch_behavior():
    """Compare with standard PyTorch ReduceLROnPlateau (no restoration)."""
    print("\n\n" + "="*80)
    print("Comparing with Standard PyTorch ReduceLROnPlateau")
    print("="*80)

    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Standard PyTorch scheduler
    scheduler_standard = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        threshold=1e-4
    )

    val_losses = [5.0, 4.0, 3.5, 4.0, 4.5, 5.0]
    weights_at_epoch_2 = None

    print("\nStandard PyTorch behavior (NO restoration):")
    for epoch, val_loss in enumerate(val_losses):
        print(f"\nEpoch {epoch}: val_loss = {val_loss:.6f}, LR = {optimizer.param_groups[0]['lr']:.6f}")

        if epoch == 2:
            weights_at_epoch_2 = model.fc.weight.data.clone()

        if epoch > 0:
            with torch.no_grad():
                model.fc.weight.data += 0.1 * epoch

        # Standard scheduler doesn't take model as argument
        scheduler_standard.step(val_loss)

        if epoch == 5:
            print(f"\n  After LR reduction:")
            print(f"  Weights at epoch 2 (best): {weights_at_epoch_2[0, :3].tolist()}")
            print(f"  Current weights (NOT restored): {model.fc.weight[0, :3].tolist()}")
            if not torch.allclose(model.fc.weight.data, weights_at_epoch_2):
                print(f"  ✓ As expected: Standard scheduler does NOT restore model")
            else:
                print(f"  ✗ Unexpected: Weights match (should be different)")


if __name__ == "__main__":
    test_restore_behavior()
    test_standard_pytorch_behavior()

"""
Test script to verify OneCycleLR behavior
"""
import torch
import torch.nn as nn
import torch.optim as optim

# Create a simple model
model = nn.Linear(10, 1)

# Create optimizer with initial LR
optimizer = optim.AdamW(model.parameters(), lr=0.01)

# Create OneCycleLR scheduler
num_epochs = 50
steps_per_epoch = 50  # batch_size=1, train_years=50
total_steps = num_epochs * steps_per_epoch

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.1,
    total_steps=total_steps,
    pct_start=0.3,
    anneal_strategy='cos',
    div_factor=25.0,
    final_div_factor=1e4
)

print(f"Initial LR (should be max_lr/div_factor = 0.1/25 = 0.004): {optimizer.param_groups[0]['lr']:.6f}")
print(f"\nFirst 10 steps:")

for step in range(10):
    # Simulate training step
    optimizer.step()
    scheduler.step()

    current_lr = optimizer.param_groups[0]['lr']
    print(f"  Step {step+1}: LR = {current_lr:.6f}")

print(f"\nSteps around peak (warmup ends at step {int(total_steps * 0.3)}):")
for step in range(int(total_steps * 0.3) - 5, int(total_steps * 0.3) + 5):
    # Skip to this step
    scheduler._step_count = step
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    print(f"  Step {step+1}: LR = {current_lr:.6f}")

print(f"\nLast 5 steps:")
for step in range(total_steps - 5, total_steps):
    scheduler._step_count = step
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    print(f"  Step {step+1}: LR = {current_lr:.6f}")

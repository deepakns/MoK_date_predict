"""
Transform functions for ERA5 data preprocessing.

This module provides various transformation functions that can be applied
to the ERA5 dataset tensors.
"""

import torch
import numpy as np


class Normalize:
    """
    Normalize each channel to have zero mean and unit variance.

    This transform normalizes each channel independently across the spatial dimensions.
    """

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization to the tensor.

        Args:
            tensor: Input tensor of shape (channels, lat, lon)

        Returns:
            Normalized tensor of same shape
        """
        # Normalize each channel independently
        # Calculate mean and std across spatial dimensions (lat, lon)
        mean = tensor.mean(dim=(1, 2), keepdim=True)  # Shape: (channels, 1, 1)
        std = tensor.std(dim=(1, 2), keepdim=True)    # Shape: (channels, 1, 1)

        # Avoid division by zero
        return (tensor - mean) / (std + 1e-8)


class StandardizeWithStats:
    """
    Standardize using pre-computed statistics.

    Useful when you want to use training set statistics for both
    training and validation/test sets.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """
        Initialize with pre-computed statistics.

        Args:
            mean: Mean for each channel, shape (channels,) or (channels, 1, 1)
            std: Standard deviation for each channel, shape (channels,) or (channels, 1, 1)
        """
        self.mean = mean
        self.std = std

        # Reshape if needed to allow broadcasting
        if self.mean.dim() == 1:
            self.mean = self.mean.view(-1, 1, 1)
        if self.std.dim() == 1:
            self.std = self.std.view(-1, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply standardization using the stored statistics."""
        return (tensor - self.mean) / (self.std + 1e-8)


class MinMaxScale:
    """
    Scale each channel to a specified range [min_val, max_val].

    Default is [0, 1].
    """

    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        """
        Initialize the min-max scaler.

        Args:
            min_val: Minimum value of output range
            max_val: Maximum value of output range
        """
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply min-max scaling to the tensor."""
        # Scale each channel independently
        min_per_channel = tensor.amin(dim=(1, 2), keepdim=True)
        max_per_channel = tensor.amax(dim=(1, 2), keepdim=True)

        # Normalize to [0, 1]
        normalized = (tensor - min_per_channel) / (max_per_channel - min_per_channel + 1e-8)

        # Scale to [min_val, max_val]
        return normalized * (self.max_val - self.min_val) + self.min_val


class Compose:
    """
    Compose multiple transforms together.

    Similar to torchvision.transforms.Compose
    """

    def __init__(self, transforms):
        """
        Initialize with a list of transforms.

        Args:
            transforms: List of callable transforms
        """
        self.transforms = transforms

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply all transforms in sequence."""
        for transform in self.transforms:
            tensor = transform(tensor)
        return tensor


class ClipValues:
    """
    Clip values to a specified range.

    Useful for removing outliers.
    """

    def __init__(self, min_val: float = None, max_val: float = None):
        """
        Initialize the clipper.

        Args:
            min_val: Minimum value (None for no lower bound)
            max_val: Maximum value (None for no upper bound)
        """
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Clip tensor values to the specified range."""
        if self.min_val is not None and self.max_val is not None:
            return torch.clamp(tensor, self.min_val, self.max_val)
        elif self.min_val is not None:
            return torch.clamp(tensor, min=self.min_val)
        elif self.max_val is not None:
            return torch.clamp(tensor, max=self.max_val)
        else:
            return tensor


class AddGaussianNoise:
    """
    Add Gaussian noise to the data for data augmentation.
    """

    def __init__(self, mean: float = 0.0, std: float = 0.01):
        """
        Initialize the noise adder.

        Args:
            mean: Mean of the Gaussian noise
            std: Standard deviation of the Gaussian noise
        """
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to the tensor."""
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise


# Functional API (simpler for quick use)
def normalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Functional version of Normalize.

    Args:
        tensor: Input tensor of shape (channels, lat, lon)

    Returns:
        Normalized tensor
    """
    mean = tensor.mean(dim=(1, 2), keepdim=True)
    std = tensor.std(dim=(1, 2), keepdim=True)
    return (tensor - mean) / (std + 1e-8)


def min_max_scale(tensor: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0) -> torch.Tensor:
    """
    Functional version of MinMaxScale.

    Args:
        tensor: Input tensor
        min_val: Minimum value of output range
        max_val: Maximum value of output range

    Returns:
        Scaled tensor
    """
    min_per_channel = tensor.amin(dim=(1, 2), keepdim=True)
    max_per_channel = tensor.amax(dim=(1, 2), keepdim=True)
    normalized = (tensor - min_per_channel) / (max_per_channel - min_per_channel + 1e-8)
    return normalized * (max_val - min_val) + min_val


# Example usage documentation
if __name__ == "__main__":
    # Example: Create a dummy tensor
    dummy_tensor = torch.randn(35, 100, 200)  # 35 channels, 100 lat, 200 lon

    print("Original tensor:")
    print(f"  Shape: {dummy_tensor.shape}")
    print(f"  Mean: {dummy_tensor.mean():.4f}")
    print(f"  Std: {dummy_tensor.std():.4f}")

    # Test Normalize
    normalizer = Normalize()
    normalized = normalizer(dummy_tensor)
    print("\nAfter Normalize:")
    print(f"  Shape: {normalized.shape}")
    print(f"  Mean per channel: {normalized.mean(dim=(1,2)).mean():.4f}")
    print(f"  Std per channel: {normalized.std(dim=(1,2)).mean():.4f}")

    # Test MinMaxScale
    scaler = MinMaxScale(min_val=0, max_val=1)
    scaled = scaler(dummy_tensor)
    print("\nAfter MinMaxScale:")
    print(f"  Min: {scaled.min():.4f}")
    print(f"  Max: {scaled.max():.4f}")

    # Test Compose
    composed = Compose([
        ClipValues(min_val=-3, max_val=3),
        Normalize(),
        MinMaxScale(0, 1)
    ])
    result = composed(dummy_tensor)
    print("\nAfter Compose (clip -> normalize -> scale):")
    print(f"  Min: {result.min():.4f}")
    print(f"  Max: {result.max():.4f}")

"""
PyTorch Dataset for loading monthly ERA5 data from NetCDF files.

This module provides a dataset class for loading ERA5 monthly averaged data
and converting it to PyTorch tensors with specific channel ordering.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset


class MonthlyERA5Dataset(Dataset):
    """
    PyTorch Dataset for loading monthly ERA5 NetCDF files.

    The dataset reads NetCDF files from the specified directory and creates
    tensors with time steps stacked as separate channels.

    Output tensor shape: (36, latitude, longitude) for default configuration

    Channel structure (36 channels total):
        - Channels 0-14: Surface variables (5 vars × 3 time steps)
          ttr_t0, ttr_t1, ttr_t2, msl_t0, msl_t1, msl_t2, t2m_t0, t2m_t1, t2m_t2,
          sst_t0, sst_t1, sst_t2, tcc_t0, tcc_t1, tcc_t2
          Note: SST NaN values (over land) are filled with 0

        - Channels 15-32: Pressure level variables (3 vars × 2 levels × 3 time steps)
          u1_t0, u1_t1, u1_t2, v1_t0, v1_t1, v1_t2, z1_t0, z1_t1, z1_t2,
          u2_t0, u2_t1, u2_t2, v2_t0, v2_t1, v2_t2, z2_t0, z2_t1, z2_t2

        - Channels 33-35: Static fields (land_sea_mask, lat, lon)
          land_sea_mask: 1.0 = ocean/valid SST, 0.0 = land/NaN SST

    Variables with dimensions (valid_time, latitude, longitude):
        - ttr: Top net thermal radiation
        - msl: Mean sea level pressure
        - t2m: 2m temperature
        - sst: Sea surface temperature
        - tcc: Total cloud cover

    Variables with dimensions (valid_time, pressure_levels, latitude, longitude):
        - u: U component of wind
        - v: V component of wind
        - z: Geopotential height

    For pressure level variables, we extract the first two pressure levels
    and denote them as u1 (level 0) and u2 (level 1), etc.

    Attributes:
        data_dir (Path): Directory containing the NetCDF files
        years (List[int]): List of years to load
        num_time_steps (int): Number of time steps to load (default: 3)
        pressure_levels (List[int]): Indices of pressure levels to use (default: [0, 1])
    """

    def __init__(
        self,
        data_dir: str = "/gdata2/ERA5/monthly",
        years: Optional[List[int]] = None,
        num_time_steps: int = 3,
        pressure_levels: Optional[List[int]] = None,
        transform: Optional[callable] = None,
        target_file: Optional[str] = None,
    ):
        """
        Initialize the MonthlyERA5Dataset.

        Args:
            data_dir: Path to directory containing NetCDF files named as <year>.nc
            years: List of years to include. If None, all available years are used.
            num_time_steps: Number of time steps to extract from each file (default: 3)
            pressure_levels: Indices of pressure levels to extract (default: [0, 1])
            transform: Optional transform to apply to the data
            target_file: Optional path to CSV file with targets (Year, DateRelJun01 columns)
        """
        self.data_dir = Path(data_dir)
        self.num_time_steps = num_time_steps
        self.pressure_levels = pressure_levels if pressure_levels is not None else [0, 1]
        self.transform = transform
        self.target_file = target_file

        # Variable names in the NetCDF files
        self.surface_vars = ['ttr', 'msl', 't2m', 'sst', 'tcc']
        self.pressure_vars = ['u', 'v', 'z']

        # Load target data if provided
        self.targets: Optional[Dict[int, float]] = None
        if target_file:
            self.targets = self._load_targets(target_file)

        # Get available years from directory if not specified
        if years is None:
            self.years = self._get_available_years()
        else:
            self.years = sorted(years)

        # Build index: list of (year, time_idx) tuples
        self.data_index = []
        self._build_index()

    def _get_available_years(self) -> List[int]:
        """Scan directory for available .nc files and extract years."""
        nc_files = sorted(self.data_dir.glob("*.nc"))
        years = []
        for nc_file in nc_files:
            try:
                year = int(nc_file.stem)
                years.append(year)
            except ValueError:
                continue
        return sorted(years)

    def _load_targets(self, target_file: str) -> Dict[int, float]:
        """
        Load target values from CSV file.

        Args:
            target_file: Path to CSV file with columns 'Year' and 'DateRelJun01'

        Returns:
            Dictionary mapping year to target value
        """
        df = pd.read_csv(target_file)
        # Create a dictionary mapping year to target value
        targets = dict(zip(df['Year'].values, df['DateRelJun01'].values))
        return targets

    def _build_index(self):
        """Build an index of all available samples (year, time_step combinations)."""
        for year in self.years:
            nc_file = self.data_dir / f"{year}.nc"
            if not nc_file.exists():
                print(f"Warning: File {nc_file} not found, skipping year {year}")
                continue

            # Open dataset to get number of time steps
            try:
                with xr.open_dataset(nc_file) as ds:
                    num_times = len(ds['valid_time'])
                    # Create samples for each valid starting point
                    # We need num_time_steps consecutive time steps
                    for time_idx in range(num_times - self.num_time_steps + 1):
                        self.data_index.append((year, time_idx))
            except Exception as e:
                print(f"Warning: Error reading {nc_file}: {e}")
                continue

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.data_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (data_tensor, metadata_dict) where:
                - data_tensor: Tensor of shape (channels, lat, lon)
                  channels = 36 (5 surface vars × 3 time steps + 3 pressure vars × 2 levels × 3 time steps + land_sea_mask + lat + lon)
                - metadata_dict: Dictionary containing year, time_idx, and other info
        """
        year, time_idx = self.data_index[idx]
        nc_file = self.data_dir / f"{year}.nc"

        # Load the NetCDF file
        ds = xr.open_dataset(nc_file)

        # Extract time slice
        time_slice = slice(time_idx, time_idx + self.num_time_steps)

        # List to hold all channels
        all_channels = []

        # Create land-sea mask from SST (before filling NaNs)
        # Use first time step to create mask (mask is constant across time)
        land_sea_mask = None
        if 'sst' in ds:
            sst_data = ds['sst'].isel(valid_time=time_slice).values  # (time, lat, lon)
            # Create mask: 1.0 where SST is valid (ocean), 0.0 where NaN (land)
            land_sea_mask = (~np.isnan(sst_data[0])).astype(np.float32)  # (lat, lon)

        # Extract surface variables: shape (time, lat, lon)
        # Stack each time step as a separate channel
        for var_name in self.surface_vars:
            if var_name not in ds:
                raise ValueError(f"Variable {var_name} not found in {nc_file}")
            var_data = ds[var_name].isel(valid_time=time_slice).values  # (time, lat, lon)

            # Fill NaN values for SST with 0
            if var_name == 'sst':
                var_data = np.nan_to_num(var_data, nan=0.0)

            # Add each time step as a separate channel
            for t in range(self.num_time_steps):
                all_channels.append(var_data[t])  # (lat, lon)

        # Extract pressure level variables: shape (time, pressure, lat, lon)
        # For each variable and each pressure level, stack time steps as channels
        for var_name in self.pressure_vars:
            if var_name not in ds:
                raise ValueError(f"Variable {var_name} not found in {nc_file}")
            # Get data for the specified time and pressure levels
            var_data_full = ds[var_name].isel(valid_time=time_slice).values  # (time, pressure, lat, lon)

            # Extract specified pressure levels
            for p_level in self.pressure_levels:
                if p_level >= var_data_full.shape[1]:
                    raise ValueError(
                        f"Pressure level {p_level} out of range for variable {var_name}"
                    )
                # Add each time step as a separate channel for this pressure level
                for t in range(self.num_time_steps):
                    all_channels.append(var_data_full[t, p_level, :, :])  # (lat, lon)

        # Get lat/lon coordinates
        lat = ds['latitude'].values
        lon = ds['longitude'].values

        # Create coordinate grids (single 2D grids, not repeated for time)
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        # Add land-sea mask (if available)
        if land_sea_mask is not None:
            all_channels.append(land_sea_mask)  # (lat, lon)

        all_channels.append(lat_grid)  # (lat, lon)
        all_channels.append(lon_grid)  # (lat, lon)

        # Convert to numpy array and then to torch tensor
        # Shape: (channels, lat, lon)
        data_array = np.array(all_channels, dtype=np.float32)
        data_tensor = torch.from_numpy(data_array)

        # Close the dataset
        ds.close()

        # Create channel names
        channel_names = []
        # Surface variables
        for var in self.surface_vars:
            for t in range(self.num_time_steps):
                channel_names.append(f"{var}_t{t}")
        # Pressure variables
        for var in self.pressure_vars:
            for p_idx in range(len(self.pressure_levels)):
                for t in range(self.num_time_steps):
                    channel_names.append(f"{var}{p_idx+1}_t{t}")
        # Static fields
        if land_sea_mask is not None:
            channel_names.append('land_sea_mask')
        channel_names.extend(['lat', 'lon'])

        # Get target value if available
        target = None
        if self.targets is not None:
            if year in self.targets:
                target = torch.tensor([self.targets[year]], dtype=torch.float32)
            else:
                # If target not available for this year, use NaN or raise warning
                print(f"Warning: No target found for year {year}")
                target = torch.tensor([float('nan')], dtype=torch.float32)

        # Create metadata dictionary
        metadata = {
            'year': year,
            'time_idx': time_idx,
            'num_time_steps': self.num_time_steps,
            'shape': data_tensor.shape,
            'channel_names': channel_names,
            'target': target
        }

        # Apply transform if specified
        if self.transform:
            data_tensor = self.transform(data_tensor)

        return data_tensor, metadata

    def get_channel_info(self) -> dict:
        """
        Get information about the channels in the output tensor.

        Returns:
            Dictionary with channel indices and names
        """
        channel_names = []

        # Surface variables (each var × num_time_steps)
        for var in self.surface_vars:
            for t in range(self.num_time_steps):
                channel_names.append(f"{var}_t{t}")

        # Pressure level variables (each var × num_pressure_levels × num_time_steps)
        for var in self.pressure_vars:
            for p_idx in range(len(self.pressure_levels)):
                for t in range(self.num_time_steps):
                    channel_names.append(f"{var}{p_idx + 1}_t{t}")

        # Static fields
        channel_names.append('land_sea_mask')
        channel_names.extend(['lat', 'lon'])

        return {
            'channel_names': channel_names,
            'num_channels': len(channel_names),
            'surface_vars': self.surface_vars,
            'pressure_vars': self.pressure_vars,
            'pressure_levels': self.pressure_levels,
            'num_time_steps': self.num_time_steps
        }


def create_dataloaders(
    data_dir: str = "/gdata2/ERA5/monthly",
    train_years: Optional[List[int]] = None,
    val_years: Optional[List[int]] = None,
    test_years: Optional[List[int]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create training, validation, and test dataloaders.

    Args:
        data_dir: Path to directory containing NetCDF files
        train_years: List of years for training
        val_years: List of years for validation
        test_years: List of years for testing
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        **dataset_kwargs: Additional arguments for MonthlyERA5Dataset

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = MonthlyERA5Dataset(
        data_dir=data_dir,
        years=train_years,
        **dataset_kwargs
    )

    val_dataset = MonthlyERA5Dataset(
        data_dir=data_dir,
        years=val_years,
        **dataset_kwargs
    )

    test_dataset = MonthlyERA5Dataset(
        data_dir=data_dir,
        years=test_years,
        **dataset_kwargs
    )

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

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    dataset = MonthlyERA5Dataset(
        data_dir="/gdata2/ERA5/monthly",
        num_time_steps=3,
        pressure_levels=[0, 1]
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Channel info: {dataset.get_channel_info()}")

    # Get first sample
    if len(dataset) > 0:
        data, metadata = dataset[0]
        print(f"\nFirst sample:")
        print(f"  Shape: {data.shape}")
        print(f"  Metadata: {metadata}")
        print(f"  Data range: [{data.min():.2f}, {data.max():.2f}]")

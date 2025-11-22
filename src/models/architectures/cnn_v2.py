import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_normalization_layer(norm_type: int, num_features: int, num_groups: int = 32):
    """
    Factory function to create normalization layers based on norm_type.

    Args:
        norm_type: Type of normalization
            0: BatchNormalization - normalizes across batch dimension
            1: LayerNormalization - normalizes across channel dimension
            2: InstanceNormalization - normalizes per sample and channel
            3: GroupNormalization - normalizes in groups of channels
        num_features: Number of channels/features to normalize
        num_groups: Number of groups for GroupNorm (default: 32)

    Returns:
        nn.Module: Appropriate normalization layer
    """
    if norm_type == 0:
        return nn.BatchNorm2d(num_features)
    elif norm_type == 1:
        return nn.GroupNorm(1, num_features)
    elif norm_type == 2:
        return nn.InstanceNorm2d(num_features, affine=True)
    elif norm_type == 3:
        actual_groups = min(num_groups, num_features)
        return nn.GroupNorm(actual_groups, num_features)
    else:
        raise ValueError(f"Invalid norm_type: {norm_type}. Must be 0 (Batch), 1 (Layer), 2 (Instance), or 3 (Group).")


class PositionalEmbedding(nn.Module):
    """
    Creates learnable positional embeddings from latitude and longitude coordinates.

    The embeddings are created using sinusoidal positional encoding (similar to Transformers)
    followed by a learnable linear projection. These embeddings are then added to each
    channel of the input tensor.

    Args:
        embedding_dim: Dimension of the positional embedding
        max_lat: Maximum latitude value (default: 90.0)
        min_lat: Minimum latitude value (default: -90.0)
        max_lon: Maximum longitude value (default: 180.0)
        min_lon: Minimum longitude value (default: -180.0)
        num_frequencies: Number of sinusoidal frequencies to use (default: 16)
    """
    def __init__(self, embedding_dim: int, max_lat: float = 90.0, min_lat: float = -90.0,
                 max_lon: float = 180.0, min_lon: float = -180.0, num_frequencies: int = 16):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_lat = max_lat
        self.min_lat = min_lat
        self.max_lon = max_lon
        self.min_lon = min_lon
        self.num_frequencies = num_frequencies

        # Learnable projection from sinusoidal features to embedding_dim
        # Input: 2 * num_frequencies * 2 (lat and lon, each with sin and cos for each frequency)
        sinusoidal_dim = 4 * num_frequencies
        self.projection = nn.Linear(sinusoidal_dim, embedding_dim)

    def forward(self, lat, lon):
        """
        Args:
            lat: Latitude tensor of shape (H, W) or (B, 1, H, W)
            lon: Longitude tensor of shape (H, W) or (B, 1, H, W)

        Returns:
            Positional embedding of shape (B, embedding_dim, H, W)
        """
        # Handle different input shapes
        if lat.dim() == 2:
            H, W = lat.shape
            lat = lat.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            lon = lon.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif lat.dim() == 4:
            _, _, H, W = lat.shape
        else:
            raise ValueError(f"Expected lat/lon to have 2 or 4 dimensions, got {lat.dim()}")

        B = lat.size(0)

        # Normalize lat and lon to [-1, 1]
        lat_norm = 2.0 * (lat - self.min_lat) / (self.max_lat - self.min_lat) - 1.0
        lon_norm = 2.0 * (lon - self.min_lon) / (self.max_lon - self.min_lon) - 1.0

        # Create sinusoidal positional encodings
        # Generate frequencies: [1, 2, 4, 8, ..., 2^(num_frequencies-1)]
        frequencies = 2.0 ** torch.arange(self.num_frequencies, dtype=lat.dtype, device=lat.device)
        frequencies = frequencies.view(1, -1, 1, 1)  # (1, num_frequencies, 1, 1)

        # Apply sinusoidal encoding to lat and lon
        lat_encoded = []
        lon_encoded = []

        for freq in frequencies.squeeze(0):
            lat_encoded.append(torch.sin(math.pi * freq * lat_norm))
            lat_encoded.append(torch.cos(math.pi * freq * lat_norm))
            lon_encoded.append(torch.sin(math.pi * freq * lon_norm))
            lon_encoded.append(torch.cos(math.pi * freq * lon_norm))

        # Concatenate all encodings: (B, 4*num_frequencies, H, W)
        positional_features = torch.cat(lat_encoded + lon_encoded, dim=1)

        # Reshape to (B, H, W, 4*num_frequencies) for linear projection
        positional_features = positional_features.permute(0, 2, 3, 1)

        # Project to embedding_dim: (B, H, W, embedding_dim)
        embeddings = self.projection(positional_features)

        # Reshape back to (B, embedding_dim, H, W)
        embeddings = embeddings.permute(0, 3, 1, 2)

        return embeddings


class SpatialPyramidPooling(nn.Module):
    """
    Spatial Pyramid Pooling layer that outputs a fixed-length representation
    regardless of input size by pooling features at multiple scales.

    Args:
        output_pool_sizes: List of pool sizes for each pyramid level (e.g., [1, 2, 4])
        pool_type: Pooling type - 0: max only, 1: avg only, 2: both max and avg (concatenated)
    """
    def __init__(self, output_pool_sizes=[1, 2, 4], pool_type=2):
        super().__init__()
        self.output_pool_sizes = output_pool_sizes
        self.pool_type = pool_type

    def forward(self, x):
        batch_size, num_channels = x.size()[:2]

        pooled_outputs = []

        # Max pooling
        if self.pool_type in [0, 2]:
            for pool_size in self.output_pool_sizes:
                pooled = F.adaptive_max_pool2d(x, (pool_size, pool_size))
                pooled = pooled.view(batch_size, num_channels, -1)
                pooled_outputs.append(pooled)

        # Avg pooling
        if self.pool_type in [1, 2]:
            for pool_size in self.output_pool_sizes:
                pooled = F.adaptive_avg_pool2d(x, (pool_size, pool_size))
                pooled = pooled.view(batch_size, num_channels, -1)
                pooled_outputs.append(pooled)

        # Concatenate all pooled features along the feature dimension
        output = torch.cat(pooled_outputs, dim=2)

        # Flatten to (batch_size, num_channels * total_bins)
        output = output.view(batch_size, -1)

        return output


class ResSpatialProcessorBlock(nn.Module):
    """Residual Block to process stacked variables"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm_type=0, norm_num_groups=32):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=1, bias=False)
        self.bn1 = get_normalization_layer(norm_type, out_channels, norm_num_groups)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=1, bias=False)
        self.bn2 = get_normalization_layer(norm_type, out_channels, norm_num_groups)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                get_normalization_layer(norm_type, out_channels, norm_num_groups)
            )

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class MoK_CNN_V2_Predictor(nn.Module):
    """
    CNN model with positional embeddings from latitude and longitude.

    Key difference from v1: Uses lat/lon to create positional embeddings that are
    merged (concatenated) with the input channels before processing.

    Input: 12 variables from ERA5 dataset (5 surface vars + 6 pressure vars + land_sea_mask)
           x 3 time steps (for surface and pressure vars) = 33 channels (excluding lat/lon)
           Variables: ttr, msl, t2m, sst, tcc (surface) + u1, u2, v1, v2, z1, z2 (pressure) + land_sea_mask

           Additional inputs: lat, lon (used for positional embedding, not as input channels)

    Input spatial dimensions: 1440 x 481

    Architecture:
        - Creates positional embeddings from lat/lon
        - Concatenates positional embeddings with input channels
        - Coarsening avg pool (4x4 kernel, stride 4) reduces input: 1440x481 -> 360x120
        - Configurable activation in input lifting pathway
        - Uses residual blocks to progressively increase channels
        - Max pooling reduces spatial dimensions
        - Spatial Pyramid Pooling for multi-scale feature extraction
        - Dense layers followed by output layer
    """

    def __init__(self, in_channels: int = 33, dropout_rate: float = 0.5, num_classes: int = 1,
                 activation_type: str = "relu", prelu_mode: str = "shared", leaky_relu_slope: float = 0.01,
                 l2_reg: float = 0.0001, spp_option: int = 2, spp_ops: list = [1], norm_type: int = 0,
                 norm_num_groups: int = 32, pos_embedding_dim: int = 16, num_frequencies: int = 16):
        """
        Args:
            in_channels: Number of input channels (excluding lat/lon)
            dropout_rate: Dropout probability
            num_classes: Number of output classes (1 for regression)
            activation_type: Type of activation ("relu", "prelu", "leaky_relu")
            prelu_mode: PReLU mode ("shared" or "channel")
            leaky_relu_slope: Negative slope for LeakyReLU
            l2_reg: L2 regularization strength
            spp_option: Spatial Pyramid Pooling type (0: max only, 1: avg only, 2: both)
            spp_ops: List of output pool sizes for pyramid levels
            norm_type: Type of normalization (0: Batch, 1: Layer, 2: Instance, 3: Group)
            norm_num_groups: Number of groups for GroupNormalization
            pos_embedding_dim: Dimension of positional embedding from lat/lon
            num_frequencies: Number of sinusoidal frequencies for positional encoding
        """
        super().__init__()

        self.num_classes = num_classes
        self.activation_type = activation_type
        self.l2_reg = l2_reg
        self.norm_type = norm_type
        self.norm_num_groups = norm_num_groups
        self.in_channels = in_channels
        self.pos_embedding_dim = pos_embedding_dim

        # Positional embedding module
        self.pos_embedding = PositionalEmbedding(
            embedding_dim=pos_embedding_dim,
            num_frequencies=num_frequencies
        )

        # After concatenating positional embedding with input
        total_channels = in_channels + pos_embedding_dim

        # Coarsening avg pool layer (4x4 kernel with stride 4)
        self.coarsen_pool = nn.AvgPool2d(kernel_size=4, stride=4)

        # Initial convolution layers
        self.conv1 = nn.Conv2d(in_channels=total_channels, out_channels=total_channels,
                               groups=1, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=total_channels, out_channels=64,
                               kernel_size=1, bias=False)
        self.bn1 = get_normalization_layer(norm_type, num_features=64, num_groups=norm_num_groups)

        # Create activation functions
        if activation_type == "prelu":
            if prelu_mode == "channel":
                self.act1 = nn.PReLU(num_parameters=total_channels)
                self.act2 = nn.PReLU(num_parameters=64)
            else:
                self.act1 = nn.PReLU(num_parameters=1)
                self.act2 = nn.PReLU(num_parameters=1)
        elif activation_type == "leaky_relu":
            self.act1 = nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True)
            self.act2 = nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True)
        else:
            self.act1 = nn.ReLU(inplace=True)
            self.act2 = nn.ReLU(inplace=True)

        self.relu = nn.ReLU(inplace=True)

        # Residual processing blocks
        self.process_block1 = ResSpatialProcessorBlock(64, 64, norm_type=norm_type, norm_num_groups=norm_num_groups)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.process_block2 = ResSpatialProcessorBlock(64, 128, norm_type=norm_type, norm_num_groups=norm_num_groups)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.process_block3 = ResSpatialProcessorBlock(128, 128, norm_type=norm_type, norm_num_groups=norm_num_groups)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.process_block4 = ResSpatialProcessorBlock(128, 256, norm_type=norm_type, norm_num_groups=norm_num_groups)

        # Spatial Pyramid Pooling
        self.spp = SpatialPyramidPooling(output_pool_sizes=spp_ops, pool_type=spp_option)

        self.flatten = nn.Flatten(start_dim=1)

        # Dropout layers
        self.dropout1 = nn.Dropout(p=dropout_rate)

        # Lazy linear layer for flexibility
        self.fc1 = nn.LazyLinear(out_features=num_classes)

    def forward(self, x, lat, lon):
        """
        Args:
            x: Input tensor of shape (B, C, H, W) where C = in_channels
            lat: Latitude tensor of shape (H, W) or (B, 1, H, W)
            lon: Longitude tensor of shape (H, W) or (B, 1, H, W)

        Returns:
            Output predictions of shape (B, num_classes)
        """
        # Generate positional embeddings from lat/lon
        pos_emb = self.pos_embedding(lat, lon)  # (B, pos_embedding_dim, H, W)

        # Expand positional embedding to match batch size if needed
        if pos_emb.size(0) == 1 and x.size(0) > 1:
            pos_emb = pos_emb.expand(x.size(0), -1, -1, -1)

        # Concatenate positional embeddings with input channels
        x = torch.cat([x, pos_emb], dim=1)  # (B, C + pos_embedding_dim, H, W)

        # Coarsen spatial dimensions
        x = self.coarsen_pool(x)

        # Input lifting with configurable activation
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.act2(x)

        # Process through residual blocks
        x = self.process_block1(x)
        x = self.maxpool1(x)

        x = self.process_block2(x)
        x = self.maxpool2(x)

        x = self.process_block3(x)
        x = self.maxpool3(x)

        x = self.process_block4(x)

        # Spatial Pyramid Pooling
        x = self.spp(x)

        # Final projection with dropout
        x = self.fc1(x)
        x = self.dropout1(x)

        return x

    def get_parameter_groups(self):
        """
        Get parameter groups for optimizer with different weight decay settings.

        Returns:
            List of parameter groups:
                - Conv weights: with L2 regularization (weight_decay = self.l2_reg)
                - Other parameters: without L2 regularization (weight_decay = 0)
        """
        conv_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if 'conv' in name and 'weight' in name:
                conv_params.append(param)
            else:
                no_decay_params.append(param)

        return [
            {'params': conv_params, 'weight_decay': self.l2_reg},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]


def create_model(in_channels: int = 33, dropout_rate: float = 0.5, num_classes: int = 1,
                 activation_type: str = "relu", prelu_mode: str = "shared", leaky_relu_slope: float = 0.01,
                 l2_reg: float = 0.0001, spp_option: int = 2, spp_ops: list = [1], norm_type: int = 0,
                 norm_num_groups: int = 32, pos_embedding_dim: int = 16, num_frequencies: int = 16):
    """
    Create a MoK_CNN_V2_Predictor model with positional embeddings.

    Args:
        in_channels: Number of input channels (excluding lat/lon, default: 33)
        dropout_rate: Dropout probability
        num_classes: Number of output classes (1 for regression)
        activation_type: Type of activation function
        prelu_mode: PReLU parameter sharing
        leaky_relu_slope: Negative slope for LeakyReLU
        l2_reg: L2 regularization strength
        spp_option: Spatial Pyramid Pooling type
        spp_ops: List of output pool sizes for pyramid levels
        norm_type: Type of normalization
        norm_num_groups: Number of groups for GroupNormalization
        pos_embedding_dim: Dimension of positional embedding from lat/lon
        num_frequencies: Number of sinusoidal frequencies for positional encoding

    Returns:
        MoK_CNN_V2_Predictor model instance
    """
    model = MoK_CNN_V2_Predictor(
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
        norm_num_groups=norm_num_groups,
        pos_embedding_dim=pos_embedding_dim,
        num_frequencies=num_frequencies
    )
    return model


def test():
    import os
    import yaml

    try:
        from torchview import draw_graph
        torchview_available = True
    except ImportError:
        torchview_available = False
        print("Warning: torchview not available. Install with: pip install torchview")

    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                               'config', 'model_config_with_wandb.yml')
    model_name = "MoK_CNN_V2_Predictor"
    num_classes = 1
    dropout_rate = 0.5
    in_channels = 33  # Excluding lat/lon
    spatial_shape = [1440, 481]
    norm_type = 0
    norm_num_groups = 32
    pos_embedding_dim = 16

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            model_name = config.get('model', {}).get('name', 'MoK_CNN_V2_Predictor')
            num_classes = config.get('model', {}).get('num_classes', 1)
            dropout_rate = config.get('model', {}).get('dropout_rate', 0.5)
            spatial_shape = config.get('model', {}).get('spatial_input_shape', [1440, 481])
            norm_type = config.get('model', {}).get('cnn_norm', 0)
            norm_num_groups = config.get('model', {}).get('cnn_norm_num_groups', 32)
            pos_embedding_dim = config.get('model', {}).get('pos_embedding_dim', 16)

            # Calculate channels (excluding lat/lon for v2)
            data_config = config.get('data', {})
            surface_vars = data_config.get('input_geo_var_surf', [])
            pressure_vars = data_config.get('input_geo_var_press', [])
            time_steps = data_config.get('time_steps', [])
            pressure_levels = data_config.get('pressure_levels', [])
            include_landsea = data_config.get('include_landsea', False)

            num_surface_channels = len(surface_vars) * len(time_steps)
            num_pressure_channels = len(pressure_vars) * len(pressure_levels) * len(time_steps)
            num_static_channels = int(include_landsea)
            in_channels = num_surface_channels + num_pressure_channels + num_static_channels

            print(f"Configuration loaded from: {config_path}")
            print(f"  Surface vars: {surface_vars} (× {len(time_steps)} time steps = {num_surface_channels} channels)")
            print(f"  Pressure vars: {pressure_vars} (× {len(pressure_levels)} levels × {len(time_steps)} time steps = {num_pressure_channels} channels)")
            print(f"  Static channels: {num_static_channels} (landsea={include_landsea})")
            print(f"  Total input channels: {in_channels}")
            print(f"  Positional embedding dim: {pos_embedding_dim}")
            print()

    except Exception as e:
        print(f"Warning: Could not load config file ({e}). Using default values.")
        print(f"  in_channels={in_channels}, num_classes={num_classes}")
        print()

    # Test model
    print("Testing CNN V2 Model with Positional Embeddings:")
    print("-" * 60)

    l2_reg = 0.0001
    try:
        l2_reg = config.get('training', {}).get('weight_decay', 0.0001)
    except:
        pass

    model = MoK_CNN_V2_Predictor(
        in_channels=in_channels,
        num_classes=num_classes,
        l2_reg=l2_reg,
        norm_type=norm_type,
        norm_num_groups=norm_num_groups,
        pos_embedding_dim=pos_embedding_dim
    )

    # Create sample inputs
    x = torch.randn(2, in_channels, spatial_shape[0], spatial_shape[1])
    lat = torch.randn(spatial_shape[0], spatial_shape[1])  # Can be (H, W)
    lon = torch.randn(spatial_shape[0], spatial_shape[1])

    output = model(x, lat, lon)

    print(f"Input shape: {x.shape}")
    print(f"Lat shape: {lat.shape}")
    print(f"Lon shape: {lon.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"L2 regularization strength: {model.l2_reg}")
    print(f"Positional embedding dimension: {model.pos_embedding_dim}")

    # Show parameter groups
    param_groups = model.get_parameter_groups()
    conv_params = sum(p.numel() for p in param_groups[0]['params'])
    other_params = sum(p.numel() for p in param_groups[1]['params'])
    print(f"  Conv params (with L2): {conv_params:,}")
    print(f"  Other params (no L2): {other_params:,}")


if __name__ == "__main__":
    test()

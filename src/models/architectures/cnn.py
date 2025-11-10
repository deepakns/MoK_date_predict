import torch
import torch.nn as nn
import torch.nn.functional as F


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
                # Perform adaptive max pooling to get pool_size x pool_size output
                pooled = F.adaptive_max_pool2d(x, (pool_size, pool_size))

                # Flatten the pooled output
                pooled = pooled.view(batch_size, num_channels, -1)
                pooled_outputs.append(pooled)

        # Avg pooling
        if self.pool_type in [1, 2]:
            for pool_size in self.output_pool_sizes:
                # Perform adaptive avg pooling to get pool_size x pool_size output
                pooled = F.adaptive_avg_pool2d(x, (pool_size, pool_size))

                # Flatten the pooled output
                pooled = pooled.view(batch_size, num_channels, -1)
                pooled_outputs.append(pooled)

        # Concatenate all pooled features along the feature dimension
        output = torch.cat(pooled_outputs, dim=2)

        # Flatten to (batch_size, num_channels * total_bins)
        output = output.view(batch_size, -1)

        return output

class ResSpatialProcessorBlock(nn.Module):
    """Residual Block to process stacked variables"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__() # Required by PyTorch to call the init constructor of the parent class nn.Module so that all layers defined here are tracked correctly

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride,
                               padding=1, bias=False)
        # conv1 is an instance variable that stores a 2D convolution operation. Use self.conv1 allows us to use conv1 throughout the class ResSpatialProcessorBlock.
        # When stride = 1 is used, there is no spatial reduction. Using padding=1 ensures that input and output size are equal. If padding=0, the kernel_size will reduce the spatial dimension
        # bias = False is set so that when a batch normalization is used after the conv1 with a bias term, there is no redundancy of variables
        self.bn1 = nn.BatchNorm2d(out_channels) # scales the input to bn1 such that mean = 0, std = 1 and then scales it tby gamma and shifts it by beta, both learnable
        self.relu = nn.ReLU (inplace = True)  # inplace=True modifies the input tensor directly in memory (saves memory); generally inplace=False is safer as it creates a new tensor

        # Now repeat conv, bn. ReLU can be reused as it has no learnable parameters (it is simply a function application)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, stride = stride,
                               padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)

        # define the skip connection and ensure that skip connection works even with dimensions change in conv1 and conv2

        self.skip = nn.Sequential() # Sequential is a container module
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride=stride, bias=False), # this performs 1x1 convolution that changes the number of channels. padding is not needed as kernel_size is 1
                nn.BatchNorm2d(out_channels)
            )


# In PyTorch, nn.Module is the base class for all neural network components. 
# Think of it as a building block that can 
#   (i) Have learnable parameters (weights, biases); 
#  (ii) Perform computations (forward pass); and 
# (iii) Be composed with other modules

# nn.Sequential() is a container module that can hold other modules such as layers
# We use nn.Sequential and an if statement to implement Residual connection as it makes the forward simple and clean

# other modules in PyTorch are layer modules, activation modules, loss modules, custom modules.
        
    def forward(self, x):
        identity = self.skip(x) # cleverly adjusts the dimension of x so that the out += identity works correctly

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity         # this step performs the actual residual connection
        out = self.relu(out) 

        return out


class MoK_CNN_Predictor(nn.Module):
    """
    Puts together the CNN model for regression or classification
    Input: 12 variables from ERA5 dataset (5 surface vars + 6 pressure vars + land_sea_mask + lat + lon)
           x 3 time steps (for surface and pressure vars) = 36 channels
           Variables: ttr, msl, t2m, sst, tcc (surface) + u1, u2, v1, v2, z1, z2 (pressure) + land_sea_mask + lat, lon
           Note: SST NaN values (over land) are filled with 0, land_sea_mask indicates ocean (1.0) vs land (0.0)
    Input spatial dimensions: 1440 x 481

    Architecture:
        - Coarsening avg pool (4x4 kernel, stride 4) reduces input: 1440x481 -> 360x120
        - Configurable activation in input lifting pathway (ReLU/PReLU/LeakyReLU)
        - Uses residual blocks with ReLU to progressively increase channels (36 -> 64 -> 128 -> 256)
        - Max pooling reduces spatial dimensions by factor of 2 each stage
        - Spatial Pyramid Pooling (SPP) for multi-scale feature extraction (replaces adaptive pooling)
        - Four dense layers (100->100->100->10) followed by output layer
        - Output layer produces num_classes logits for classification or 1 value for regression

    Regularization:
        - L2 regularization on convolutional layers (applied via optimizer weight_decay)
        - Dropout on fc1, fc2, fc3, and fc4 layers
        - Batch normalization after initial convolutions
    """

    def __init__(self, in_channels: int = 36, dropout_rate: float = 0.5, num_classes: int = 1,
                 activation_type: str = "relu", prelu_mode: str = "shared", leaky_relu_slope: float = 0.01,
                 l2_reg: float = 0.0001, spp_option: int = 2, spp_ops: list = [1]):
        """
        Args:
            in_channels: Number of input channels
            dropout_rate: Dropout probability
            num_classes: Number of output classes (1 for regression)
            activation_type: Type of activation ("relu", "prelu", "leaky_relu")
            prelu_mode: PReLU mode ("shared" or "channel") - only used if activation_type="prelu"
            leaky_relu_slope: Negative slope for LeakyReLU - only used if activation_type="leaky_relu"
            l2_reg: L2 regularization strength (default: 0.0001)
                    Applied via optimizer weight_decay parameter during training
                    Typical values: 1e-4 to 1e-5
            spp_option: Spatial Pyramid Pooling type (0: max only, 1: avg only, 2: both)
            spp_ops: List of output pool sizes for pyramid levels (e.g., [1], [1, 2], [1, 2, 4])
        """
        super().__init__() # call the constructor of nn.Module to ensure all params are tracked

        self.num_classes = num_classes  # Store for forward pass logic
        self.activation_type = activation_type
        self.l2_reg = l2_reg  # Store for reference (actual application is via optimizer)

        # Coarsening avg pool layer before processing (4x4 kernel with stride 4)
        # Reduces spatial dimensions: 1440x481 -> 360x120
        self.coarsen_pool = nn.AvgPool2d(kernel_size=4, stride=4)

        # Initial convolution layer to process input channels
        # Note: Adjusted to use in_channels parameter for flexibility
        # Using groups=1 for now to process all channels together
        # TO DO: Have to implement logic to process groups of channels per variable for their three day together first

        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = in_channels, groups = 1, kernel_size = 3, padding = 1, bias = False )
        self.conv2 = nn.Conv2d(in_channels = in_channels, out_channels = 64, kernel_size = 1, bias = False )
        self.bn1 = nn.BatchNorm2d(num_features = 64)

        # Create activation functions based on type
        if activation_type == "prelu":
            # PReLU: Learnable negative slopes
            if prelu_mode == "channel":
                # Channel-wise: Each channel learns its own slope
                self.act1 = nn.PReLU(num_parameters=in_channels)
                self.act2 = nn.PReLU(num_parameters=64)
            else:  # shared
                # Shared: All channels share same slope
                self.act1 = nn.PReLU(num_parameters=1)
                self.act2 = nn.PReLU(num_parameters=1)
        elif activation_type == "leaky_relu":
            # LeakyReLU: Fixed negative slope
            self.act1 = nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True)
            self.act2 = nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True)
        else:  # relu (default)
            # ReLU: Standard activation
            self.act1 = nn.ReLU(inplace=True)
            self.act2 = nn.ReLU(inplace=True)

        # Keep ReLU for use in residual blocks
        self.relu = nn.ReLU(inplace=True)

        self.process_block1 = ResSpatialProcessorBlock(64, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.process_block2 = ResSpatialProcessorBlock(64, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.process_block3 = ResSpatialProcessorBlock(128, 128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.process_block4 = ResSpatialProcessorBlock(128, 256)

        # Spatial Pyramid Pooling replaces AdaptiveAvgPool2d
        # Output features depend on spp_ops and spp_option
        # e.g., spp_ops=[1], spp_option=2: bins = 1*1*2 = 2 (max + avg)
        # For 256 channels with [1]: 256 * 2 = 512 features
        self.spp = SpatialPyramidPooling(output_pool_sizes=spp_ops, pool_type=spp_option)

        # Flatten does not need a layer definition, but we will define for style consistency
        self.flatten = nn.Flatten(start_dim=1) # keeps the batch axis=0 and flattens the rest

        # Dropout layers for regularization
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.dropout3 = nn.Dropout(p=dropout_rate)
        self.dropout4 = nn.Dropout(p=dropout_rate)

        #Now we have to add linear layers. Ideally we must calculate the number of output from the flatten, but we are lazy. So we do the following.
        self.fc1 = nn.LazyLinear(out_features=num_classes) # LazyLinear does not require explicit computation of the input feature size. CAUTION: Run forward by invoking the class before savi

        # Additional fully connected layers
        #self.fc2 = nn.Linear(in_features=100, out_features=100)
        #self.fc3 = nn.Linear(in_features=100, out_features=100)

        #self.fc4 = nn.Linear(in_features = 100, out_features = 10)

        # Output layer: num_classes for classification, 1 for regression
        #self.out = nn.Linear(in_features = 10, out_features = num_classes)

    def forward(self, x):
        # Coarsen spatial dimensions before processing
        x = self.coarsen_pool(x)

        # input lifting with configurable activation
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.act2(x)

        # Process 1: Kernel Integral Operations
        x = self.process_block1(x)
        x = self.maxpool1(x)

        # Process 2
        x = self.process_block2(x)
        x = self.maxpool2(x)

        # Process 3
        x = self.process_block3(x)
        x = self.maxpool3(x)

        # Process 4
        x = self.process_block4(x)

        # Spatial Pyramid Pooling (replaces adaptive pooling)
        x = self.spp(x)

        # Final Projection with dropout
        x = self.fc1(x)
        #x = self.relu(x)
        x = self.dropout1(x)  # Dropout after fc1

       # x = self.fc2(x)
       # x = self.relu(x)
       # x = self.dropout2(x)  # Dropout after fc2

       # x = self.fc3(x)
       # x = self.relu(x)
       # x = self.dropout3(x)  # Dropout after fc3

       # x = self.fc4(x)
       # x = self.relu(x)
       # x = self.dropout4(x)  # Dropout after fc4

       # x = self.out(x)

        return x

    def get_parameter_groups(self):
        """
        Get parameter groups for optimizer with different weight decay settings.

        Returns:
            List of parameter groups:
                - Conv weights: with L2 regularization (weight_decay = self.l2_reg)
                - BatchNorm and biases: without L2 regularization (weight_decay = 0)

        Usage example:
            # Create model with L2 regularization
            model = MoK_CNN_Predictor(in_channels=36, l2_reg=0.0001)

            # Create optimizer using parameter groups (recommended)
            optimizer = torch.optim.AdamW(model.get_parameter_groups(), lr=0.001)

            # Alternative: Apply weight_decay uniformly to all parameters
            # optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

        Note:
            Using get_parameter_groups() is recommended as it applies L2 regularization
            only to convolutional weights, not to batch norm parameters or biases.
            This is the standard practice in deep learning.
        """
        # Parameters with L2 regularization (convolutional weights)
        conv_params = []
        # Parameters without L2 regularization (batch norm, biases, output layer)
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # Apply L2 reg to conv weights only (not batch norm or biases)
            if 'conv' in name and 'weight' in name:
                conv_params.append(param)
            else:
                no_decay_params.append(param)

        return [
            {'params': conv_params, 'weight_decay': self.l2_reg},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

# We will make a create_model just as a scope for future expansion
# In the simple implementation this function is not needed
def create_model(in_channels: int = 36, dropout_rate: float = 0.5, num_classes: int = 1,
                 activation_type: str = "relu", prelu_mode: str = "shared", leaky_relu_slope: float = 0.01,
                 l2_reg: float = 0.0001, spp_option: int = 2, spp_ops: list = [1]):
    """
    Create a MoK_CNN_Predictor model with specified number of input channels, dropout rate, and output classes.

    Args:
        in_channels: Number of input channels (default: 36 for 3 time steps)
                     Calculated as: (5 surface vars + 3 pressure vars × num_pressure_levels) × num_time_steps + 3 static
                     Example: (5 + 3×2) × 3 + 3 = 36 channels for time_steps=[0,1,2] and pressure_levels=[0,1]
        dropout_rate: Dropout probability for fc1, fc2, fc3, and fc4 layers (default: 0.5)
        num_classes: Number of output classes for classification (default: 1 for regression)
                     For classification, set to number of classes (e.g., 9 for binned onset dates 0-8)
        activation_type: Type of activation function ("relu", "prelu", "leaky_relu")
        prelu_mode: PReLU parameter sharing ("shared" or "channel")
        leaky_relu_slope: Negative slope for LeakyReLU
        l2_reg: L2 regularization strength (default: 0.0001, typical: 1e-4 to 1e-5)
        spp_option: Spatial Pyramid Pooling type (0: max only, 1: avg only, 2: both max and avg)
        spp_ops: List of output pool sizes for pyramid levels (e.g., [1], [1, 2], [1, 2, 4])

    Returns:
        MoK_CNN_Predictor model instance

    Note:
        - Weight initialization should be done separately using the initialize_weights() function
          from training.utils.weight_init after model creation.
        - L2 regularization is applied by using model.get_parameter_groups() when creating the optimizer
    """
    model = MoK_CNN_Predictor(
        in_channels=in_channels,
        dropout_rate=dropout_rate,
        num_classes=num_classes,
        activation_type=activation_type,
        prelu_mode=prelu_mode,
        leaky_relu_slope=leaky_relu_slope,
        l2_reg=l2_reg,
        spp_option=spp_option,
        spp_ops=spp_ops
    )
    return model
        
        
# We will make a __main__ function to test the model in this class
# This __main__ will not be called when the model is imported elsewhere
# The utility of the following is to write a "unit test" equivalent during development

def test():
    import os
    import yaml

    try:
        from torchview import draw_graph
        torchview_available = True
    except ImportError:
        torchview_available = False
        print("Warning: torchview not available. Install with: pip install torchview")

    # Load model name and configuration from config file
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                               'config', 'model_config_with_wandb.yml')
    model_name = "MoK_CNN_Predictor"  # Default name
    num_classes = 1  # Default
    dropout_rate = 0.5  # Default
    in_channels = 36  # Default
    spatial_shape = [1440, 481]  # Default

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            model_name = config.get('model', {}).get('name', 'MoK_CNN_Predictor')
            num_classes = config.get('model', {}).get('num_classes', 1)
            dropout_rate = config.get('model', {}).get('dropout_rate', 0.5)
            spatial_shape = config.get('model', {}).get('spatial_input_shape', [1440, 481])

            # Calculate number of input channels based on data configuration
            data_config = config.get('data', {})
            surface_vars = data_config.get('input_geo_var_surf', [])
            pressure_vars = data_config.get('input_geo_var_press', [])
            time_steps = data_config.get('time_steps', [])
            pressure_levels = data_config.get('pressure_levels', [])
            include_lat = data_config.get('include_lat', False)
            include_lon = data_config.get('include_lon', False)
            include_landsea = data_config.get('include_landsea', False)

            # Calculate channels: (surface_vars × time_steps) + (pressure_vars × pressure_levels × time_steps) + static
            num_surface_channels = len(surface_vars) * len(time_steps)
            num_pressure_channels = len(pressure_vars) * len(pressure_levels) * len(time_steps)
            num_static_channels = int(include_lat) + int(include_lon) + int(include_landsea)
            in_channels = num_surface_channels + num_pressure_channels + num_static_channels

            print(f"Configuration loaded from: {config_path}")
            print(f"  Surface vars: {surface_vars} (× {len(time_steps)} time steps = {num_surface_channels} channels)")
            print(f"  Pressure vars: {pressure_vars} (× {len(pressure_levels)} levels × {len(time_steps)} time steps = {num_pressure_channels} channels)")
            print(f"  Static channels: {num_static_channels} (lat={include_lat}, lon={include_lon}, landsea={include_landsea})")
            print(f"  Total input channels: {in_channels}")
            print()

    except Exception as e:
        print(f"Warning: Could not load config file ({e}). Using default values.")
        print(f"  in_channels={in_channels}, num_classes={num_classes}, dropout_rate={dropout_rate}")
        print()

    # Create output directory for architecture visualizations
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                              'results', 'models')
    os.makedirs(output_dir, exist_ok=True)

    # Test regression model
    print("Testing Regression Model (num_classes=1):")
    print("-" * 60)

    # Read L2 regularization from config if available
    l2_reg = 0.0001  # Default
    try:
        l2_reg = config.get('training', {}).get('weight_decay', 0.0001)
    except:
        pass

    model_reg = MoK_CNN_Predictor(in_channels=in_channels, num_classes=1, l2_reg=l2_reg)
    input = torch.randn(2, in_channels, spatial_shape[0], spatial_shape[1])
    output_reg = model_reg(input)

    print(f"Input shape: {input.shape}")
    print(f"Output shape: {output_reg.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model_reg.parameters()):,}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model_reg.parameters() if p.requires_grad):,}")
    print(f"L2 regularization strength: {model_reg.l2_reg}")

    # Show parameter groups for L2 regularization
    param_groups = model_reg.get_parameter_groups()
    conv_params = sum(p.numel() for p in param_groups[0]['params'])
    other_params = sum(p.numel() for p in param_groups[1]['params'])
    print(f"  Conv params (with L2): {conv_params:,}")
    print(f"  Other params (no L2): {other_params:,}")

    # Test classification model (if configured)
    if num_classes > 1:
        print("\n" + "="*60)
        print(f"Testing Classification Model (num_classes={num_classes}):")
        print("-" * 60)
        model_cls = MoK_CNN_Predictor(in_channels=in_channels, num_classes=num_classes)
        output_cls = model_cls(input)

        print(f"Input shape: {input.shape}")
        print(f"Output shape: {output_cls.shape}")
        print(f"Number of parameters: {sum(p.numel() for p in model_cls.parameters()):,}")
        print(f"Number of trainable parameters: {sum(p.numel() for p in model_cls.parameters() if p.requires_grad):,}")
        print(f"\nFor classification, output logits for {model_cls.num_classes} classes")

    # Generate model architecture visualization
    if torchview_available:
        print("\n" + "="*60)
        print("Generating Model Architecture Visualization:")
        print("-" * 60)

        filename = f"{model_name}_architecture"

        try:
            model_graph = draw_graph(
                model_reg,
                input_size=(2, in_channels, spatial_shape[0], spatial_shape[1]),
                expand_nested=True,
                graph_name=f'{model_name}_Architecture',
                save_graph=True,
                filename=filename,
                directory=output_dir,
            )
            output_path = os.path.join(output_dir, f"{filename}.png")
            print(f"Model architecture saved as '{output_path}'")
            print("Visualization includes layer dimensions and parameter counts")
        except Exception as e:
            print(f"Error generating visualization: {e}")
            print("Attempting alternative method...")

            try:
                # Alternative: simpler visualization
                filename_simple = f"{model_name}_architecture_simple"
                model_graph = draw_graph(
                    model_reg,
                    input_data=torch.randn(1, in_channels, spatial_shape[0], spatial_shape[1]),
                    expand_nested=False,
                    save_graph=True,
                    filename=filename_simple,
                    directory=output_dir,
                )
                output_path = os.path.join(output_dir, f"{filename_simple}.png")
                print(f"Simplified model architecture saved as '{output_path}'")
            except Exception as e2:
                print(f"Alternative method also failed: {e2}")

if __name__ == "__main__":
    test()

# Explanation: In Python, when this file cnn.py is run directly, the special string __name__ is set to __main__. Else it will be set to what we are importing.
# By comparing __name__ with __main__, we get a pythonic way to testing this code directly.
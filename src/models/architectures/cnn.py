import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # Now repear conv, bn. ReLU can be reused as it has no learnable parameters (it is simply a function application)

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
    Puts together the CNN model
    Input: 13 variables (sst, t2m, msl, ttr, tcc, u1, u2, v1, v2, z1, z2, lat, lon)
           x 3 time steps = 39 channels
    Input spatial dimensions: 1440 x 481

    Architecture:
        - Uses residual blocks to progressively increase channels (39 -> 64 -> 128)
        - Max pooling reduces spatial dimensions by factor of 2 each stage
        - Final flatten to 200 units using an adaptive avg pooling so that arbitrary input spatial dimensions can be used
        - Two dense layers to produce single output
    """

    def __init__(self):
        super().__init__() # call the constructor of nn.Module to ensure all params are tracked

        # Initial convolution layer to lift 39 channels to 64 channels
        # TO DO: Change 64 to be read from the config file
        # group = 3 processes each variable separately as we are stacking channels each variable on each month 
        # TO DO: Change out_channels

        self.conv1 = nn.Conv2d(in_channels = 39, out_channels = 39, groups = 3, kernel_size = 3, padding = (1,0), padding_mode = 'circular', bias = True )
        self.conv2 = nn.Conv2d(in_channels = 39, out_channels = 64, kernel_size = 1, bias = False )
        self.bn1 = nn.BatchNorm2d(num_features = 64)

        self.relu = nn.ReLU(inplace=True) # will be reused everytime nonlinearity is needed

        self.process_block1 = ResSpatialProcessorBlock(64, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.process_block2 = ResSpatialProcessorBlock(64, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.process_block3 = ResSpatialProcessorBlock(128, 128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 2))

        # Flatten does not need a layer definition, but we will define for style consistency
        self.flatten = nn.Flatten(start_dim=1) # keeps the batch axis=0 and flattens the rest

        #Now we have to add linear layers. Ideally we must calculate the number of output from the flatten, but we are lazy. So we do the following.
        self.fc1 = nn.LazyLinear(out_features=100) # LazyLinear does not require explicit computation of the input feature size. CAUTION: Run forward by invoking the class before savi
        
        self.fc2 = nn.Linear(in_features = 100, out_features = 10)

        self.out = nn.Linear(in_features = 10, out_features = 1)

    def forward(self, x):
        # input lifting

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Process 1: Kernel Integral Operations
        x = self.process_block1(x)
        x = self.maxpool1(x)

        # Process 2
        x = self.process_block2(x)
        x = self.maxpool2(x)

        # Process 3
        x = self.process_block3(x)
        x = self.maxpool3(x)

        # Adaptive Pooling
        x = self.adaptive_pool(x)

        # Flatten
        x = self.flatten(x)

        # Final Projection
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.out(x)

        return x

# We will make a create_model just as a scope for future expansion
# In the simple implementation this function is not needed
def create_model():
    return MoK_CNN_Predictor()
        
        
# We will make a __main__ function to test the model in this class
# This __main__ will not be called when the model is imported elsewhere
# The utility of the following is to write a "unit test" equivalent during development

def test():
    model = MoK_CNN_Predictor()
    input = torch.randn(2, 39, 1440, 481)
    output = model(input)

    print(f"Input shape: {input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

if __name__ == "__main__":
    test()

# Explanation: In Python, when this file cnn.py is run directly, the special string __name__ is set to __main__. Else it will be set to what we are importing.
# By comparing __name__ with __main__, we get a pythonic way to testing this code directly.
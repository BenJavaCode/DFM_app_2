"""
Convolutional model, with residual skip connections, to alleviate the degradation problem.

"""



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ResidualBlock(nn.Module):
    """
    Residual block. Defines the layers of a resiudal block,
    And also the method, for reshaping x to be of same shape as F(x)
    """
    def __init__(self, in_channels, out_channels, stride=1):
        """
        __init__(in_channels, out_channels, stride)
        Description: Defines internal variables, and method for reshaping x to same size as F(x)
        Params: in_channels = Input channels. Self.in_channels from ResNet class.
                out_channels = Output channels. hidden_size item.
                stride = Convolution stride.  A Resnet _make_layer strides item.
        Latest update: 13-06-2021. Added more comments.
        """
        super(ResidualBlock, self).__init__()  # Inherit nn.module methods and vars

        # LAYERS
        self.strideprint = stride
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5,
                               stride=stride, padding=2, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5,
                               stride=1, padding=2, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        # -

        # SKIP CONNECTION
        self.shortcut = nn.Sequential()  # When x and F(x) is the same shape.

        # When x and F(x) is not the same shape.
        # If stride is not == 1, then the number of features will have halved(because it will be 2).
        # Do to [(w-k+2p)/s]+1 = n_features. Where w = input features, k = kernel size, p = padding, s = stride.
        # If in_channels != out_channels, number of filters will not be equal, and this needs to be corrected.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels))
        # -

    # Sequential forward function
    def forward(self, x):
        """
        forward(x)
        Description: Defines the sequence, that the layers will be executed in.
        Params: x = Input.
        Latest update: 13-06-2021. Added more comments
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # F(x) + x (This is the skip connection)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    Classifier that used Convolutions for spatial differentiation, and skip connections, for identity functions.
    """
    def __init__(self, hidden_sizes, num_blocks, input_dim=1000,
                 in_channels=64, n_classes=30):
        """
        __init__(hidden_sizes, num_blocks, input_dim, in_channels, n_classes)
        Description: Defines internal variables, and creates a sequential containing all but the last layer.
        Params: hidden_sizes = Used to define length of strides var, also used to define output channel size of
                               residual block in _make_layer. Also used, to define number of residual blocks.
                num_blocks = Used for _make_layer internal strides list.
                input_dim = Is the len of a input datapoint. Used for _get_encoding_size to define last dimension.
                in_channels = Initial filter size.
                n_classes = number of classes.
        """
        super(ResNet, self).__init__()  # Inherit nn.Module methods and vars.
        assert len(num_blocks) == len(hidden_sizes)  # If condition returns False, AssertionError is raised.

        self.input_dim = input_dim  # Used to get input size for output layer (Used by _get_encoding_size)
        self.in_channels = in_channels  # Input size
        self.n_classes = n_classes  # Number of classes

        # First layer has 1 input channel
        self.conv1 = nn.Conv1d(1, self.in_channels, kernel_size=5, stride=1,
                               padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)

        # FLEXIBLE NUMBER OF RESIDUAL ENCODING LAYERS
        strides = [1] + [2] * (len(hidden_sizes) - 1)  # Stride of first layer: 1, all others: 2
        layers = []
        for idx, hidden_size in enumerate(hidden_sizes):
            layers.append(self._make_layer(hidden_size, num_blocks[idx],
                                           stride=strides[idx]))
        self.encoder = nn.Sequential(*layers)  # Unpack, as sequential does not take list.
        # -

        self.z_dim = self._get_encoding_size()  # Input size for output layer.
        self.linear = nn.Linear(self.z_dim, self.n_classes)  # Output layer.

    def encode(self, x):
        """
        encode(x)
        Description: Makes layer-structure, from start until the output layer(Linear)
        Params: x = the input to the model.
        Latest update: 13-06-2021. Added more comments.
        """
        x = F.relu(self.bn1(self.conv1(x)))  # First layer, no residual connections.
        x = self.encoder(x)  # Residual blocks.

        # "reshape" tensor into size [0]: x.size(0), [1]: x.size(1:)
        # This is so it is a [?,?] shape tensor, that is suitable for linear.
        z = x.view(x.size(0), -1)
        return z

    def forward(self, x):
        """
        forward(x)
        Description: Defines the sequence, that the layers will be executed in.
        Params: x = The model input.
        Latest update: 13-06-2021. Added more comments
        """
        z = self.encode(x)  # Run input through hidden layers.
        return self.linear(z)  # Output layer. Has z_dim as input dim and n_classes as output dim.

    def _make_layer(self, out_channels, num_blocks_val, stride=1):
        """
        _make_layer(out_channels, num_block_val, stride)
        Description: For making a residual block.
        Params: out_channels = Output channels. Is a item in hidden_sizes.
                num_blocks_val = Used for the second blocks stride. Is an item from num_blocks
                stride = Used for first block stride. Is an item from strides.
        Latest update: 13-06-2021. Added more comments.
                                   Changed name of num_blocks_val, from num_blocks.
        """
        strides = [stride] + [1] * (num_blocks_val - 1)  # List of stride value and num_blocks value -1. eg [2,1]
        blocks = []
        # Make residual block
        for stride in strides:
            blocks.append(ResidualBlock(self.in_channels, out_channels, stride=stride))
            self.in_channels = out_channels
        return nn.Sequential(*blocks)  # Return a sequential

    def _get_encoding_size(self):
        """
        _get_encoding_size()
        Description: Returns the dimension of the encoded input.
                    Does this by creating a dummy tensor with only one datapoint.
                    And then sending this, through the layers, to similate a real run,
                    and get the viewed [?,?] dim tensor.
                    That is fed, to the output layer (Linear).
                    Lastly takes, the size at position 1.
                    This var named z_dim, is used to define the input size to the linear layer.
        Last updated: 13-06-2021. Added more comments.
        """
        temp = Variable(torch.rand(1, 1, self.input_dim))  # Tensor wrap containing 1 datapoint, with input len
        z = self.encode(temp)  # Go though layers, to get same output in last dimension as when, real data go through
        z_dim = z.data.size(1)  # Get size of tensor at position 1
        return z_dim


def add_activation(activation='relu'):
    """
    add_activation(activation)
    Description: Adds specified activation layer, choices include:
    - 'relu'
    - 'elu' (alpha)
    - 'selu'
    - 'leaky relu' (negative_slope)
    - 'sigmoid'
    - 'tanh'
    - 'softplus' (beta, threshold)
    Params: activation =  The activation function, you would like to use.
    Last updated: 13-06-2021. Added more comments.
    """
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'elu':
        return nn.ELU(alpha=1.0)
    elif activation == 'selu':
        return nn.SELU()
    elif activation == 'leaky relu':
        return nn.LeakyReLU(negative_slope=0.1)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    # SOFTPLUS DOESN'T WORK with automatic differentiation in pytorch (autograd wont work)
    elif activation == 'softplus':
        return nn.Softplus(beta=1, threshold=20)
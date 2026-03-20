import torch.nn as nn
from torch import Tensor

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )

class WideResBlock(nn.Module):
    """
    A residual block
    """

    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()

        # the input is only added back to the output if they are of the same size
        self.io_same_size = (in_channels == out_channels) and (stride == 1)
        # if input and output are of the same size, add the original signal. Else, downsample the signal by a convolution
        self.res = nn.Identity() if self.io_same_size else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False,)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True) #inplace=True means change the input in place
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        # only apply dropout if dro[out rate is non-zero
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        output = self.relu1(self.bn1(x))
        if not self.io_same_size: x = output # get x ready for the possible downsampling
        res_add = self.res(x)

        output = self.conv1(output)
        output = self.relu2(self.bn2(output))
        output = self.dropout(output)
        output = self.conv2(output)

        return output + res_add

class WideResNet(nn.Module):
    """
    a list over several residual blocks
    """
    def __init__(self, num_blocks, in_channels, out_channels, stride, dropout):
        super().__init__()

        layers = []
        for i in range(num_blocks):
            # Notice that only the first block changes the channels or the size
            # Subsequent blocks stay the same channel and same size
            layers.append(
                WideResBlock(
                in_channels=in_channels if i == 0 else out_channels, 
                out_channels=out_channels,
                stride=stride if i == 0 else 1,
                dropout=dropout)
            )
        self.layers = nn.Sequential(*layers) # stack them all together
    
    def forward(self, x: Tensor) -> Tensor: return self.layers(x)

class CustomModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # YOUR CODE BEGIN.
        depth = 16 # how many res layers the whole network has
        # this is what wide ResNet is different from general ResNets
        widen_factor = 8 # multiplies the channel width of the main stages, larger means each layers have more feature channels
        conv1_channels = 16
        dropout = 0.1

        # as we'll have 3 stages, each stage has num_blocks blocks,
        # each block has 2 convs, also there're 4 fixed layers (bn, relu, pool, fc)
        # so depth = 6 * (num_blocks) + 4
        num_blocks = (depth - 4) // 6

        channels = [
            conv1_channels,
            16 * widen_factor,
            32 * widen_factor,
            64 * widen_factor
        ]
        layers = []

        # 1 conv1 layer and three detail layers
        # the three layers learn the details from narrow to wide features
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, conv1_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv1_channels),
            nn.ReLU(inplace=True)
        )
        for i in range(3):
            layers.append(
                WideResNet(
                    num_blocks=num_blocks,
                    in_channels=channels[i],out_channels=channels[i+1],
                    stride=1 if i == 0 else 2, 
                    dropout=dropout,
                )
            )
        self.layers = nn.Sequential(*layers)

        # 4 fixed layers
        self.bn=nn.BatchNorm2d(channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1,1)) # reduce the size to (1,1)
        self.fc = nn.Linear(channels[3], 200) # final output classes is 200

        self._init_weights()
        # YOUR CODE END.

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d): # use Kaiming Initialization for Conv2s
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)
            else: pass

    def forward(self, x: Tensor) -> Tensor:
        # YOUR CODE BEGIN.
        x = self.conv1(x)
        x = self.layers(x)
        x = self.relu(self.bn(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
        # YOUR CODE END.
import torch.nn as nn
from torch import flatten
from torch import Tensor

def conv3x3(in_channels, out_channels, stride=1, groups=1):
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )

class CustomModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.inplanes = 64
        self.groups = 16
        self.width_per_group = 4

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, 200)


    def forward(self, x: Tensor) -> Tensor:
       x = self.conv1(x)
       x = self.bn1(x)
       x = self.relu(x)

       x = self.layer1(x)
       x = self.layer2(x)
       x = self.layer3(x)
       x = self.layer4(x)

       x = self.avgpool(x)
       x = flatten(x, 1)
       x = self.fc(x)
       
       return x
    
    def _make_layer(self, planes, blocks, stride):
        downsample = None
        out_channels = planes * Bottleneck.expansion

        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, out_channels, stride),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        layers.append(
            Bottleneck(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                groups=self.groups,
                width_per_group=self.width_per_group,
            )
        )

        self.inplanes = out_channels

        for _ in range(1, blocks):
            layers.append(
                Bottleneck(
                    self.inplanes,
                    planes,
                    downsample=None,
                    groups=self.groups,
                    width_per_group=self.width_per_group,
                )
            )

        return nn.Sequential(*layers)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            groups=16,
            width_per_group=4,
    ):
        super().__init__()
        width = int(planes * width_per_group / 64.0) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)

        self.conv2 = conv3x3(width, width, stride=stride, groups=groups)
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
    

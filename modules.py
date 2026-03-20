import torch.nn as nn
from torch import Tensor


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class WideBasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()

        self.equal_in_out = (in_channels == out_channels) and (stride == 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.conv2 = conv3x3(out_channels, out_channels)
        self.shortcut = (
            nn.Identity()
            if self.equal_in_out
            else nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.equal_in_out:
            out = self.relu1(self.bn1(x))
            shortcut = x
        else:
            x = self.relu1(self.bn1(x))
            out = x
            shortcut = self.shortcut(x)

        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        out = self.dropout(out)
        out = self.conv2(out)

        return out + shortcut


class NetworkBlock(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        in_channels: int,
        out_channels: int,
        stride: int,
        dropout_p: float,
    ) -> None:
        super().__init__()

        layers = []
        for block_idx in range(num_blocks):
            layers.append(
                WideBasicBlock(
                    in_channels=in_channels if block_idx == 0 else out_channels,
                    out_channels=out_channels,
                    stride=stride if block_idx == 0 else 1,
                    dropout_p=dropout_p,
                )
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class CustomModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        depth = 16
        widen_factor = 8
        stem_channels = 16
        dropout_p = 0.1

        num_blocks = (depth - 4) // 6
        channels = [
            stem_channels,
            16 * widen_factor,
            32 * widen_factor,
            64 * widen_factor,
        ]

        # Lighter stem plus immediate spatial reduction for better throughput on 64x64.
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )
        self.layer1 = NetworkBlock(
            num_blocks=num_blocks,
            in_channels=channels[0],
            out_channels=channels[1],
            stride=1,
            dropout_p=dropout_p,
        )
        self.layer2 = NetworkBlock(
            num_blocks=num_blocks,
            in_channels=channels[1],
            out_channels=channels[2],
            stride=2,
            dropout_p=dropout_p,
        )
        self.layer3 = NetworkBlock(
            num_blocks=num_blocks,
            in_channels=channels[2],
            out_channels=channels[3],
            stride=2,
            dropout_p=dropout_p,
        )
        self.bn = nn.BatchNorm2d(channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[3], 200)

        self._init_weights()

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.relu(self.bn(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

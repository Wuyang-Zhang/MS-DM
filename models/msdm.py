"""MS-DM network for multi-species pest counting.

The model uses a shared VGG19 feature extractor and two density-regression
branches:

* Whitefly branch: FPN-style fusion of 1/16 and 1/8 scale features.
* Fruit-fly branch: ASPP context extraction followed by CBAM attention.

Both branches produce a non-negative density map and its normalized spatial
distribution. Summing a density map gives the predicted object count, while
the normalized map is used by the optimal-transport and total-variation losses.
"""

from typing import Iterable, List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

__all__ = ["CBAM", "ASPP", "VGG", "vgg19"]

VGG19_URL = "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"
VGG19_CONFIG = [
    64, 64, "M",
    128, 128, "M",
    256, 256, 256, 256, "M",
    512, 512, 512, 512, "M",
    512, 512, 512, 512,
]

LayerConfig = Sequence[Union[int, str]]
ModelOutputs = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _initialize_module(module: nn.Module) -> None:
    """Apply the initialization scheme used by the original implementation."""
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.BatchNorm2d):
            init.constant_(layer.weight, 1)
            init.constant_(layer.bias, 0)


class CBAM(nn.Module):
    """Channel-guided attention module used by the fruit-fly branch.

    This class preserves the attention formulation from the published project:
    pooled channel descriptors produce channel weights, after which the
    channel-refined and original features are concatenated to produce the final
    per-position, per-channel attention mask.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if reduction <= 0 or channels // reduction == 0:
            raise ValueError("reduction must produce at least one hidden channel")

        hidden_channels = channels // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_after_concat = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        average_descriptor = self.fc2(self.relu(self.fc1(self.avg_pool(features))))
        maximum_descriptor = self.fc2(self.relu(self.fc1(self.max_pool(features))))
        channel_weights = self.sigmoid_channel(average_descriptor + maximum_descriptor)

        channel_refined = features * channel_weights
        combined = torch.cat((channel_refined, features), dim=1)
        attention = self.sigmoid_spatial(self.conv_after_concat(combined))
        return features * attention


class ASPPConv(nn.Sequential):
    """A 3x3 atrous-convolution branch in ASPP."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class ASPPPooling(nn.Sequential):
    """Global-context pooling branch in ASPP."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output_size = features.shape[-2:]
        pooled = features
        for layer in self:
            pooled = layer(pooled)
        return F.interpolate(
            pooled, size=output_size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling with local and global context branches."""

    def __init__(
        self,
        in_channels: int,
        atrous_rates: Iterable[int],
        out_channels: int = 512,
    ) -> None:
        super().__init__()

        branches: List[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        ]
        branches.extend(
            ASPPConv(in_channels, out_channels, rate)
            for rate in tuple(atrous_rates)
        )
        branches.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.Conv2d(
                len(self.convs) * out_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # Retained to preserve the random-initialization sequence of the
        # original implementation. The parent model initializes all modules a
        # second time after every branch and head has been constructed.
        _initialize_module(self)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        multi_scale_features = [branch(features) for branch in self.convs]
        return self.project(torch.cat(multi_scale_features, dim=1))


class VGG(nn.Module):
    """Final MS-DM network built on a convolutional VGG19 backbone.

    The class name is kept as ``VGG`` for checkpoint and API compatibility.
    Inputs may have any spatial size supported by the backbone, but training
    uses dimensions divisible by 16. Returned density maps are at 1/8 scale.
    """

    def __init__(self, features: nn.Sequential) -> None:
        super().__init__()
        self.features = features

        # Keep these registered names stable: existing checkpoints depend on
        # the resulting state_dict keys.
        self.cbam = CBAM(channels=512)
        self.aspp = ASPP(512, [1, 2, 5], 512)

        self.reg_layer1 = self._make_regression_layer()
        self.density_layer1 = self._make_density_head()
        self.reg_layer2 = self._make_regression_layer()
        self.density_layer2 = self._make_density_head()

        # Historical checkpoint compatibility: this module was registered in
        # the published code and therefore appears in trained state_dicts, but
        # it is not called by forward(). Removing it would break strict loading.
        self.convs_bn = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        _initialize_module(self)

    @staticmethod
    def _make_regression_layer() -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _make_density_head() -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ReLU(),
        )

    @staticmethod
    def _normalize_density(density: torch.Tensor) -> torch.Tensor:
        batch_size = density.shape[0]
        total_mass = density.reshape(batch_size, -1).sum(dim=1).view(
            batch_size, 1, 1, 1)
        return density / (total_mass + 1e-6)

    def forward(self, image: torch.Tensor) -> ModelOutputs:
        """Predict whitefly and fruit-fly density distributions."""
        stage1 = self.features[:4](image)          # 64 channels, 1x scale
        stage2 = self.features[4:9](stage1)        # 128 channels, 1/2 scale
        stage3 = self.features[9:18](stage2)       # 256 channels, 1/4 scale
        stage4 = self.features[18:27](stage3)      # 512 channels, 1/8 scale
        stage5 = self.features[27:35](stage4)      # 512 channels, 1/16 scale

        # Whitefly branch: FPN-style lateral fusion at 1/8 scale.
        whitefly_features = F.interpolate(
            stage5, scale_factor=2, mode="nearest") + stage4
        whitefly_features = self.reg_layer1(whitefly_features)
        whitefly_density = self.density_layer1(whitefly_features)
        whitefly_normalized = self._normalize_density(whitefly_density)

        # Fruit-fly branch: multi-scale context followed by attention.
        fruit_fly_features = self.aspp(stage5)
        fruit_fly_features = F.interpolate(
            fruit_fly_features, scale_factor=2, mode="nearest")
        fruit_fly_features = self.cbam(fruit_fly_features)
        fruit_fly_features = self.reg_layer2(fruit_fly_features)
        fruit_fly_density = self.density_layer2(fruit_fly_features)
        fruit_fly_normalized = self._normalize_density(fruit_fly_density)

        return (
            whitefly_density,
            whitefly_normalized,
            fruit_fly_density,
            fruit_fly_normalized,
        )


def make_layers(config: LayerConfig, batch_norm: bool = False) -> nn.Sequential:
    """Build the convolutional VGG19 feature extractor."""
    layers: List[nn.Module] = []
    in_channels = 3

    for value in config:
        if value == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue

        out_channels = int(value)
        convolution = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1)
        layers.append(convolution)
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels

    return nn.Sequential(*layers)


def vgg19(pretrained: bool = True) -> VGG:
    """Create the MS-DM model, optionally initializing VGG19 from ImageNet."""
    model = VGG(make_layers(VGG19_CONFIG))
    if pretrained:
        model.load_state_dict(model_zoo.load_url(VGG19_URL), strict=False)
    return model


def print_model(model: nn.Module, output_path: str) -> None:
    """Write a readable module-by-module model description to a text file."""
    with open(output_path, "w", encoding="utf-8") as stream:
        for index, module in enumerate(model.modules()):
            stream.write("Layer {} ({})\n".format(index, module.__class__.__name__))
            stream.write("-" * 50 + "\n")
            stream.write(str(module) + "\n")
            stream.write("-" * 50 + "\n")
    print("Model structure written to {}".format(output_path))


if __name__ == "__main__":
    print_model(vgg19(), "model_structure.txt")

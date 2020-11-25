import torch
from dropblock import DropBlock2D, LinearScheduler
from torch import nn, Tensor
from typing import Callable, Optional


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        network_type: Optional[str] = "plain",
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer=None,
        output_block={"class": None, "params": {}},
    ) -> None:
        super().__init__()
        self.network_type = network_type

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU(inplace=True)
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        if self.network_type in ("preact", "pyramid"):
            self.bn1 = norm_layer(inplanes)
        else:
            self.bn1 = norm_layer(planes)
        self.activation_layer = activation_layer
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        if self.network_type == "pyramid":
            self.bn3 = norm_layer(planes * self.expansion)
        self.output_block = (
            None
            if output_block["class"] is None
            else output_block["class"](
                planes * self.expansion, **output_block["params"]
            )
        )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = x
        if self.network_type in ("preact", "pyramid"):
            out = self.bn1(out)
            if self.network_type == "preact":
                out = self.activation_layer(out)
        out = self.conv1(out)
        if self.network_type == "plain":
            out = self.bn1(out)
            out = self.activation_layer(out)

        if self.network_type in ("preact", "pyramid"):
            out = self.bn2(out)
            out = self.activation_layer(out)
        out = self.conv2(out)
        if self.network_type == "plain":
            out = self.bn2(out)

        if self.network_type == "pyramid":
            out = self.bn3(out)

        if self.output_block is not None:
            out = self.output_block(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        if self.network_type == "plain":
            out = self.activation_layer(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        network_type="plain",
        norm_layer=None,
        activation_layer=None,
        output_block={"class": None, "params": {}},
    ):
        super().__init__()
        self.network_type = network_type
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU(inplace=True)
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        if self.network_type in ("preact", "pyramid"):
            self.bn1 = norm_layer(inplanes)
        else:
            self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        if self.network_type in ("preact", "pyramid"):
            self.bn3 = norm_layer(width)
        else:
            self.bn3 = norm_layer(planes * self.expansion)
        if self.network_type == "pyramid":
            self.bn4 = norm_layer(planes * self.expansion)
        self.activation_layer = activation_layer
        self.output_block = (
            None
            if output_block["class"] is None
            else output_block["class"](
                planes * self.expansion, **output_block["params"]
            )
        )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = x
        if self.network_type in ("preact", "pyramid"):
            out = self.bn1(out)
            if self.network_type == "preact":
                out = self.activation_layer(out)
        out = self.conv1(out)
        if self.network_type == "plain":
            out = self.bn1(out)
            out = self.activation_layer(out)

        if self.network_type in ("preact", "pyramid"):
            out = self.bn2(out)
            out = self.activation_layer(out)
        out = self.conv2(out)
        if self.network_type == "plain":
            out = self.bn2(out)
            out = self.activation_layer(out)

        if self.network_type in ("preact", "pyramid"):
            out = self.bn3(out)
            out = self.activation_layer(out)
        out = self.conv3(out)
        if self.network_type == "plain":
            out = self.bn3(out)

        if self.network_type == "pyramid":
            out = self.bn4(out)

        if self.output_block is not None:
            out = self.output_block(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        if self.network_type == "plain":
            out = self.activation_layer(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        network_type="plain",
        norm_layer=None,
        activation_layer=None,
        output_block={"class": None, "params": {}},
        dropblock={"drop_prob": 0.0},
    ):
        super().__init__()
        self.network_type = network_type

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        if activation_layer is None:
            activation_layer = nn.ReLU(inplace=True)

        self.output_block = output_block

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.activation_layer = activation_layer
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if dropblock["drop_prob"] != 0.0 and dropblock.get("max_steps", None) is None:
            raise ValueError("Max steps must be specified when using dropblock")

        self.dropblock = LinearScheduler(
            DropBlock2D(block_size=dropblock.get("block_size", 7), drop_prob=0.0),
            start_value=0.0,
            stop_value=dropblock["drop_prob"],
            nr_steps=dropblock.get("max_steps", 0),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if (isinstance(m, Bottleneck) and self.network_type == "plain") or (
                    isinstance(m, BasicBlock) and self.network_type == "pyramid"
                ):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, Bottleneck) and self.network_type == "pyramid":
                    nn.init.constant_(m.bn4.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                self.network_type,
                norm_layer,
                self.activation_layer,
                self.output_block,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    network_type=self.network_type,
                    norm_layer=norm_layer,
                    activation_layer=self.activation_layer,
                    output_block=self.output_block,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation_layer(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.dropblock(self.layer3(x))
        x = self.dropblock(self.layer4(x))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

    def step(self):
        self.dropblock.step()


def _resnet(arch, block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def custom_resnet18(**kwargs):
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], **kwargs)


def custom_resnet34(**kwargs):
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], **kwargs)


def custom_resnet50(**kwargs):
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], **kwargs)

from .output_block import OutputBlock
from torch import nn


class AccuracyBoosterBlock(OutputBlock):
    def __init__(
        self,
        inplanes,
        pooling_layer=nn.AdaptiveAvgPool2d(1),
        activation_layer=nn.Sigmoid(),
        plus=False,
    ):
        super().__init__(
            inplanes,
            nn.Sequential(
                nn.Conv2d(
                    inplanes, inplanes, 1, groups=1 if plus else inplanes, bias=False
                ),
                nn.BatchNorm2d(inplanes),
            ),
            pooling_layer=pooling_layer,
            activation_layer=activation_layer,
        )


class AccuracyBoosterPlusBlock(AccuracyBoosterBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, plus=True)

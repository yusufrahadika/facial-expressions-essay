from .output_block import OutputBlock
from torch import nn


# class SqueezeExcitationBlock(nn.Module):
#     def __init__(
#         self,
#         inplanes,
#         ratio=16,
#         pooling_layer=nn.AdaptiveAvgPool2d(1),
#         mid_activation_layer=nn.ReLU(inplace=True),
#         activation_layer=nn.Sigmoid(),
#     ):
#         super().__init__()

#         self.pooling_layer = pooling_layer
#         self.fc1 = nn.Linear(inplanes, inplanes // ratio, bias=False)
#         self.mid_activation_layer = mid_activation_layer
#         self.fc2 = nn.Linear(inplanes // ratio, inplanes, bias=False)
#         self.activation_layer = activation_layer

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         w = self.pooling_layer(x).view(-1)
#         w = self.fc1(w)
#         w = self.mid_activation_layer(w)
#         w = self.fc2(w)
#         w = self.activation_layer(w)

#         return x * w.view(b, c, 1, 1)


class SqueezeExcitationBlock(OutputBlock):
    def __init__(
        self,
        inplanes,
        ratio=16,
        pooling_layer=nn.AdaptiveAvgPool2d(1),
        mid_activation_layer=nn.ReLU(inplace=True),
        activation_layer=nn.Sigmoid(),
    ):
        super().__init__(
            inplanes,
            nn.Sequential(
                nn.Conv2d(inplanes, inplanes // ratio, 1, bias=False),
                mid_activation_layer,
                nn.Conv2d(inplanes // ratio, inplanes, 1, bias=False),
            ),
            pooling_layer=pooling_layer,
            activation_layer=activation_layer,
        )

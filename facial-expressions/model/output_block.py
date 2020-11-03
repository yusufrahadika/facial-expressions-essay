from torch import nn


class OutputBlock(nn.Module):
    def __init__(
        self,
        inplanes,
        mid_conn,
        pooling_layer=nn.AdaptiveAvgPool2d(1),
        activation_layer=nn.Sigmoid(),
    ):
        super().__init__()

        self.pooling_layer = pooling_layer
        self.mid_conn = mid_conn
        self.activation_layer = activation_layer

    def forward(self, x):
        w = self.pooling_layer(x)
        w = self.mid_conn(w)
        w = self.activation_layer(w)

        return x * w

from torch import nn
from utils import ResidualConnection

def SingleBlockDownSampling(in_channels, out_channels, pooling: int = None):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 0, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace = True),
        nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace = True),
    )
    if pooling is not None:
        block.add_module(f'Downsampling: (size, size) -> (size/{pooling}, size/{pooling})', nn.AvgPool2d(pooling, pooling))
    return block

def SingleBlockUpSampling(in_channels, out_channels):
    block = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 1,1,0, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ConvTranspose2d(out_channels, out_channels, 3,1,0, bias = False),
        nn.SiLU(True)
    )
    return block

class Backbone(nn.Module):
    def __init__(self, architecture: SingleBlockDownSampling, in_channels: int = 1, mid_layers: tuple[int] = (32, 64, 128)):
        super().__init__()
        self.residual_connections = nn.ModuleList([ResidualConnection(None, None) for _ in mid_layers])
        layers = list()
        for out_channel in mid_layers:
            architecture(in_channels, out_channel)
            in_channels = out_channel
        self.layers = nn.ModuleList(layers)
    def forward(self, x):
        for layer, connection in zip(self.layers, self.residual_connections):
            x = connection(x, lambda x: layer(x))
        return x

class ReconstructionAutoencoder(nn.Module):
    def __init__(
            self
    ):
        super().__init__()
        self.encoder = Backbone(SingleBlockDownSampling)
        self.decoder = Backbone(SingleBlockUpSampling, 128, (64, 32, 1))
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
        
        
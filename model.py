from utils import *

class CoronagraphReconstructor(TrainingPhase):
    def __init__(
            self,
            num_layers: int = 3,
            in_channels: int = 1,
            mid_channels: tuple[int, int, int] = (64,128,256)
    ):
        assert (num_layers == len(mid_channels)), 'Channel sizes must be same size as num_layers'
        super().__init__()
        ## Defining the amount of residual connections
        self.residual_connections = nn.ModuleList([ResidualConnection(None, None) for _ in mid_channels])
        
        ## Defining the amount of single blocks
        layers = list()

        for channel_size in mid_channels:
            layers.append(SingleBlock(in_channels, channel_size, ))
            in_channels = channel_size

        self.layers = nn.ModuleList(layers)

    def forward(self, batch):
        x, mask = batch['x'], batch['mask']

        for layer, connection in zip(self.layers, self.residual_connections):
            x = connection(x, lambda x: layer(x, mask))
        
        return x, mask if self.return_mask else x
    

    
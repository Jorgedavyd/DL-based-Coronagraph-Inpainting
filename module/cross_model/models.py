# General utils
from typing import List
from torch import Tensor
from torch import nn
from torch.fft import fftn, ifftn
import torch
from lightning.pytorch import LightningModule

# Module utils
from .loss import Loss
from ..utils import (
    ComplexBatchNorm,
    ComplexReLU,
    ComplexReLU6,
    ComplexSiLU,
    ComplexSigmoid,
    ComplexMaxPool2d,
    _FourierConv,
    fourier_conv2d,
    ComplexUpsamplingBilinear2d
)

VALID_OPTIMIZERS = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
    'rms': torch.optim.RMSprop
}

class SingleFourierBlock(nn.Module):
    def __init__(
        self,
        in_channels: Tensor,
        height: Tensor,
        width: Tensor,
        layers: int,
        bias: bool = True,
        activation: str = None,
        pool: int = None
    ) -> None:
        super().__init__()
        self.layer = nn.Sequential()
        
        # pooling layer
        if pool is not None:
            self.layer.add_module('Max pool', ComplexMaxPool2d(pool, pool))

        for layer in range(layers):
            self.layer.add_module(f'FourierConv_{layer}', _FourierConv(in_channels, height, width, bias))
        
        if activation is not None:
            match activation:
                case "relu":
                    self.layer.add_module('ReLU', ComplexReLU())
                case "relu6":
                    self.layer.add_module('ReLU6' , ComplexReLU6())
                case "silu":
                    self.layer.add_module('SiLU', ComplexSiLU())
                case "sigmoid":
                    self.layer.add_module('Sigmoid', ComplexSigmoid())

        self.layer.add_module('Batch Norm', ComplexBatchNorm(in_channels))


    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


class UpsamplingFourierBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        height: int,
        width: int,
        bias: bool,
        layers: int,
        activation: str,
        pool = None,
    ) -> None:
        super().__init__()
        self.conv_layers = nn.ModuleList([])

        if pool is not None:
            # Upsamling for both spaces
            self.upsampling = ComplexUpsamplingBilinear2d(scale_factor=pool, mode = 'bilinear')
        else:
            self.upsampling = nn.Identity()

        for _ in range(layers):
            self.conv_layers.append(_FourierConv(in_channels, height, width, bias))

        match activation:
            case "relu":
                self.activation = ComplexReLU()
            case "relu6":
                self.activation = ComplexReLU6()
            case "silu":
                self.activation = ComplexSiLU()
            case "sigmoid":
                self.activation = ComplexSigmoid()
            case 'None':
                self.activation = nn.Identity()
            case None:
                self.activation = nn.Identity()
        # Batch normalization
        self.norm = ComplexBatchNorm(in_channels)
        

    def forward(self, x: Tensor, weight: Tensor) -> Tensor:
        # Upsampling
        if self.upsampling is not None:
            x = self.upsampling(x)
        # Forward with residual weight
        res = fourier_conv2d(x, weight)
        # New layers
        for layer in self.conv_layers:
            x = layer(x)
        if self.activation is not None:
            x = self.activation(x)
        x = self.norm(x)
        # Add residual
        x += res
        return x
        

class FourierVAE(LightningModule):
    def __init__(self, **hparams) -> None:
        super().__init__()
        # Setting hyperparameters values
        for k, v in hparams.items():
            setattr(self, k, v)
                
        # Setting defaults
        self.criterion = Loss(self.beta, self.alpha)

        self.save_hyperparameters()

        # Encoder to fourier space
        self.fft = fftn
        self.ifft = ifftn

        setattr(
            self,
            f"block_1",
            SingleFourierBlock(
                1,
                1024,
                1024,
                3,
                False,
                self.activation,
                None
            )
        )
        setattr(
            self,
            f"block_2",
            SingleFourierBlock(
                1,
                512,
                512,
                3,
                False,
                self.activation,
                2
            ),
        )
        setattr(
            self,
            f"block_3",
            SingleFourierBlock(
                1,
                256,
                256,
                3,
                False,
                self.activation,
                2
            ),
        )
        setattr(
            self,
            f"block_4",
            SingleFourierBlock(
                1,
                128,
                128,
                3,
                False,
                self.activation,
                2
            ),
        )
        setattr(
            self,
            f"block_5",
            SingleFourierBlock(
                1,
                64,
                64,
                3,
                False,
                self.activation,
                2
            ),
        )

        # Reparametrization
        setattr(
            self, f"encoder_fc", nn.Linear(64 * 64, 64 * 64 * 2)
        )  # -> 1, 64, 64

        # Upsampling block
        setattr(
            self,
            f"upblock_1",
            UpsamplingFourierBlock(
                1,
                64,
                64,
                False,
                3,
                self.activation,
                None
            ),
        )
        setattr(
            self,
            f"upblock_2",
            UpsamplingFourierBlock(
                1,
                128,
                128,
                False,
                3,
                self.activation,
                2
            ),
        )
        setattr(
            self,
            f"upblock_3",
            UpsamplingFourierBlock(
                1,
                256,
                256,
                False,
                3,
                self.activation,
                2
            ),
        )
        setattr(
            self,
            f"upblock_4",
            UpsamplingFourierBlock(
                1,
                512,
                512,
                False,
                3,
                self.activation,
                2
            ),
        )
        setattr(
            self,
            f"upblock_5",
            UpsamplingFourierBlock(
                1,
                1024,
                1024,
                False,
                3,
                self.activation,
                2
            ),
        )

        self.fc_act = nn.Sigmoid()

    def reparametrization(self, out: Tensor) -> Tensor:
        # b, 64^2 -> 1, 64, 64
        b, _ = out.shape
        logvar = out[:, :64**2].view(b, 1, 64, 64)
        mu = out[:, 64**2:].view(b, 1, 64, 64)
        # computing the standard deviation
        std = torch.exp(0.5 * logvar)
        # computing the latent variable on normal distribution
        epsilon = torch.randn(b, 1, 64, 64).to('cuda')
        # Return the normalized distribution
        return mu + epsilon * std, mu, logvar

    def forward(self, x: Tensor, mask_in: Tensor) -> Tensor:
        # To fourier space
        out = self.fft(x * mask_in, dim = (-2, -1))

        #Saves history
        fourier_hist: List[Tensor] = []

        for i in range(1, 6):
            out = getattr(self, f"block_{i}")(out)
            fourier_hist.append(out)
        
        out = self.ifft(out, dim = (-2,-1)).real

        out = getattr(self, f"encoder_fc")(
            out.view(out.shape[0], -1)
        )  # b, 64^2

        out, mu, logvar = self.reparametrization(out)

        out = self.fft(out, dim = (-2,-1))

        for i in range(1, 6):
            out = getattr(self, f"upblock_{i}")(out, fourier_hist[-i])

        out = self.ifft(out).real

        out = self.fc_act(out)

        return out, mu, logvar

    def training_step(self, batch: Tensor) -> Tensor:
        I_gt, mask_in = batch
        I_out, mu, logvar = self(I_gt, mask_in)
        args = self.criterion(I_out, I_gt, mu, logvar)
        loss = args[-1]
        metrics = {f"Training/{k}": v for k, v in zip(self.criterion.labels, args)}
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch = True)
        return loss

    def validation_step(self, batch: Tensor, idx) -> Tensor:
        I_gt, mask_in = batch
        I_out, mu, logvar = self(I_gt, mask_in)
        args = self.criterion(I_out, I_gt, mu, logvar)
        self.log_dict(
            {f"Validation/{k}": v for k, v in zip(self.criterion.labels, args)}
        )
        self.log('hp_metric', args[-1])
        
    def configure_optimizers(self):
        encoder_param_group : List[nn.Module] = []
        decoder_param_group : List[nn.Module] = []

        optimizer = VALID_OPTIMIZERS[self.optimizer]

        for name, param in self.named_parameters():
            if name.startswith(("block", "encoder")):
                encoder_param_group.append(param)
            else:
                decoder_param_group.append(param)

        optimizer = optimizer(
            [
                {
                    "params": encoder_param_group,
                    "lr": self.encoder_lr,
                    "weight_decay": self.encoder_wd,
                },
                {
                    "params": decoder_param_group,
                    "lr": self.decoder_lr,
                    "weight_decay": self.decoder_wd,
                },
            ]
        )

        return optimizer
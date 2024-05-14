# General utils
from typing import List, Dict
from torch import Tensor
from torch import nn
from collections import defaultdict
from torch.fft import fftn, ifftn
import torch.nn.functional as F
import torch
from lightning.pytorch import LightningModule

# Module utils
from .loss import Loss, SecondLoss
from ..utils import fourier_conv2d
from ..utils import (
    ComplexBatchNorm,
    ComplexReLU,
    ComplexReLU6,
    ComplexSiLU,
    ComplexSigmoid,
    ComplexMaxPool2d,
    _FourierConv,
    PartialConv2d,
)


class SingleFourierBlock(_FourierConv):
    def __init__(
        self,
        in_channels: Tensor,
        height: Tensor,
        width: Tensor,
        bias: bool = True,
        activation: str = None,
        pool: int = None,
        eps: float = 0.00001,
        momentum: float = 0.1,
    ) -> None:
        super().__init__(in_channels, height, width, bias)
        self.layer = nn.Sequential()

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

        self.layer.add_module('Batch Norm', ComplexBatchNorm(in_channels, eps, momentum))

        # pooling layer
        if pool is not None:
            self.layer.add_module('Max pool', ComplexMaxPool2d(pool, pool))

    def forward(self, x: Tensor) -> Tensor:
        out = super(SingleFourierBlock, self).forward(x)
        return self.layer(out)


class UpsamplingFourierBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        fourier_activation: str,
        normal_activation: str,
        eps: float,
        momentum: float,
        pool: int,
    ) -> None:
        super().__init__()
        # fourier transform
        self.fft = fftn
        self.ifft = ifftn
        # Upsamling for both spaces
        self.n_upsampling = lambda x, mask_in: (
            F.interpolate(x, scale_factor=pool, mode="nearest"),
            F.interpolate(mask_in, scale_factor=pool, mode="nearest"),
        )

        # Batch normalization
        self.f_norm = ComplexBatchNorm(out_channels, eps, momentum)
        self.n_norm = nn.BatchNorm2d(out_channels, eps, momentum)
        self.fc_norm = nn.BatchNorm2d(out_channels, eps, momentum)

        # Partial convolution
        self.partial_conv = PartialConv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )

        if fourier_activation is not None:
            match fourier_activation:
                case "relu":
                    self.fourier_activation = ComplexReLU()
                case "relu6":
                    self.fourier_activation = ComplexReLU6()
                case "silu":
                    self.fourier_activation = ComplexSiLU()
                case "sigmoid":
                    self.fourier_activation = ComplexSigmoid()

        if normal_activation is not None:
            match normal_activation:
                case "relu":
                    self.normal_activation = nn.ReLU()
                case "relu6":
                    self.normal_activation = nn.ReLU6()
                case "silu":
                    self.normal_activation = nn.SiLU()
                case "sigmoid":
                    self.normal_activation = nn.Sigmoid()

    def forward(self, x: Tensor, mask_in: Tensor, weight: Tensor) -> Tensor:
        # Partial convolution
        out, mask_out = self.partial_conv(x, mask_in)
        if self.normal_activation is not None:
            out = self.normal_activation(out)
        n_out = self.n_norm(out)

        # Fourier convolution with encoder weights
        out = self.fft(n_out, dim=(-2, -1))
        out = fourier_conv2d(out, weight)
        if self.fourier_activation is not None:
            out = self.fourier_activation(out)
        out = self.f_norm(out)
        #Convolution in fourier space with encoder weights
        out = self.ifft(out)

        out = self.fc_norm(n_out + out.real + out.imag)

        out, mask_out = self.n_upsampling(out, mask_out)

        return out, mask_out


class FourierVAE(LightningModule):
    def __init__(self, **hparams) -> None:
        super().__init__()
        # Setting hyperparameters values
        for k, v in hparams.items():
            setattr(self, k, v)

        match self.optimizer:
            case 'adam':
                self.optimizer = torch.optim.Adam
            case 'rms':
                self.optimizer = torch.optim.RMSprop
            case 'sgd':
                self.optimizer = torch.optim.SGD
                
        # Setting defaults
        self.criterion = SecondLoss(self.alpha)

        self.save_hyperparameters()

        # Encoder to fourier space
        self.fft = fftn

        for layer in range(1):
            # Downsampling in fourier space
            setattr(
                self,
                f"block{layer}_1",
                SingleFourierBlock(
                    1, 
                    1024,
                    1024,
                    True,
                    self.fourier_activation,
                    2,
                    self.eps,
                    self.momentum
                )
            )
            setattr(
                self,
                f"block{layer}_2",
                SingleFourierBlock(
                    1,
                    512,
                    512,
                    True,
                    self.fourier_activation,
                    2,
                    self.eps,
                    self.momentum,
                ),
            )
            setattr(
                self,
                f"block{layer}_3",
                SingleFourierBlock(
                    1,
                    256,
                    256,
                    True,
                    self.fourier_activation,
                    2,
                    self.eps,
                    self.momentum,
                ),
            )
            setattr(
                self,
                f"block{layer}_4",
                SingleFourierBlock(
                    1,
                    128,
                    128,
                    True,
                    self.fourier_activation,
                    2,
                    self.eps,
                    self.momentum,
                ),
            )
            setattr(
                self,
                f"block{layer}_5",
                SingleFourierBlock(
                    1, 64, 64, True, self.fourier_activation, 2, self.eps, self.momentum
                ),
            )

            # Reparametrization
            setattr(
                self, f"encoder_fc_{layer}", nn.Linear(32 * 32, 32 * 32 * 2)
            )  # -> 512, 32, 32

            # Upsampling block
            setattr(
                self,
                f"upblock{layer}_1",
                UpsamplingFourierBlock(
                    512,
                    256,
                    11,
                    1,
                    5,
                    self.fourier_activation,
                    self.normal_activation,
                    self.eps,
                    self.momentum,
                    2, 
                ),
            )
            setattr(
                self,
                f"upblock{layer}_2",
                UpsamplingFourierBlock(
                    256,
                    128,
                    9,
                    1,
                    4,
                    self.fourier_activation,
                    self.normal_activation,
                    self.eps,
                    self.momentum,
                    2,
                ),
            )
            setattr(
                self,
                f"upblock{layer}_3",
                UpsamplingFourierBlock(
                    128,
                    64,
                    7,
                    1,
                    3,
                    self.fourier_activation,
                    self.normal_activation,
                    self.eps,
                    self.momentum,
                    2,
                ),
            )
            setattr(
                self,
                f"upblock{layer}_4",
                UpsamplingFourierBlock(
                    64,
                    32,
                    5,
                    1,
                    2,
                    self.fourier_activation,
                    self.normal_activation,
                    self.eps,
                    self.momentum,
                    2,
                ),
            )
            setattr(
                self,
                f"upblock{layer}_5",
                UpsamplingFourierBlock(
                    32,
                    1,
                    3,
                    1,
                    1,
                    self.fourier_activation,
                    self.normal_activation,
                    self.eps,
                    self.momentum,
                    2,
                ),
            )

            setattr(self, f"decoder_fc_conv_{layer}", PartialConv2d(1, 1, 3, 1, 1))

        self.fc_act = nn.Sigmoid()
        self.ifft = ifftn
    def reparametrization(self, out: Tensor) -> Tensor:
        # b, 1024 -> 512, 32, 32
        b, _ = out.shape
        logvar = out[:, :1024].view(b, 1, 32, 32)
        mu = out[:, 1024:].view(b, 1, 32, 32)
        # computing the standard deviation
        std = torch.exp(0.5 * logvar)
        # computing the latent variable on normal distribution
        epsilon = torch.randn(b, 512, 32, 32).to('cuda')
        # Return the normalized distribution
        return mu + epsilon * std, mu, logvar

    def _single_forward(self, x: Tensor, mask_in: Tensor, layer: int) -> Tensor:
        out = self.fft(x * mask_in, dim = (-2, -1))
        fourier_hist: List[Tensor] = []
        for i in range(1, 6):
            out = getattr(self, f"block{layer}_{i}")(out)
            fourier_hist.append(out)

        out = self.ifft(out, dim=(-2, -1))
        
        out = out.real + out.imag

        out = getattr(self, f"encoder_fc_{layer}")(
            out.view(out.shape[0], -1)
        )  # b, 32*32

        out, mu, logvar = self.reparametrization(out)

        mask_out = F.max_pool2d(mask_in, 32, 32, 0)

        mask_out = mask_out.repeat_interleave(512, 1)

        for i in range(1, 6):
            out, mask_out = getattr(self, f"upblock{layer}_{i}")(
                out, mask_out, fourier_hist[-i]
            )

        out, mask_out = getattr(self, f"decoder_fc_conv_{layer}")(out, mask_out)

        out = self.fc_act(out)

        return out, mu, logvar

    def forward(self, x: Tensor, mask_in: Tensor) -> Tensor:
        x, mu, logvar = self._single_forward(x, mask_in, 0)
        return x, mu, logvar

    def training_step(self, batch: Tensor) -> Tensor:
        I_gt, mask_in = batch
        I_out, mu, logvar = self(I_gt, mask_in)
        args = self.criterion(I_out, I_gt, mu, logvar)
        loss = args[-1]
        metrics = {f"Training/{k}": v for k, v in zip(self.criterion.labels, args)}
        self.log_dict(metrics, prog_bar=True)
        return loss

    def validation_step(self, batch: Tensor, idx) -> Tensor:
        I_gt, mask_in = batch
        I_out, mu, logvar = self(I_gt, mask_in)
        args = self.criterion(I_out, I_gt, mu, logvar)
        self.log_dict(
            {f"Validation/{k}": v for k, v in zip(self.criterion.labels, args)}
        )
        self.log('hp_metric', Tensor([loss/factor for loss, factor in zip(args[:-1], self.criterion.alpha)]).sum().item())

    def configure_optimizers(self):
        encoder_param_group = defaultdict(list)
        decoder_param_group = defaultdict(list)

        for name, param in self.named_parameters():
            if name.startswith("block"):
                encoder_param_group["params"].append(param)
            else:
                decoder_param_group["params"].append(param)

        optimizer = torch.optim.Adam(
            [
                {
                    "params": encoder_param_group["params"],
                    "lr": self.encoder_lr,
                    "weight_decay": self.encoder_wd,
                },
                {
                    "params": decoder_param_group["params"],
                    "lr": self.decoder_lr,
                    "weight_decay": self.decoder_wd,
                },
            ]
        )

        return optimizer

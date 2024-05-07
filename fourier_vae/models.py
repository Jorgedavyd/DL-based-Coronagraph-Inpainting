# General utils
from typing import Tuple, List, Dict, Callable
from torch import Tensor
from torch import nn
from collections import defaultdict
from torch.fft import fftn
import torch.nn.functional as F
import torch
import numpy as np
from lightning import LightningModule

#Module utils
from .loss import Loss
from ..utils import fourier_conv2d
from ..utils import (
    ComplexBatchNorm,
    ComplexSwiGLU,
    ComplexReLU,
    ComplexReLU6,
    ComplexSiLU,
    ComplexSigmoid,
    ComplexMaxPool2d,
    _FourierConv,
    ChannelWiseSelfAttention,
    PartialConv2d
)

class SingleFourierBlock(_FourierConv):
    def __init__(self, in_channels: Tensor, height: Tensor, width: Tensor, bias: bool = True, activation: str = None, pool: int = None, eps: float = 0.00001, momentum: float = 0.1) -> None:
        super().__init__(in_channels, height, width, bias)
        self.layer = nn.Sequential()

        if activation is not None:
            match activation:
                case 'relu':
                    self.layer.add_module(ComplexReLU())
                case 'relu6': 
                    self.layer.add_module(ComplexReLU6())
                case 'silu':
                    self.layer.add_module(ComplexSiLU())
                case 'swiglu':
                    self.layer.add_module(ComplexSwiGLU())
                case 'sigmoid':
                    self.layer.add_module(ComplexSigmoid())
        
        self.layer.add_module(ComplexBatchNorm(in_channels, eps, momentum))
        
        # pooling layer
        if pool is not None:
            self.layer.add_module(ComplexMaxPool2d(pool, pool))

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
            batch_norm: bool,
            dropout: float,
            num_heads: int,
            eps: float,
            momentum: float,
            pool: int
    ) -> None:
        super().__init__()
        # fourier transform
        self.fft = fftn

        #Upsamling for both spaces
        self.n_upsampling = lambda x, mask_in: (F.interpolate(x, scale_factor=pool, mode = 'nearest'), F.interpolate(mask_in, scale_factor=pool, mode = 'nearest'))

        # Batch normalization
        if batch_norm:
            self.f_norm = ComplexBatchNorm(in_channels, eps, momentum)
            self.n_norm = nn.BatchNorm2d(out_channels, eps, momentum)
            self.fc_norm = nn.BatchNorm2d(out_channels, eps, momentum)
        #Partial convolution
        self.partial_conv = PartialConv2d(in_channels, out_channels, kernel_size, stride, padding)
        
        if fourier_activation is not None:
            match fourier_activation:
                case 'relu':
                    self.fourier_activation = ComplexReLU()
                case 'relu6': 
                    self.fourier_activation =ComplexReLU6()
                case 'silu':
                    self.fourier_activation =ComplexSiLU()
                case 'swiglu':
                    self.fourier_activation =ComplexSwiGLU()
                case 'sigmoid':
                    self.fourier_activation =ComplexSigmoid()
        
        if normal_activation is not None:
            match normal_activation:
                case 'relu':
                    self.normal_activation=ComplexReLU()
                case 'relu6': 
                    self.normal_activation=ComplexReLU6()
                case 'silu':
                    self.normal_activation=ComplexSiLU()
                case 'swiglu':
                    self.normal_activation=ComplexSwiGLU()
                case 'sigmoid':
                    self.normal_activation=ComplexSigmoid()

        self.attention = ChannelWiseSelfAttention(embed_dim = in_channels, num_heads=num_heads, dropout=dropout, kdim = in_channels, vdim = out_channels)
        
    def forward(self, x: Tensor, mask_in: Tensor, weight: Tensor) -> Tensor:
        # Partial convolution
        n_out, mask_out = self.partial_conv(x, mask_in)
        if self.normal_activation is not None:
            n_out = self.normal_activation(n_out)
        if self.n_norm is not None:
            n_out = self.n_norm(n_out)

        # Fourier convolution with encoder weights
        f_out = self.fftn(x * mask_in, dim = (-2, -1))
        f_out = fourier_conv2d(f_out, weight)
        if self.fourier_activation is not None:
            f_out = self.fourier_activation(f_out)
        if self.f_norm is not None:
            f_out = self.f_norm(f_out)

        out = self.attention(
            f_out.real.view(*f_out.real.shape[:2], -1),
            f_out.imag.view(*f_out.imag.shape[:2], -1),
            n_out.view(*n_out.shape[:2], -1)
        ).view(*n_out.shape)

        out += n_out

        out = self.fc_norm(out)

        # To 
        out, mask_out = self.n_upsampling(out, mask_out)

        return out, mask_out


class FourierVAE(LightningModule):
    def __init__(
            self,
            hparams: Dict[str, any]
    ) -> None:
        super().__init__()

        # Setting hyperparameters values
        for k,v in hparams.items():
            setattr(self, k, v)

        # Setting defaults
        self.criterion = Loss(
            self.alpha
        )

        self.save_hyperparameters()

        # Encoder to fourier space
        self.fft = fftn

        
        for layer in range(self.layers):
            #Downsampling in fourier space
            setattr(self, f'block{layer}_1',SingleFourierBlock(1, 1024, 1024, True, self.fourier_activation, 2, self.eps, self.momentum))
            setattr(self, f'block{layer}_2',SingleFourierBlock(1, 512, 512, True, self.fourier_activation, 2, self.eps, self.momentum))
            setattr(self, f'block{layer}_3',SingleFourierBlock(1, 256, 256, True, self.fourier_activation, 2, self.eps, self.momentum))
            setattr(self, f'block{layer}_4',SingleFourierBlock(1, 128, 128, True, self.fourier_activation, 2, self.eps, self.momentum))
            setattr(self, f'block{layer}_5',SingleFourierBlock(1, 64, 64, True, self.fourier_activation, 2, self.eps, self.momentum))
            setattr(self, f'block{layer}_6',SingleFourierBlock(1, 32, 32, True, self.fourier_activation, 2, self.eps, self.momentum))

            # Reparametrization
            setattr(self, f'encoder_fc_{layer}', nn.Linear(32*32, 32*32*2)) # -> 512, 32, 32
            
            #Upsampling block
            setattr(self, f'upblock{layer}_1', UpsamplingFourierBlock(512, 256, 11, 1, 5, self.fourier_activation, self.normal_activation, self.batch_norm, self.dropout, self.num_heads, self.eps, self.momentum, 2))
            setattr(self, f'upblock{layer}_2', UpsamplingFourierBlock(256, 128, 9, 1, 4, self.fourier_activation, self.normal_activation, self.batch_norm, self.dropout, self.num_heads, self.eps, self.momentum, 2))
            setattr(self, f'upblock{layer}_3', UpsamplingFourierBlock(128, 64, 7, 1, 3, self.fourier_activation, self.normal_activation, self.batch_norm, self.dropout, self.num_heads, self.eps, self.momentum, 2))
            setattr(self, f'upblock{layer}_4', UpsamplingFourierBlock(64, 32, 5, 1, 2, self.fourier_activation, self.normal_activation, self.batch_norm, self.dropout, self.num_heads, self.eps, self.momentum, 2))
            setattr(self, f'upblock{layer}_5', UpsamplingFourierBlock(32, 1, 3, 1, 1, self.fourier_activation, self.normal_activation, self.batch_norm, self.dropout, self.num_heads, self.eps, self.momentum, 2))

            setattr(self, f'decoder_fc_conv_{layer}', PartialConv2d(1, 3, 1, 1))
        
        self.fc_act = nn.Sigmoid()

    def reparametrization(self, out: Tensor) -> Tensor:
        # b, 1024 -> 512, 32, 32
        logvar = out[: 1024].view(out.shape[0], 32, 32)
        mu = out[1024:].view(out.shape[0], 32, 32)
        # computing the standard deviation
        std = torch.exp(0.5 * logvar)
        # computing the latent variable on normal distribution
        epsilon = torch.randn(512, 32, 32)
        # Return the normalized distribution
        return mu + epsilon * std, mu, logvar
    
    def _single_forward(self, x: Tensor, mask_in: Tensor, layer: int) -> Tensor:
        out = x * mask_in
        fourier_hist: List[Tensor] = []
        for i in range(1, 7):
            out = getattr(self, f'block{layer}_{i}')(out)
            fourier_hist.append(fourier_hist)
        
        out = getattr(self, f'encoder_fc_{layer}')(out.view(out.shape[0], -1)) # b, 1024

        out = self.reparametrization(out)
        
        mask_out = mask_in.clone()

        for i in range(1,6):
            out, mask_out = getattr(self, f'upblock{layer}_{i}')(out, mask_out, fourier_hist[-i])
        
        out, mask_out = getattr(self, f'decoder_fc_conv_{layer}')(out, mask_out)

        out = self.fc_act(out)
        
        return out * ~mask_in.bool() + x * mask_in.bool(), mask_out

    def forward(self, x: Tensor, mask_in: Tensor) -> Tensor:
        for layer in range(self.layers):
            x, mask_in = self._single_forward(x, mask_in, layer)
        return x, mask_in
    
    def training_step(self, batch: Tensor) -> Tensor:
        I_gt, mask_in = batch
        loss = 0

        for layer in range(self.layers):
            I_out, mask_out = self._single_forward(I_gt, mask_in, layer)
            args = self.criterion(I_out, I_gt, mask_in, mask_out)
            loss +=args[-1]
            metrics = {f'Training/{k}':v for k,v in zip(self.criterion.labels, args)}
            self.log_dict(metrics, prog_bar=True)
            mask_in = mask_out
        
        return loss

    def validation_step(self, batch: Tensor) -> Tensor:
        I_gt, mask_in = batch
        I_out, mask_out = self(I_gt, mask_in)
        args = self.criterion(I_out, I_gt, mask_in, mask_out)
        self.log_dict(
            {
                f'Validation/{k}': v for k,v in zip(self.criterion.labels, args)
            }
        )
        
    def configure_optimizers(self):
        encoder_param_group = defaultdict(list)
        decoder_param_group = defaultdict(list)

        for name, param in self.named_parameters():
            if name.startswith('block'):
                encoder_param_group['params'].append(param)
            else:
                decoder_param_group['params'].append(param)

        optimizer = torch.optim.Adam([
            {'params': encoder_param_group['params'], 'lr': self.encoder_lr, 'weight_decay': self.encoder_wd},
            {'params': decoder_param_group['params'], 'lr': self.decoder_lr, 'weight_decay': self.decoder_wd}
        ])

        return optimizer

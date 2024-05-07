from typing import Tuple, Callable, Iterable, List
from utils import PartialConv2d, ChannelWiseSelfAttention, _FourierConv
from torch import Tensor, nn
from collections import defaultdict
import torch.nn.functional as F
import torch
from lightning import LightningModule
from .loss import Loss
from torch.fft import fftn


class SingleFourierBlock(nn.Module):
    pool = None
    normal_activation = None
    fourier_activation = None

    def __init__(
        self,
        height: int,
        width: int,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        pool: int = None,
        batch_norm: bool = True,
        normal_activation: str = None,
        fourier_activation: str = None,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.normal_activation = normal_activation
        self.fourier_activation = fourier_activation
        self.batch_norm = batch_norm
        self.pool = pool
        self.fft = fftn
        self.conv1 = PartialConv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        if pool is not None:
            self.pool = lambda x: F.max_pool2d(x, pool, pool)

        self.norm1 = nn.BatchNorm2d(out_channels)
        self.Re_fnorm1 = nn.BatchNorm2d(in_channels)
        self.Im_fnorm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)

        if normal_activation is not None:
            match normal_activation:
                case "relu":
                    self.normal_act = F.relu
                case "sigmoid":
                    self.normal_act = F.sigmoid
                case "swiglu":
                    self.normal_act = lambda x: F.hardswish(x) * F.glu(x)
                case "silu":
                    self.normal_act = F.silu

        if fourier_activation is not None:
            match fourier_activation:
                case "relu":
                    self.fourier_act = F.relu
                case "sigmoid":
                    self.fourier_act = F.sigmoid
                case "swiglu":
                    self.fourier_act = lambda x: F.hardswish(x) * F.glu(x)
                case "silu":
                    self.fourier_act = F.silu

        self.fconv1 = _FourierConv(in_channels, height, width)

        self.attention = ChannelWiseSelfAttention(
            in_channels, num_heads, dropout, kdim=in_channels, vdim=out_channels
        )

    def forward(self, x: Tensor, mask_in: Tensor) -> Tuple[Tensor, Tensor]:
        b, _, h, w = b.shape

        # Normal forward
        n_out, updated_mask = self.conv1(x, mask_in)
        if self.normal_activation is not None:
            n_out = self.normal_act(n_out)
        if self.batch_norm:
            n_out = self.norm1(n_out)

        # Fourier Forward
        f_out = self.fft(x * mask_in)
        f_out = self.fconv1(f_out)

        if self.fourier_activation is not None:
            f_out = self.fourier_act(f_out)

        if self.batch_norm:
            Re_out = self.Re_fnorm1(f_out.real)  # in_channels, h, w
            Im_out = self.Im_fnorm1(f_out.imag)  # in_channels, h, w

        # Perform multihead attention
        out = self.attention(
            Re_out.view(b, self.in_channels, -1),
            Im_out.view(b, self.in_channels, -1),
            n_out.view(b, self.out_channels, -1),
        ).view(b, self.out_channels, h, w)

        out = self.norm2(out + n_out)

        return out + n_out, updated_mask, Re_out, Im_out


class DefaultUpsamplingBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        pool: int = None,
        activation: str = None,
        batch_norm: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.upsample = lambda x: F.interpolate(x, scale_factor=2, mode="nearest")
        self.upsampling: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]] = (
            lambda x, mask_in: (self.upsample(x), self.upsample(mask_in))
        )
        self.conv1 = PartialConv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        if pool is not None:
            self.pool = lambda x: F.max_pool2d(x, pool, pool)
        if activation is not None:
            match activation:
                case "relu":
                    self.activation = F.relu
                case "sigmoid":
                    self.activation = F.sigmoid
                case "swiglu":
                    self.activation = lambda x: F.hardswish(x) * F.glu(x)
                case "silu":
                    self.activation = F.silu
        if batch_norm:
            self.norm1 = nn.BatchNorm2d(in_channels * 2)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask_in: Tensor) -> Tuple[Tensor, Tensor]:
        x, mask_in = self.upsampling(x, mask_in)
        out, mask_out = self.conv1(x, mask_in)
        out = self.activation(out)
        out = self.norm1(out)
        out = self.dropout(out)

        return out, mask_out


class DeluxeFourierModel(LightningModule):
    def __init__(
        self,
        encoder_lr: float,
        encoder_wd: float,
        decoder_lr: float,
        decoder_wd: float,
        optimizer: torch.optim.Optimizer,
        layers: int,
        alpha: Iterable,
        normal_activation: str,
        fourier_activation: str,
        dropout: float,
        num_heads: int,
    ) -> None:
        super().__init__()

        self.criterion = Loss(alpha)
        self.encoder_lr = encoder_lr
        self.encoder_wd = encoder_wd
        self.decoder_lr = decoder_lr
        self.decoder_wd = decoder_wd
        self.optimizer = optimizer
        self.layer = layers

        self.save_hyperparameters()

        for layer in range(layers):
            setattr(
                self,
                f"block{layer}_1",
                SingleFourierBlock(
                    1024,
                    1024,
                    1,
                    32,
                    3,
                    1,
                    1,
                    2,
                    True,
                    normal_activation,
                    fourier_activation,
                    num_heads,
                    dropout,
                ),
            )
            # -> 32, 512, 512
            setattr(
                self,
                f"block{layer}_2",
                SingleFourierBlock(
                    512,
                    512,
                    32,
                    64,
                    5,
                    1,
                    2,
                    2,
                    True,
                    normal_activation,
                    fourier_activation,
                    num_heads,
                    dropout,
                ),
            )
            # -> 64, 256, 256
            setattr(
                self,
                f"block{layer}_3",
                SingleFourierBlock(
                    256,
                    256,
                    64,
                    128,
                    7,
                    1,
                    3,
                    2,
                    True,
                    normal_activation,
                    fourier_activation,
                    num_heads,
                    dropout,
                ),
            )
            # -> 128, 128, 128
            setattr(
                self,
                f"block{layer}_4",
                SingleFourierBlock(
                    128,
                    128,
                    128,
                    256,
                    9,
                    1,
                    4,
                    2,
                    True,
                    normal_activation,
                    fourier_activation,
                    num_heads,
                    dropout,
                ),
            )
            # -> 256, 64, 64
            setattr(
                self,
                f"block{layer}_5",
                SingleFourierBlock(
                    64,
                    64,
                    256,
                    512,
                    11,
                    1,
                    5,
                    2,
                    True,
                    normal_activation,
                    fourier_activation,
                    num_heads,
                    dropout,
                ),
            )
            # -> 512, 32, 32
            setattr(
                self,
                f"block{layer}_5",
                SingleFourierBlock(
                    32,
                    32,
                    512,
                    1024,
                    13,
                    1,
                    6,
                    2,
                    True,
                    normal_activation,
                    fourier_activation,
                    num_heads,
                    dropout,
                ),
            )
            # -> 1024, 16, 16

            setattr(
                self,
                f"upblock{layer}_1",
                DefaultUpsamplingBlock(
                    1024, 512, 13, 1, 6, 2, normal_activation, True, dropout
                ),
            )
            # -> 512, 32, 32
            setattr(
                self,
                f"upblock{layer}_2",
                DefaultUpsamplingBlock(
                    512, 256, 11, 1, 5, 2, normal_activation, True, dropout
                ),
            )
            # -> 256, 64,64
            setattr(
                self,
                f"upblock{layer}_3",
                DefaultUpsamplingBlock(
                    256, 128, 9, 1, 4, 2, normal_activation, True, dropout
                ),
            )
            # -> 128, 128, 128
            setattr(
                self,
                f"upblock{layer}_4",
                DefaultUpsamplingBlock(
                    128, 64, 7, 1, 3, 2, normal_activation, True, dropout
                ),
            )
            # -> 64, 256, 256
            setattr(
                self,
                f"upblock{layer}_5",
                DefaultUpsamplingBlock(
                    64, 32, 5, 1, 2, 2, normal_activation, True, dropout
                ),
            )
            # -> 32, 512, 512
            setattr(
                self,
                f"upblock{layer}_5",
                DefaultUpsamplingBlock(
                    32, 1, 3, 1, 1, 2, normal_activation, True, dropout
                ),
            )
            # -> 1, 1024, 1024
            setattr(self, f"fc_conv_{layer}", PartialConv2d(1, 1, 3, 1, 1))

        self.fc_act = F.sigmoid

    def _single_forward(
        self, x: Tensor, mask_in: Tensor, layer: int
    ) -> Tuple[Tensor, Tensor]:
        real_hist: List[Tensor] = []
        imag_hist: List[Tensor] = []
        for i in range(1, 6):
            x, mask_in, Re_out, Im_out = getattr(self, f"block{i}")(x, mask_in)
            real_hist.append(Re_out)
            imag_hist.append(Im_out)

        for i in range(1, 7):
            x, mask_in = getattr(self, f"upblock{i}")(
                x + real_hist[-i] + imag_hist[-i], mask_in
            )

        x, mask_in = getattr(self, f"fc_conv_{layer}")(x, mask_in)
        x = self.fc_act(x)

        return x, mask_in

    def forward(self, x: Tensor, mask_in: Tensor) -> Tuple[Tensor, Tensor]:
        for layer in range(self.layers):
            x, mask_in = self._single_forward(x, mask_in, layer)
        return x, mask_in

    def training_step(self, batch: Tensor) -> Tensor:
        I_gt, mask_in = batch
        loss = 0.0
        for layer in self.layers:
            I_out, mask_out = self._single_forward(I_gt, mask_in, layer)
            args = self.criterion(I_out, I_gt, mask_in, mask_out)
            metrics = {
                f"{k} Layer {layer}": v for k, v in zip(self.criterion.labels, args)
            }
            self.log_dict(metrics, prog_bar=True)
            loss += args[-1]
            mask_in = mask_out

        return loss

    @torch.no_grad()
    def validation_step(self, batch: Tensor) -> Tuple[Tensor, Tensor]:
        I_gt, mask_in = batch
        I_out, mask_out = self(I_gt, mask_in)
        args = self.criterion(I_out, I_gt, mask_in, mask_out)
        metrics = {f"Validation {k}": v for k, v in zip(self.criterion.labels, args)}
        self.log_dict(metrics)

    def configure_optimizers(self):
        encoder_param_group = defaultdict(list)
        decoder_param_group = defaultdict(list)

        for name, param in self.named_parameters():
            if name.startswith("block"):
                encoder_param_group["params"].append(param)
            else:
                decoder_param_group["params"].append(param)

        optimizer = self.optimizer(
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

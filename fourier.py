from torch.fft import fftn, ifftn
from typing import Tuple, Union
from torch import nn
from torch import Tensor
from partial_conv import PartialConv2d, FourierPartialConv2d
import torch
from torch.nn import functional as F
from model import TrainingPhase
from typing import Dict
import lightning as L

from collections import defaultdict

class FourierMultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.fft_attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.fft = torch.fft.fftn

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        fft_x = self.fft(x, dim=(-1, -2))
        x = x.view(b, c, -1)
        out, _ = self.attention(x, x, x)
        out_fft, _ = self.fft_attention(
            fft_x.real.view(b, c, -1), fft_x.imag.view(b, c, -1), x
        )
        return (out + out_fft).view(b, c, h, w)

class ChannelWiseSelfAttention(nn.MultiheadAttention):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            embed_dim,
            num_heads,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            kdim,
            vdim,
            batch_first,
            device,
            dtype,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = True,
        attn_mask: Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[Tensor | None]:
        query = query.transpose(-1, -2)
        key = key.transpose(-1, -2)
        value = value.transpose(-1, -2)
        out, _ = super().forward(
            query,
            key,
            value,
            key_padding_mask,
            need_weights,
            attn_mask,
            average_attn_weights,
            is_causal,
        )
        return out.transpose(-1, -2)

class FourierLayerConv2d(nn.Conv2d):
    def __init__(
        self,
        height: int,
        width: int,
        n_heads: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int],
        stride: int | Tuple[int] = 1,
        padding: str | int | Tuple[int] = 0,
        dilation: int | Tuple[int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        update_mask: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        self.update_mask = update_mask
        # Parameters for the normal partial convolution
        self.sum_1 = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.mask_updater = torch.ones_like(self.weight, requires_grad=False).to("cuda")

        # Parameters for the fourier space convolution
        self.Re_weight = nn.init.kaiming_uniform_(
            nn.Parameter(torch.zeros(out_channels, in_channels, height, width), True)
        )
        self.Im_weight = nn.init.kaiming_uniform_(
            nn.Parameter(torch.zeros(out_channels, in_channels, height, width), True)
        )

        self.Re_b = nn.Parameter(torch.zeros(out_channels), True)
        self.Im_b = nn.Parameter(torch.zeros(out_channels), True)

        # Multihead attention
        self.attention = ChannelWiseSelfAttention(in_channels, n_heads, dropout=0.1)

        self.height = height
        self.width = width

    def _part_conv_forward(
        self, input: torch.Tensor, mask_in: torch.Tensor
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        b, _, _, _ = input.shape
        # The mask update could be seen as a maxpooling operation
        with torch.no_grad():
            # This outputs an image that represents each sum(M)
            sum_m = F.conv2d(
                mask_in,
                self.mask_updater,
                None,
                self.stride,
                self.padding,
                self.dilation,
            )
            if self.update_mask:
                updated_mask = torch.clamp_max(sum_m, 1)

        img = self._img_space_convolution(input, mask_in, sum_m)

        Re_f, Im_f = self._fourier_space_convolution(input, mask_in, sum_m)

        out = self.attention(
            Re_f.view(b, self.in_channels, -1),
            Im_f.view(b, self.in_channels, -1),
            img.view(b, self.out_channels, -1),
        )

        return out + img, updated_mask

    def _img_space_convolution(
        self, x: Tensor, mask_in: Tensor, sum_m: Tensor
    ) -> Tensor:
        # W^T (X \odot M) + b
        out = super(FourierLayerConv2d, self).forward(torch.mul(x, mask_in))

        # W^T (X \odot M) * \frac{sum(1)}{sum(M)} + b
        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1)
            out = torch.mul(out - bias, self.sum_1 / (sum_m + 1e-8)) + bias

        return out

    def _fourier_space_convolution(
        self, x: Tensor, mask_in: Tensor, sum_m: Tensor
    ) -> Tuple[Tensor, Tensor]:
        out = self.fft(x * mask_in, dim=(-2, -1))

        Re_out = (
            torch.abs(self.sum_1 / sum_m) * self.Re_weight * out.real + self.Re_b
        )  # Convolution in fourier space as product
        Im_out = (
            torch.abs(self.sum_1 / sum_m) * self.Im_weight * out.real + self.Im_b
        )  # Convolution in fourier space as product

        return Re_out, Im_out

    def forward(
        self, input: Tensor, mask_in: Tensor
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        return self._part_conv_forward(input, mask_in)

def fourier_conv2d(x: Tensor, weights: Tensor, bias: Tensor) -> Tensor:
    return (
        x.real * weights.real
        + bias.imag.view(1, -1, 1, 1)
        + 1j * (x.imag * weights.imag + bias.imag.view(1, -1, 1, 1))
    )

class _FourierConv(nn.Module):
    def __init__(
        self,
        in_channels: Tensor,
        height: Tensor,
        width: Tensor,
        bias: bool = True,
        device="cpu",
        dtype=torch.float32,
    ) -> None:
        super(_FourierConv, self).__init__()
        self.weight = nn.init.kaiming_uniform_(
            nn.Parameter(
                torch.zeros(
                    in_channels,
                    height,
                    width,
                    dtype=dtype,
                    device=device,
                    requires_grad=True,
                )
                + 1j
                * torch.zeros(
                    in_channels, dtype=dtype, device=device, requires_grad=True
                ),
                True,
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(in_channels, dtype=dtype, device=device, requires_grad=True)
                + 1j
                * torch.zeros(
                    in_channels, dtype=dtype, device=device, requires_grad=True
                ),
                True,
            )
        else:
            self.bias = nn.Parameter(
                torch.zeros(
                    in_channels, dtype=dtype, device=device, requires_grad=False
                )
                + 1j
                * torch.zeros(
                    in_channels, dtype=dtype, device=device, requires_grad=False
                ),
                False,
            )

    def forward(self, x: Tensor):
        out: Tensor = fourier_conv2d(x, self.weight, self.bias)
        return out

class FourierConv2d(_FourierConv):
    """
    # FourierConv2d
    Pass the input through Fast Fourier Transformation
    into a new linear space where the convolution is
    the product between the tensors. Computes the
    convolution and finally the Inverse Fourier
    Transformation.
    """

    def __init__(self, in_channels: Tensor, height: Tensor, width: Tensor) -> None:
        super().__init__(in_channels, height, width)
        self.fft = fftn
        self.ifft = ifftn

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.fft(x, dim=(-2, -1))
        out = super(FourierConv2d, self).forward(out)
        out = self.ifft(out, dim=(-2, -1))
        return out

class ResidualFourierConvTranspose2d(nn.ConvTranspose2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int],
        stride: int | Tuple[int] = 1,
        padding: int | Tuple[int] = 0,
        output_padding: int | Tuple[int] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int | Tuple[int] = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
            device,
            dtype,
        )
        self.ifft = ifftn
        self.fft = fftn
        if bias:
            self.f_bias = nn.Parameter(
                torch.zeros(
                    out_channels, dtype=dtype, device=device, requires_grad=True
                )
                + 1j
                * torch.zeros(
                    out_channels, dtype=dtype, device=device, requires_grad=True
                ),
                True,
            )
        else:
            self.f_bias = nn.Parameter(
                torch.zeros(
                    out_channels, dtype=dtype, device=device, requires_grad=False
                )
                + 1j
                * torch.zeros(
                    out_channels, dtype=dtype, device=device, requires_grad=False
                ),
                False,
            )

    def forward(self, x: Tensor, f_weights: Tensor):
        x = super(ResidualFourierConvTranspose2d, self).forward(x)
        out = self.fft(x, dim=(-2, -1))
        out = fourier_conv2d(out, f_weights, self.f_bias)

        return self.ifft(out).real

# UNet like architectures with residual connections
class FourierPartial(L.LightningModule):
    def __init__(self, criterion, hyperparams):
        super().__init__()
        self.criterion = criterion
        
        self.save_hyperparameters({
            f'{base}_{arg}': dict_[arg]  for base, dict_ in zip(['encoder', 'decoder'], hyperparams) for arg in dict_
        }.update(self.criterion.factors))

        self.hyperparams = hyperparams
        
        self.maxpool = lambda x: F.max_pool2d(x, 2, 2)
        self.upsample = lambda x: F.interpolate(x, scale_factor=2, mode="nearest")

        self.hid_act = lambda x: F.relu(x)
        self.last_act = lambda x: F.sigmoid(x)

        self.fconv1 = FourierPartialConv2d(1, 32, 3, 1, 1)  # -> 32, 1024, 1024
        self.norm1 = nn.BatchNorm2d(32)

        self.fconv2 = FourierPartialConv2d(32, 64, 3, 1, 1)  # -> 64, 512, 512
        self.norm2 = nn.BatchNorm2d(64)

        self.fconv3 = FourierPartialConv2d(64, 128, 3, 1, 1)  # -> 128, 256, 256
        self.norm3 = nn.BatchNorm2d(128)

        self.fconv4 = FourierPartialConv2d(128, 256, 3, 1, 1)  # -> 256, 128, 128
        self.norm4 = nn.BatchNorm2d(256)

        self.fconv5 = FourierPartialConv2d(256, 512, 5, 1, 2)  # -> 512, 64, 64
        self.norm5 = nn.BatchNorm2d(512)

        self.fconv6 = FourierPartialConv2d(512, 1024, 7, 1, 3)  # -> 1024, 32, 32

        self.fupconv1 = FourierPartialConv2d(1024, 512, 7, 1, 3)  # -> 512, 32, 32
        self.upnorm1 = nn.BatchNorm2d(512)

        self.fupconv2 = FourierPartialConv2d(512, 256, 3, 1, 1)  # -> 256, 64, 64
        self.upnorm2 = nn.BatchNorm2d(256)

        self.fupconv3 = FourierPartialConv2d(256, 128, 3, 1, 1)  # -> 128, 128, 128
        self.upnorm3 = nn.BatchNorm2d(128)

        self.fupconv4 = FourierPartialConv2d(128, 64, 3, 1, 1)  # -> 64, 256, 256
        self.upnorm4 = nn.BatchNorm2d(64)

        self.fupconv5 = FourierPartialConv2d(64, 32, 3, 1, 1)  # ->  32, 512, 512
        self.upnorm5 = nn.BatchNorm2d(32)

        self.upconv6 = PartialConv2d(32, 1, 3, 1, 1)  # ->  1, 1024, 1024

        self.attention = FourierMultiheadAttention(1024, 8)

    def joint_maxpool(self, x: Tensor, mask_in: Tensor) -> Tuple[Tensor, Tensor]:
        return self.maxpool(x), self.maxpool(mask_in)

    def joint_upsample(self, x: Tensor, mask_in: Tensor) -> Tuple[Tensor, Tensor]:
        return self.upsample(x), self.upsample(mask_in)

    def forward(self, x: Tensor, mask_in: Tensor) -> Tuple[Tensor, Tensor]:
        # for output
        gt = x.clone()
        init_mask = mask_in.clone()
        # First layer
        x, mask_in = self.fconv1(x, mask_in)  # -> 32, 1024, 1024
        x = self.norm1(x)
        x_1 = self.hid_act(x)
        x, mask_in = self.joint_maxpool(x_1, mask_in)  #  -> 32, 512, 512

        # Second layer
        x, mask_in = self.fconv2(x, mask_in)  # -> 64, 512, 512
        x = self.norm2(x)
        x_2 = self.hid_act(x)
        x, mask_in = self.joint_maxpool(x_2, mask_in)  # -> 64, 256, 256

        # Third layer
        x, mask_in = self.fconv3(x, mask_in)  # -> 128, 256, 256
        x = self.norm3(x)
        x_3 = self.hid_act(x)
        x, mask_in = self.joint_maxpool(x_3, mask_in)  # -> 128, 128, 128

        # Fourth layer
        x, mask_in = self.fconv4(x, mask_in)  # -> 256, 128, 128
        x = self.norm4(x)
        x_4 = self.hid_act(x)
        x, mask_in = self.joint_maxpool(x_4, mask_in)  # -> 256, 64, 64

        # Fifth layer
        x, mask_in = self.fconv5(x, mask_in)  # -> 512, 64, 64
        x = self.norm5(x)
        x_5 = self.hid_act(x)
        x, mask_in = self.joint_maxpool(x_5, mask_in)  # -> 512, 32, 32

        # Sixth layer
        x, mask_in = self.fconv6(x, mask_in)  # -> 1024, 32, 32

        # Fourier attention layer and residual connection
        x, mask_in = self.fupconv1(x + self.attention(x), mask_in)  # -> 512, 32, 32
        x = self.upnorm1(x)
        x = self.hid_act(x)

        x, mask_in = self.joint_upsample(x, mask_in)  # -> 512, 64, 64

        x += x_5

        x, mask_in = self.fupconv2(x, mask_in)  # -> 256, 64, 64
        x = self.upnorm2(x)
        x = self.hid_act(x)

        x, mask_in = self.joint_upsample(x, mask_in)  # -> 256, 128, 128

        x += x_4

        x, mask_in = self.fupconv3(x, mask_in)  # -> 512, 128, 128
        x = self.upnorm3(x)
        x = self.hid_act(x)

        x, mask_in = self.joint_upsample(x, mask_in)  # -> 512, 256, 256

        x += x_3

        x, mask_in = self.fupconv4(x, mask_in)  # -> 512, 128, 128
        x = self.upnorm4(x)
        x = self.hid_act(x)

        x, mask_in = self.joint_upsample(x, mask_in)  # -> 512, 256, 256

        x += x_2

        x, mask_in = self.fupconv5(x, mask_in)  # -> 512, 128, 128
        x = self.upnorm5(x)
        x = self.hid_act(x)

        x, mask_in = self.joint_upsample(x, mask_in)  # -> 512, 256, 256

        x += x_1

        x, mask_in = self.upconv6(x, mask_in)

        x = self.last_act(x)

        return x * ~init_mask.bool() + gt * init_mask.bool(), mask_in

    def training_step(self, batch: Tensor) -> Tensor:
        I_gt, M_l_1 = batch
        I_out, M_l_2 = self(I_gt, M_l_1)
        args = self.criterion(I_out, I_gt, M_l_1, M_l_2)
        metrics = {k: v.item() for k, v in zip(self.criterion.labels, args)}
        self.log_dict(metrics, prog_bar=True, enable_graph=True)
        return args[-1]
        
    def configure_optimizers(self):
        encoder_param_group = defaultdict(list)
        decoder_param_group = defaultdict(list)

        for name, param in self.named_parameters():
            if name.startswith(('fconv', 'norm')):
                encoder_param_group['params'].append(param)
            else:
                decoder_param_group['params'].append(param)

        optimizer = torch.optim.Adam([
            {'params': encoder_param_group['params'], **self.hyperparams[0]},
            {'params': decoder_param_group['params'], **self.hyperparams[1]}
        ])

        return optimizer

# Generative VAE for image inpainting with fourier
class FourierVAE(L.LightningModule):
    def __init__(self, hyperparams, criterion) -> None:
        super().__init__()
        self.criterion = criterion
        
        self.save_hyperparameters({
            f'{base}_{arg}': dict_[arg]  for base, dict_ in zip(['encoder', 'decoder'], hyperparams) for arg in dict_
        }.update(self.criterion.factors))

        self.hyperparams = hyperparams
        self.fft = fftn
        self.ifft = ifftn
        self.maxpool = lambda x: F.max_pool2d(x, 2, 2)
        self.upsample = lambda x: F.interpolate(x, scale_factor=2, mode="nearest")

        self.hid_act = lambda x: F.relu(x)
        self.last_act = lambda x: F.sigmoid(x)

        self.fconv1 = _FourierConv(1, 1024, 1024)  # -> 1, 1024, 1024

        self.fconv2 = _FourierConv(1, 512, 512)  # -> 1, 512, 512

        self.fconv3 = _FourierConv(1, 256, 256)  # -> 1, 256, 256

        self.fconv4 = _FourierConv(1, 128, 128)  # -> 1, 128, 128

        self.fconv5 = _FourierConv(1, 64, 64)  # -> 1, 64, 64

        self.fconv6 = _FourierConv(
            1, 32, 32
        )  # -> 1, 32, 32 -> normal space distribution with ifftn

        # This goes all in reparametrization
        self.flatten = nn.Flatten()

        self.mu_fc = nn.Linear(32 * 32, 32 * 32)
        self.logvar_fc = nn.Linear(32 * 32, 32 * 32)

        self.upconv1 = ResidualFourierConvTranspose2d(
            1024, 512, 3, 1, 1
        )  # -> 512, 32, 32
        self.upnorm1 = nn.BatchNorm2d(512)

        self.upconv2 = ResidualFourierConvTranspose2d(
            512, 256, 4, 2, 1
        )  # -> 256, 64, 64
        self.upnorm2 = nn.BatchNorm2d(256)

        self.upconv3 = ResidualFourierConvTranspose2d(
            256, 128, 4, 2, 1
        )  # -> 128, 128, 128
        self.upnorm3 = nn.BatchNorm2d(128)

        self.upconv4 = ResidualFourierConvTranspose2d(
            128, 64, 4, 2, 1
        )  # -> 64, 256, 256
        self.upnorm4 = nn.BatchNorm2d(64)

        self.upconv5 = ResidualFourierConvTranspose2d(
            64, 32, 4, 2, 1
        )  # ->  32, 512, 512
        self.upnorm5 = nn.BatchNorm2d(32)

        self.upconv6 = ResidualFourierConvTranspose2d(
            32, 1, 4, 2, 1
        )  # ->  1, 1024, 1024

        self.fc = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1), nn.Sigmoid())

    def reparametrization(self, out: Tensor, in_channels: int) -> Tensor:
        b, c, h, w = out.shape
        out = self.flatten(out)
        logvar = self.logvar_fc(out).view(b, c, h, w)  # std: b, c, h, w (b, 1, 32, 32)
        mu = self.mu_fc(out).view(b, c, h, w)  # mu: b, c, h, w (b, 1, 32, 32)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(b, in_channels, h, w, device="cuda")
        return mu + eps * std, mu, logvar

    def _joint_maxpool(self, Re: Tensor, Im: Tensor) -> Tuple[Tensor, Tensor]:
        return self.maxpool(Re) + 1j * self.maxpool(Im)

    def forward(self, x: Tensor, mask_in: Tensor) -> Tensor:
        # To fourier space
        out = self.fft(x * mask_in, dim=(-2, -1))

        weight_1 = self.fconv1(out)
        out = self._joint_maxpool(weight_1.real, weight_1.imag)  # -> 1, 512, 512
        weight_2 = self.fconv2(out)
        out = self._joint_maxpool(weight_2.real, weight_2.imag)  # -> 1, 256, 256
        weight_3 = self.fconv3(out)
        out = self._joint_maxpool(weight_3.real, weight_3.imag)  # -> 1, 128,128
        weight_4 = self.fconv4(out)
        out = self._joint_maxpool(weight_4.real, weight_4.imag)  # -> 1, 64,64
        weight_5 = self.fconv5(out)
        out = self._joint_maxpool(weight_5.real, weight_5.imag)  # -> 1, 32, 32
        weight_6 = self.fconv6(out)  # -> 1, 32,32
        out = self.ifft(out).real  # -> Real

        out, mu, logvar = self.reparametrization(out, 1024)  # -> 1024, 32, 32

        z = self.upconv1(out, weight_6)  # -> 512, 32, 32
        z = self.hid_act(z)
        z = self.upnorm1(z)
        z = self.upconv2(z, weight_5)  # -> 256, 64, 64
        z = self.hid_act(z)
        z = self.upnorm2(z)
        z = self.upconv3(z, weight_4)  # -> 128, 128, 128
        z = self.hid_act(z)
        z = self.upnorm3(z)
        z = self.upconv4(z, weight_3)  # -> 64, 256, 256
        z = self.hid_act(z)
        z = self.upnorm4(z)
        z = self.upconv5(z, weight_2)  # -> 32, 512, 512
        z = self.hid_act(z)
        z = self.upnorm5(z)
        z = self.upconv6(z, weight_1)  # -> 1, 1024, 1024
        z = self.hid_act(z)
        z = self.fc(z)

        return z*~mask_in.bool() + x*mask_in, mu, logvar

    def training_step(self, batch: Tensor) -> Tensor:
        I_gt, mask = batch
        I_out, mu, logvar = self(I_gt, mask)
        args = self.criterion(I_out, I_gt, mask, mu, logvar)
        metrics = {k: v.item() for k, v in zip(self.criterion.labels, args)}
        self.log_dict(metrics, prog_bar=True, enable_graph=True)
        return args[-1]
        
    def configure_optimizers(self):
        encoder_param_group = defaultdict(list)
        decoder_param_group = defaultdict(list)

        for name, param in self.named_parameters():
            if name.startswith('fconv'):
                encoder_param_group['params'].append(param)
            else:
                decoder_param_group['params'].append(param)

        optimizer = torch.optim.Adam([
            {'params': encoder_param_group['params'], **self.hyperparams[0]},
            {'params': decoder_param_group['params'], **self.hyperparams[1]}
        ])

        return optimizer
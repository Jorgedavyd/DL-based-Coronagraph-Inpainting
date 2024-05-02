from torch.fft import fftn, ifftn
from typing import Tuple, Union
from torch import nn
from torch import Tensor
import torch
from torch.nn import functional as F
from model import TrainingPhase


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

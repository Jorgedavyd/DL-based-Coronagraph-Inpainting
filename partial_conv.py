from torch import nn
import torch
from typing import Union, Tuple
import torch.nn.functional as F
from torch import Tensor


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        if "update_mask" in kwargs:
            self.update_mask = kwargs["update_mask"]
            kwargs.pop("update_mask")
        else:
            self.update_mask = True
        super(PartialConv2d, self).__init__(*args, **kwargs)
        self.sum_1 = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.mask_updater = torch.ones_like(self.weight, requires_grad=False).to("cuda")

    def _part_conv_forward(
        self, input: torch.Tensor, mask_in: torch.Tensor
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
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

        # W^T (X \odot M) + b
        out = super(PartialConv2d, self).forward(torch.mul(input, mask_in))

        # W^T (X \odot M) * \frac{sum(1)}{sum(M)} + b
        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1)
            out = torch.mul(out - bias, self.sum_1 / (sum_m + 1e-8)) + bias
        if self.update_mask:
            return out, updated_mask
        else:
            return out

    def forward(
        self, input: Tensor, mask_in: Tensor
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        return self._part_conv_forward(input, mask_in)


class FourierPartialConv2d(PartialConv2d):

    def __init__(self, *args, **kwargs):
        super(FourierPartialConv2d, self).__init__(*args, **kwargs)
        self.fft = torch.fft.fftn
        self.ifft = torch.fft.ifftn
        self.fft_conv = PartialConv2d(
            self.in_channels * 2,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            update_mask=False,
        )

    def forward(self, input: Tensor, mask_in: Tensor) -> Tuple[Tensor, Tensor] | Tensor:
        # Forward pass with Fourier transform
        out_complex = self.fft(input * mask_in, dim=(-2, -1))  # Compute FFT
        # Separate real and imaginary parts and compute convolution
        input_fft = torch.cat([out_complex.real, out_complex.imag], dim=1)
        out_fft = self.fft_conv(input_fft, torch.cat([mask_in, mask_in], dim=1))
        # Partial convolution forward
        out, mask = super(FourierPartialConv2d, self).forward(input, mask_in)
        return out + out_fft, mask

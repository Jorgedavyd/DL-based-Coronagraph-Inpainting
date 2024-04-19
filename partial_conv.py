from torch import nn
import torch
from typing import Union, Tuple
import torch.nn.functional as F
from torch import Tensor


class PartialConv2d(nn.Conv2d):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = True
        super(PartialConv2d, self).__init__(*args, **kwargs)
        self.sum_1 = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.mask_updater = torch.ones_like(self.weight, requires_grad=False).to('cuda')

    def _part_conv_forward(self, input: torch.Tensor, mask_in: torch.Tensor) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        # The mask update could be seen as a maxpooling operation
        with torch.no_grad():
            # This outputs an image that represents each sum(M)
            sum_m = F.conv2d(mask_in, self.mask_updater, None, self.stride, self.padding, self.dilation)
            
            updated_mask = torch.clamp_max(sum_m, 1)

        # W^T (X \odot M) + b
        out = super(PartialConv2d, self).forward(torch.mul(input, mask_in))

        # W^T (X \odot M) * \frac{sum(1)}{sum(M)} + b
        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1)
            out = torch.mul(out - bias, self.sum_1 / (sum_m + 1e-8)) + bias

        return out, updated_mask if self.return_mask else out

    def forward(self, input: Tensor, mask_in: Tensor) -> Union[Tuple[Tensor, Tensor], Tensor]:
        return self._part_conv_forward(input, mask_in)

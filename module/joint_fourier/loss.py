from typing import Iterable, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .. import pytorch_ssim


class Loss(nn.Module):
    def __init__(self, alpha: Iterable = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) -> None:
        super().__init__()
        self.ssim_loss = pytorch_ssim.SSIM()
        self.l1_loss = F.l1_loss
        self.labels = [
            "L1 Inner Loss",
            "L1 XOR Loss",
            "PSNR Inner Loss",
            "PSNR XOR Loss",
            "SSIM Inner Loss",
            "SSIM XOR Loss",
            "Overall",
        ]
        self.alpha = alpha
        self.factors = {k: v for k, v in zip(self.labels[:-1], self.alpha)}

    def forward(self, I_out, I_gt, mask_in, mask_out) -> Tuple[Tensor, ...]:
        # Getting the batch_size (b), number of channels (c), heigh (h) and width (w)
        b, c, h, w = I_out.shape
        # N_{I_{gt}} = C \times H \times W
        N_I_gt: float = c * h * w
        # for diff terms of the loss function
        mathcal_M: Tensor = mask_in.bool() ^ mask_out.bool()

        I_out_masked_inner = ~mask_out.bool() * I_out
        I_out_masked_diff = mathcal_M * I_out

        I_gt_masked_inner = ~mask_out.bool() * I_gt
        I_gt_masked_diff = mathcal_M * I_gt

        L_pixel_inner: Tensor = (
            (1 / N_I_gt)
            * self.alpha[0]
            * self.l1_loss(I_out_masked_inner, I_gt_masked_inner)
        )

        L_pixel_diff: Tensor = (
            (1 / N_I_gt)
            * self.alpha[1]
            * self.l1_loss(I_out_masked_diff, I_gt_masked_diff)
        )

        L_psnr_inner = (
            self.alpha[2]
            * 20
            * torch.log10(torch.sqrt(F.mse_loss(I_out_masked_inner, I_gt_masked_inner)))
        )

        L_psnr_diff = (
            self.alpha[3]
            * 20
            * torch.log10(torch.sqrt(F.mse_loss(I_out_masked_diff, I_gt_masked_diff)))
        )

        L_ssim_inner = self.alpha[4] * (
            -self.ssim_loss(I_out_masked_inner, I_gt_masked_inner)
        )

        L_ssim_diff = self.alpha[5](
            -self.ssim_loss(I_out_masked_diff, I_gt_masked_diff)
        )

        return (
            L_pixel_inner,
            L_pixel_diff,
            L_psnr_inner,
            L_psnr_diff,
            L_ssim_inner,
            L_ssim_diff,
            L_pixel_inner
            + L_pixel_diff
            + L_psnr_inner
            + L_psnr_diff
            + L_ssim_inner
            + L_ssim_diff,
        )

# General utils
from torch import nn
from typing import Tuple, Iterable, List, Callable
from torch import Tensor
import torch
from torch.nn.functional import mse_loss
from ..utils import (
    FeatureExtractor
)

class Loss(nn.Module):
    """
    # Fourier Variational Autoencoder Loss
    nn.Module implementation for inpainting training
    """

    def __init__(self, beta: float = 1, alpha: Iterable = [4, 6, 0.05, 110, 120, 0.1, 30]) -> None:
        super().__init__()
        self.alpha: Iterable = alpha
        self.fe = FeatureExtractor()
        self.beta = beta
        self.labels = [
            "Pixel",
            "Perceptual",
            "Style",
            "Total variance",
            "KL Divergence",
            "Overall",
        ]

        self.factors = {
            "Pixel": alpha[0],
            "Perceptual": alpha[1],
            "Style": alpha[2],
            "Total variance": alpha[3],
            "KL Divergence": self.beta,
        }

        sample_tensor: Tensor = torch.randn(32, 1, 1024, 1024)
        self.dim_per_layer: List[List[int]] = []
        F_p: List[int] = []
        N_phi_p: List[int] = []

        for feature_layer in self.fe(sample_tensor):
            c, h, w = feature_layer.shape[1:]
            F_p.append(c**3 * h * w)
            N_phi_p.append(c * h * w)

        self.F_p: Tensor = Tensor(F_p)
        self.N_phi_p: Tensor = Tensor(N_phi_p)
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        self.l1_loss = nn.L1Loss(reduction='sum')
    def forward(
        self, I_out: Tensor, I_gt: Tensor, mu: Tensor, logvar: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # Getting the batch_size (b), number of channels (c), heigh (h) and width (w)
        b, c, h, w = I_out.shape
        # N_{I_{gt}} = C \times H \times W
        N_I_gt: float = c * h * w

        L_pixel: Tensor = (1 / N_I_gt) * self.l1_loss(I_out, I_gt)

        psi_masked_out: List[Tensor] = self.fe(I_out)
        psi_masked_gt: List[Tensor] = self.fe(I_gt)

        L_perceptual: Tensor = (
            (
                Tensor(
                    [
                        torch.norm(phi_out - phi_gt, p=1)
                        for phi_out, phi_gt in zip(
                            psi_masked_out,
                            psi_masked_gt,
                        )
                    ]
                )
                / self.N_phi_p
            ).sum()
        )
        # Style loss
        ## Changing size of the features from the images
        change_dim: Callable[[List[Tensor]], List[Tensor]] = lambda P: [
            tensor.view(tensor.shape[0], tensor.shape[1], -1) for tensor in P
        ]

        psi_masked_out: List[Tensor] = change_dim(psi_masked_out)
        psi_masked_gt: List[Tensor] = change_dim(psi_masked_gt)

        L_n: Callable[[List[Tensor], List[Tensor]], List[Tensor]] = (
            lambda out_list, gt_list: Tensor(
                [
                    torch.norm(
                        out @ out.transpose(-2, -1) - gt @ gt.transpose(-2, -1), p=1
                    )
                    for out, gt in zip(out_list, gt_list)
                ]
            )
        )

        L_style: Tensor = (
            (L_n(psi_masked_out, psi_masked_gt)) / self.F_p
        ).sum()

        L_tv = torch.mean(
            torch.abs(I_out[:, :, :, :-1] - I_out[:, :, :, 1:])
        ) + torch.mean(torch.abs(I_out[:, :, :-1, :] - I_out[:, :, 1:, :]))

        # KL divergence
        L_kl =  (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))

        return (
            L_pixel,
            L_perceptual,
            L_style,
            L_tv,
            L_kl,
            self.alpha[0] * L_pixel + self.alpha[1] * L_perceptual + self.alpha[2] * L_style + self.alpha[3] * L_tv + self.beta * L_kl,
        )
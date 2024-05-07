from torch import nn
from typing import Iterable, List, Callable, Tuple
import torch
from torch import Tensor
from ..utils import FeatureExtractor


class NewInpaintingLoss(nn.Module):
    """
    # Inpainting Loss
    nn.Module implementation for inpainting training
    """

    def __init__(self, alpha: Iterable = [4, 6, 0.05, 110, 120, 0.1]) -> None:
        super().__init__()
        self.alpha: Iterable = alpha
        self.fe = FeatureExtractor()

        self.labels = [
            "Pixel Loss",
            "Perceptual Loss",
            "Style Loss",
            "Total variance",
            "Overall",
        ]
        self.factors = {
            "Pixel inner": alpha[0],
            "Pixel diff": alpha[1],
            "Perceptual": alpha[2],
            "Style inner": alpha[3],
            "Style diff": alpha[4],
            "Total variance": alpha[5],
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

    def forward(
        self, I_out: Tensor, I_gt: Tensor, M_l_1: Tensor, M_l_2: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # Getting the batch_size (b), number of channels (c), heigh (h) and width (w)
        b, c, h, w = I_out.shape
        # N_{I_{gt}} = C \times H \times W
        N_I_gt: float = c * h * w
        # Default
        diff: Tensor = I_out - I_gt
        # for diff terms of the loss function
        mathcal_M: Tensor = M_l_1.bool() ^ M_l_2.bool()

        L_inner: Tensor = torch.norm(~M_l_2.bool() * diff, p=1)
        L_diff: Tensor = torch.norm(mathcal_M * diff, p=1)
        L_pixel: Tensor = (1 / N_I_gt) * (
            self.alpha[0] * L_inner + self.alpha[1] * L_diff
        )

        psi_out_mathcal_masked: List[Tensor] = self.fe(I_out * mathcal_M)
        psi_gt_mathcal_masked: List[Tensor] = self.fe(I_gt * mathcal_M)

        psi_masked_out: List[Tensor] = self.fe(I_out * ~M_l_2.bool())
        psi_masked_gt: List[Tensor] = self.fe(I_gt * ~M_l_2.bool())

        L_perceptual: Tensor = (
            self.alpha[2]
            * (
                Tensor(
                    [
                        torch.norm(phi_out - phi_gt, p=1)
                        + torch.norm(psi_out_m - psi_gt_m, p=1)
                        for phi_out, phi_gt, psi_out_m, psi_gt_m in zip(
                            psi_masked_out,
                            psi_masked_gt,
                            psi_out_mathcal_masked,
                            psi_gt_mathcal_masked,
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
        psi_mathcal_M_out: List[Tensor] = change_dim(psi_out_mathcal_masked)
        psi_mathcal_M_gt: List[Tensor] = change_dim(psi_gt_mathcal_masked)

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
            (
                self.alpha[3] * L_n(psi_masked_out, psi_masked_gt)
                + self.alpha[4] * L_n(psi_mathcal_M_out, psi_mathcal_M_gt)
            )
            / self.F_p
        ).sum()

        L_tv = self.alpha[5] * (
            torch.mean(torch.abs(I_out[:, :, :, :-1] - I_out[:, :, :, 1:]))
            + torch.mean(torch.abs(I_out[:, :, :-1, :] - I_out[:, :, 1:, :]))
        )

        return (
            L_pixel,
            L_perceptual,
            L_style,
            L_tv,
            L_pixel + L_perceptual + L_style + L_tv,
        )


class OldInpaintingLoss(nn.Module):
    def __init__(self, alpha: Iterable = [4, 6, 4, 0.05, 110, 120, 110]) -> None:
        super().__init__()
        self.alpha: Iterable = alpha
        self.fe = FeatureExtractor()
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

    def forward(
        self, I_out: Tensor, I_gt: Tensor, M_l_1: Tensor, M_l_2: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # Getting the batch_size (b), number of channels (c), heigh (h) and width (w)
        b, c, h, w = I_out.shape
        # N_{I_{gt}} = C \times H \times W
        N_I_gt: float = c * h * w
        # Default
        diff: Tensor = I_out - I_gt
        # for diff terms of the loss function
        mathcal_M: Tensor = M_l_1.bool() ^ M_l_2.bool()

        # Per pixel loss
        L_outter: Tensor = torch.norm(M_l_1 * diff, p=1)
        L_inner: Tensor = torch.norm(~M_l_2.bool() * diff, p=1)
        L_diff: Tensor = torch.norm(mathcal_M * diff, p=1)

        L_pixel: Tensor = (1 / N_I_gt) * (
            self.alpha[0] * L_outter + self.alpha[1] * L_inner + self.alpha[2] * L_diff
        )

        # Extracting features for the perceptual and style losses
        psi_out: List[Tensor] = self.fe(I_out)
        psi_gt: List[Tensor] = self.fe(I_gt)

        psi_masked_out: List[Tensor] = self.fe(I_out * ~M_l_2.bool())
        psi_masked_gt: List[Tensor] = self.fe(I_gt * ~M_l_2.bool())

        L_perceptual: Tensor = (
            self.alpha[3]
            * (
                Tensor(
                    [
                        torch.norm(phi_out - phi_gt, p=1)
                        + torch.norm(psi_out_m - psi_gt_m, p=1)
                        for phi_out, phi_gt, psi_out_m, psi_gt_m in zip(
                            psi_out, psi_gt, psi_masked_out, psi_masked_gt
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
        psi_out: List[Tensor] = change_dim(psi_out)
        psi_gt: List[Tensor] = change_dim(psi_gt)
        psi_masked_out: List[Tensor] = change_dim(psi_masked_out)
        psi_masked_gt: List[Tensor] = change_dim(psi_masked_gt)
        psi_mathcal_M_out: List[Tensor] = change_dim(self.fe(mathcal_M * I_out))
        psi_mathcal_M_gt: List[Tensor] = change_dim(self.fe(mathcal_M * I_gt))

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
            (
                self.alpha[4] * L_n(psi_out, psi_gt)
                + self.alpha[5] * L_n(psi_masked_out, psi_masked_gt)
                + self.alpha[6] * L_n(psi_mathcal_M_out, psi_mathcal_M_gt)
            )
            / self.F_p
        ).sum()

        # Total variance loss

        return L_pixel, L_perceptual, L_style, L_pixel + L_perceptual + L_style

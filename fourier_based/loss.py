from torch import nn
from typing import Iterable, List, Callable, Tuple
import torch
from torch import Tensor
from torchvision.models import vgg19, VGG19_Weights
import torch.nn.functional as F
from joint_fourier import pytorch_ssim

def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + torch.mean(
        torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])
    )
    return loss

class FeatureExtractor(nn.Module):
    def __init__(self, layers: Iterable = [4, 9, 18]) -> None:
        super().__init__()
        self.layers = list(map(str, layers))
        # Setting vgg19
        self.model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        # Freeze gradients
        for param in self.model.parameters():
            param.requires_grad = False
        # Setting the transformation
        self.transform = VGG19_Weights.IMAGENET1K_V1.transforms(antialias=True)

    def forward(self, input: Tensor) -> List[Tensor]:
        x = self.transform(torch.cat((input, input, input), -3))
        features = []
        for name, layer in self.model.features.named_children():
            x = layer(x)
            if name in self.layers:
                features.append(x)
                if name == str(18):
                    break
        return features

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

        L_tv = self.alpha[5] * (torch.mean(
            torch.abs(I_out[:, :, :, :-1] - I_out[:, :, :, 1:])
        ) + torch.mean(torch.abs(I_out[:, :, :-1, :] - I_out[:, :, 1:, :])))

        return (
            L_pixel,
            L_perceptual,
            L_style,
            L_tv,
            L_pixel + L_perceptual + L_style + L_tv,
        )

class FourierModelCriterion(nn.Module):
    """
    # Inpainting Loss
    nn.Module implementation for inpainting training
    """

    def __init__(self, alpha: Iterable = [4, 6, 0.05, 110, 120, 0.1, 30]) -> None:
        super().__init__()
        self.alpha: Iterable = alpha
        self.fe = FeatureExtractor()

        self.labels = [
            "Pixel",
            "Perceptual",
            "Style",
            "Total variance",
            "ELBO",
            "Overall"
        ]

        self.factors = {
            "Pixel": alpha[0],
            "Perceptual": alpha[1],
            "Style": alpha[2],
            "Total variance": alpha[3],
            "ELBO": alpha[4]
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
    def gaussian_likelihood(self, mean, logscale, sample):
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(dim = (-1, -2, -3))
        return kl
    
    def forward(
        self, I_out: Tensor, I_gt: Tensor, mask: Tensor, mean: Tensor, log_var: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # Getting the batch_size (b), number of channels (c), heigh (h) and width (w)
        b, c, h, w = I_out.shape
        # N_{I_{gt}} = C \times H \times W
        N_I_gt: float = c * h * w
        # Default
        diff: Tensor = I_out - I_gt

        L_pixel: Tensor = self.alpha[0] *(1 / N_I_gt) *torch.norm(~mask.bool() * diff, p=1)

        psi_masked_out: List[Tensor] = self.fe(I_out * ~mask.bool())
        psi_masked_gt: List[Tensor] = self.fe(I_gt * ~mask.bool())

        L_perceptual: Tensor = (
            self.alpha[1]
            * (
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
            (
                self.alpha[2] * L_n(psi_masked_out, psi_masked_gt)
            )
            / self.F_p
        ).sum()

        L_tv = self.alpha[3] * torch.mean(
            torch.abs(I_out[:, :, :, :-1] - I_out[:, :, :, 1:])
        ) + torch.mean(torch.abs(I_out[:, :, :-1, :] - I_out[:, :, 1:, :]))

        # elbo
        L_elbo = self.alpha[4] *(F.binary_cross_entropy(I_out, I_gt, reduction='sum') + 
                                 (- 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())))

        return (
            L_pixel,
            L_perceptual,
            L_style,
            L_tv,
            L_elbo,
            L_pixel + L_perceptual + L_style + L_tv + L_elbo,
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

        return L_pixel, L_perceptual, L_style

class UNetLoss(nn.Module):
    def __init__(self, alpha: Iterable = [1, 6, 0.05, 110, 120]) -> None:
        super().__init__()
        self.alpha = alpha
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
        self, I_out: Tensor, I_gt: Tensor, mask_in: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # Getting the batch_size (b), number of channels (c), heigh (h) and width (w)
        b, c, h, w = I_out.shape
        # N_{I_{gt}} = C \times H \times W
        N_I_gt: float = c * h * w
        # Default
        diff: Tensor = I_out - I_gt

        # Per pixel loss
        L_outter: Tensor = torch.norm(mask_in * diff, p=1)
        L_inner: Tensor = torch.norm(~mask_in.bool() * diff, p=1)

        L_pixel: Tensor = (1 / N_I_gt) * (
            self.alpha[0] * L_outter + self.alpha[1] * L_inner
        )

        # Extracting features for the perceptual and style losses
        psi_out: List[Tensor] = self.fe(I_out)
        psi_gt: List[Tensor] = self.fe(I_gt)

        psi_masked_out: List[Tensor] = self.fe(I_out * ~mask_in.bool())
        psi_masked_gt: List[Tensor] = self.fe(I_gt * ~mask_in.bool())

        L_perceptual: Tensor = (
            self.alpha[2]
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
                self.alpha[3] * L_n(psi_out, psi_gt)
                + self.alpha[4] * L_n(psi_masked_out, psi_masked_gt)
            )
            / self.F_p
        ).sum()

        return L_pixel, L_perceptual, L_style

class CoronagraphConstraint(nn.Module):
    def __init__(self, lambd: Tensor) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, pred: Tensor) -> Tensor:

        return

class FourierDeluxeCriterion(nn.Module):
    def __init__(self, alpha: Iterable = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) -> None:
        self.ssim_loss = pytorch_ssim.SSIM()
        self.l1_loss = F.l1_loss
        self.labels = ['L1 Inner Loss', 'L1 XOR Loss', 'PSNR Inner Loss', 'PSNR XOR Loss', 'SSIM Inner Loss', 'SSIM XOR Loss', 'Overall']
        self.alpha = alpha
        self.factors = {k:v for k, v in zip(self.labels[:-1], self.alpha)}

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

        L_pixel_inner: Tensor = (1 / N_I_gt) * self.alpha[0] * self.l1_loss(I_out_masked_inner, I_gt_masked_inner)

        L_pixel_diff: Tensor = (1 / N_I_gt) * self.alpha[1] * self.l1_loss(I_out_masked_diff, I_gt_masked_diff)
        
        L_psnr_inner = self.alpha[2] * 20 * torch.log10(torch.sqrt(F.mse_loss(I_out_masked_inner, I_gt_masked_inner)))
        
        L_psnr_diff = self.alpha[3] * 20 * torch.log10(torch.sqrt(F.mse_loss(I_out_masked_diff, I_gt_masked_diff)))

        L_ssim_inner = self.alpha[4] * (- self.ssim_loss(I_out_masked_inner, I_gt_masked_inner))
            
        L_ssim_diff = self.alpha[5] (- self.ssim_loss(I_out_masked_diff, I_gt_masked_diff))

        return (
            L_pixel_inner,
            L_pixel_diff,
            L_psnr_inner,
            L_psnr_diff,
            L_ssim_inner,
            L_ssim_diff,
            L_pixel_inner + L_pixel_diff + L_psnr_inner + L_psnr_diff + L_ssim_inner + L_ssim_diff
        )
    


        
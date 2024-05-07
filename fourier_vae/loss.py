# General utils
from torch import nn
from typing import Iterable, List, Callable, Tuple
from torch import Tensor
import torch.nn.functional as F
import torch
# Feature extraction
from torchvision.models import vgg19, VGG19_Weights

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

class Loss(nn.Module):
    """
    # Fourier Variational Autoencoder Loss
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



        
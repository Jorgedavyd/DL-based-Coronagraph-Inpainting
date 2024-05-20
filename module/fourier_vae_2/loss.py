# General utils
from torch import nn
from typing import Tuple
from torch import Tensor
import torch
from torch.nn.functional import mse_loss

class Loss(nn.Module):
    """
    # Fourier Variational Autoencoder Loss
    nn.Module implementation for inpainting training
    """

    def __init__(self, beta = 1) -> None:
        super().__init__()
        
        self.beta = beta
        
        self.labels = [
            "Reconstruction",
            "KL Divergence",
            "Overall",
        ]

    def forward(self, I_out: Tensor, I_gt: Tensor, mu: Tensor, logvar: Tensor) -> Tuple[Tensor, ...]:

        rec = mse_loss(I_out, I_gt, reduction = 'sum')
        
        L_kl =  - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return (
            rec, 
            L_kl, 
            rec + self.beta * L_kl
        )
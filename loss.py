from torch import nn
from typing import Iterable, List, Dict
import torch
from torchvision.models.vgg import vgg16
from torch import Tensor
import torchvision.transforms as tt
from torchvision.models import vgg19, VGG19_Weights


class FeatureExtractor(nn.Module):
    def __init__(self, layers: Iterable = [4, 9, 18]):
        super().__init__()
        self.layers = list(map(str, layers))
        #Setting vgg19
        self.model = vgg19(weights = VGG19_Weights.IMAGENET1K_V1)
        #Freeze gradients
        for param in self.model.parameters():
            param.requires_grad = False
        # Setting the transformation
        self.transform = VGG19_Weights.IMAGENET1K_V1.transforms(antialias = True)

    def forward(self, input: Tensor):
        x = self.transform(torch.cat((input, input, input), -3))
        features = []
        for name, layer in self.model.features.named_children():
            x = layer(x)
            if name in self.layers:
                features.append(x)
                if name == str(18):
                    break
        return features

class InpaintingLoss(nn.Module):
    """
    # Inpainting Loss
    nn.Module implementation for inpainting training
    """
    def __init__(self, factors: Iterable = [6, 1, 0.05, 120, 120]) -> None:
        assert (len(factors) == 5), 'Must be 5 factor parameters for: L_holes, L_valid, L_percerptual, L_style_out, L_style_comp'
        super().__init__()
        self.factors = factors
        self.fe = FeatureExtractor()

        sample_tensor: Tensor = torch.randn(32, 1, 1024, 1024)
        self.dim_per_layer: List[List[int]] = []
        K_p_final: List[int] = []
        N_gt: List[int] = []

        for feature_layer in self.fe(sample_tensor):
            c, h, w = feature_layer.shape[1:]
            K_p_final.append(1/(c**3*h*w))
            N_gt.append(c*h*w)

        self.K_p_final: Tensor = Tensor(K_p_final)
        self.N_gt: Tensor = Tensor(N_gt)

    def forward(self, prediction: Tensor, ground_truth: Tensor, prior_mask: Tensor, updated_mask: Tensor) -> Tensor:
        #Getting the shape of the output tensor, expected batch_size, 1, 1024, 1024
        b, c, h, w = prediction.shape
        pixel_factor: float = 1/(c*h*w) # 1/N_{I_{gt}}
        diff: Tensor = prediction - ground_truth

        # Per pixel loss
        L_hole: Tensor = self.factors[0] * pixel_factor * torch.norm(torch.mul(~prior_mask.bool(),diff), p = 1)
        L_valid: Tensor = self.factors[1] * pixel_factor * torch.norm(torch.mul(prior_mask, diff), p = 1)

        # Feature extraction psi_out, 
        psi_out: List[Tensor] = self.fe(prediction)
        psi_gt: List[Tensor] = self.fe(ground_truth)

        #Creating a new mask for the region of interest        
        new_mask: Tensor = ~prior_mask.bool() ^ ~updated_mask.bool() # This generates a mask that should be filled by the n-th layer

        #Masked for content
        psi_masked_out: List[Tensor] = self.fe(torch.mul(prediction, new_mask))
        psi_masked_gt: List[Tensor] = self.fe(torch.mul(ground_truth, new_mask))

        # Content and style terms for perceptual loss
        cont_term: Tensor = (Tensor([torch.norm(out-gt, p = 1) for out, gt in zip(psi_masked_out, psi_masked_gt)]) / self.N_gt).sum()
        style_term: Tensor = (Tensor([torch.norm(out-gt, p = 1) for out, gt in zip(psi_out, psi_gt)]) / self.N_gt).sum()

        #Perceptual loss
        L_perceptual: Tensor = self.factors[2] * (cont_term + style_term)

        # Style loss
        ## Performing all matrix multiplications for Gram Matrix
        psi_out: List[Tensor] = [tensor.view(tensor.shape[0], tensor.shape[1], -1) for tensor in psi_out]
        psi_gt: List[Tensor] = [tensor.view(tensor.shape[0], tensor.shape[1], -1) for tensor in psi_gt]
        psi_masked_out: List[Tensor] = [tensor.view(tensor.shape[0], tensor.shape[1], -1) for tensor in psi_masked_out]
        psi_masked_gt: List[Tensor] = [tensor.view(tensor.shape[0], tensor.shape[1], -1) for tensor in psi_masked_gt]
        
        L_style_out: Tensor = self.factors[3]*(self.K_p_final*Tensor([torch.norm(torch.matmul(out, out.transpose(-2,-1)) - torch.matmul(gt, gt.transpose(-2, -1)), p = 1) for out, gt in zip(psi_out, psi_gt)])).sum()
        L_style_comp: Tensor = self.factors[4]*(self.K_p_final*Tensor([torch.norm(torch.matmul(out, out.transpose(-2,-1)) - torch.matmul(gt, gt.transpose(-2, -1)), p = 1) for out, gt in zip(psi_masked_out, psi_masked_gt)])).sum()

        return L_hole, L_valid, L_perceptual, L_style_out, L_style_comp
    


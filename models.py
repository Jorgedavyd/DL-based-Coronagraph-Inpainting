from typing import Tuple, Union, List, Dict, Any, Callable, Optional, Iterable
from utils import PartialConv2d, _FourierConv, FourierMultiheadAttention, FourierPartialConv2d, ResidualFourierConvTranspose2d
from torch import Tensor
from torch import nn
from collections import defaultdict
from loss import NewInpaintingLoss
from torch.fft import fftn, ifftn
import torch.nn.functional as F
import torch
import numpy as np
from astropy.visualization import HistEqStretch, ImageNormalize
import matplotlib.pyplot as plt
from lightning import LightningModule
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

class InpaintingBase(LightningModule):

    peak_signal = PeakSignalNoiseRatio().to('cuda')
    ssim = StructuralSimilarityIndexMeasure().to('cuda')

    def training_step(self, batch) -> Tensor:
        x, mask_in = batch
        y, mask_out = self(x, mask_in)

        args = self.criterion(y, x, mask_in, mask_out)

        self.log_dict(
            {
                k:v for k,v in zip(self.criterion.labels, args)
            }
        )

        return args[-1]
    
    def validation_step(self, batch, batch_idx) -> Tensor:
        x, mask_in = batch 
        y, mask_out = self(x, mask_in)

        args = self.criterion(y, x, mask_in, mask_out)

        peak_signal = self.peak_signal(y, x)
        ssim = self.ssim(y, x)

        self.log_dict({
            "val_loss": args[-1],
            "val_peak_signal": peak_signal,
            "val_ssim": ssim
        }, prog_bar = True)

class VAEbase(LightningModule):
    peak_signal = PeakSignalNoiseRatio()
    ssim = StructuralSimilarityIndexMeasure()
    
    def training_step(self, batch) -> Tensor:
        x, mask_in = batch
        out = self(x, mask_in)

        args = self.criterion(out, x, mask_in)

        self.log_dict({k:v for k,v in zip(self.criterion.labels, args)})

        return args[-1]
    
    def validation_step(self, batch) -> Tensor:
        x, mask_in = batch
        out = self(x, mask_in)

        args = self.criterion(out, x, mask_in)

        peak_noise = self.peak_signal(out, x)

        ssim = self.ssim(out, x)

        self.log_dict({
            "val_loss": args[-1],
            "val_peak_signal": peak_noise,
            "val_ssim": ssim
        })

class SingleLayer(LightningModule):

    def __init__(self, res_arch: Tuple[int, ...], criterion_args):
        super().__init__()
        self.criterion = NewInpaintingLoss(*criterion_args)
        # Activation function
        self.act = nn.SiLU()
        # Partial convolutional layers
        self.conv_1 = PartialConv2d(1, 32, 3, 1, 1)
        self.conv_2 = PartialConv2d(32, 64, 3, 1, 1)
        self.res_1 = DefaultResidual(64, res_arch)
        self.conv_3 = PartialConv2d(64, 32, 3, 1, 1)
        self.conv_4 = PartialConv2d(32, 1, 3, 1, 1)

    def forward(self, input: Tensor, mask_in: Tensor) -> Tuple[Tensor, Tensor]:
        out, mask = self.conv_1(input, mask_in)
        out = self.act(out)
        out, mask = self.conv_2(out, mask)
        out = self.act(out)
        out, mask = self.res_1(out, mask)
        out = self.act(out)
        out, mask = self.conv_3(out, mask)
        out = self.act(out)
        out, mask = self.conv_4(out, mask)
        return out * ~mask_in.bool() + input * mask_in, mask

    def training_step(self, batch) -> None:
        # Defining the img and mask
        ground_truth, prior_mask = batch
        # forward pass for single layer
        x, updated_mask = self(ground_truth, prior_mask)
        # Compute each term of the loss function
        args = self.criterion(
            x, ground_truth, prior_mask, updated_mask
        )

        metrics = {k:v for k,v in zip(self.criterion.labels, args)}

        self.log_dict(metrics, prog_bar=True)
        
        return args[-1]

class UNetArchitecture(LightningModule):
    def __init__(self, criterion) -> None:
        super().__init__()
        self.criterion = criterion
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()
        self.max_pool = nn.MaxPool2d(2, 2)

        # Downsampling
        self.conv1 = PartialConv2d(1, 32, 3, 1, 1)
        self.norm1 = nn.BatchNorm2d(32)
        self.conv2 = PartialConv2d(32, 32, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(32)
        self.conv3 = PartialConv2d(32, 64, 3, 1, 1)
        self.norm3 = nn.BatchNorm2d(64)
        self.conv4 = PartialConv2d(64, 64, 3, 1, 1)
        self.norm4 = nn.BatchNorm2d(64)
        self.conv5 = PartialConv2d(64, 128, 3, 1, 1)
        self.norm5 = nn.BatchNorm2d(128)
        self.conv6 = PartialConv2d(128, 256, 3, 1, 1)
        self.norm6 = nn.BatchNorm2d(256)

        # Upsampling
        self.upconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  # -> 64, 64
        self.upnorm1 = nn.BatchNorm2d(128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)  # -> 128, 128
        self.upnorm2 = nn.BatchNorm2d(64)
        self.upconv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)  # -> 256, 256
        self.upnorm3 = nn.BatchNorm2d(64)
        self.upconv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # -> 512, 512
        self.upnorm4 = nn.BatchNorm2d(32)
        self.upconv5 = nn.ConvTranspose2d(32, 1, 4, 2, 1)  # -> 1024, 1024

    def encoder(self, x: Tensor, mask_in: Tensor) -> Tuple[Tensor, ...]:
        # First layer
        x_0, mask_in = self.conv1(x, mask_in)  # 1, 1024, 1024 -> 32, 1024, 1024
        x_1 = self.norm1(x_0)
        x_1 = self.relu(x_1)  # activation function
        # Downsampling
        x_1, mask_in = self.max_pool(x_1), self.max_pool(
            mask_in
        )  # 32, 1024, 1024 -> 32, 512, 512

        # Second Layer
        x_1, mask_in = self.conv2(x_1, mask_in)  # 32, 512, 512-> 32, 512, 512
        x_2 = self.norm2(x_1)
        x_2 = self.relu(x_2)

        # Downsampling
        x_2, mask_in = self.max_pool(x_2), self.max_pool(
            mask_in
        )  # 32, 512, 512 -> 32, 256, 256

        x_2, mask_in = self.conv3(x_2, mask_in)  # 32, 256, 256 -> 64, 256, 256
        x_3 = self.norm3(x_2)
        x_3 = self.relu(x_3)

        # Downsampling
        x_3, mask_in = self.max_pool(x_3), self.max_pool(
            mask_in
        )  # 64, 256, 256  -> 64, 128, 128

        x_3, mask_in = self.conv4(x_3, mask_in)  # 64, 128, 128 -> 64, 128, 128
        x_4 = self.norm4(x_3)
        x_4 = self.relu(x_4)

        # Downsampling
        x_4, mask_in = self.max_pool(x_4), self.max_pool(
            mask_in
        )  # 64, 128, 128 -> 64, 64, 64

        x_4, mask_in = self.conv5(x_4, mask_in)  # 64, 64, 64 -> 128, 64, 64
        x_5 = self.norm5(x_4)
        x_5 = self.relu(x_5)

        # Downsampling
        x_5, mask_in = self.max_pool(x_5), self.max_pool(
            mask_in
        )  # 128, 64, 64 -> 128, 32, 32

        x_5, mask_in = self.conv6(x_5, mask_in)  # 128, 32, 32 -> 256, 32, 32
        out = self.norm6(x_5)
        out = self.relu(out)

        return x_1, x_2, x_3, x_4, out

    def decoder(
        self, x_1: Tensor, x_2: Tensor, x_3: Tensor, x_4: Tensor, out: Tensor
    ) -> Tensor:
        out = self.upconv1(out) + x_4  # 256, 32, 32 -> 128, 64, 64
        out = self.upnorm1(out)
        out = self.silu(out)

        out = self.upconv2(out) + x_3  # 128, 64, 64 -> 64, 128, 128
        out = self.upnorm2(out)
        out = self.silu(out)

        out = self.upconv3(out) + x_2  # 64, 128, 128 -> 64, 256, 256
        out = self.upnorm3(out)
        out = self.silu(out)

        out = self.upconv4(out) + x_1  # 64, 256, 256 -> 32, 512, 512
        out = self.upnorm4(out)
        out = self.silu(out)

        out = self.upconv5(out)  # 32, 512, 512 -> 1, 1024, 1024
        out = self.silu(out)

        return out

    def forward(self, x: Tensor, mask_in: Tensor) -> Tensor:
        x_1, x_2, x_3, x_4, out = self.encoder(x, mask_in)
        out = self.decoder(x_1, x_2, x_3, x_4, out)
        return out

    def training_step(self, batch):
        img, mask = batch
        pred = self(img, mask)
        # Computing the loss
        args = self.criterion(pred, img, mask)
        # Defining the metrics that will be wrote
        metrics = {k:v for k,v in zip(self.criterion.labels, args)}
        
        self.log_dict(metrics)
        
        return args[-1]

    @torch.no_grad()
    def validation_step(self, batch) -> None:
        img, mask = batch
        pred = self(img, mask)
        # Computing the loss
        args = self.criterion(pred, img, mask)
        # Defining the metrics that will be wrote
        metrics = {k:v for k,v in zip(self.criterion.labels, args)}
        
        self.log_dict(metrics)
        
        return args[-1]

class UNetArchitectureDeluxe(LightningModule):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion
        self.up_act = lambda input: F.relu(6 * F.sigmoid(input))
        self.down_act = nn.SiLU()
        self.upsample = lambda input: nn.functional.interpolate(
            input=input, scale_factor=2, mode="nearest"
        )
        self.max_pool = nn.MaxPool2d(2, 2)
        # Encoder
        self.conv1 = PartialConv2d(1, 32, 3, 1, 1)
        self.norm1 = nn.BatchNorm2d(32)

        self.conv2 = PartialConv2d(32, 64, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(64)

        self.conv3_init = PartialConv2d(64, 128, 3, 1, 1)
        self.norm3_init = nn.BatchNorm2d(128)

        ## Residual layer
        self.conv3_1 = PartialConv2d(128, 128, 15, 1, 7)
        self.norm3_1 = nn.BatchNorm2d(128)

        self.conv3_2 = PartialConv2d(128, 128, 13, 1, 6)
        self.norm3_2 = nn.BatchNorm2d(128)

        self.conv3_3 = PartialConv2d(128, 128, 11, 1, 5)
        self.norm3_3 = nn.BatchNorm2d(128)

        self.conv4_init = PartialConv2d(128, 256, 3, 1, 1)
        self.norm4_init = nn.BatchNorm2d(256)

        # 3 Residual layer
        self.conv4_1 = PartialConv2d(256, 256, 17, 1, 8)
        self.norm4_1 = nn.BatchNorm2d(256)

        self.conv4_2 = PartialConv2d(256, 256, 15, 1, 7)
        self.norm4_2 = nn.BatchNorm2d(256)

        self.conv4_3 = PartialConv2d(256, 256, 11, 1, 5)
        self.norm4_3 = nn.BatchNorm2d(256)

        self.conv5 = PartialConv2d(256, 512, 3, 1, 1)
        self.norm5 = nn.BatchNorm2d(512)

        # Decoder
        self.upconv1_init = PartialConv2d(512, 256, 3, 1, 1)
        self.upnorm1_init = nn.BatchNorm2d(256)

        ## Residual layer
        self.upconv1_1 = PartialConv2d(256, 256, 11, 1, 5)
        self.upnorm1_1 = nn.BatchNorm2d(256)

        self.upconv1_2 = PartialConv2d(256, 256, 13, 1, 6)
        self.upnorm1_2 = nn.BatchNorm2d(256)

        self.upconv1_3 = PartialConv2d(256, 256, 15, 1, 7)
        self.upnorm1_3 = nn.BatchNorm2d(256)

        self.upconv2_init = PartialConv2d(256, 128, 3, 1, 1)
        self.upnorm2_init = nn.BatchNorm2d(128)

        ## Residual layer
        self.upconv2_1 = PartialConv2d(128, 128, 11, 1, 5)
        self.upnorm2_1 = nn.BatchNorm2d(128)

        self.upconv2_2 = PartialConv2d(128, 128, 13, 1, 6)
        self.upnorm2_2 = nn.BatchNorm2d(128)

        self.upconv2_3 = PartialConv2d(128, 128, 15, 1, 7)
        self.upnorm2_3 = nn.BatchNorm2d(128)

        self.upconv3 = PartialConv2d(128, 64, 3, 1, 1)
        self.upnorm3 = nn.BatchNorm2d(128)
        self.upconv4 = PartialConv2d(64, 32, 3, 1, 1)
        self.upnorm4 = nn.BatchNorm2d(128)
        self.upconv5 = PartialConv2d(32, 1, 3, 1, 1)
        self.upnorm5 = nn.BatchNorm2d(128)

    def encoder(self, x: Tensor, mask_in: Tensor) -> Tuple[Tensor, Tensor]:
        x, mask_in = self.conv1(x, mask_in)
        x = self.norm1(x)
        x = self.down_act(x)

        x, mask_in = self.conv2(x, mask_in)
        x = self.norm2(x)
        x = self.down_act(x)

        x, mask_in = self.conv3_init(x, mask_in)
        x = self.norm3_init(x)
        x = self.down_act(x)

        # Max pooling layer
        x, mask_in = self.max_pool(x), self.max_pool(mask_in)  # -> 512,512

        # Residual layer
        x_1, mask_in = self.conv3_1(x, mask_in)
        x_1 = self.norm3_1(x_1)
        x_1 = self.down_act(x_1)

        x_1, mask_in = self.conv3_2(x_1, mask_in)
        x_1 = self.norm3_2(x_1)
        x_1 = self.down_act(x_1)

        x_1, mask_in = self.conv3_3(x_1, mask_in)
        x_1 = self.norm3_3(x_1)
        x_1 = self.down_act(x_1)

        x += x_1  # 512, 512

        x, mask_in = self.max_pool(x), self.max_pool(mask_in)  # -> 256,256

        x, mask_in = self.conv4_init(x, mask_in)
        x = self.norm4_init(x)
        x = self.down_act(x)

        # Residual layer
        x_2, mask_in = self.conv4_1(x, mask_in)
        x_2 = self.norm4_1(x_2)
        x_2 = self.down_act(x_2)

        x_2, mask_in = self.conv4_2(x_2, mask_in)
        x_2 = self.norm4_2(x_2)
        x_2 = self.down_act(x_2)

        x_2, mask_in = self.conv4_3(x_2, mask_in)
        x_2 = self.norm4_3(x_2)
        x_2 = self.down_act(x_2)

        x += x_2  # 256, 256

        x, mask_in = self.max_pool(x), self.max_pool(mask_in)  # -> 128,128

        x, mask_in = self.conv5(x, mask_in)
        x = self.down_act(x)
        x = self.norm5(x)

        x, mask_in = self.max_pool(x), self.max_pool(mask_in)  # -> 64, 64

        return x, x_1, x_2, mask_in

    def decoder(
        self, x: Tensor, x_1: Tensor, x_2: Tensor, mask_in: Tensor
    ) -> Tuple[Tensor, Tensor]:

        x, mask_in = self.upsample(x), self.upsample(mask_in)  # -> 512, 128,128

        x, mask_in = self.upconv1_init(x, mask_in)
        x = self.up_act(x)
        x = self.upnorm1_init(x)

        x, mask_in = self.upsample(x) + x_2, self.upsample(mask_in)  # -> 256, 256,256

        last_x = x.clone()
        # Residual layer
        x, mask_in = self.upconv1_1(x, mask_in)
        x = self.upnorm1_1(x)
        x = self.up_act(x)

        x, mask_in = self.upconv1_2(x, mask_in)
        x = self.upnorm1_2(x)
        x = self.up_act(x)

        x, mask_in = self.upconv1_2(x, mask_in)
        x = self.upnorm1_2(x)
        x = self.up_act(x)

        x += last_x

        x, mask_in = self.upsample(x) + x_1, self.upsample(mask_in)  # -> 256, 512,512

        x, mask_in = self.upconv2_init(x, mask_in)
        x = self.upnorm2_init(x)
        x = self.up_act(x)

        last_x = x.clone()

        # Residual layer
        x, mask_in = self.upconv2_1(x, mask_in)
        x = self.upnorm2_1(x)
        x = self.up_act(x)

        x, mask_in = self.upconv2_2(x, mask_in)
        x = self.upnorm2_2(x)
        x = self.up_act(x)

        x, mask_in = self.upconv2_2(x, mask_in)
        x = self.upnorm2_2(x)
        x = self.up_act(x)

        x += last_x

        x, mask_in = self.upconv3(x, mask_in)
        x = self.upnorm3(x)
        x = self.up_act(x)

        x, mask_in = self.upsample(x) + x_1, self.upsample(mask_in)  # -> 256, 1024,1024

        x, mask_in = self.upconv4(x, mask_in)
        x = self.upnorm4(x)
        x = self.up_act(x)

        x, mask_in = self.upconv5(x, mask_in)
        x = self.upnorm5(x)
        x = self.up_act(x)

        return x, mask_in

    def forward(self, x: Tensor, mask_in: Tensor) -> Tensor:
        gt = x.clone()
        first_mask = mask_in.clone()
        x, x_1, x_2, mask_in = self.encoder(x, mask_in)
        out, mask_in = self.decoder(x, x_1, x_2, mask_in)
        return out * ~first_mask.bool() + gt * first_mask, mask_in

    def training_step(self, batch: Tensor) -> Tuple[Dict[str, float], Tensor]:
        I_gt, M_l_1 = batch

        I_out, M_l_2 = self(I_gt, M_l_1)

        L_pixel, L_perceptual, L_style = self.criterion(I_out, I_gt, M_l_1, M_l_2)

        loss = L_pixel + L_perceptual + L_style

        metrics = {
            f"Pixel-wise Loss": L_pixel.item(),
            f"Perceptual loss": L_perceptual.item(),
            f"Style Loss": L_style.item(),
            f"Overall": loss.item(),
        }

        return metrics, loss

    @torch.no_grad()
    def validation_step(self, batch) -> None:
        I_gt, M_l_1 = batch

        I_out, M_l_2 = self(I_gt, M_l_1)

        args = self.criterion(I_out, I_gt, M_l_1, M_l_2)

        metrics = {k:v for k, v in zip(self.criterion.labels, args)}

        self.log_dict(metrics)

        return args[-1]
    def imshow(self, train_loader):
        for batch in train_loader:
            I_gt, M_l_1 = batch

            I_out, M_l_2 = self(I_gt, M_l_1)

            I_out: np.array = self.inverse_transform(I_out[0, :, :, :].unsqueeze(0))
            I_gt: np.array = self.inverse_transform(I_gt[0, :, :, :].unsqueeze(0))

            mathcal_M = (
                (M_l_1.bool() ^ M_l_2.bool()).cpu().detach().view(1024, 1024).numpy()
            )

            inner_out = I_out * ~M_l_2.bool()

            M_l_2 = M_l_2[0, :, :, :].cpu().detach().view(1024, 1024).numpy()

            gt_norm = ImageNormalize(stretch=HistEqStretch(I_gt[np.isfinite(I_gt)]))(
                I_gt
            )

            out_norm = ImageNormalize(stretch=HistEqStretch(I_out[np.isfinite(I_out)]))(
                I_out
            )

            I_out = I_out * mathcal_M

            diff_norm = ImageNormalize(
                stretch=HistEqStretch(I_out[np.isfinite(I_out)])
            )(I_out)

            inner_norm = ImageNormalize(
                stretch=HistEqStretch(inner_out[np.isfinite(inner_out)])
            )(inner_out)
            # Make plot
            fig, ax = plt.subplots(1, 4)
            ax[0].imshow(gt_norm)
            ax[0].yticks([])
            ax[0].xticks([])
            ax[1].imshow(out_norm)
            ax[1].yticks([])
            ax[1].xticks([])
            ax[2].imshow(diff_norm)
            ax[2].yticks([])
            ax[2].xticks([])
            ax[3].imshow(inner_norm)
            ax[3].yticks([])
            ax[3].xticks([])
            plt.show()
            break

class SmallUNet(LightningModule):
    def __init__(
            self,
            layers: float,
            encoder_lr: float,
            encoder_wd: float,
            decoder_lr: float,
            decoder_wd: float,
            alpha_1: float, 
            alpha_2: float,
            alpha_3: float,
            alpha_4: float,
            alpha_5: float,
            alpha_6: float
    ) -> None:
        super().__init__()
        self.criterion = NewInpaintingLoss([alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6])
        # General utils
        self.relu = nn.ReLU()
        self.spe_act: Callable[[Tensor], Tensor] = lambda x: F.relu(6 * F.sigmoid(x))
        self.upsampling: Callable[[Tensor], Tensor] = (
            lambda input: nn.functional.interpolate(
                input, scale_factor=2, mode="nearest"
            )
        )
        self.upsample: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]] = (
            lambda x, mask_in: (self.upsampling(x), self.upsampling(mask_in))
        )
        self.pool: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]] = (
            lambda x, mask_in: (F.max_pool2d(x, 2, 2), F.max_pool2d(mask_in, 2, 2))
        )
        self.layers = layers
        # Backbone
        for layer in range(layers):
            # Downsampling
            self.__setattr__(
                f"conv{layer}_1", PartialConv2d(1, 32, 9, 1, 4)
            )  # -> 32, 1024, 1024
            self.__setattr__(f"norm{layer}_1", nn.BatchNorm2d(32))
            self.__setattr__(
                f"conv{layer}_2", PartialConv2d(32, 64, 7, 1, 3)
            )  # -> 64, 512, 512
            self.__setattr__(f"norm{layer}_2", nn.BatchNorm2d(64))
            self.__setattr__(
                f"conv{layer}_3", PartialConv2d(64, 128, 5, 1, 2)
            )  # -> 128, 256, 256
            self.__setattr__(f"norm{layer}_3", nn.BatchNorm2d(128))
            self.__setattr__(
                f"conv{layer}_4", PartialConv2d(128, 256, 3, 1, 1)
            )  # -> 256, 128, 128
            self.__setattr__(f"norm{layer}_4", nn.BatchNorm2d(256))
            self.__setattr__(
                f"conv{layer}_5", PartialConv2d(256, 512, 3, 1, 1)
            )  # -> 512, 64, 64
            self.__setattr__(f"norm{layer}_5", nn.BatchNorm2d(512))
            self.__setattr__(
                f"conv{layer}_6", PartialConv2d(512, 1024, 3, 1, 1)
            )  # -> 1024, 32, 32

            # Multihead attention
            self.__setattr__(f"att{layer}", nn.MultiheadAttention(1024, 8))
            # Upsampling
            self.__setattr__(
                f"upconv{layer}_1", PartialConv2d(1024, 512, 9, 1, 4)
            )  # -> 512, 32, 32
            self.__setattr__(f"upnorm{layer}_1", nn.BatchNorm2d(512))
            self.__setattr__(
                f"upconv{layer}_2", PartialConv2d(512, 256, 7, 1, 3)
            )  # -> 256, 64, 64
            self.__setattr__(f"upnorm{layer}_2", nn.BatchNorm2d(256))
            self.__setattr__(
                f"upconv{layer}_3", PartialConv2d(256, 128, 5, 1, 2)
            )  # -> 128, 128, 128
            self.__setattr__(f"upnorm{layer}_3", nn.BatchNorm2d(128))
            self.__setattr__(
                f"upconv{layer}_4", PartialConv2d(128, 64, 3, 1, 1)
            )  # -> 64, 256, 256
            self.__setattr__(f"upnorm{layer}_4", nn.BatchNorm2d(64))
            self.__setattr__(
                f"upconv{layer}_5", PartialConv2d(64, 32, 3, 1, 1)
            )  # -> 32, 512, 512
            self.__setattr__(f"upnorm{layer}_5", nn.BatchNorm2d(32))
            self.__setattr__(
                f"upconv{layer}_6", PartialConv2d(32, 1, 3, 1, 1)
            )  # -> 1, 1024, 1024

    def _act_maxpool(
        self, x: Tensor, mask_in: Tensor, act: Optional[bool] = True
    ) -> Tuple[Tensor, Tensor]:
        if act:
            x = self.relu(x)
        x, mask_in = self.pool(x, mask_in)
        return x, mask_in

    def _act_upsample(
        self, x: Tensor, mask_in: Tensor, act: Optional[bool] = True
    ) -> Tuple[Tensor, Tensor]:
        if act:
            x = self.spe_act(x)
        x, mask_in = self.upsample(x, mask_in)
        return x, mask_in

    def encoder(self, x: Tensor, mask_in: Tensor, layer: int) -> Tuple[Tensor, Tensor]:
        x, mask_in = self.__getattr__(f"conv{layer}_1")(x, mask_in)  # -> 32, 1024, 1024
        x_1 = self.__getattr__(f"norm{layer}_1")(x)
        x, mask_in = self._act_maxpool(x_1, mask_in)  # -> 32, 512, 512

        x, mask_in = self.__getattr__(f"conv{layer}_2")(x, mask_in)  # -> 64, 512, 512
        x_2 = self.__getattr__(f"norm{layer}_2")(x)
        x, mask_in = self._act_maxpool(x_2, mask_in)  # -> 64, 256, 256

        x, mask_in = self.__getattr__(f"conv{layer}_3")(x, mask_in)  # -> 128, 256, 256
        x_3 = self.__getattr__(f"norm{layer}_3")(x)
        x, mask_in = self._act_maxpool(x_3, mask_in)  # -> 128, 128, 128

        x, mask_in = self.__getattr__(f"conv{layer}_4")(x, mask_in)  # -> 256, 128, 128
        x_4 = self.__getattr__(f"norm{layer}_4")(x)
        x, mask_in = self._act_maxpool(x_4, mask_in)  # -> 256, 64, 64

        x, mask_in = self.__getattr__(f"conv{layer}_5")(x, mask_in)  # -> 512, 64, 64
        x_5 = self.__getattr__(f"norm{layer}_5")(x)
        x, mask_in = self._act_maxpool(x_5, mask_in)  # -> 512, 32, 32

        x, mask_in = self.__getattr__(f"conv{layer}_6")(x, mask_in)  # -> 1024, 32, 32
        return x_5, x_4, x_3, x_2, x_1, x, mask_in

    def decoder(
        self,
        x_5: Tensor,
        x_4: Tensor,
        x_3: Tensor,
        x_2: Tensor,
        x_1: Tensor,
        x: Tensor,
        mask_in: Tensor,
        layer: int,
    ) -> Tuple[Tensor, Tensor]:
        # Upsampling
        x, mask_in = self.__getattr__(f"upconv{layer}_1")(x, mask_in)  # -> 512, 32, 32
        x = self.__getattr__(f"upnorm{layer}_1")(x)
        x, mask_in = self._act_upsample(x, mask_in)  # -> 512, 64, 64

        x += x_5

        x, mask_in = self.__getattr__(f"upconv{layer}_2")(x, mask_in)  # -> 256, 64, 64
        x = self.__getattr__(f"upnorm{layer}_2")(x)
        x, mask_in = self._act_upsample(x, mask_in)  # -> 256, 128, 128

        x += x_4

        x, mask_in = self.__getattr__(f"upconv{layer}_3")(
            x, mask_in
        )  # -> 128, 128, 128
        x = self.__getattr__(f"upnorm{layer}_3")(x)
        x, mask_in = self._act_upsample(x, mask_in)  # -> 128, 256, 256

        x += x_3

        x, mask_in = self.__getattr__(f"upconv{layer}_4")(x, mask_in)  # -> 64, 256, 256
        x = self.__getattr__(f"upnorm{layer}_4")(x)
        x, mask_in = self._act_upsample(x, mask_in)  # -> 64, 512, 512

        x += x_2

        x, mask_in = self.__getattr__(f"upconv{layer}_5")(x, mask_in)  # -> 32, 512, 512
        x = self.__getattr__(f"upnorm{layer}_5")(x)
        x, mask_in = self._act_upsample(x, mask_in)  # -> 32, 1024, 1024

        x += x_1

        x, mask_in = self.__getattr__(f"upconv{layer}_6")(
            x, mask_in
        )  # -> 1, 1024, 1024

        x = self.spe_act(x)

        return x, mask_in

    def _single_forward(
        self, x: Tensor, mask_in: Tensor, layer: int
    ) -> Tuple[Tensor, Tensor]:
        b, _, _, _ = x.shape
        x_5, x_4, x_3, x_2, x_1, encoded, mask = self.encoder(x, mask_in, layer)
        att_input = encoded.view(b, 1024, -1)
        att_output, _ = self.__getattr__(f"att{layer}")(att_input, att_input, att_input)
        out, mask_out = self.decoder(
            x_5,
            x_4,
            x_3,
            x_2,
            x_1,
            encoded + att_output.view(b, 1024, 32, 32),
            mask,
            layer,
        )

        return out * ~mask_in.bool() + x * mask_in.bool(), mask_out

    def forward(self, x: Tensor, mask_in: Tensor) -> Tuple[Tensor, Tensor]:

        for layer in range(self.layers):
            x, mask_in = self._single_forward(x, mask_in, layer)

        return x, mask_in


class DefaultResidual(nn.Module):
    def __init__(
        self, channels: int, n_architecture: Tuple[int, ...] = (3, 2, 1)
    ) -> None:
        super().__init__()
        self.act = nn.ReLU()
        self.conv_layers = nn.ModuleList(
            [
                (
                    PartialConv2d(channels, channels, 2 * n + 1, 1, n),
                    nn.BatchNorm2d(channels),
                )
                for n in n_architecture
            ]
        )

    def forward(self, I_out: Tensor, mask_in: Tensor) -> Tuple[Tensor, Tensor]:
        x = I_out.clone()

        for conv, norm in self.conv_layers:
            x, mask_in = conv(x, mask_in)
            x = norm(x)
            x = self.act(x)

        return I_out + x, mask_in
    
from loss import FourierModelCriterion
# UNet like architectures with residual connections
class FourierPartial(InpaintingBase):
    def __init__(
            self,
            encoder_lr: float,
            encoder_wd: float,
            decoder_lr: float,
            decoder_wd: float,
            alpha_1: float, 
            alpha_2: float,
            alpha_3: float,
            alpha_4: float,
            alpha_5: float,
            alpha_6: float,
    ):
        super().__init__()

        self.encoder_lr = encoder_lr
        self.encoder_wd = encoder_wd
        self.decoder_lr = decoder_lr
        self.decoder_wd = decoder_wd

        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.alpha_3 = alpha_3
        self.alpha_4 = alpha_4
        self.alpha_5 = alpha_5
        self.alpha_6 = alpha_6

        self.criterion = NewInpaintingLoss(
            [alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6]
        )
        
        self.save_hyperparameters()

        self.maxpool = lambda x: F.max_pool2d(x, 2, 2)
        self.upsample = lambda x: F.interpolate(x, scale_factor=2, mode="nearest")

        self.hid_act = lambda x: F.relu(x)
        self.last_act = lambda x: F.sigmoid(x)

        self.fconv1 = FourierPartialConv2d(1, 32, 3, 1, 1)  # -> 32, 1024, 1024
        self.norm1 = nn.BatchNorm2d(32)

        self.fconv2 = FourierPartialConv2d(32, 64, 3, 1, 1)  # -> 64, 512, 512
        self.norm2 = nn.BatchNorm2d(64)

        self.fconv3 = FourierPartialConv2d(64, 128, 3, 1, 1)  # -> 128, 256, 256
        self.norm3 = nn.BatchNorm2d(128)

        self.fconv4 = FourierPartialConv2d(128, 256, 3, 1, 1)  # -> 256, 128, 128
        self.norm4 = nn.BatchNorm2d(256)

        self.fconv5 = FourierPartialConv2d(256, 512, 5, 1, 2)  # -> 512, 64, 64
        self.norm5 = nn.BatchNorm2d(512)

        self.fconv6 = FourierPartialConv2d(512, 1024, 7, 1, 3)  # -> 1024, 32, 32

        self.fupconv1 = FourierPartialConv2d(1024, 512, 7, 1, 3)  # -> 512, 32, 32
        self.upnorm1 = nn.BatchNorm2d(512)

        self.fupconv2 = FourierPartialConv2d(512, 256, 3, 1, 1)  # -> 256, 64, 64
        self.upnorm2 = nn.BatchNorm2d(256)

        self.fupconv3 = FourierPartialConv2d(256, 128, 3, 1, 1)  # -> 128, 128, 128
        self.upnorm3 = nn.BatchNorm2d(128)

        self.fupconv4 = FourierPartialConv2d(128, 64, 3, 1, 1)  # -> 64, 256, 256
        self.upnorm4 = nn.BatchNorm2d(64)

        self.fupconv5 = FourierPartialConv2d(64, 32, 3, 1, 1)  # ->  32, 512, 512
        self.upnorm5 = nn.BatchNorm2d(32)

        self.upconv6 = PartialConv2d(32, 1, 3, 1, 1)  # ->  1, 1024, 1024

        self.attention = FourierMultiheadAttention(1024, 8)

    def joint_maxpool(self, x: Tensor, mask_in: Tensor) -> Tuple[Tensor, Tensor]:
        return self.maxpool(x), self.maxpool(mask_in)

    def joint_upsample(self, x: Tensor, mask_in: Tensor) -> Tuple[Tensor, Tensor]:
        return self.upsample(x), self.upsample(mask_in)

    def forward(self, x: Tensor, mask_in: Tensor) -> Tuple[Tensor, Tensor]:
        # for output
        gt = x.clone()
        init_mask = mask_in.clone()
        # First layer
        x, mask_in = self.fconv1(x, mask_in)  # -> 32, 1024, 1024
        x = self.norm1(x)
        x_1 = self.hid_act(x)
        x, mask_in = self.joint_maxpool(x_1, mask_in)  #  -> 32, 512, 512

        # Second layer
        x, mask_in = self.fconv2(x, mask_in)  # -> 64, 512, 512
        x = self.norm2(x)
        x_2 = self.hid_act(x)
        x, mask_in = self.joint_maxpool(x_2, mask_in)  # -> 64, 256, 256

        # Third layer
        x, mask_in = self.fconv3(x, mask_in)  # -> 128, 256, 256
        x = self.norm3(x)
        x_3 = self.hid_act(x)
        x, mask_in = self.joint_maxpool(x_3, mask_in)  # -> 128, 128, 128

        # Fourth layer
        x, mask_in = self.fconv4(x, mask_in)  # -> 256, 128, 128
        x = self.norm4(x)
        x_4 = self.hid_act(x)
        x, mask_in = self.joint_maxpool(x_4, mask_in)  # -> 256, 64, 64

        # Fifth layer
        x, mask_in = self.fconv5(x, mask_in)  # -> 512, 64, 64
        x = self.norm5(x)
        x_5 = self.hid_act(x)
        x, mask_in = self.joint_maxpool(x_5, mask_in)  # -> 512, 32, 32

        # Sixth layer
        x, mask_in = self.fconv6(x, mask_in)  # -> 1024, 32, 32

        # Fourier attention layer and residual connection
        x, mask_in = self.fupconv1(x + self.attention(x), mask_in)  # -> 512, 32, 32
        x = self.upnorm1(x)
        x = self.hid_act(x)

        x, mask_in = self.joint_upsample(x, mask_in)  # -> 512, 64, 64

        x += x_5

        x, mask_in = self.fupconv2(x, mask_in)  # -> 256, 64, 64
        x = self.upnorm2(x)
        x = self.hid_act(x)

        x, mask_in = self.joint_upsample(x, mask_in)  # -> 256, 128, 128

        x += x_4

        x, mask_in = self.fupconv3(x, mask_in)  # -> 512, 128, 128
        x = self.upnorm3(x)
        x = self.hid_act(x)

        x, mask_in = self.joint_upsample(x, mask_in)  # -> 512, 256, 256

        x += x_3

        x, mask_in = self.fupconv4(x, mask_in)  # -> 512, 128, 128
        x = self.upnorm4(x)
        x = self.hid_act(x)

        x, mask_in = self.joint_upsample(x, mask_in)  # -> 512, 256, 256

        x += x_2

        x, mask_in = self.fupconv5(x, mask_in)  # -> 512, 128, 128
        x = self.upnorm5(x)
        x = self.hid_act(x)

        x, mask_in = self.joint_upsample(x, mask_in)  # -> 512, 256, 256

        x += x_1

        x, mask_in = self.upconv6(x, mask_in)

        x = self.last_act(x)

        return x * ~init_mask.bool() + gt * init_mask.bool(), mask_in
        
    def configure_optimizers(self):
        encoder_param_group = defaultdict(list)
        decoder_param_group = defaultdict(list)

        for name, param in self.named_parameters():
            if name.startswith(('fconv', 'norm')):
                encoder_param_group['params'].append(param)
            else:
                decoder_param_group['params'].append(param)

        optimizer = torch.optim.Adam([
            {'params': encoder_param_group['params'], 'lr': self.encoder_lr, 'weight_decay':self.encoder_wd},
            {'params': decoder_param_group['params'], 'lr': self.decoder_lr, 'weight_decay':self.decoder_wd}
        ])

        return optimizer

# Generative VAE for image inpainting with fourier
class FourierVAE(LightningModule):
    def __init__(self, hyperparams, criterion) -> None:
        super().__init__()
        self.criterion = criterion
        
        self.save_hyperparameters({
            f'{base}_{arg}': dict_[arg]  for base, dict_ in zip(['encoder', 'decoder'], hyperparams) for arg in dict_
        }.update(self.criterion.factors))

        self.hyperparams = hyperparams
        self.fft = fftn
        self.ifft = ifftn
        self.maxpool = lambda x: F.max_pool2d(x, 2, 2)
        self.upsample = lambda x: F.interpolate(x, scale_factor=2, mode="nearest")

        self.hid_act = lambda x: F.relu(x)
        self.last_act = lambda x: F.sigmoid(x)

        self.fconv1 = _FourierConv(1, 1024, 1024)  # -> 1, 1024, 1024

        self.fconv2 = _FourierConv(1, 512, 512)  # -> 1, 512, 512

        self.fconv3 = _FourierConv(1, 256, 256)  # -> 1, 256, 256

        self.fconv4 = _FourierConv(1, 128, 128)  # -> 1, 128, 128

        self.fconv5 = _FourierConv(1, 64, 64)  # -> 1, 64, 64

        self.fconv6 = _FourierConv(
            1, 32, 32
        )  # -> 1, 32, 32 -> normal space distribution with ifftn

        # This goes all in reparametrization
        self.flatten = nn.Flatten()

        self.mu_fc = nn.Linear(32 * 32, 32 * 32)
        self.logvar_fc = nn.Linear(32 * 32, 32 * 32)

        self.upconv1 = ResidualFourierConvTranspose2d(
            1024, 512, 3, 1, 1
        )  # -> 512, 32, 32
        self.upnorm1 = nn.BatchNorm2d(512)

        self.upconv2 = ResidualFourierConvTranspose2d(
            512, 256, 4, 2, 1
        )  # -> 256, 64, 64
        self.upnorm2 = nn.BatchNorm2d(256)

        self.upconv3 = ResidualFourierConvTranspose2d(
            256, 128, 4, 2, 1
        )  # -> 128, 128, 128
        self.upnorm3 = nn.BatchNorm2d(128)

        self.upconv4 = ResidualFourierConvTranspose2d(
            128, 64, 4, 2, 1
        )  # -> 64, 256, 256
        self.upnorm4 = nn.BatchNorm2d(64)

        self.upconv5 = ResidualFourierConvTranspose2d(
            64, 32, 4, 2, 1
        )  # ->  32, 512, 512
        self.upnorm5 = nn.BatchNorm2d(32)

        self.upconv6 = ResidualFourierConvTranspose2d(
            32, 1, 4, 2, 1
        )  # ->  1, 1024, 1024

        self.fc = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1), nn.Sigmoid())

    def reparametrization(self, out: Tensor, in_channels: int) -> Tensor:
        b, c, h, w = out.shape
        out = self.flatten(out)
        logvar = self.logvar_fc(out).view(b, c, h, w)  # std: b, c, h, w (b, 1, 32, 32)
        mu = self.mu_fc(out).view(b, c, h, w)  # mu: b, c, h, w (b, 1, 32, 32)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(b, in_channels, h, w)
        return mu + eps * std, mu, logvar

    def _joint_maxpool(self, Re: Tensor, Im: Tensor) -> Tuple[Tensor, Tensor]:
        return self.maxpool(Re) + 1j * self.maxpool(Im)

    def forward(self, x: Tensor, mask_in: Tensor) -> Tensor:
        # To fourier space
        out = self.fft(x * mask_in, dim=(-2, -1))

        weight_1 = self.fconv1(out)
        out = self._joint_maxpool(weight_1.real, weight_1.imag)  # -> 1, 512, 512
        weight_2 = self.fconv2(out)
        out = self._joint_maxpool(weight_2.real, weight_2.imag)  # -> 1, 256, 256
        weight_3 = self.fconv3(out)
        out = self._joint_maxpool(weight_3.real, weight_3.imag)  # -> 1, 128,128
        weight_4 = self.fconv4(out)
        out = self._joint_maxpool(weight_4.real, weight_4.imag)  # -> 1, 64,64
        weight_5 = self.fconv5(out)
        out = self._joint_maxpool(weight_5.real, weight_5.imag)  # -> 1, 32, 32
        weight_6 = self.fconv6(out)  # -> 1, 32,32
        out = self.ifft(out).real  # -> Real

        out, mu, logvar = self.reparametrization(out, 1024)  # -> 1024, 32, 32

        z = self.upconv1(out, weight_6)  # -> 512, 32, 32
        z = self.hid_act(z)
        z = self.upnorm1(z)
        z = self.upconv2(z, weight_5)  # -> 256, 64, 64
        z = self.hid_act(z)
        z = self.upnorm2(z)
        z = self.upconv3(z, weight_4)  # -> 128, 128, 128
        z = self.hid_act(z)
        z = self.upnorm3(z)
        z = self.upconv4(z, weight_3)  # -> 64, 256, 256
        z = self.hid_act(z)
        z = self.upnorm4(z)
        z = self.upconv5(z, weight_2)  # -> 32, 512, 512
        z = self.hid_act(z)
        z = self.upnorm5(z)
        z = self.upconv6(z, weight_1)  # -> 1, 1024, 1024
        z = self.hid_act(z)
        z = self.fc(z)

        return z*~mask_in.bool() + x*mask_in, mu, logvar

    def training_step(self, batch: Tensor) -> Tensor:
        I_gt, mask = batch
        I_out, mu, logvar = self(I_gt, mask)
        args = self.criterion(I_out, I_gt, mask, mu, logvar)
        metrics = {k: v.item() for k, v in zip(self.criterion.labels, args)}
        self.log_dict(metrics, prog_bar=True, enable_graph=True)
        return args[-1]
        
    def configure_optimizers(self):
        encoder_param_group = defaultdict(list)
        decoder_param_group = defaultdict(list)

        for name, param in self.named_parameters():
            if name.startswith('fconv'):
                encoder_param_group['params'].append(param)
            else:
                decoder_param_group['params'].append(param)

        optimizer = torch.optim.Adam([
            {'params': encoder_param_group['params'], **self.hyperparams[0]},
            {'params': decoder_param_group['params'], **self.hyperparams[1]}
        ])

        return optimizer

from utils import ChannelWiseSelfAttention

class SingleFourierBlock(nn.Module):
    def __init__(
            self,
            height: int,
            width: int,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            pool: int = None,
            batch_norm: bool = True,
            normal_activation: str = None,
            fourier_activation: str = None,
            num_heads: int = 8,
            dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.conv1 = PartialConv2d(in_channels, out_channels, kernel_size, stride, padding)
        if pool is not None:
            self.pool = lambda x: F.max_pool2d(x, pool, pool)
        if batch_norm:
            self.norm1 = nn.BatchNorm2d(out_channels)
            self.Re_fnorm1 = nn.BatchNorm(in_channels)
            self.Im_fnorm1 = nn.BatchNorm(in_channels)
        if normal_activation is not None:
            match normal_activation:
                case 'relu':
                    self.normal_act = F.relu
                case 'sigmoid':
                    self.normal_act = F.sigmoid
                case 'swiglu':
                    self.normal_act = lambda x: F.hardswish(x)*F.glu(x)
                case 'silu':
                    self.normal_act = F.silu

        if fourier_activation is not None:
            match fourier_activation:
                case 'relu':
                    self.fourier_act = F.relu
                case 'sigmoid':
                    self.fourier_act = F.sigmoid
                case 'swiglu':
                    self.fourier_act = lambda x: F.hardswish(x)*F.glu(x)
                case 'silu':
                    self.fourier_act = F.silu

        self.fconv1 = _FourierConv(in_channels, height, width)

        self.attention = ChannelWiseSelfAttention(in_channels, num_heads, dropout = dropout)
        
    def forward(self, x: Tensor, mask_in: Tensor) -> Tuple[Tensor, Tensor]:
        b, _, h, w = b.shape
        # Normal forward
        n_out, updated_mask= self.conv1(x, mask_in)
        n_out = self.normal_act(n_out)
        n_out = self.norm1(n_out)
        # Fourier Forward
        f_out = self.fconv1(x * mask_in)
        f_out = self.fourier_act(f_out)
        Re_out = self.Re_fnorm1(f_out.real) # in_channels, h, w
        Im_out = self.Im_fnorm1(f_out.imag) # in_channels, h, w
        # Perform multihead attention
        out = self.attention(
            Re_out.view(b, self.in_channels, -1),
            Im_out.view(b, self.in_channels, -1),
            n_out.view(b, self.out_channels, -1),
        ).view(b, self.out_channels, h, w)
        

        return out + n_out, updated_mask, Re_out, Im_out

class DefaultUpsamplingBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size, 
            stride,
            padding,
            pool: int = None,
            activation: str = None,
            batch_norm: bool = True,
            dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.upsample = lambda x: F.interpolate(x, scale_factor=2, mode = 'nearest')
        self.upsampling: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]] = lambda x, mask_in: (self.upsample(x), self.upsample(mask_in))
        self.conv1 = PartialConv2d(in_channels, out_channels , kernel_size, stride, padding)
        if pool is not None:
            self.pool = lambda x: F.max_pool2d(x, pool, pool)
        if activation is not None:
            match activation:
                case 'relu':
                    self.activation = F.relu
                case 'sigmoid':
                    self.activation = F.sigmoid
                case 'swiglu':
                    self.activation = lambda x: F.hardswish(x)*F.glu(x)
                case 'silu':
                    self.activation = F.silu
        if batch_norm:
            self.norm1 = nn.BatchNorm2d(in_channels*2)
        
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: Tensor, mask_in: Tensor) -> Tuple[Tensor, Tensor]:
        x, mask_in = self.upsampling(x, mask_in)
        out, mask_out = self.conv1(x, mask_in)
        out = self.activation(out)
        out = self.norm1(out)
        out = self.dropout(out)

        return out, mask_out

from loss import FourierDeluxeCriterion

class DeluxeFourierModel(LightningModule):
    def __init__(
            self,
            encoder_lr: float,
            encoder_wd: float,
            decoder_lr: float,
            decoder_wd: float,
            optimizer: torch.optim.Optimizer,
            layers: int,
            alpha: Iterable,
            normal_activation: str,
            fourier_activation: str,
            dropout: float,
            num_heads: int
    ) -> None:
        super().__init__()

        self.criterion = FourierDeluxeCriterion(alpha)
        self.encoder_lr = encoder_lr
        self.encoder_wd = encoder_wd
        self.decoder_lr = decoder_lr
        self.decoder_wd = decoder_wd
        self.optimizer = optimizer
        self.layer = layers
        
        self.save_hyperparameters()

        for layer in range(layers):
            setattr(self, f'block{layer}_1', SingleFourierBlock(1024,1024, 1, 32, 3, 1, 1, 2, True, normal_activation, fourier_activation, num_heads, dropout))
            # -> 32, 512, 512
            setattr(self, f'block{layer}_2', SingleFourierBlock(512,512,32,64, 5, 1, 2, 2, True, normal_activation, fourier_activation, num_heads, dropout))
            # -> 64, 256, 256
            setattr(self, f'block{layer}_3', SingleFourierBlock(256, 256, 64, 128, 7, 1, 3, 2, True, normal_activation, fourier_activation, num_heads, dropout))
            # -> 128, 128, 128
            setattr(self, f'block{layer}_4', SingleFourierBlock(128, 128, 128, 256, 9, 1, 4, 2, True, normal_activation, fourier_activation, num_heads, dropout))
            # -> 256, 64, 64
            setattr(self, f'block{layer}_5', SingleFourierBlock(64, 64, 256, 512, 11, 1, 5, 2, True, normal_activation, fourier_activation, num_heads, dropout))
            # -> 512, 32, 32
            setattr(self, f'block{layer}_5', SingleFourierBlock(32, 32, 512, 1024, 13, 1, 6, 2, True, normal_activation, fourier_activation, num_heads, dropout))
            # -> 1024, 16, 16

            setattr(self, f'upblock{layer}_1', DefaultUpsamplingBlock(1024, 512, 13, 1, 6, 2, normal_activation, True, dropout))
            # -> 512, 32, 32
            setattr(self, f'upblock{layer}_2', DefaultUpsamplingBlock(512, 256, 11, 1, 5, 2, normal_activation, True, dropout))
            # -> 256, 64,64
            setattr(self, f'upblock{layer}_3', DefaultUpsamplingBlock(256, 128, 9, 1, 4, 2, normal_activation, True, dropout))
            # -> 128, 128, 128
            setattr(self, f'upblock{layer}_4', DefaultUpsamplingBlock(128, 64, 7, 1, 3, 2, normal_activation, True, dropout))
            # -> 64, 256, 256
            setattr(self, f'upblock{layer}_5', DefaultUpsamplingBlock(64, 32, 5, 1, 2, 2, normal_activation, True, dropout))
            # -> 32, 512, 512
            setattr(self, f'upblock{layer}_5', DefaultUpsamplingBlock(32, 1, 3, 1, 1, 2, normal_activation, True, dropout))
            # -> 1, 1024, 1024
            setattr(self, f'fc_conv_{layer}', PartialConv2d(1, 1, 3, 1, 1))           

        self.fc_act = F.sigmoid

    def _single_forward(self, x: Tensor, mask_in: Tensor, layer: int) -> Tuple[Tensor, Tensor]:
        real_hist: List[Tensor] = []
        imag_hist: List[Tensor] = []
        for i in range(1,6):
            x, mask_in, Re_out, Im_out = getattr(self, f'block{i}')(x, mask_in)
            real_hist.append(Re_out)
            imag_hist.append(Im_out)
            
        for i in range(1,7):
            x, mask_in = getattr(self, f'upblock{i}')(x + real_hist[-i] + imag_hist[-i], mask_in)

        x, mask_in = getattr(self, f'fc_conv_{layer}')(x, mask_in)
        x = self.fc_act(x)

        return x, mask_in

    def forward(self, x: Tensor, mask_in: Tensor) -> Tuple[Tensor, Tensor]:
        for layer in range(self.layers):
            x, mask_in = self._single_forward(x, mask_in, layer)
        return x, mask_in 
    
    def training_step(self, batch: Tensor) -> Tensor:
        I_gt, mask_in = batch
        loss = 0.
        for layer in self.layers:
            I_out, mask_out = self._single_forward(I_gt, mask_in, layer)
            args = self.criterion(I_out, I_gt, mask_in, mask_out)
            metrics = {f'{k} Layer {layer}':v for k, v in zip(self.criterion.labels, args)}
            self.log_dict(metrics, prog_bar=True)
            loss += args[-1]
            mask_in = mask_out
        
        return loss
    
    def validation_step(self, batch: Tensor) -> Tuple[Tensor, Tensor]:
        I_gt, mask_in = batch
        I_out, mask_out = self(I_gt, mask_in)
        args = self.criterion(I_out, I_gt, mask_in, mask_out)
        metrics = {f'Validation {k}':v for k,v in zip(self.criterion.labels, args)}
        self.log_dict(metrics)

    def configure_optimizers(self):
        encoder_param_group = defaultdict(list)
        decoder_param_group = defaultdict(list)

        for name, param in self.named_parameters():
            if name.startswith('block'):
                encoder_param_group['params'].append(param)
            else:
                decoder_param_group['params'].append(param)

        optimizer = self.optimizer([
            {'params': encoder_param_group['params'], 'lr': self.encoder_lr, 'weight_decay': self.encoder_wd},
            {'params': decoder_param_group['params'], 'lr': self.decoder_lr, 'weight_decay': self.decoder_wd}
        ])

        return optimizer
    
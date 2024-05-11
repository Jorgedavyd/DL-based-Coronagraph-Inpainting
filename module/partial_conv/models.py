from typing import Tuple, Dict, Callable, Optional
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from astropy.visualization import HistEqStretch, ImageNormalize
import matplotlib.pyplot as plt
from lightning import LightningModule
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from collections import defaultdict

from .loss import NewInpaintingLoss
from ..utils import PartialConv2d

class InpaintingBase(LightningModule):

    peak_signal = PeakSignalNoiseRatio().to("cuda")
    ssim = StructuralSimilarityIndexMeasure().to("cuda")

    def training_step(self, batch) -> Tensor:
        x, mask_in = batch
        y, mask_out = self(x, mask_in)

        args = self.criterion(y, x, mask_in, mask_out)

        self.log_dict({k: v for k, v in zip(self.criterion.labels, args)})

        return args[-1]

    def validation_step(self, batch, batch_idx) -> Tensor:
        x, mask_in = batch
        y, mask_out = self(x, mask_in)

        args = self.criterion(y, x, mask_in, mask_out)

        peak_signal = self.peak_signal(y, x)
        ssim = self.ssim(y, x)

        self.log_dict(
            {"val_loss": args[-1], "val_peak_signal": peak_signal, "val_ssim": ssim},
            prog_bar=True,
        )


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
        metrics = {k: v for k, v in zip(self.criterion.labels, args)}

        self.log_dict(metrics)

        return args[-1]

    @torch.no_grad()
    def validation_step(self, batch) -> None:
        img, mask = batch
        pred = self(img, mask)
        # Computing the loss
        args = self.criterion(pred, img, mask)
        # Defining the metrics that will be wrote
        metrics = {k: v for k, v in zip(self.criterion.labels, args)}

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

        metrics = {k: v for k, v in zip(self.criterion.labels, args)}

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
    def __init__(self, hparams: Dict[str, any]) -> None:
        super().__init__()
        self.save_hyperparameters()

        for k, v in hparams.items():
            setattr(self, k, v)

        self.criterion = NewInpaintingLoss(self.alpha)
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
        # Backbone
        for layer in range(self.layers):
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

    def training_step(self, batch: Tensor) -> Tensor:
        I_gt, mask_in = batch
        loss = 0

        for layer in range(self.layers):
            I_out, mask_out = self._single_forward(I_gt, mask_in, layer)
            args = self.criterion(I_out, I_gt, mask_in, mask_out)
            loss += args[-1]
            metrics = {
                f"Training/Layer_{layer}_{k}": v
                for k, v in zip(self.criterion.labels, args)
            }
            self.log_dict(metrics, prog_bar=True)
            mask_in = mask_out

        return loss

    def validation_step(self, batch: Tensor, batch_idx) -> Tensor:
        I_gt, mask_in = batch
        I_out, mask_out = self(I_gt, mask_in)
        args = self.criterion(I_out, I_gt, mask_in, mask_out)
        metrics = {f"Validation/{k}": v for k, v in zip(self.criterion.labels, args)}
        self.log_dict(metrics)

    def configure_optimizers(self):
        encoder_param_group = defaultdict(list)
        decoder_param_group = defaultdict(list)

        for name, param in self.named_parameters():
            if name.startswith(("conv", "norm", "att")):
                encoder_param_group["params"].append(param)
            else:
                decoder_param_group["params"].append(param)

        optimizer = self.optimizer(
            [
                {
                    "params": encoder_param_group["params"],
                    "lr": self.encoder_lr,
                    "weight_decay": self.encoder_wd,
                },
                {
                    "params": decoder_param_group["params"],
                    "lr": self.decoder_lr,
                    "weight_decay": self.decoder_wd,
                },
            ]
        )

        return optimizer


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

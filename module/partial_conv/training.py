from lightning.pytorch.cli import LightningCLI

from .models import SmallUNet
from ..data import CoronagraphDataModule

## Run

if __name__ == "__main__":

    cli = LightningCLI(
        model_class=SmallUNet,
        datamodule_class=CoronagraphDataModule,
        seed_everything_default=42,
        trainer_defaults={
            "max_epochs": 200,
            "accelerator": "gpu",
        },
    )

    model = SmallUNet(
        {
            "encoder_lr": cli.model.encoder_lr,
            "encoder_wd": cli.model.encoder_wd,
            "decoder_lr": cli.model.decoder_lr,
            "decoder_wd": cli.model.decoder_wd,
            "optimizer": cli.model.optimizer,
            "layers": cli.model.layers,
            "alpha": cli.model.alpha,
        }
    )

from lightning.pytorch.cli import LightningCLI
from joint_fourier.models import DeluxeFourierModel
from ..data import CoronagraphDataModule
## Run
if __name__ == '__main__':    
    cli = LightningCLI(
        model_class = DeluxeFourierModel,
        datamodule_class=CoronagraphDataModule,
        seed_everything_default=42,
        trainer_defaults={
            'max_epochs': 200,
            'accelerator': 'gpu',
        },
        
    )

    model = DeluxeFourierModel(
		cli.model.encoder_lr,
        cli.model.encoder_wd,
        cli.model.decoder_lr,
        cli.model.decoder_wd,
        cli.model.optimizer,
        cli.model.layers,
        cli.model.alpha,
        cli.model.normal_activation,
        cli.model.fourier_activation,
        cli.model.dropout,
        cli.model.num_heads

	)

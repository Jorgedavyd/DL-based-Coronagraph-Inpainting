from lightning.pytorch.cli import LightningCLI
from .models import FourierVAE
from data import CoronagraphDataModule

if __name__ == '__main__':   

    cli = LightningCLI(
        model_class = FourierVAE,
        datamodule_class=CoronagraphDataModule,
        seed_everything_default=42,
        trainer_defaults={
            'max_epochs': 200,
            'accelerator': 'gpu',
        },
        
    )

    model = FourierVAE(
		cli.model.hparams
	)

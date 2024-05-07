from lightning.pytorch.cli import LightningCLI
from .models import UNetArchitecture
from data import CoronagraphDataModule

## Run

if __name__ == '__main__':  
      
    cli = LightningCLI(
        model_class = UNetArchitecture,
        datamodule_class=CoronagraphDataModule,
        seed_everything_default=42,
        trainer_defaults={
            'max_epochs': 200,
            'accelerator': 'gpu',
        },
        
    )

    model = UNetArchitecture(
		{'encoder_lr': cli.model.encoder_lr,
        'encoder_wd': cli.model.encoder_wd,
        'decoder_lr': cli.model.decoder_lr,
        'decoder_wd': cli.model.decoder_wd,
        'optimizer': cli.model.optimizer,
        'layers': cli.model.layers,
        'alpha': cli.model.alpha}
	)

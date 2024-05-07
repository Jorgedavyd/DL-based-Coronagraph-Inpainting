from lightning.pytorch.cli import LightningCLI
from models import FourierPartial
from data import CoronagraphDataModule
import torch
from lightning import Trainer

## Run
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    cli = LightningCLI(
        model_class = FourierPartial,
        datamodule_class=CoronagraphDataModule,
        seed_everything_default=42,
        trainer_defaults={
            'max_epochs': 200,
            'accelerator': 'gpu',
            'resume_from_checkpoint': 'lightning_logs/version_19/checkpoints/epoch=99-step=65800.ckpt'  
        },
        
    )

    model = FourierPartial(
		cli.model.encoder_lr, 
		cli.model.encoder_wd,
		cli.model.decoder_lr,
		cli.model.decoder_wd,
		cli.model.alpha_1,
		cli.model.alpha_2,
		cli.model.alpha_3,
		cli.model.alpha_4,
		cli.model.alpha_5,
		cli.model.alpha_6,
	)

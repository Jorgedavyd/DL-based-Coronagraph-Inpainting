from lightning.pytorch.cli import LightningCLI
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from models import FourierPartial
from data import CoronagraphDataModule
import torch

if __name__ == '__main__':
	torch.set_float32_matmul_precision('medium')
	
	cli = LightningCLI(
		model_class = FourierPartial,
		datamodule_class=CoronagraphDataModule,
		run = False,
		save_config_kwargs = {'overwrite': True},
		seed_everything_default=42,
		trainer_defaults={
			"max_epochs": 10,
			"precision": "mixed-bf16",
			"callbacks": [ModelCheckpoint(monitor="val_loss")]
		}
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
		cli.model.alpha_7
	)

	cli.trainer.fit(model, datamodule = cli.datamodule)
	cli.trainer.test(model, datamodule= cli.datamodule)
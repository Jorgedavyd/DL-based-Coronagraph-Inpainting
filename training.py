from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import seed_everything
from module.data import CoronagraphDataModule
import torch

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')    

    cli = LightningCLI(
        datamodule_class=CoronagraphDataModule,
        trainer_defaults={
            'deterministic': True, # Reproducibility
        },
    )


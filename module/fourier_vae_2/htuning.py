# Model and data module
from .models import FourierVAE
from ..data import CoronagraphDataModule

# Utils for script automation
import optuna
from lightning.pytorch import Trainer
import torch

def objective(trial: optuna.trial.Trial):

    dataset = CoronagraphDataModule(12)

    hparams = dict(
    encoder_lr = trial.suggest_float('encoder_lr', 1e-7, 1e-2, log = True),
    encoder_wd = trial.suggest_float('encoder_wd', 1e-7, 1e-2, log = True),
    decoder_lr = trial.suggest_float('decoder_lr', 1e-7, 1e-2, log = True),
    decoder_wd = trial.suggest_float('decoder_wd', 1e-7, 1e-2, log = True),
    optimizer = trial.suggest_categorical('opt', ['adam', 'rms', 'sgd']),
    activation = trial.suggest_categorical('activation', ['relu', 'relu6', 'silu', None]),
    beta = trial.suggest_float('beta', 0.01, 1, log = True)
    )

    model = FourierVAE(**hparams)

    trainer = Trainer(
        logger = True,
        enable_checkpointing=False,
        max_epochs=10,
        accelerator="cuda",
        devices = 1,
        log_every_n_steps=22,
        precision="32",
        limit_train_batches=1 / 3,
        limit_val_batches=1 / 3,
        deterministic=True
    )

    trainer.fit(model, datamodule=dataset)

    return trainer.callback_metrics["Validation/Reconstruction"].item()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    study = optuna.create_study(direction = 'minimize')

    study.optimize(objective, n_trials=150, gc_after_trial=True)

    # Access the best hyperparameters
    best_params = study.best_params
    best_score = study.best_value

    print(best_params, '\n', best_score)
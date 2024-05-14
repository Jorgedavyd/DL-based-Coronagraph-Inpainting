# Model and data module
from .models import FourierVAE
from ..data import CoronagraphDataModule
from .loss import Loss, SecondLoss

# Utils for script automation
import optuna
from lightning.pytorch import Trainer
import torch

ghost_criterion = SecondLoss()

def objective(trial: optuna.trial.Trial):

    dataset = CoronagraphDataModule(6)

    hparams = dict(
    encoder_lr = 1e-3,
    encoder_wd = 0,
    decoder_lr = 1e-4,
    decoder_wd = 0,
    alpha = [trial.suggest_float('alpha_1', 0.001, 1, log = True),
            trial.suggest_float('alpha_2', 0.001, 1, log = True),
            trial.suggest_float('alpha_3', 0.001, 1, log = True),
            trial.suggest_float('alpha_4', 0.001, 1, log = True),
            trial.suggest_float('alpha_5', 0.001, 1, log = True)],
    optimizer = 'adam',
    normal_activation = 'silu',
    fourier_activation = 'sigmoid',
    eps = 0.1,
    momentum = 0.1
    )

    model = FourierVAE(**hparams)

    trainer = Trainer(
        logger = True,
        enable_checkpointing=False,
        max_epochs=5,
        accelerator="cuda",
        devices = 1,
        log_every_n_steps=22,
        precision="bf16-mixed",
        limit_train_batches=1 / 5,
        limit_val_batches=1 / 4,
        deterministic=True
    )

    trainer.fit(model, datamodule=dataset)

    return (
        trainer.callback_metrics["Validation/SSIM"].item(),
        trainer.callback_metrics["Validation/PSNR"].item(),
        trainer.callback_metrics["Validation/TV"].item(),
        trainer.callback_metrics["Validation/KL"].item(),
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    study = optuna.create_study(directions=["maximize","maximize","minimize","minimize"])

    study.optimize(objective, n_trials=10, gc_after_trial=True)

    print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

    for i, name in enumerate(ghost_criterion.labels[:-1]):
        best_param = max(study.best_trials, key = lambda t: t.values[i])
        print(f'Trial with best {name}:')
        print(f"\tnumber: {best_param.number}")
        print(f"\tparams: {best_param.params}")
        print(f"\tvalues: {best_param.values}")


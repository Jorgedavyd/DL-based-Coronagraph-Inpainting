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

    dataset = CoronagraphDataModule(12)

    hparams = dict(
    encoder_lr = trial.suggest_float('encoder_lr', 1e-8, 1e-3, log = True),
    encoder_wd = trial.suggest_float('encoder_wd', 1e-8, 1e-3, log = True),
    decoder_lr = trial.suggest_float('decoder_lr', 1e-8, 1e-3, log = True),
    decoder_wd = trial.suggest_float('decoder_wd', 1e-8, 1e-3, log = True),
    alpha = [trial.suggest_float('alpha_1', 0.001, 100),
            trial.suggest_float('alpha_2', 0.001, 100),
            trial.suggest_float('alpha_3', 0.001, 100),
            trial.suggest_float('alpha_4', 0.001, 100),
            trial.suggest_float('alpha_5', 0.001, 100)],
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'rms', 'sgd']),
    normal_activation = trial.suggest_categorical('normal activation', ['relu', 'sigmoid', 'silu', 'relu6']),
    fourier_activation = trial.suggest_categorical('fourier activation', ['relu', 'sigmoid', 'silu', 'relu6']),
    eps = trial.suggest_float('eps', 0.1, 0.3),
    momentum = trial.suggest_float('momentum', 0.1, 0.3)
    )
    model = FourierVAE(hparams)

    trainer = Trainer(
        logger = True,
        enable_checkpointing=False,
        max_epochs=10,
        accelerator="auto",
        devices = 1,
        log_every_n_steps=11,
        precision="32",
        limit_train_batches=1 / 5,
        limit_val_batches=1 / 4
    )

    trainer.fit(model, datamodule=dataset)

    return (
        trainer.callback_metrics["Validation/SSIM"].item() / model.criterion.factors['SSIM'],
        trainer.callback_metrics["Validation/PSNR"].item() / model.criterion.factors['PSNR'],
        trainer.callback_metrics["Validation/L1"].item() / model.criterion.factors['L1'],
        trainer.callback_metrics["Validation/TV"].item() / model.criterion.factors['TV'],
        trainer.callback_metrics["Validation/KL"].item() / model.criterion.factors['KL'],
    )


if __name__ == "__main__":
    # Reproducibility
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)

    study = optuna.create_study(directions=["minimize","minimize","minimize","minimize","minimize"])

    study.optimize(objective, n_trials=150, gc_after_trial=True)

    print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

    for i, name in enumerate(ghost_criterion.labels[:-1]):
        best_param = max(study.best_trials, key = lambda t: t.values[i])
        print(f'Trial with best {name}:')
        print(f"\tnumber: {best_param.number}")
        print(f"\tparams: {best_param.params}")
        print(f"\tvalues: {best_param.values}")


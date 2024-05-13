import optuna
from lightning import Trainer
from optuna_integration import PyTorchLightningPruningCallback
import argparse
import torch

from .models import DeluxeFourierModel
from ..data import CoronagraphDataModule

def define_hyp(trial: optuna.trial.Trial):
    encoder_lr = trial.suggest_float("encoder_lr", 1e-8, 1e-2, log = True)
    encoder_wd = trial.suggest_float("encoder_wd", 1e-8, 1e-3, log = True)
    decoder_lr = trial.suggest_float("decoder_lr", 1e-8, 1e-2, log = True)
    decoder_wd = trial.suggest_float("decoder_wd", 1e-8, 1e-3, log = True)
    layers = trial.suggest_int("layers", 1, 1)
    alpha_1 = trial.suggest_float("alpha_1", 1e-4, 120)
    alpha_2 = trial.suggest_float("alpha_2", 1e-4, 120)
    alpha_3 = trial.suggest_float("alpha_3", 1e-4, 120)
    alpha_4 = trial.suggest_float("alpha_4", 1e-4, 120)
    alpha_5 = trial.suggest_float("alpha_5", 1e-4, 120)
    alpha_6 = trial.suggest_float("alpha_6", 1e-4, 120)
    normal_activation = trial.suggest_categorical(
        "normal_activation", ["relu", "sigmoid", "silu"]
    )
    fourier_activation = trial.suggest_categorical(
        "fourier_activation", ["relu", "sigmoid", "silu"]
    )
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    optimizer = trial.suggest_categorical(
        "optimizer", ['adam', 'rms', 'sgd']
    )
    num_heads = trial.suggest_categorical("number of heads", [2, 4, 8, 16])
    
    match optimizer:
        case 'adam':
            optimizer = torch.optim.Adam
        case 'rms':
            optimizer = torch.optim.RMSprop
        case 'sgd':
            optimizer = torch.optim.SGD

    return (
        encoder_lr,
        encoder_wd,
        decoder_lr,
        decoder_wd,
        optimizer,
        layers,
        [alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6],
        normal_activation,
        fourier_activation,
        dropout,
        num_heads,
    )


dataset = CoronagraphDataModule(1)


def objective(trial):
    model = DeluxeFourierModel(*define_hyp(trial))

    trainer = Trainer(
        enable_checkpointing=False,
        max_epochs=5,
        accelerator="gpu",
        precision="bf16",
        limit_train_batches=1 / 6,
        limit_val_batches=1 / 4,
        callbacks=[
            PyTorchLightningPruningCallback(trial, monitor="Validation Overall")
        ],
    )
    trainer.fit(model, dataset)

    return trainer.callback_metrics["Validation/Overall"].item()


if __name__ == "__main__":
    # Reproducibility
    torch.set_float32_matmul_precision("medium")
    torch.manual_seed(42)
    # pruning
    parser = argparse.ArgumentParser(description="Hyperparameter tuning.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    args = parser.parse_args()

    pruner = (
        optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    )

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

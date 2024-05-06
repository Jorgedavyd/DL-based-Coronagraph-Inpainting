from models import FourierPartial
import optuna
from lightning import Trainer
from data import CoronagraphDataModule
from optuna_integration import PyTorchLightningPruningCallback
import argparse

def define_hyp(trial):
    encoder_lr = trial.suggest_float('encoder_lr', 1e-8, 1e-2) 
    encoder_wd = trial.suggest_float('encoder_wd', 1e-8, 1e-3)
    decoder_lr = trial.suggest_float('decoder_lr', 1e-8, 1e-2)
    decoder_wd = trial.suggest_float('decoder_wd', 1e-8, 1e-3)
    alpha_1 = trial.suggest_float('alpha_1', 1e-4, 120)
    alpha_2 = trial.suggest_float('alpha_2', 1e-4, 120)
    alpha_3 = trial.suggest_float('alpha_3', 1e-4, 120)
    alpha_4 = trial.suggest_float('alpha_4', 1e-4, 120)
    alpha_5 = trial.suggest_float('alpha_5', 1e-4, 120)
    alpha_6 = trial.suggest_float('alpha_6', 1e-4, 120)

    return (
        encoder_lr,
        encoder_wd,
        decoder_lr,
        decoder_wd,
        alpha_1,
        alpha_2,
        alpha_3,
        alpha_4,
        alpha_5,
        alpha_6
    )

dataset = CoronagraphDataModule(2)

def objective(trial):
    model = FourierPartial(*define_hyp(trial))

    trainer = Trainer(
        enable_checkpointing=False,
        max_epochs = 5,
        accelerator='gpu',
        precision='bf16',
        limit_train_batches=1/6,
        limit_val_batches=1/4,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor = 'val_loss')]
    )
    trainer.fit(
        model,
        dataset
    )

    return trainer.callback_metrics['val_loss'].item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hyperparameter tuning.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    args = parser.parse_args()

    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
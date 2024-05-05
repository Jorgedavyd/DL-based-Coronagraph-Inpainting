import os.path as ops

import optuna
from lightning import LightningApp
from lightning_hpo import Sweep
from lightning_hpo.algorithm.optuna import OptunaAlgorithm
from lightning_hpo.distributions.distributions import (
    Categorical,
    IntUniform,
    LogUniform,
)

app = LightningApp(
    Sweep(
        script_path=ops.join(ops.dirname(__file__), "./random_search.py"),
        total_experiments=3,
        parallel_experiments=1,
        distributions={
            "--model.encoder_wd": LogUniform(1e-6, 1e-3),
            "--model.encoder_lr": LogUniform(1e-7, 1e-3),
            "--model.decoder_lr": LogUniform(1e-6, 1e-2),
            "--model.decoder_wd": LogUniform(1e-6, 1e-2),
            "--data.batch_size": Categorical([32, 64]),
            "--trainer.max_epochs": IntUniform(1, 3),
        },
        algorithm=OptunaAlgorithm(optuna.create_study(direction="maximize")),
        framework="pytorch_lightning",
    )
)
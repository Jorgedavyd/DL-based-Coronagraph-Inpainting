import os.path as ops

import optuna
from lightning import LightningApp
from lightning_hpo import Sweep
from lightning_hpo.algorithm.optuna import OptunaAlgorithm
from lightning_hpo.distributions.distributions import LogUniform

app = LightningApp(
    Sweep(
        script_path=ops.join(ops.dirname(__file__), "./random_search.py"),
        total_experiments=50,
        parallel_experiments=2,
        distributions={
            "--model.encoder_wd": LogUniform(1e-6, 1e-3),
            "--model.encoder_lr": LogUniform(1e-7, 1e-3),
            "--model.decoder_lr": LogUniform(1e-6, 1e-2),
            "--model.decoder_wd": LogUniform(1e-6, 1e-2),
            "--model.alpha_1": LogUniform(1e-2, 120.),
            "--model.alpha_2": LogUniform(1e-2, 120.),
            "--model.alpha_3": LogUniform(1e-2, 120.),
            "--model.alpha_4": LogUniform(1e-2, 120.),
            "--model.alpha_5": LogUniform(1e-2, 120.),
            "--model.alpha_6": LogUniform(1e-2, 120.),
            "--model.alpha_7": LogUniform(1e-2, 120.)
        },
        algorithm=OptunaAlgorithm(optuna.create_study(direction="minimize")),
        framework="pytorch_lightning",
    )
)
import torch

import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from ax.service.ax_client import AxClient, ObjectiveProperties
from torch._tensor import Tensor
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from typing import List, Dict
from loss import NewInpaintingLoss
from models import FourierVAE, FourierPartial
from data import CoronagraphDataset
from torch.utils.data import random_split
import json

# Hyperparameter grids
hyperparameters: List[Dict[str,str]] = [
    {"name": "encoder_lr", "value_type": 'float', "type": "range", "bounds": [1e-7, 1e-1], "log_scale": True},
    {"name": "encoder_weight_decay", "value_type": 'float', "type": "range", "bounds": [1e-8, 1e-3], "log_scale": True},
    {"name": "decoder_lr", "value_type": 'float', "type": "range", "bounds": [1e-7, 1e-1], "log_scale": True},
    {"name": "decoder_weight_decay", "value_type": 'float', "type": "range", "bounds": [1e-8, 1e-3], "log_scale": True},
    {"name": "alpha_1", "value_type": 'float', "type": "range", "bounds": [0.1, 120.], "log_scale": True},
    {"name": "alpha_2", "value_type": 'float', "type": "range", "bounds": [0.1, 120.], "log_scale": True},
    {"name": "alpha_3", "value_type": 'float', "type": "range", "bounds": [0.1, 120.], "log_scale": True},
    {"name": "alpha_4", "value_type": 'float', "type": "range", "bounds": [0.1, 120.], "log_scale": True},
    {"name": "alpha_5", "value_type": 'float', "type": "range", "bounds": [0.1, 120.], "log_scale": True},
    {"name": "alpha_6", "value_type": 'float', "type": "range", "bounds": [0.1, 120.], "log_scale": True},
]

batch_size = 1

dataset = CoronagraphDataset(tool="c3")

# 0.8 - 0.1 - 0.1
train_len = round(0.8*len(dataset))
val_len = len(dataset) - train_len

#random split
train_ds, val_ds = random_split(dataset, [train_len, val_len])

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=10)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8)

def evaluator(parametrization):
# Optimizer parameters
    encoder_lr = parametrization.get('encoder_lr')
    encoder_weight_decay = parametrization.get('encoder_weight_decay')
    decoder_lr = parametrization.get('decoder_lr') 
    decoder_weight_decay = parametrization.get('decoder_weight_decay') 
    # Criterion parameters
    alpha_1 = parametrization.get('alpha_1')
    alpha_2 = parametrization.get('alpha_2')
    alpha_3 = parametrization.get('alpha_3')
    alpha_4 = parametrization.get('alpha_4')
    alpha_5 = parametrization.get('alpha_5')
    alpha_6 = parametrization.get('alpha_6')
    
    criterion = NewInpaintingLoss(
        Tensor([alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6])
    )
    
    # Create LightningModule with suggested hyperparameters
    model = FourierPartial(criterion, [
        {'lr': encoder_lr, 'weight_decay': encoder_weight_decay},
        {'lr': decoder_lr, 'weight_decay': decoder_weight_decay}
    ])

    # Train the model
    trainer = L.Trainer(max_epochs=5, limit_train_batches=1/5, enable_checkpointing=False, enable_model_summary=False)
    
    trainer.fit(model, train_dl)

    # Evaluate the model
    val_loss = trainer.validate(model, val_dl)[0]['val_loss']

    torch.cuda.empty_cache()

    return val_loss


if __name__ == '__main__':
    #Reproducibility
    L.seed_everything(42, True)
    torch.set_float32_matmul_precision('medium')

    # Parse arguments for the experiment
    argparser = ArgumentParser(description='Hyperparameter tuning for the models...')
    argparser.add_argument('-n', dest = 'name')
    argparser.add_argument('-t', dest = 'trials', type=int)
    args = argparser.parse_args()

    ax_client = AxClient(verbose_logging=False)

    ax_client.create_experiment(
        name=args.name,  # The name of the experiment.
        parameters=hyperparameters,
        objectives={"val_loss": ObjectiveProperties(minimize=True)},  # The objective name and minimization setting.
        # parameter_constraints: Optional, a list of strings of form "p1 >= p2" or "p1 + p2 <= some_bound".
        # outcome_constraints: Optional, a list of strings of form "constrained_metric <= some_bound".
    )

    for i in range(args.trials):
        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluator(parameters))

    ax_client.get_trials_data_frame().to_csv('htuning.csv')

    ax_client.save_to_json_file('htuning.json')
    
    best_parameters, values = ax_client.get_best_parameters()
    
    with open("best_parameters.json", "w") as outfile: 
        json.dump(best_parameters, outfile)

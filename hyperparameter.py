from lightning.pytorch import seed_everything
from fourier import FourierVAE
from torch import Tensor
from loss import FourierModelCriterion
import lightning as L
from ax.service.managed_loop import optimize
from data import CoronagraphDataModule
from typing import List, Dict
import torch

# Hyperparameter grids
hyperparameters: List[Dict[str,str]] = [
    {"name": "encoder_lr", "value_type": float, "type": "range", "bounds": [1e-7, 1e-1], "log_scale": True},
    {"name": "encoder_weight_decay", "value_type": float, "type": "range", "bounds": [1e-8, 1e-3], "log_scale": True},
    {"name": "decoder_lr", "value_type": float, "type": "range", "bounds": [1e-7, 1e-1], "log_scale": True},
    {"name": "decoder_weight_decay", "value_type": float, "type": "range", "bounds": [1e-8, 1e-3], "log_scale": True},
    {"name": "alpha_1", "value_type": float, "type": "range", "bounds": [0.1, 120.], "log_scale": True},
    {"name": "alpha_2", "value_type": float, "type": "range", "bounds": [0.1, 120.], "log_scale": True},
    {"name": "alpha_3", "value_type": float, "type": "range", "bounds": [0.1, 120.], "log_scale": True},
    {"name": "alpha_4", "value_type": float, "type": "range", "bounds": [0.1, 120.], "log_scale": True},
    {"name": "alpha_5", "value_type": float, "type": "range", "bounds": [0.1, 120.], "log_scale": True},
]

# Defining the Data module
dataset = CoronagraphDataModule(2)

def objective(hyperparams):
    # Optimizer parameters
    encoder_lr = hyperparams.get('encoder_lr')
    encoder_weight_decay = hyperparams.get('encoder_weight_decay')
    decoder_lr = hyperparams.get('decoder_lr') 
    decoder_weight_decay = hyperparams.get('decoder_weight_decay') 
    # Criterion parameters
    alpha_1 = hyperparams.get('alpha_1')
    alpha_2 = hyperparams.get('alpha_2')
    alpha_3 = hyperparams.get('alpha_3')
    alpha_4 = hyperparams.get('alpha_4')
    alpha_5 = hyperparams.get('alpha_5')
    
    criterion = FourierModelCriterion(
        Tensor([alpha_1, alpha_2, alpha_3, alpha_4, alpha_5])
    )
    
    # Create LightningModule with suggested hyperparameters
    model = FourierVAE([
        {'lr': encoder_lr, 'weight_decay': encoder_weight_decay},
        {'lr': decoder_lr, 'weight_decay': decoder_weight_decay}
    ], criterion)

    # Train the model
    trainer = L.Trainer(max_epochs=25, limit_train_batches=0.25)
    trainer.fit(model, dataset)

    # Evaluate the model
    val_loss = trainer.validate(dataset)[0]['val_loss']

    return val_loss

if __name__ == '__main__':
    # Reproducibility
    seed_everything(42, True)
    torch.set_float32_matmul_precision('medium')
    # Multi-objective optimization
    best_parameters, values, experiment, model = optimize(
        parameters=hyperparameters,
        evaluation_function=objective,
        minimize=True,
        total_trials=100
    )

    print("Best hyperparameters:")
    print(best_parameters)
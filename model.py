from typing import Tuple, Union, List, Dict, Any, Callable, Optional, Iterable
from partial_conv import PartialConv2d
from torch import Tensor
from torch import nn
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
import torchmetrics.functional as f
import torch.nn.functional as F
from tqdm import tqdm
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import HistEqStretch, ImageNormalize
import matplotlib.pyplot as plt
from copy import deepcopy
from utils import create_config

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    
class TrainingPhase(nn.Module):
    def __init__(
        self, name_run: str, criterion
    ) -> None:
        super().__init__()
        self.name_run = name_run
        self.config = create_config(name_run)
        # Tensorboard writer
        self.writer: SummaryWriter = SummaryWriter(f"{name_run}")
        self.name: str = name_run
        self.criterion = criterion
        # History of changes
        self.train_epoch: List[List[int]] = []
        self.val_epoch: List[List[int]] = []
        self.lr_epoch: Dict[str, List[float]] = defaultdict(list)
        self.num_runs: int = 0
        os.makedirs(f'{self.name_run}/images/', exist_ok=True)
    
    @torch.no_grad()
    def batch_metrics(self, metrics: dict, card: str) -> None:

        if card == "Training" or "Training" in card.split("/"):
            self.train()
            mode = "Training"
        elif card == "Validation" or "Validation" in card.split("/"):
            self.eval()
            mode = "Validation"
        else:
            raise ValueError(
                f"{card} is not a valid card, it should at least have one of these locations: [Validation, Training]"
            )

        self.writer.add_scalars(
            f"{card}/metrics",
            metrics,
            (
                self.config["global_step_train"]
                if mode == "Training"
                else self.config["global_step_val"]
            ),
        )
        self.writer.flush()

        if mode == "Training":
            self.train_epoch.append(list(metrics.values()))
        elif mode == "Validation":
            self.val_epoch.append(list(metrics.values()))

    @torch.no_grad()
    def validation_step(self, batch) -> None:
        metrics = self.compute_val_metrics(batch)
        self.batch_metrics(metrics, "Validation")
        self.config["global_step_val"] += 1

    def training_step(self, batch) -> None:
        # Compute loss and metrics
        metrics, loss = self.compute_train_loss(batch)
        # Write metrics to TensorBoard
        if metrics is not None:
            self.batch_metrics(metrics, "Training")
        # Backpropagation
        loss.backward()
        # Gradient clipping
        if self.grad_clip:
            for grad_clip, params in zip(self.grad_clip, self.params):
                nn.utils.clip_grad_value_(params, grad_clip)
        # Optimizer and scheduler steps
        for optimizer, scheduler in zip(self.optimizer, self.scheduler):
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                for param_group in optimizer.param_groups:
                    lr = param_group['lr']
                    idx = self.optimizer.index(optimizer)
                    self.lr_epoch[f'{optimizer.__class__.__name__}_{idx}'].append(lr)
                    self.writer.add_scalars(
                        "Training", {f"lr_{optimizer.__class__.__name__}_{idx}": lr}, self.config["global_step_train"]
                    )
                    scheduler.step()
        # Increment global step
        self.config["global_step_train"] += 1
        self.writer.flush()

    @torch.no_grad()
    def end_of_epoch(self) -> None:
        train_metrics = Tensor(self.train_epoch).mean(dim=-1)
        val_metrics = Tensor(self.val_epoch).mean(dim=-1)
        lr_epoch = {key:Tensor(value).mean().item() for key, value in self.lr_epoch.items()}

        train_metrics = {
            key: value for key, value in zip(self.criterion.labels, train_metrics)
        }
        val_metrics = {
            key: value for key, value in zip(self.criterion.labels, val_metrics)
        }

        self.writer.add_scalars(
            "Training/Epoch", train_metrics, global_step=self.config["epoch"]
        )
        self.writer.add_scalars(
            "Validation/Epoch", val_metrics, global_step=self.config["epoch"]
        )

        hparams = {
            "Epochs": self.config["epoch"],
            "Batch Size": self.batch_size
        }
        # Criterion factors
        hparams.update(self.criterion.factors)
        # Learning rate
        hparams.update(lr_epoch)
        # Weight decay
        hparams.update({
            f'weight_decay_{optimizer.__class__.__name__}_{idx}': param_group['weight_decay'] for optimizer in self.optimizer for idx, param_group in enumerate(optimizer.param_groups)
        })
        # Gradient clipping
        hparams.update({
            f'grad_clip_{optimizer.__class__.__name__}': clip for optimizer, clip in zip(self.optimizer, self.grad_clip)
        })
        self.writer.add_hparams(hparams, train_metrics)

        self.writer.flush()

        self.train_epoch = []
        self.val_epoch = []
        self.lr_epoch = defaultdict(list)
    
    def save_config(self) -> None:
        os.makedirs(f"{self.name}/models", exist_ok=True)
        # Model weights
        self.config["optimizer_state_dicts"] = [deepcopy(optimizer.state_dict()) for optimizer in self.optimizer]
        if self.scheduler is not None:
            self.config["scheduler_state_dicts"] = [deepcopy(scheduler.state_dict()) for scheduler in self.scheduler]
        self.config["model_state_dict"] = deepcopy(self.state_dict())

        torch.save(
            self.config,
            f'./{self.name}/models/{self.config["name"]}_{self.config["epoch"]}.pt',
        )
    
    @torch.no_grad()
    def load_config(self, epochs, train_loader) -> None:
        try:
            root_path = f"./{self.name_run}/models"
            files = sorted(os.listdir(root_path))
            if len(files) == 0:
                raise FileNotFoundError
            for idx, file in enumerate(files):
                print(f"{idx+1}. {file}")
            config = int(input("Choose the config: ")) - 1
        except (ValueError, FileNotFoundError):
            return None
        
        config: Dict[str, Any] = torch.load(
            os.path.join(root_path, files[config])
        )
        # Modules
        self.load_state_dict(config['model_state_dict'])    
        #Optimizer
        for optimizer, state in zip(self.optimizer, config['optimizer_state_dict']):
            optimizer.load_state_dict(state)
        #Scheduler
        if self.scheduler is not None and config['scheduler_state_dict'] is not None:
            for scheduler, state in zip(self.scheduler, config['scheduler_state_dict']):
                state['total_steps'] += epochs * int(len(train_loader))
                scheduler.load_state_dict(state)
        #Other parameters
        self.config['epoch'] = config['epoch']
        self.config['device'] = config['device']
        self.config['global_step_train'] = config['global_step_train']
        self.config['global_step_val'] = config['global_step_val']

    scheduler = [None]
    def get_module_parameters(modules: Iterable[Union[nn.Module, str]], hyperparams: Iterable[Dict[str, float]]):
        assert(len(modules)==len(hyperparams))
        return [
            {
                'params': module
            }.update(hyperparam) for module, hyperparam in zip(modules, hyperparams)
        ]
    
    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int,
        batch_size: int,
        lr: Union[float, Iterable[float]],
        weight_decay: Union[float, Iterable[float]] = 0.0,
        grad_clip: bool = False,
        opt_func: Any = torch.optim.Adam,
        lr_sched: Any = None,
        saving_div: int = 5,
        graph: bool = False,
        sample_input: Tensor = None,
        modules: Iterable = None
    ) -> None:
        self.num_runs += 1
        torch.cuda.empty_cache()
        #Given a list of learning rates
        if isinstance(lr, list):
            assert (isinstance(weight_decay, (tuple, list, set))), 'Must define the regularization term for every param group or optimizer'
            assert (len(lr) == len(weight_decay)), 'Should have the same amount of learning rates and regularization parameters'
            if grad_clip:
                assert (isinstance(grad_clip, (tuple, list, set))), 'You should define the gradient clipping term for every param group or optimizer'
                assert (len(grad_clip) == len(lr)), 'Should have the same amount of learning rates and gradient clipping parameters'
            assert (modules is not None), 'Should define by param groups with modules argument for different learning rates or optimizers'
            assert (len(modules)>1), 'Should have more than one module'
            if isinstance(opt_func, (tuple, list, set)):
                if isinstance(lr_sched, (tuple, list, set)):
                    assert(len(lr_sched) == len(opt_func) == len(lr) or len(lr_sched) == 1 or lr_sched is None), 'Learning rate scheduler not correctly defined'
            else:
                assert (lr_sched is None or not isinstance(lr_sched, (tuple, list, set)))

        # If it's the first run on notebook, we need to define and load parameters from previous loops
        if self.num_runs == 1:
            #More than one optimizer
            if isinstance(opt_func, list):
                self.params = self.get_module_parameters(modules)
                # Defining each of the optimizers
                self.optimizer = [optimizer(module, lr = LR, weight_decay = wd) for optimizer, module, LR, wd in zip(opt_func, self.params, lr, weight_decay)]
                # Defining each of the schedulers
                if isinstance(lr_sched, (list, tuple, set)):
                    if len(lr_sched) == 1:
                        self.scheduler = [lr_sched[0](optimizer, epoch = epochs, step_per_epoch = len(train_loader)) for optimizer in self.optimizer]
                    else:
                        self.scheduler = [scheduler(optimizer, epoch = epochs, steps_per_epoch = len(train_loader)) if scheduler is not None else None for scheduler, optimizer in zip(lr_sched, self.optimizer)]
                else:
                    if lr_sched is None:
                        self.scheduler = [lr_sched]*len(self.optimizer)
                    else:
                        self.scheduler = [lr_sched(optimizer, epoch = epochs, step_per_epoch = len(train_loader)) for optimizer in self.optimizer]
                # Since first run, we load the configuration of prior trainings
                self.load_config(epochs, train_loader)
            # If not more than one optimizer but different learning rates
            elif isinstance(lr, (list, tuple, set)):
                self.params = self.get_module_parameters(modules)
                # Defining just one optimizer with param groups
                self.optimizer = opt_func(self.get_module_parameters(modules, [{
                    "lr": lr_inst, "weight_decay": wd
                } for lr_inst, wd in zip(lr, weight_decay)]))
                
                if lr_sched is not None:
                    self.scheduler = [lr_sched(self.optimizer, epochs = epochs, steps_per_epoch = len(train_loader))]

                self.optimizer = [self.optimizer]

                self.load_config(epochs, train_loader)
            # If none of that happens, normal training loop
            else:
                self.params = self.parameters()
                # Defining the optimizer
                self.optimizer = opt_func(self.params, lr, weight_decay=weight_decay)
                # Defining the scheduler
                if lr_sched is not None:
                    self.scheduler = [lr_sched(self.optimizer,lr,  epochs = epochs, steps_per_epoch = len(train_loader))]
                # Putting into list for sintax
                self.optimizer = [self.optimizer]
                #Loading last config
                self.load_config(epochs, train_loader)
        
        if isinstance(lr, (float, int)): lr = [lr]
        if isinstance(weight_decay, (float, int)): weight_decay = [weight_decay]
        if isinstance(grad_clip, (float, int)): grad_clip = [grad_clip]
        
        if len(self.optimizer) > 1:
            for optimizer, LR, wd in zip(self.optimizer, lr, weight_decay):
                for parameter in optimizer.param_groups:
                    parameter['lr'] = LR
                    parameter['weight_decay'] = wd
        else:
            for optimizer in self.optimizer:
                for parameter, LR, wd in zip(optimizer.param_groups, lr, weight_decay):
                    parameter['lr'] = LR
                    parameter['weight_decay'] = wd
        #Clean the GPU cache
        if graph:
            assert (
                sample_input is not None
            ), "If you want to visualize a graph, you must pass through a sample tensor"
        #Build graph in tensorboard
        if graph and sample_input:
            self.writer.add_graph(self, sample_input)

        # Defining hyperparameters as attributes of the model and training object
        self.batch_size = batch_size
        self.grad_clip = grad_clip

        for epoch in range(self.config["epoch"], self.config["epoch"] + epochs):
            # Define epoch
            self.config["epoch"] = epoch
            # Training loop
            self.train()
            for train_batch in tqdm(train_loader, desc=f"Training - Epoch: {epoch}"):
                # training step
                self.training_step(train_batch)
            # Validation loop
            self.eval()
            for val_batch in tqdm(val_loader, desc=f"Validation - Epoch: {epoch}"):
                self.validation_step(val_batch)
            # Save model and config if epoch mod(saving_div) = 0
            if epoch % saving_div == 0:
                self.save_config()
            # Show sample of data
            self.imshow(val_loader)
            # End of epoch
            self.end_of_epoch()

    def imshow(self, loader):
        for batch in loader:
            I_gt, M_l_1 = batch
            
            I_gt = I_gt[0, :, :, :]
            M_l_1 = M_l_1[0, :, :, :]

            I_out, M_l_2 = self(I_gt.unsqueeze(0), M_l_1.unsqueeze(0))
            
            I_out = I_out[0, :, :, :]
            M_l_2 = M_l_2[0, :, :, :]

            I_out: np.array = I_out.view(1024, 1024).detach().cpu().numpy()
            I_gt: np.array = I_gt.view(1024, 1024).detach().cpu().numpy()

            mathcal_M = (M_l_1.bool() ^ M_l_2.bool()).cpu().detach().view(1024, 1024).numpy()

            M_l_2 = M_l_2.cpu().detach().view(1024, 1024).numpy()

            diff = I_out*mathcal_M

            #Make plot
            fig, axes = plt.subplots(1, 3)
            axes[0].imshow(I_gt)
            axes[1].imshow(I_out)
            axes[2].imshow(diff)

            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])

            plt.show()

            fig.savefig(f'{self.name_run}/images/{self.config["epoch"]}.png')
            break
class SingleLayer(TrainingPhase):
    
    def __init__(
            self,
            name_run: str,
            criterion,
            res_arch: Tuple[int, ...]
    ):
        super().__init__(name_run, criterion)
        #Activation function
        self.act = nn.SiLU()
        #Partial convolutional layers
        self.conv_1 = PartialConv2d(1, 32, 3, 1, 1)
        self.conv_2 = PartialConv2d(32, 64, 3, 1, 1)
        self.res_1 = DefaultResidual(64, res_arch)
        self.conv_3 = PartialConv2d(64, 32, 3, 1, 1)
        self.conv_4 = PartialConv2d(32, 1, 3, 1, 1)

    def forward(self, input: Tensor, mask_in: Tensor) -> Tuple[Tensor, Tensor]:
        out, mask = self.conv_1(input, mask_in)
        out = self.act(out)
        out, mask = self.conv_2(out, mask)
        out = self.act(out)
        out, mask = self.res_1(out, mask)
        out = self.act(out)
        out, mask = self.conv_3(out, mask)
        out = self.act(out)
        out, mask = self.conv_4(out, mask)
        return out*~mask_in.bool() + input*mask_in, mask
    
    def training_step(self, batch) -> None:
        # Defining the img and mask
        ground_truth, prior_mask = batch
        #forward pass for single layer
        x, updated_mask = self(ground_truth, prior_mask)
        # Compute each term of the loss function
        L_pixel, L_perceptual, L_style = self.criterion(x, ground_truth, prior_mask, updated_mask)
        loss = L_pixel + L_perceptual + L_style

        metrics = {
            f"Pixel-wise Loss": L_pixel.item(),
            f"Perceptual loss": L_perceptual.item(),
            f"Style Loss": L_style.item(),
            f"Overall": loss.item(),
        }

        self.batch_metrics(metrics, f"Training")

        loss.backward()

        # gradient clipping
        if self.grad_clip:
            nn.utils.clip_grad_value_(self.parameters(), self.grad_clip)
        # optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        # scheduler step
        if self.scheduler:
            lr = get_lr(self.optimizer)
            self.lr_epoch.append(lr)
            self.writer.add_scalars(
                "Training", {"lr": lr}, self.config["global_step_train"]
            )
            self.scheduler.step()
        self.writer.flush()
        # Global step shift
        self.config["global_step_train"] += 1
    def imshow(self, train_loader):
        for batch in train_loader:
            img, mask = batch
            x, updated_mask = self(img, mask)
            mathcal_mask: Tensor = updated_mask.bool()^mask.bool()
            
            x: np.array = self.inverse_transform(x)
            mask = mask.cpu().detach().view(1024, 1024).numpy()
            mathcal_mask = mathcal_mask.cpu().detach().view(1024, 1024).numpy()

            #Make plot
            fig, ax = plt.subplots(1, 4)
            ax[0].imshow(img)
            ax[1].imshow(x)
            ax[2].imshow(x*mask)
            ax[3].imshow(x*mathcal_mask)
            plt.yticks([])
            plt.xticks([])
            plt.show()
            break

class UNetArchitecture(TrainingPhase):
    def __init__(
            self,
            name_run: str,
            criterion
    ) -> None:
        super().__init__(name_run, criterion)
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()
        self.max_pool = nn.MaxPool2d(2, 2)

        #Downsampling
        self.conv1 = PartialConv2d(1, 32, 3, 1, 1)
        self.norm1 = nn.BatchNorm2d(32)
        self.conv2 = PartialConv2d(32, 32, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(32)
        self.conv3 = PartialConv2d(32, 64, 3, 1, 1)
        self.norm3 = nn.BatchNorm2d(64)
        self.conv4 = PartialConv2d(64, 64, 3, 1, 1)
        self.norm4 = nn.BatchNorm2d(64)
        self.conv5 = PartialConv2d(64, 128, 3, 1, 1)
        self.norm5 = nn.BatchNorm2d(128)
        self.conv6 = PartialConv2d(128, 256, 3, 1, 1)
        self.norm6 = nn.BatchNorm2d(256)

        #Upsampling
        self.upconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  # -> 64, 64
        self.upnorm1 = nn.BatchNorm2d(128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)  # -> 128, 128
        self.upnorm2 = nn.BatchNorm2d(64)
        self.upconv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1) # -> 256, 256
        self.upnorm3 = nn.BatchNorm2d(64)
        self.upconv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1) # -> 512, 512
        self.upnorm4 = nn.BatchNorm2d(32)
        self.upconv5 = nn.ConvTranspose2d(32, 1, 4, 2, 1) # -> 1024, 1024

    def encoder(self, x: Tensor, mask_in: Tensor) -> Tuple[Tensor, ...]:
        #First layer
        x_0, mask_in = self.conv1(x, mask_in) #1, 1024, 1024 -> 32, 1024, 1024
        x_1 = self.norm1(x_0) 
        x_1 = self.relu(x_1) #activation function
        #Downsampling
        x_1, mask_in = self.max_pool(x_1), self.max_pool(mask_in) # 32, 1024, 1024 -> 32, 512, 512
        
        #Second Layer
        x_1, mask_in = self.conv2(x_1, mask_in)# 32, 512, 512-> 32, 512, 512
        x_2 = self.norm2(x_1)
        x_2 = self.relu(x_2)
        
        #Downsampling
        x_2, mask_in = self.max_pool(x_2), self.max_pool(mask_in) # 32, 512, 512 -> 32, 256, 256
        
        x_2, mask_in = self.conv3(x_2, mask_in) # 32, 256, 256 -> 64, 256, 256
        x_3 = self.norm3(x_2)
        x_3 = self.relu(x_3)

        #Downsampling
        x_3, mask_in = self.max_pool(x_3), self.max_pool(mask_in) # 64, 256, 256  -> 64, 128, 128
        
        x_3, mask_in = self.conv4(x_3, mask_in) # 64, 128, 128 -> 64, 128, 128
        x_4 = self.norm4(x_3) 
        x_4 = self.relu(x_4)

        #Downsampling
        x_4, mask_in = self.max_pool(x_4), self.max_pool(mask_in) # 64, 128, 128 -> 64, 64, 64
        
        x_4, mask_in = self.conv5(x_4, mask_in) # 64, 64, 64 -> 128, 64, 64
        x_5 = self.norm5(x_4)
        x_5 = self.relu(x_5)
        
        #Downsampling
        x_5, mask_in = self.max_pool(x_5), self.max_pool(mask_in) # 128, 64, 64 -> 128, 32, 32
        
        x_5, mask_in = self.conv6(x_5, mask_in) # 128, 32, 32 -> 256, 32, 32
        out = self.norm6(x_5)
        out = self.relu(out)

        return x_1, x_2, x_3, x_4, out
    def decoder(self, x_1: Tensor, x_2: Tensor, x_3: Tensor, x_4: Tensor, out: Tensor) -> Tensor:
        out = self.upconv1(out) + x_4 # 256, 32, 32 -> 128, 64, 64
        out = self.upnorm1(out)
        out = self.silu(out)
        
        out = self.upconv2(out) + x_3 # 128, 64, 64 -> 64, 128, 128
        out = self.upnorm2(out)
        out = self.silu(out)
        
        out = self.upconv3(out) + x_2 # 64, 128, 128 -> 64, 256, 256
        out = self.upnorm3(out)
        out = self.silu(out)
        
        out = self.upconv4(out) + x_1 # 64, 256, 256 -> 32, 512, 512
        out = self.upnorm4(out)
        out = self.silu(out)

        out = self.upconv5(out) # 32, 512, 512 -> 1, 1024, 1024
        out = self.silu(out)

        return out
    def forward(self, x: Tensor, mask_in: Tensor) -> Tensor:
        x_1, x_2, x_3, x_4, out = self.encoder(x, mask_in)
        out = self.decoder(x_1, x_2, x_3, x_4, out)
        return out
    def loss(self, batch):
        img, mask = batch
        pred = self(img, mask)
        # Computing the loss
        L_pixel, L_perceptual, L_style = self.criterion(pred, img, mask)
        loss = L_pixel + L_perceptual + L_style
        # Defining the metrics that will be wrote
        metrics = {
            f"Pixel-wise Loss": L_pixel.item(),
            f"Perceptual loss": L_perceptual.item(),
            f"Style Loss": L_style.item(),
            f"Overall": loss.item(),
        }

        return metrics, loss
    @torch.no_grad()
    def validation_step(self, batch) -> None:
        img, mask = batch
        x = self(img, mask)
        L_pixel, L_perceptual, L_style = self.criterion(
            x, img, mask
        )
        metrics = {
            f"Pixel-wise Loss": L_pixel.item(),
            f"Perceptual loss": L_perceptual.item(),
            f"Style Loss": L_style.item(),
            f"Overall": (L_pixel + L_perceptual + L_style).item(),
        }
        self.batch_metrics(metrics, "Validation")
        self.config["global_step_val"] += 1

    def imshow(self, train_loader):
        for batch in train_loader:
            img, mask = batch
            x = self(img, mask)

            x: np.array = self.inverse_transform(x[0, :, :, :].unsqueeze(0))
            img: np.array = self.inverse_transform(img[0, :, :, :].unsqueeze(0))
            mask = mask[0, :, :, :].cpu().detach().view(1024, 1024).numpy()

            img_norm = ImageNormalize(stretch = HistEqStretch(img[np.isfinite(img)]))(img)
            x_norm = ImageNormalize(stretch = HistEqStretch(x[np.isfinite(x)]))(x)

            #Make plot
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(img_norm)
            ax[1].imshow(x_norm)
            plt.yticks([])
            plt.xticks([])
            plt.show()
            break

class UNetArchitectureDeluxe(TrainingPhase):
    def __init__(self, name_run: str, criterion, in_channels):
        super().__init__(name_run, criterion)
        self.up_act = lambda input: F.relu(6*F.sigmoid(input))
        self.down_act = nn.SiLU()
        self.upsample = lambda input: nn.functional.interpolate(input = input, scale_factor=2, mode = 'nearest')
        self.max_pool = nn.MaxPool2d(2, 2)
        # Encoder
        self.conv1 = PartialConv2d(in_channels, 32, 3, 1, 1)
        self.norm1 = nn.BatchNorm2d(32)

        self.conv2 = PartialConv2d(32, 64, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(64)
        
        self.conv3_init = PartialConv2d(64, 128, 3, 1, 1)
        self.norm3_init = nn.BatchNorm2d(128)
        
        ## Residual layer
        self.conv3_1 = PartialConv2d(128, 128, 15, 1, 7)
        self.norm3_1 = nn.BatchNorm2d(128)

        self.conv3_2 = PartialConv2d(128, 128, 13, 1, 6)
        self.norm3_2 = nn.BatchNorm2d(128)

        self.conv3_3 = PartialConv2d(128, 128, 11, 1, 5)
        self.norm3_3 = nn.BatchNorm2d(128)


        self.conv4_init = PartialConv2d(128, 256, 3, 1, 1)
        self.norm4_init = nn.BatchNorm2d(256)

        #3 Residual layer
        self.conv4_1 = PartialConv2d(256, 256, 17, 1, 8)
        self.norm4_1 = nn.BatchNorm2d(256)

        self.conv4_2 = PartialConv2d(256, 256, 15, 1, 7)
        self.norm4_2 = nn.BatchNorm2d(256)

        self.conv4_3 = PartialConv2d(256, 256, 11, 1, 5)
        self.norm4_3 = nn.BatchNorm2d(256)

        self.conv5 = PartialConv2d(256, 512, 3, 1, 1)
        self.norm5 = nn.BatchNorm2d(512)

        # Decoder
        self.upconv1_init = PartialConv2d(512, 256, 3, 1, 1)
        self.upnorm1_init = nn.BatchNorm2d(256)
        
        ## Residual layer
        self.upconv1_1 = PartialConv2d(256, 256, 11, 1, 5)
        self.upnorm1_1 = nn.BatchNorm2d(256)

        self.upconv1_2 = PartialConv2d(256, 256, 13, 1, 6)
        self.upnorm1_2 = nn.BatchNorm2d(256)

        self.upconv1_3 = PartialConv2d(256, 256, 15, 1, 7)
        self.upnorm1_3 = nn.BatchNorm2d(256)

        self.upconv2_init = PartialConv2d(256, 128, 3, 1, 1)
        self.upnorm2_init = nn.BatchNorm2d(128)

        ## Residual layer
        self.upconv2_1 = PartialConv2d(128, 128, 11, 1, 5)
        self.upnorm2_1 = nn.BatchNorm2d(128)

        self.upconv2_2 = PartialConv2d(128, 128, 13, 1, 6)
        self.upnorm2_2 = nn.BatchNorm2d(128)

        self.upconv2_3 = PartialConv2d(128, 128, 15, 1, 7)
        self.upnorm2_3 = nn.BatchNorm2d(128)

        self.upconv3 = PartialConv2d(128, 64, 3, 1, 1)
        self.upnorm3 = nn.BatchNorm2d(128)
        self.upconv4 = PartialConv2d(64, 32, 3, 1, 1)
        self.upnorm4 = nn.BatchNorm2d(128)
        self.upconv5 = PartialConv2d(32, 1, 3, 1, 1)
        self.upnorm5 = nn.BatchNorm2d(128)

    def encoder(self, x: Tensor, mask_in: Tensor) -> Tuple[Tensor, Tensor]:
        x, mask_in = self.conv1(x, mask_in)
        x = self.norm1(x)
        x = self.down_act(x)

        x, mask_in = self.conv2(x, mask_in)
        x = self.norm2(x)
        x = self.down_act(x)
        
        x, mask_in = self.conv3_init(x, mask_in)
        x = self.norm3_init(x)
        x = self.down_act(x)

        # Max pooling layer
        x, mask_in = self.max_pool(x), self.max_pool(mask_in) #-> 512,512

        #Residual layer
        x_1, mask_in = self.conv3_1(x, mask_in)
        x_1 = self.norm3_1(x_1)
        x_1 = self.down_act(x_1)

        x_1, mask_in = self.conv3_2(x_1, mask_in)
        x_1 = self.norm3_2(x_1)
        x_1 = self.down_act(x_1)
        
        x_1, mask_in = self.conv3_3(x_1, mask_in)
        x_1 = self.norm3_3(x_1)
        x_1 = self.down_act(x_1)
        
        x += x_1 # 512, 512

        x, mask_in = self.max_pool(x), self.max_pool(mask_in) #-> 256,256

        x, mask_in = self.conv4_init(x, mask_in)
        x = self.norm4_init(x)
        x = self.down_act(x)

        # Residual layer
        x_2, mask_in = self.conv4_1(x, mask_in)
        x_2 = self.norm4_1(x_2)
        x_2 = self.down_act(x_2)

        x_2, mask_in = self.conv4_2(x_2, mask_in)
        x_2 = self.norm4_2(x_2)
        x_2 = self.down_act(x_2)

        x_2, mask_in = self.conv4_3(x_2, mask_in)
        x_2 = self.norm4_3(x_2)
        x_2 = self.down_act(x_2)

        x+=x_2 # 256, 256

        x, mask_in = self.max_pool(x), self.max_pool(mask_in) #-> 128,128

        x, mask_in = self.conv5(x, mask_in)
        x = self.down_act(x)
        x = self.norm5(x)

        x, mask_in = self.max_pool(x), self.max_pool(mask_in)#-> 64, 64

        return x, x_1, x_2, mask_in
    
    def decoder(self, x: Tensor, x_1: Tensor, x_2: Tensor, mask_in: Tensor) -> Tuple[Tensor, Tensor]:
        
        x, mask_in = self.upsample(x), self.upsample(mask_in) #-> 512, 128,128
        
        x, mask_in = self.upconv1_init(x, mask_in)
        x = self.up_act(x)
        x = self.upnorm1_init(x)

        x, mask_in = self.upsample(x) + x_2, self.upsample(mask_in) #-> 256, 256,256

        last_x = x.clone()
        # Residual layer
        x, mask_in = self.upconv1_1(x, mask_in)
        x = self.upnorm1_1(x)
        x = self.up_act(x)

        x, mask_in = self.upconv1_2(x, mask_in)
        x = self.upnorm1_2(x)
        x = self.up_act(x)

        x, mask_in = self.upconv1_2(x, mask_in)
        x = self.upnorm1_2(x)
        x = self.up_act(x)

        x += last_x
        
        x, mask_in = self.upsample(x) + x_1, self.upsample(mask_in) #-> 256, 512,512

        x, mask_in = self.upconv2_init(x, mask_in)
        x = self.upnorm2_init(x)
        x = self.up_act(x)

        last_x = x.clone()

        # Residual layer
        x, mask_in = self.upconv2_1(x, mask_in)
        x = self.upnorm2_1(x)
        x = self.up_act(x)

        x, mask_in = self.upconv2_2(x, mask_in)
        x = self.upnorm2_2(x)
        x = self.up_act(x)

        x, mask_in = self.upconv2_2(x, mask_in)
        x = self.upnorm2_2(x)
        x = self.up_act(x)

        x+= last_x

        x, mask_in = self.upconv3(x, mask_in)
        x = self.upnorm3(x)
        x = self.up_act(x)

        x, mask_in = self.upsample(x) + x_1, self.upsample(mask_in) #-> 256, 1024,1024
        
        x, mask_in = self.upconv4(x, mask_in)
        x = self.upnorm4(x)
        x = self.up_act(x)
        
        x, mask_in = self.upconv5(x, mask_in)
        x = self.upnorm5(x)
        x = self.up_act(x)

        return x, mask_in
    
    def forward(self, x: Tensor, mask_in: Tensor) -> Tensor:
        gt = x.clone()
        first_mask = mask_in.clone()
        x, x_1, x_2, mask_in = self.encoder(x, mask_in)
        out, mask_in = self.decoder(x, x_1, x_2, mask_in) 
        return out*~first_mask.bool() + gt*first_mask, mask_in
    
    def loss(self, batch: Tensor) -> Tuple[Dict[str, float], Tensor]:
        I_gt, M_l_1 = batch

        I_out, M_l_2 = self(I_gt, M_l_1)
        
        L_pixel, L_perceptual, L_style = self.criterion(I_out, I_gt, M_l_1, M_l_2)

        loss = L_pixel + L_perceptual + L_style
        
        metrics = {
            f"Pixel-wise Loss": L_pixel.item(),
            f"Perceptual loss": L_perceptual.item(),
            f"Style Loss": L_style.item(),
            f"Overall": loss.item(),
        }

        return metrics, loss
    @torch.no_grad()
    def validation_step(self, batch) -> None:
        I_gt, M_l_1 = batch

        I_out, M_l_2 = self(I_gt, M_l_1)
        
        L_pixel, L_perceptual, L_style = self.criterion(I_out, I_gt, M_l_1, M_l_2)

        metrics = {
            f"Pixel-wise Loss": L_pixel.item(),
            f"Perceptual loss": L_perceptual.item(),
            f"Style Loss": L_style.item(),
            f"Overall": (L_pixel + L_perceptual + L_style).item(),
        }

        self.batch_metrics(metrics, "Validation")
        self.config["global_step_val"] += 1

    def imshow(self, train_loader):
        for batch in train_loader:
            I_gt, M_l_1 = batch

            I_out, M_l_2 = self(I_gt, M_l_1)
            
            I_out: np.array = self.inverse_transform(I_out[0, :, :, :].unsqueeze(0))
            I_gt: np.array = self.inverse_transform(I_gt[0, :, :, :].unsqueeze(0))
            
            mathcal_M = (M_l_1.bool() ^ M_l_2.bool()).cpu().detach().view(1024, 1024).numpy()
            
            inner_out = I_out*~M_l_2.bool()

            M_l_2 = M_l_2[0, :, :, :].cpu().detach().view(1024, 1024).numpy()


            gt_norm = ImageNormalize(stretch = HistEqStretch(I_gt[np.isfinite(I_gt)]))(I_gt)

            out_norm = ImageNormalize(stretch = HistEqStretch(I_out[np.isfinite(I_out)]))(I_out)

            I_out = I_out*mathcal_M

            diff_norm = ImageNormalize(stretch = HistEqStretch(I_out[np.isfinite(I_out)]))(I_out)

            inner_norm = ImageNormalize(stretch = HistEqStretch(inner_out[np.isfinite(inner_out)]))(inner_out)
            #Make plot
            fig, ax = plt.subplots(1, 4)
            ax[0].imshow(gt_norm)
            ax[0].yticks([])
            ax[0].xticks([])
            ax[1].imshow(out_norm)
            ax[1].yticks([])
            ax[1].xticks([])
            ax[2].imshow(diff_norm)
            ax[2].yticks([])
            ax[2].xticks([])
            ax[3].imshow(inner_norm)
            ax[3].yticks([])
            ax[3].xticks([])
            plt.show()
            break

class SmallUNet(TrainingPhase):
    def __init__(
            self,
            name_run: str, 
            criterion ,
            layers: int
    ) -> None:
        super().__init__(name_run, criterion)
        # General utils
        self.relu = nn.ReLU()
        self.spe_act: Callable[[Tensor], Tensor] = lambda x: F.relu(6*F.sigmoid(x))
        self.upsampling: Callable[[Tensor], Tensor] = lambda input: nn.functional.interpolate(input, scale_factor=2, mode = 'nearest')
        self.upsample: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]] = lambda x, mask_in: (self.upsampling(x), self.upsampling(mask_in))
        self.pool: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]] = lambda x, mask_in: (F.max_pool2d(x, 2, 2), F.max_pool2d(mask_in, 2, 2))
        self.layers = layers
        # Backbone
        for layer in range(layers):
            #Downsampling
            self.__setattr__(f'conv{layer}_1', PartialConv2d(1, 32, 9, 1, 4)) # -> 32, 1024, 1024
            self.__setattr__(f'norm{layer}_1', nn.BatchNorm2d(32)) 
            self.__setattr__(f'conv{layer}_2', PartialConv2d(32, 64, 7, 1, 3)) # -> 64, 512, 512
            self.__setattr__(f'norm{layer}_2', nn.BatchNorm2d(64))
            self.__setattr__(f'conv{layer}_3', PartialConv2d(64, 128, 5, 1, 2)) # -> 128, 256, 256
            self.__setattr__(f'norm{layer}_3', nn.BatchNorm2d(128))
            self.__setattr__(f'conv{layer}_4', PartialConv2d(128, 256, 3, 1, 1)) # -> 256, 128, 128
            self.__setattr__(f'norm{layer}_4', nn.BatchNorm2d(256))
            self.__setattr__(f'conv{layer}_5', PartialConv2d(256, 512, 3, 1, 1)) # -> 512, 64, 64
            self.__setattr__(f'norm{layer}_5', nn.BatchNorm2d(512))
            self.__setattr__(f'conv{layer}_6', PartialConv2d(512, 1024, 3, 1, 1)) # -> 1024, 32, 32
            
            # Multihead attention
            self.__setattr__(f'att{layer}', nn.MultiheadAttention(1024, 8))
            #Upsampling
            self.__setattr__(f'upconv{layer}_1', PartialConv2d(1024, 512, 9, 1 ,4)) # -> 512, 32, 32
            self.__setattr__(f'upnorm{layer}_1', nn.BatchNorm2d(512))
            self.__setattr__(f'upconv{layer}_2', PartialConv2d(512, 256, 7, 1, 3)) # -> 256, 64, 64
            self.__setattr__(f'upnorm{layer}_2', nn.BatchNorm2d(256))
            self.__setattr__(f'upconv{layer}_3', PartialConv2d(256, 128, 5, 1, 2)) # -> 128, 128, 128
            self.__setattr__(f'upnorm{layer}_3', nn.BatchNorm2d(128))
            self.__setattr__(f'upconv{layer}_4', PartialConv2d(128, 64, 3, 1, 1)) # -> 64, 256, 256
            self.__setattr__(f'upnorm{layer}_4', nn.BatchNorm2d(64))
            self.__setattr__(f'upconv{layer}_5', PartialConv2d(64, 32, 3, 1, 1)) # -> 32, 512, 512
            self.__setattr__(f'upnorm{layer}_5', nn.BatchNorm2d(32))
            self.__setattr__(f'upconv{layer}_6', PartialConv2d(32, 1, 3, 1, 1)) # -> 1, 1024, 1024

    def _act_maxpool(self, x: Tensor, mask_in: Tensor, act: Optional[bool] = True) -> Tuple[Tensor, Tensor]:
        if act:
            x = self.relu(x)
        x, mask_in = self.pool(x, mask_in)
        return x, mask_in
    
    def _act_upsample(self, x: Tensor, mask_in: Tensor, act: Optional[bool] = True) -> Tuple[Tensor, Tensor]:
        if act:
            x = self.spe_act(x)
        x, mask_in = self.upsample(x, mask_in)
        return x, mask_in

    def encoder(self, x: Tensor, mask_in: Tensor, layer: int) -> Tuple[Tensor, Tensor]:
        x, mask_in = self.__getattr__(f'conv{layer}_1')(x, mask_in) # -> 32, 1024, 1024
        x_1 = self.__getattr__(f'norm{layer}_1')(x)
        x, mask_in = self._act_maxpool(x_1, mask_in) # -> 32, 512, 512

        x, mask_in = self.__getattr__(f'conv{layer}_2')(x, mask_in) # -> 64, 512, 512
        x_2 = self.__getattr__(f'norm{layer}_2')(x)
        x, mask_in = self._act_maxpool(x_2, mask_in)# -> 64, 256, 256

        x, mask_in = self.__getattr__(f'conv{layer}_3')(x, mask_in) # -> 128, 256, 256
        x_3 = self.__getattr__(f'norm{layer}_3')(x)
        x, mask_in = self._act_maxpool(x_3, mask_in)# -> 128, 128, 128

        x, mask_in = self.__getattr__(f'conv{layer}_4')(x, mask_in)  # -> 256, 128, 128
        x_4 = self.__getattr__(f'norm{layer}_4')(x)
        x, mask_in = self._act_maxpool(x_4, mask_in) # -> 256, 64, 64
        
        x, mask_in = self.__getattr__(f'conv{layer}_5')(x, mask_in)  # -> 512, 64, 64
        x_5 = self.__getattr__(f'norm{layer}_5')(x)
        x, mask_in = self._act_maxpool(x_5, mask_in) # -> 512, 32, 32

        x, mask_in = self.__getattr__(f'conv{layer}_6')(x, mask_in)  # -> 1024, 32, 32
        return x_5, x_4, x_3, x_2, x_1, x, mask_in
    
    def decoder(self, x_5: Tensor, x_4: Tensor, x_3: Tensor, x_2: Tensor, x_1: Tensor, x: Tensor, mask_in: Tensor, layer: int) -> Tuple[Tensor, Tensor]:
        #Upsampling
        x, mask_in = self.__getattr__(f'upconv{layer}_1')(x, mask_in) # -> 512, 32, 32
        x = self.__getattr__(f'upnorm{layer}_1')(x)
        x, mask_in = self._act_upsample(x, mask_in) # -> 512, 64, 64

        x+=x_5

        x, mask_in = self.__getattr__(f'upconv{layer}_2')(x, mask_in) # -> 256, 64, 64
        x = self.__getattr__(f'upnorm{layer}_2')(x)
        x, mask_in = self._act_upsample(x, mask_in) # -> 256, 128, 128
        
        x+=x_4

        x, mask_in = self.__getattr__(f'upconv{layer}_3')(x, mask_in) # -> 128, 128, 128
        x = self.__getattr__(f'upnorm{layer}_3')(x)
        x, mask_in = self._act_upsample(x, mask_in) # -> 128, 256, 256

        x += x_3

        x, mask_in = self.__getattr__(f'upconv{layer}_4')(x, mask_in) # -> 64, 256, 256
        x = self.__getattr__(f'upnorm{layer}_4')(x)
        x, mask_in = self._act_upsample(x, mask_in) # -> 64, 512, 512

        x += x_2

        x, mask_in = self.__getattr__(f'upconv{layer}_5')(x, mask_in) # -> 32, 512, 512
        x = self.__getattr__(f'upnorm{layer}_5')(x)
        x, mask_in = self._act_upsample(x, mask_in) # -> 32, 1024, 1024

        x += x_1
        
        x, mask_in = self.__getattr__(f'upconv{layer}_6')(x, mask_in) # -> 1, 1024, 1024
        
        x = self.spe_act(x)

        return x, mask_in
    
    def _single_forward(self, x: Tensor, mask_in: Tensor, layer: int) -> Tuple[Tensor, Tensor]:
        b, _, _, _ = x.shape
        x_5, x_4, x_3, x_2, x_1, encoded, mask = self.encoder(x, mask_in, layer)
        att_input  = encoded.view(b, 1024, -1)
        att_output, _ = self.__getattr__(f'att{layer}')(att_input, att_input, att_input)
        out, mask_out = self.decoder(x_5, x_4, x_3, x_2, x_1, encoded + att_output.view(b, 1024, 32, 32), mask, layer)

        return out * ~mask_in.bool() + x * mask_in.bool(), mask_out
    
    def forward(self, x: Tensor, mask_in: Tensor) -> Tuple[Tensor, Tensor]:
        
        for layer in range(self.layers):
            x, mask_in = self._single_forward(x, mask_in, layer)

        return x, mask_in
    
    def compute_train_loss(self, batch: Tensor) -> Tuple[None, Tensor]:
        # Teacher forcing like method for training
        gt, prior_mask = batch
        overall = 0

        for layer in range(self.layers):
            I_out, updated_mask = self._single_forward(gt, prior_mask, layer) # Feeding the ground truth so the training is not dependent of prior layers 
            
            args = self.criterion(I_out, gt, prior_mask, updated_mask)

            overall+=args[-1] # The last one is always the sum of the others

            metrics = {key: value for key, value in zip(self.criterion.labels, args)}

            self.batch_metrics(metrics, f"Training/Layer_{layer}")

            prior_mask = updated_mask

        return None, overall
    
    @torch.no_grad()
    def compute_val_metrics(self, batch: Tensor) -> Tuple[Dict[str, float]]:
        gt, l1 = batch
        out, l2 = self(gt, l1)
        args = self.criterion(out, gt, l1, l2)
        metrics = {
            key: value for key, value in zip(self.criterion.labels, [arg.item() for arg in args])
        }       
        return metrics
    
    

class DefaultResidual(nn.Module):
    def __init__(
        self, channels: int, n_architecture: Tuple[int, ...] = (3, 2, 1)
    ) -> None:
        super().__init__()
        self.act = nn.ReLU()
        self.conv_layers = nn.ModuleList(
            [(PartialConv2d(channels, channels, 2 * n + 1, 1, n), nn.BatchNorm2d(channels)) for n in n_architecture]
        )

    def forward(self, I_out: Tensor, mask_in: Tensor) -> Tuple[Tensor, Tensor]:
        x= I_out.clone()

        for conv, norm in self.conv_layers:
            x, mask_in = conv(x, mask_in)
            x = norm(x)
            x = self.act(x)

        return I_out + x, mask_in

class CrossModel(SmallUNet):
    def __init__(self, name_run, criterion, layers):
        super().__init__(name_run, criterion, layers)
        self.attention = nn.MultiheadAttention(32*32, num_heads = 8)
    
    def _single_forward(self, x_1: Tensor, x_2: Tensor, time: Tensor, mask_in: Tensor, layer: int) -> Tuple[Tensor, Tensor]:
        b, _,_,_ = x_2.shape
        gt = x_2.clone()
        # Defining time step features
        f_5, f_4, f_3, f_2, f_1, x_2, mask_out = self.encoder(x_2, mask_in, layer)

        # Defining features from previous time step
        f_5_scaled, f_4_scaled, f_3_scaled, f_2_scaled, f_1_scaled, f_scaled, _ = self.encoder(x_1, torch.ones_like(x_1), layer)
        
        # Defining new features
        f_5 += (f_5_scaled/time)
        f_4 += (f_4_scaled/time)
        f_3 += (f_3_scaled/time)
        f_2 += (f_2_scaled/time)
        f_1 += (f_1_scaled/time)

        # Prepare queries, keys, and values for attention
        query = x_2.view(b, 1024, -1)  # Reshape for MultiheadAttention
        key_value = f_scaled.view(b, 1024, -1)

        # Apply cross-attention
        attn_output, _ = self.attention(query, key_value, key_value)

        # Reshape attention output and add to x_2
        attn_output = attn_output.permute(0, 2, 1).view(b, 1024, 32, 32)
        x_2 += attn_output

        # Defining
        x_2, mask_out = self.decoder(f_5, f_4, f_3, f_2, f_1, x_2, mask_out, layer)

        return gt * mask_in.bool() + x_2 * ~mask_in.bool(), mask_out
    
    def compute_train_loss(self, batch: Tensor) -> Tuple[None, Tensor]:
        # Teacher forcing like method for training
        x_1, x_2, time, prior_mask = batch
        overall = 0
         
        # Compute features
        for layer in range(self.layers):

            I_out, updated_mask = self._single_forward(x_1, x_2, time, prior_mask, layer) # Feeding the ground truth so the training is not dependent of prior layers 
            ## Computes all losses terms
            args = self.criterion(I_out, x_2, prior_mask, updated_mask)

            ## Gets the loss that will be backwarded
            overall+=args[-1] # The last one is always the sum of the others
            
            ## Loading metrics into dictionary
            metrics = {key: value for key, value in zip(self.criterion.labels, args)}

            ## Loading the metrics into tensorboard
            self.batch_metrics(metrics, f"Training/Layer_{layer}")

            ## Updating the mask for the next time step
            prior_mask = updated_mask

        return None, overall
    
    @torch.no_grad()
    def compute_val_metrics(self, batch: Tensor) -> Tuple[Dict[str, float]]:
        x_1, x_2, time, prior_mask = batch
        out, updated_mask = self(x_1, x_2, time, prior_mask)
        args = self.criterion(out, x_2, prior_mask, updated_mask)
        metrics = {
            key: value for key, value in zip(self.criterion.labels, [arg.item() for arg in args])
        }       
        return metrics
    
    def forward(self, x_1: Tensor, x_2: Tensor, time: float, mask_in: Tensor) -> Tuple[Tensor, Tensor]:
  
        for layer in range(self.layers):
            x_2, mask_in = self._single_forward(x_1, x_2, time, mask_in, layer)

        return x_2, mask_in    
    
from partial_conv import FourierPartialConv2d
from utils import FourierMultiheadAttention


class FourierModel(TrainingPhase):
    def __init__(self, name_run: str, criterion):
        super().__init__(name_run, criterion)
        self.maxpool = lambda x: F.max_pool2d(x, 4, 4)
        self.upsample = lambda x: F.interpolate(x, scale_factor=4, mode = 'nearest')

        self.hid_act = lambda x: F.relu(x)
        self.last_act = lambda x: F.sigmoid(x)

        self.fconv1 = FourierPartialConv2d(1, 32, 3, 1, 1) # -> 32, 1024, 1024
        self.norm1= nn.BatchNorm2d(32) 

        self.fconv2 = FourierPartialConv2d(32, 64, 3, 1, 1)# -> 64, 256, 256 
        self.norm2 = nn.BatchNorm2d(64)
        
        self.fconv3 = FourierPartialConv2d(64, 128, 3, 1, 1)# -> 128, 64, 64 
        self.norm3 = nn.BatchNorm2d(128)

        self.fconv4 = FourierPartialConv2d(128, 256, 3, 1, 1) # -> 256, 16, 16
        self.norm4 = nn.BatchNorm2d(256)

        self.fupconv1 = FourierPartialConv2d(256, 128, 3, 1, 1) # -> 128, 64, 64
        self.upnorm1 = nn.BatchNorm2d(128)

        self.fupconv2 = FourierPartialConv2d(128, 64, 3, 1, 1) # -> 64, 256, 256 
        self.upnorm2 = nn.BatchNorm2d(64)

        self.fupconv3 = FourierPartialConv2d(64, 32, 3, 1, 1) # ->  32, 1024, 1024
        self.upnorm3 = nn.BatchNorm2d(32)

        self.upconv4 = PartialConv2d(32, 1, 3, 1, 1) # ->  1, 1024, 1024

        self.attention = FourierMultiheadAttention(256, 8)

    def joint_maxpool(self, x: Tensor, mask_in: Tensor) -> Tuple[Tensor, Tensor]:
        return self.maxpool(x), self.maxpool(mask_in)

    def joint_upsample(self, x: Tensor, mask_in: Tensor) -> Tuple[Tensor, Tensor]:
        return self.upsample(x), self.upsample(mask_in)

    def forward(self, x: Tensor, mask_in: Tensor) -> Tuple[Tensor, Tensor]:
        gt = x.clone()
        init_mask = mask_in.clone()
        # First layer
        x, mask_in = self.fconv1(x, mask_in) # -> 32, 1024, 1024
        x = self.norm1(x)
        x_1 = self.hid_act(x)
        x, mask_in = self.joint_maxpool(x_1, mask_in) #  -> 32, 256, 256

        # Second layer
        x, mask_in = self.fconv2(x, mask_in) # -> 64, 256, 256
        x = self.norm2(x)
        x_2 = self.hid_act(x)
        x, mask_in = self.joint_maxpool(x_2, mask_in) # -> 64, 64, 64

        # Third layer
        x, mask_in = self.fconv3(x, mask_in)# -> 128, 64, 64
        x = self.norm3(x)
        x_3 = self.hid_act(x)
        x, mask_in = self.joint_maxpool(x_3, mask_in) # -> 128, 16, 16
        
        # Fourth layer
        x, mask_in = self.fconv4(x, mask_in)# -> 256, 16, 16
        x = self.norm4(x)
        x = self.hid_act(x)

        # Fourier attention layer and residual connection
        
        x, mask_in = self.joint_upsample(self.attention(x) + x, mask_in)        
        x, mask_in = self.fupconv1(x, mask_in) # -> 128, 64, 64
        x += x_3
        x = self.upnorm1(x)
        x = self.hid_act(x)


        x, mask_in = self.joint_upsample(x, mask_in)        
        x, mask_in = self.fupconv2(x, mask_in) # -> 64, 256, 256
        x += x_2
        x = self.upnorm2(x)
        x = self.hid_act(x)


        x, mask_in = self.joint_upsample(x, mask_in)        
        x, mask_in = self.fupconv3(x, mask_in) # -> 32, 1024, 1024
        x += x_1
        x = self.upnorm3(x)
        x = self.hid_act(x)


        x, mask_in = self.upconv4(x, mask_in)
        x = self.last_act(x)

        return x * ~init_mask.bool() + gt * init_mask.bool(), mask_in

    def compute_train_loss(self, batch: Tensor) -> Tuple[Dict[str, float], Tensor]:
        I_gt, M_l_1 = batch
        I_out, M_l_2 = self(I_gt, M_l_1)
        args = self.criterion(I_out, I_gt, M_l_1, M_l_2)
        metrics = {k:v.item() for k, v in zip(self.criterion.labels, args)}
        return metrics, args[-1]
            
    @torch.no_grad()
    def compute_val_metrics(self, batch: Tensor) -> Dict[str, float]:
        out, _ = self.compute_train_loss(batch)
        return out
    
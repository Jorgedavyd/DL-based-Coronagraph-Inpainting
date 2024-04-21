from typing import Tuple, Union, List, Dict, Any, Callable
from partial_conv import PartialConv2d
from torch import Tensor
from torch import nn
from loss import InpaintingLoss
from torch.utils.tensorboard import SummaryWriter
import torchmetrics.functional as f
import torch.nn.functional as F
from tqdm import tqdm
import torch
import os
from utils import import_config

##Training phase
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
class TrainingPhase(nn.Module):
    def __init__(
            self,
            name_run: str,
            criterion: InpaintingLoss = InpaintingLoss()
    ) -> None:
        super().__init__()
        self.config: Dict[str,Any] = import_config(name_run)
        #Tensorboard writer
        self.writer: SummaryWriter = SummaryWriter(f'{name_run}')
        self.name: str = name_run
        self.criterion: InpaintingLoss = criterion
        #History of changes
        self.train_epoch: List[List[int]] = []
        self.val_epoch: List[List[int]] = []
        self.lr_epoch: List[List[int]] = []
    @torch.no_grad()
    def batch_metrics(self, metrics: dict, card: str) -> None:

        if card == 'Training' or 'Training' in card.split('/'):
            self.train()
            mode = 'Training'
        elif card == 'Validation' or 'Validation' in card.split('/'):
            self.eval()
            mode = 'Validation'
        else:
            raise ValueError(f'{card} is not a valid card, it should at least have one of these locations: [Validation, Training]')
        
        self.writer.add_scalars(
            f'{card}/metrics',
            metrics,
            self.config['global_step_train'] if mode == 'Training' else self.config['global_step_val']
        )
        self.writer.flush()

        if mode == 'Training':
            self.train_epoch.append(list(metrics.values()))
        elif mode == 'Validation':
            self.val_epoch.append(list(metrics.values()))
    @torch.no_grad()
    def validation_step(
            self,
            batch
    ) -> None:
        img, prior_mask = batch
        x = torch.clone(img)
        x, updated_mask = self(x, prior_mask)
        L_pixel, L_perceptual, L_style = self.criterion(x, img, prior_mask, updated_mask)
        metrics = {
                    f'Pixel-wise Loss': L_pixel.item(),
                    f'Perceptual loss': L_perceptual.item(),
                    f'Style Loss': L_style.item(),
                    f'Overall': (L_pixel + L_perceptual + L_style).item()
                }
        self.batch_metrics(metrics, 'Validation')
        self.config['global_step_val'] += 1
    
    def training_step(
            self, 
            batch
    ) -> None:
        torch.cuda.empty_cache()
        #Defining the img and mask
        ground_truth, prior_mask = batch
        x = torch.clone(ground_truth)
        for i, args in enumerate(self.network):
            x, updated_mask = self.single_forward(x, prior_mask, *args)
            #Compute each term of the loss function
            L_pixel, L_perceptual, L_style = self.criterion(x, ground_truth, prior_mask, updated_mask)
            #Compute the final loss function
            loss = L_pixel + L_perceptual + L_style
            #Loading the metrics to tensorboard
            metrics = {
                        f'Pixel-wise Loss': L_pixel.item(),
                        f'Perceptual loss': L_perceptual.item(),
                        f'Style Loss': L_style.item(),
                        f'Overall': loss.item()
                    }
            self.batch_metrics(metrics, f'Training/Layer_{i}')
            self.writer.flush()
            #udpating the mask
            prior_mask = updated_mask
        #backpropagate
        loss.backward()
        #gradient clipping
        if self.grad_clip:
            nn.utils.clip_grad_value_(self.parameters(), self.grad_clip)
        #optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        #scheduler step
        if self.scheduler:
            lr = get_lr(self.optimizer)
            self.lr_epoch.append(lr)
            self.writer.add_scalars('Training', {'lr': lr}, self.config['global_step_train'])
            self.scheduler.step()
        self.writer.flush()
        # Global step shift
        self.config['global_step_train'] += 1
    @torch.no_grad()
    def end_of_epoch(self) -> None:
        train_metrics = Tensor(self.train_epoch).mean(dim = -1)
        val_metrics = Tensor(self.val_epoch).mean(dim = -1)
        lr_epoch = Tensor(self.lr_epoch).mean().item()

        train_metrics = {
                f'Pixel-wise Loss': train_metrics[0],
                f'Perceptual Loss': train_metrics[1],
                f'Style Loss': train_metrics[2],
                f'Overall': train_metrics[3]
            }

        val_metrics = {
                f'Pixel-wise Loss': val_metrics[0],
                f'Perceptual Loss': val_metrics[1],
                f'Style Loss': val_metrics[2],
                f'Overall': val_metrics[3]
            }
        
        self.writer.add_scalars(
            'Training/Epoch',
            train_metrics,
            global_step = self.config['epoch']
        )
        self.writer.add_scalars(
            'Validation/Epoch',
            val_metrics,
            global_step = self.config['epoch']
        )

        self.writer.add_hparams({
            'epochs': self.config['epoch'],
            'init_learning_rate': lr_epoch,
            'batch_size': self.batch_size,
            'weight_decay': self.weight_decay,
            'grad_clip': torch.inf if not self.grad_clip else self.grad_clip,
            'Inpainting layers': self.layers,
            'Convolution deepness': self.deep,
            'Pixel: outter factor': self.criterion.alpha[0],
            'Pixel: inner factor': self.criterion.alpha[1],
            'Pixel: diff factor': self.criterion.alpha[2],
            'Perceptual factor': self.criterion.alpha[3],
            'Style: outter factor': self.criterion.alpha[4],
            'Style: inner factor': self.criterion.alpha[5],
            'Style: diff factor': self.criterion.alpha[6],
            #insert contraint lambdas
            },
            train_metrics)
        
        self.writer.flush()

        self.train_epoch = []
        self.val_epoch = []
        self.lr_epoch = []

    @torch.no_grad()
    def save_config(self) -> None:
        os.makedirs(f'{self.name}/models', exist_ok= True)
        #Model weights
        self.config['optimizer_state_dict'] = self.optimizer.state_dict()
        self.config['scheduler_state_dict'] = self.scheduler.state_dict()
        self.config['model_state_dict'] = self.state_dict()

        torch.save(self.config, f'./{self.name}/models/{self.config["name"]}_{self.layers}_{self.deep}_{self.config["epoch"]}.pt')

    def fit(
            self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            epochs: int,
            batch_size: int,
            lr: float,
            weight_decay: float = 0.,
            grad_clip: bool = False,
            opt_func: torch.optim = torch.optim.Adam,
            lr_sched: torch.optim.lr_scheduler = None,
            saving_div: int = 5,
            graph: bool = False,
            sample_input: Tensor = None,

    ) -> None:
        torch.cuda.empty_cache()
        if graph:
            assert(sample_input is not None), 'If you want to visualize a graph, you must pass through a sample tensor'

        if self.config['optimizer_state_dict'] is not None:
            self.optimizer = opt_func(self.parameters(), lr, weight_decay=weight_decay).load_state_dict(self.config['optimizer_state_dict'])
            self.optimizer.param_groups[0]['lr'] = lr
        else:
            self.optimizer = opt_func(self.parameters(), lr, weight_decay=weight_decay)
        if lr_sched is not None:
            if self.config['scheduler_state_dict'] is not None:
                self.scheduler = lr_sched(self.optimizer, lr, epochs = epochs, steps_per_epoch = len(train_loader)).load_state_dict(self.config['scheduler_state_dict'])
                self.scheduler.learning_rate = lr
            else:
                self.scheduler = lr_sched(self.optimizer, lr, epochs = epochs, steps_per_epoch = len(train_loader))
            lr_sched = True
        # Add model to graph
        if graph and sample_input:
            self.writer.add_graph(self,sample_input)

        #Defining hyperparameters as attributes of the model and training object
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

        #decorating dataloaders
        
        for epoch in range(self.config['epoch'], self.config['epoch'] + epochs):
            #Define epoch
            self.config['epoch'] = epoch
            #Training loop
            self.train()
            for train_batch in tqdm(train_loader, desc = f'Training - Epoch: {epoch}'):
                #training step
                self.training_step(train_batch)
            #Validation loop
            self.eval()
            for val_batch in tqdm(val_loader, desc = f'Validation - Epoch: {epoch}'):
                self.validation_step(val_batch)
            #Save model and config if epoch mod(saving_div) = 0
            if epoch % saving_div == 0:
                self.save_config()
            #End of epoch
            self.end_of_epoch()

class SingleResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            deep: int = 2
    ) -> None:
        super().__init__()
        assert (deep>=1), 'Each step should be compose of at least 2 convolutions'
        self.conv = nn.ModuleList([
            nn.ModuleList([
                PartialConv2d(
                    in_channels = in_channels,
                    out_channels = in_channels,
                    kernel_size = 2*n + 1,
                    stride = 1,
                    padding = n,
                    return_mask = True
                ),
                nn.SiLU()
            ]) for n in range(deep, 0, -1)
        ])

    def _res_conv_forward(self, input: Tensor, mask_in: Tensor) -> Tuple[Tensor, Tensor]:
        
        for partial_conv, activation in self.conv:
            input, mask_in = partial_conv(input, mask_in)
            input = activation(input)
        
        return input, mask_in
        

    def forward(self, input: Tensor, mask_in: Tensor) -> Tuple[Tensor, Tensor]:

        x, mask = self._res_conv_forward(input, mask_in)

        return x + input, mask

class CoronagraphReconstructor(TrainingPhase):
    def __init__(
            self,
            name_run: str,
            criterion: InpaintingLoss = InpaintingLoss(),
            in_channels: int = 1,
            deep: int = 6,
            layers: int = 6
    ):
        super().__init__(name_run, criterion)
        self.layers = layers
        self.deep = deep
        self.network = nn.ModuleList([
            nn.ModuleList([
                PartialConv2d(in_channels, 32, 3, 1, 1),
                PartialConv2d(32, 64, 3, 1, 1),
                SingleResidualBlock(64, deep),
                nn.ReLU(),
                PartialConv2d(64, 32, 3, 1, 1),
                PartialConv2d(32, in_channels, 3, 1, 1),
            ]) for _ in range(layers)
        ])
    def single_forward(self, input: Tensor, mask_im: Tensor, *args):
        x, mask = args[0](input, mask_im)
        x, mask = args[1](x, mask)
        x, mask = args[2](x, mask)
        x = args[3](x)
        x, mask= args[4](x, mask)
        x, mask = args[5](x, mask)
        return x, mask

    def forward(self, x, mask) -> Union[Tuple[torch.Tensor], torch.Tensor]:
        for args in self.network:
            x, mask = self.single_forward(x, mask, *args)
        return x, mask


class DefaultResidual(nn.Module):
    def __init__(self, channels: int, n_architecture: Tuple[int, ...] = (3,2,1)) -> None:
        super().__init__()
        self.conv_layers = nn.ModuleList([
            PartialConv2d(channels, channels, 2*n + 1, 1, n) for n in n_architecture
        ])
    def forward(self, I_out: Tensor, mask_in: Tensor) -> Tuple[Tensor, Tensor]:
        for layer in self.conv_layers:
            I_out, mask_in = layer(I_out, mask_in)

        return I_out, mask_in
            
        
class Model(nn.Module):
    layers: int = 3
    def __init__(
            self
    ) -> None:
        super().__init__()
        self.residual_connection = ResidualConnection()
        #First layer
        self.conv1_1 = PartialConv2d(1, 64, 3, 1, 1)
        self.act = nn.ReLU()
        self.res1_1 = DefaultResidual(64, (7, 6, 5))
        self.conv1_2 = PartialConv2d(64, 1, 3, 1, 1)
        #Second layer
        self.conv2_1 = PartialConv2d(1, 64, 3, 1, 1)
        self.conv2_2 = PartialConv2d(64, 128, 3, 1, 1)
        self.res2_1 = DefaultResidual(128, (5, 4, 3))
        self.conv2_3 = PartialConv2d(128, 64, 3, 1, 1)
        self.conv2_4 = PartialConv2d(64, 1, 3, 1, 1)

        #Last layer
        self.conv3_1 = PartialConv2d(1, 64, 3, 1, 1)
        self.conv3_2 = PartialConv2d(64, 128, 3, 1, 1)
        self.res3_1 = DefaultResidual(128, (4,3, 2))
        self.conv3_3 = PartialConv2d(128, 64, 3, 1, 1)
        self.conv3_4 = PartialConv2d(64, 1, 3, 1, 1)

    def __per_layer_forward(self, layer: int,  I_out: Tensor, mask_in: Tensor) -> Tensor:
        match layer:
            case 0:
                #first layer forward
                out, mask = self.conv1_1(I_out, mask_in)
                out = self.act(out)
                out, mask = self.res1_1(out, mask)
                out = self.act(out)
                out, mask = self.conv1_2(out, mask)
                return out, mask
            case 1:
                out, mask = self.conv2_1(I_out, mask_in)
                out = self.act(out)
                out, mask = self.conv2_2(out, mask)
                out = self.act(out)
                out, mask = self.res2_1(out, mask)
                out = self.act(out)
                out, mask = self.conv2_3(out, mask)
                out = self.act(out)
                out, mask = self.conv2_4(out, mask)
                return out, mask
            case 3:
                out, mask = self.conv3_1(out, mask)
                out = self.act(out)
                out, mask = self.conv3_2(out, mask)
                out = self.act(out)
                out, mask = self.res3_1(out, mask)
                out = self.act(out)
                out, mask = self.conv3_3(out, mask)
                out = self.act(out)
                out, mask = self.conv3_4(out, mask)
                return out, mask
    def forward(self, I_out: Tensor, mask_in: Tensor) -> Tuple[Tensor, Tensor]:
        for layer in range(self.layers):
            I_out, mask_in = self.residual_connection(I_out, mask_in, lambda I_out, mask_in: self.__per_layer_forward(layer, I_out, mask_in))
        return I_out, mask_in           

class ResidualConnection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x: Tensor, mask_in: Tensor, sublayer: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        y, mask_out = sublayer(x, mask_in)
        return y + x, mask_out
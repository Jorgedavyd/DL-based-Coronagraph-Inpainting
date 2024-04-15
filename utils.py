
######################################################
##               Image Reconstruction               ##
##                                                  ##
## Including partial convolution layers, residual   ##
## connections. Alternative to "fuzzy_image.pro"    ##
## from SolarSoft reconstruction scheme.            ##
##                                                  ##
######################################################

import torch.nn.functional as F
from torch import nn
import torch
import math

## Model parameter initialization
def WeightInitializer(type: str = 'xavier'):
    """
    # Weight Initializer
    Returns a function that initializes the input module.
    ## Input:
    type: ['xavier', 'gaussian', 'orthogonal', 'kaiming'] 
    ## Output:
    weight_initializer(module): function
    """
    assert (type in ['xavier', 'gaussian', 'orthogonal', 'kaiming']), 'Not valid type, choose from: (xavier, gaussian, orthogonal, kaiming)'
    forward_pass = lambda module, method, *args: method(module.weight, *args)
    if type == 'xavier':
        return lambda module: forward_pass(module, nn.init.xavier_normal_, math.sqrt(2))
    elif type == 'kaiming':
        return lambda module: forward_pass(module, nn.init.kaiming_normal_, 0, 'fan_in')
    elif type == 'orthogonal':
        return lambda module: forward_pass(module, nn.init.orthogonal_, math.sqrt(2))
    else:
        return lambda module: forward_pass(module, nn.init.normal_,0.0, 0.2)

## Inpainting loss
class Loss(nn.Module):
    """
    # Inpainting Loss
    nn.Module implementation for inpainting training
    """
    def __init__(self):
        super().__init__()
        self.feature_extractor = 
    def forward(self, *args):
        """
        $$
        """

## Single block
def SingleBlock(in_channels, out_channels, pooling: int = None):
    block = nn.Sequential(
        PartialConv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace = True),
        PartialConv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace = True),
    )
    if pooling is not None:
        block.add_module(f'Downsampling: (size, size) -> (size/{pooling}, size/{pooling})', nn.AvgPool2d(pooling, pooling))
    return block
## Partial convolution   
### NVIDIA implementation https://github.com/NVIDIA/partialconv/blob/master/models/partialconv2d.py
class PartialConv2d(nn.Conv2d): 
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False  

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
            
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2], input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in
                        
                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize/(self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)


        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output
        
## Residual connection
class ResidualConnection(nn.Module):
    def __init__(self, norm = None, prenorm: bool = None):
        assert ((norm is not None and prenorm is not None) or (norm is None and prenorm is None)), 'Not valid parameters'
        super().__init__()
        self.norm = norm #probably batch norm
        if norm is None:
            self.forw_lamb = lambda x, sublayer: x + sublayer(x)
        else:
            self.forw_lamb = lambda x, sublayer: x + sublayer(norm(x)) if prenorm \
                else lambda x, sublayer: x + norm(sublayer(x))
    def forward(self, x, sublayer):
        return self.forw_lamb(x, sublayer)

from torch.utils.tensorboard import SummaryWriter
import torchmetrics.functional as f
from tqdm import tqdm
import os

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class TrainingPhase(nn.Module):
    def __init__(
            self,
            config: dict,
            name_run: str,
            task: str,
            num_classes: int = None
    ):
        """
        task: 'reg', 'mult', 'bin'
        config: dict of training config with optimizer, last model, and configs
        name_run: trivial
        """
        super().__init__()
        assert (task in ['reg', 'mult', 'bin']), 'task names: reg, mult, bin'
        if config is None:
            if task is None:
                raise ValueError('Without config, task parameter is mandatory')
        self.task = task
        if self.task == 'mult' and num_classes is not None:
            self.num_class = num_classes
        elif self.task == 'mult' and num_classes is None:
            raise ValueError('Ought input num_classes for multiclass task')
        elif self.task != 'mult' and num_classes is not None:
            raise ValueError('Gave num_classes parameter for non-multiclass task')
        else: 
            pass
        self.config = config
        #Tensorboard writer
        self.writer = SummaryWriter(f'{name_run}')
        self.name = name_run
        self.train_epoch = []
        self.val_epoch = []
    def batch_metrics(self, preds, targets, mode: str):
        if mode == 'Training':
            self.train()
        elif mode == 'Validation':
            self.eval()
        else:
            raise ValueError(f'{mode} is not a valid mode: [Validation, Training]')
        if self.task == 'reg':
            metrics = {
                'mse': F.mse_loss(preds, targets),
                'mae': F.l1_loss(preds, targets),
                'r2': f.r2_score(preds, targets),
            }
            self.writer.add_scalars(f'{mode}/Metrics', metrics, self.global_step_val)
        elif self.task == 'mult':
            metrics = {
                'cross_entropy': F.cross_entropy(preds, targets),
                'accuracy': f.accuracy(preds, targets, 'multiclass', num_classes = self.num_class, average='weighted'),
                'recall': f.recall(preds, targets, 'multiclass', num_classes = self.num_class, average='weighted'),
                'precision': f.precision(preds, targets, 'multiclass', num_classes = self.num_class, average='weighted'),
                'f1': f.f1_score(preds, targets, 'multiclass', num_classes = self.num_class, average='weighted'),
            }
            self.writer.add_scalars(f'{mode}/Metrics', metrics, self.global_step_val)
        else:
            metrics = {
                'binary_cross_entropy': F.binary_cross_entropy(preds, targets),
                'accuracy': f.accuracy(preds, targets, task = 'binary'),
                'recall': f.recall(preds, targets, task = 'binary'),
                'precision': f.precision(preds, targets, task = 'binary'),
                'f1': f.f1_score(preds, targets, task = 'binary')
            }
            self.writer.add_scalars(f'{mode}/Metrics', metrics, self.global_step_val)
        self.writer.flush()
        if mode == 'Training':
            self.train_epoch.append(metrics)
        elif mode == 'Validation':
            self.val_epoch.append(metrics)
    @torch.no_grad()
    def validation_step(
            self,
            batch
    ):
        preds = self(batch['input'])
        self.batch_metrics(preds, batch['target'], 'Validation')
        self.config['global_step_val'] += 1
    def training_step(
            self, 
            batch, 
            criterion, #init criterion with hyperparameters
            lr_sched: torch.optim.lr_scheduler,
            grad_clip: float = False,
    ):
        pred = self(batch['x'], batch['mask'])
        loss = criterion(batch['y'], pred)
        self.writer.add_scalars('Training/Loss', {
            'Criterion Loss': loss.item()
        }, self.config['global_step_train'])
        self.batch_metrics(pred, batch['y'], 'Training')
        self.writer.flush()
        #backpropagate
        loss.backward()
        #gradient clipping
        if grad_clip:
            nn.utils.clip_grad_value_(self.parameters(), grad_clip)
        #optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        #scheduler step
        if lr_sched:
            lr = get_lr(self.optimizer)
            self.writer.add_scalars('Training/Loss', {'lr': lr}, self.config['global_step_train'])
            self.scheduler.step()
        self.writer.flush()
        # Global step shift
        self.config['global_step_train'] += 1
    @torch.no_grad()
    def end_of_epoch(self) -> None:
        keys = self.train_history[0].keys() + self.validation_keys[0].keys()
        metrics = [(train_dict.values() + val_dict.values()) for train_dict, val_dict in zip(self.train_history, self.val_history)]
        metrics = torch.tensor(metrics, dtype = torch.float32)
        metrics = [value.item() for value in metrics.mean(-2)]

        self.writer.add_scalars(
            'Overall',
            {
                k:v for k,v in zip(keys, metrics)
            },
            global_step = self.config['last_epoch']
        )
        self.writer.add_hparams(
            metric_dict = {
                k:v for k,v in zip(keys, metrics)
            },
            global_step = self.config['last_epoch']
        )
        self.writer.flush()
        self.train_epoch = []
        self.val_epoch = []     
    @torch.no_grad()
    def save_config(self) -> None:
        os.makedirs(f'{self.name}/models', exists_ok = True)
        #Model weights
        self.config['optimizer_state_dict'] = self.optimizer.state_dict()
        self.config['scheduler_state_dict'] = self.scheduler.state_dict()
        self.config['model_state_dict'] = self.state_dict()

        torch.save(self.config, f'./{self.name}/models/{self.config["name"]}_{self.config["last_epoch"]}.pt')

    def fit(
            self,
            args,
            criterion: any, #list of criterions or single criterion 
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            epochs: int,
            lr: float,
            weight_decay: float = 0.,
            grad_clip: bool = False,
            opt_func: torch.optim = torch.optim.Adam,
            lr_sched: torch.optim.lr_scheduler = None,
            saving_div: int = 5,

    ) -> None:
        self.writer.add_hparams(hparam_dict={
            'epochs': epochs,
            'init_learning_rate': lr,
            'batch_size': train_loader[0].values()[0].shape[0],
            'weight_decay': weight_decay,
            'grad_clip': 0 if not grad_clip else grad_clip,
            'arch/d_model': args.d_model,
            'arch/sequence_length': train_loader[0].values()[0].shape[1],
            'arch/num_heads': args.num_heads,
            'arch/num_layers': args.num_layers,
        })

        if self.config['optimizer_state_dict'] is not None:
            self.optimizer = opt_func(self.parameters(), lr, weight_decay=weight_decay).load_state_dict(self.config['optimizer_state_dict'])
            self.optimizer.param_groups[0]['lr'] = lr
        else:
            self.optimizer = opt_func(self.parameters(), lr, weight_decay=weight_decay)
        if lr_sched is not None:
            if self.config['scheduler_state_dict'] is not None:
                self.scheduler = lr_sched(self.optimizer, lr, len(train_loader)).load_state_dict(self.config['scheduler_state_dict'])
                self.scheduler.learning_rate = lr
            else:
                self.scheduler = lr_sched(self.optimizer, lr, len(train_loader))
            lr_sched = True
        # Add model to graph
        forward_input = torch.randn(*train_loader[0].shape)
        self.writer.add_graph(self,forward_input)
        del forward_input

        for epoch in range(self.config['last_epoch'], self.config['last_epoch'] + epochs):
            # decorating iterable dataloaders
            train_loader = tqdm(train_loader, desc = f'Training - Epoch: {epoch}')
            val_loader = tqdm(val_loader, desc = f'Validation - Epoch: {epoch}')
            self.train()
            for train_batch in train_loader:
                #training step
                self.training_step(train_batch, criterion, lr_sched, grad_clip)
            self.eval()
            for val_batch in val_loader:
                self.validation_step(val_batch)
            #Save model and config if epoch mod(saving_div) = 0
            if epoch % saving_div == 0:
                self.save_config()
            #End of epoch
            self.end_of_epoch()
            #Next epoch
            self.config['last_epoch'] = epoch

## GPU usage

def get_default_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl) 


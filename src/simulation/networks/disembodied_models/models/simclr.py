import math
from argparse import ArgumentParser
from typing import Callable, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch import nn
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer

from networks.disembodied_models.models.archs import resnets

# TODO: import all the architectures
from networks.disembodied_models.models.archs import resnet_3b
from networks.disembodied_models.models.archs import resnet_2b
from networks.disembodied_models.models.archs import resnet_1b
from networks.disembodied_models.models.archs.resnet_3b import resnet_3blocks
from networks.disembodied_models.models.archs.resnet_2b import resnet_2blocks
from networks.disembodied_models.models.archs.resnet_1b import resnet_1block


#for distributed gpus when batch size>512 and training_epochs>100
class SyncFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        return grad_input[torch.distributed.get_rank() * ctx.batch_size:(torch.distributed.get_rank() + 1) *
                          ctx.batch_size]


class Projection(nn.Module):

    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128, depth=1):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depth = depth

        layers = []
        for i in range(depth):
            if i == 0:
                layers.append(nn.Linear(self.input_dim, self.hidden_dim))
            else:
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.BatchNorm1d(self.hidden_dim))
            layers.append(nn.ReLU())

        if depth == 0:
            self.hidden_dim = self.input_dim

        layers.append(nn.Linear(self.hidden_dim, self.output_dim, bias=False))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)


class SimCLR(pl.LightningModule):

    def __init__(
        self,
        gpus: int,
        num_samples: int,
        batch_size: int,
        num_nodes: int = 1,
        arch: str = 'resnet18',
        temporal_mode: str = None,
        hidden_mlp: int = 512,
        hidden_depth: int = 1,
        feat_dim: int = 128,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        temperature: float = 0.1,
        first_conv: bool = True, # changed from True to False
        maxpool1: bool = True, # changed from True to False
        optimizer: str = 'adam',
        lars_wrapper: bool = True,
        exclude_bn_bias: bool = False,
        start_lr: float = 0.,
        learning_rate: float = 1e-3,
        final_lr: float = 0.,
        weight_decay: float = 1e-6,
        **kwargs
    ):
        """
        Args:
            batch_size: the batch size
            num_samples: num samples in the dataset
            warmup_epochs: epochs to warmup the lr for
            lr: the optimizer learning rate
            opt_weight_decay: the optimizer weight decay
            loss_temperature: the loss temperature
        """
        super().__init__()
        self.save_hyperparameters()

        self.gpus = gpus
        self.num_nodes = num_nodes
        self.arch = arch
        self.temporal_mode = temporal_mode
        self.num_samples = num_samples
        self.batch_size = batch_size

        self.hidden_mlp = hidden_mlp
        self.hidden_depth = hidden_depth
        self.feat_dim = feat_dim
        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        self.optim = optimizer
        self.lars_wrapper = lars_wrapper
        self.exclude_bn_bias = exclude_bn_bias
        self.weight_decay = weight_decay
        self.temperature = temperature

        self.start_lr = start_lr
        self.final_lr = final_lr
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.encoder = self.init_encoder()

        self.projection = Projection(
            input_dim=self.hidden_mlp,
            hidden_dim=self.hidden_mlp,
            output_dim=self.feat_dim,
            depth=self.hidden_depth,
        )

        # FIXME: update this to dynammic optimization to eliminate out-of-index error.
        # compute iters per epoch
        nb_gpus = len(self.gpus) if isinstance(gpus, (list, tuple)) else self.gpus
        assert isinstance(nb_gpus, int)
        global_batch_size = self.num_nodes * nb_gpus * self.batch_size if nb_gpus > 0 else self.batch_size
        self.train_iters_per_epoch = self.num_samples // global_batch_size

        # define LR schedule
        warmup_lr_schedule = np.linspace(
            self.start_lr, self.learning_rate, self.train_iters_per_epoch * self.warmup_epochs
        )
        iters = np.arange(self.train_iters_per_epoch * (self.max_epochs - self.warmup_epochs))
        cosine_lr_schedule = np.array([
            self.final_lr + 0.5 * (self.learning_rate - self.final_lr) *
            (1 + math.cos(math.pi * t / (self.train_iters_per_epoch * (self.max_epochs - self.warmup_epochs))))
            for t in iters
        ])

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

    def init_encoder(self):
        if self.arch.startswith('resnet'):
            # Resnet34, Resnet18
            if self.arch == 'resnet34' or self.arch == 'resnet18':
                resnet = getattr(resnets, self.arch)
                print("Architecture selected - ", self.arch)
            # Resnet18 - 3blocks
            elif self.arch == 'resnet_3blocks':
                resnet = getattr(resnet_3b, self.arch)
                print("Architecture selected - Resnet18_3Blocks")
            # Resnet18 - 2blocks
            elif self.arch == 'resnet_2blocks':
                resnet = getattr(resnet_2b, self.arch)
                print("Architecture selected - Resnet18_2Blocks")
            # Resnet18 - 1block
            elif self.arch == 'resnet_1block':
                resnet = getattr(resnet_1b, self.arch)
                print("Architecture selected - Resnet18_1Block")
            encoder = resnet(first_conv=self.first_conv, maxpool1=self.maxpool1, return_all_feature_maps=False)
        else:
            NotImplementedError("Encoder not implemented.")

        return encoder


    '''
     this forward function is required for inference purposes,
     sometimes, the code will not validate without a forward function,
     therefore, in forward function, only include the part of network through which
     the embedding will be taken out. For instance, below has only resnet encoder.
     The shared step is the one which will have the logic for training and validation.
    
    '''
    def forward(self, x):
        # bolts resnet returns a list
        return self.encoder(x)

    def shared_step(self, batch):
        # push two images together in a temporal window - 

        if self.temporal_mode == '2images':
            # len(batch) = 3 for temporal model
            if len(batch) == 3:
                img1, img2, _ = batch # (img1, img2, index)
            else:
                # final image in tuple is for online eval
                (img1, img2, _), _ = batch

            # get h representations, bolts resnet returns a list
            h1 = self(img1)
            h2 = self(img2)

            # get z representations
            z1 = self.projection(h1)
            z2 = self.projection(h2)

            loss = self.nt_xent_loss(z1, z2, self.temperature)
        
        # push 2+ images in a temporal window - 
        else:
            # window_size = 3
            if len(batch) == 4:
                flag = 0
                img1, img2, img3, _ = batch # (img1, img2, img3, index)
                #print(img1.shape)
                
            # window_size = 4
            else:
                flag = 1
                img1, img2, img3, img4, _ = batch # (img1, img2, img3, img4, index)
                
            # get h representations, bolts resnet returns a list
            h1 = self(img1)
            h2 = self(img2)
            h3 = self(img3)
                
            if flag == 1:
                h4 = self(img4)
                z4 = self.projection(h4)
                    

            # get z representations
            z1 = self.projection(h1)
            z2 = self.projection(h2)
            z3 = self.projection(h3)

            # loss between z1 and other neighboring samples
            l1 = self.nt_xent_loss(z1,z2, self.temperature) # remember that a batch of images is passed to the loss function, not a single image is passed.
            l2 = self.nt_xent_loss(z1,z3, self.temperature)
            if flag == 1:
                l3 = self.nt_xent_loss(z1,z4, self.temperature)
                    
                # gather losses - 
                loss = (l1+l2+l3)
            else:
                # gather losses - 
                loss = (l1+l2)

        return loss
    

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        # # Must clear cache at a regular interval
        # if self.global_step % 10 == 0:
        #     torch.cuda.empty_cache()
        # log LR (LearningRateLogger callback doesn't work with LARSWrapper)
        # if self.current_epoch == 5:
        #     raise StopIteration("Training stopped at epoch {}".format(self.current_epoch))
        self.log('learning_rate', self.lr_schedule[self.trainer.global_step], on_step=True, on_epoch=False)

        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=['bias', 'bn']):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [{
            'params': params,
            'weight_decay': weight_decay
        }, {
            'params': excluded_params,
            'weight_decay': 0.,
        }]

    def configure_optimizers(self):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.weight_decay)
        else:
            params = self.parameters()

        if self.optim == 'sgd':
            optimizer = torch.optim.SGD(params, lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        elif self.optim == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.lars_wrapper:
            optimizer = LARSWrapper(
                optimizer,
                eta=0.001,  # trust coefficient
                clip=False
            )

        return optimizer

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer: Optimizer = None,
        optimizer_idx: int = None,
        optimizer_closure: Optional[Callable] = None,
        on_tpu: bool = None,
        using_native_amp: bool = None,
        using_lbfgs: bool = None,
    ) -> None:
        # warm-up + decay schedule placed here since LARSWrapper is not optimizer class
        # adjust LR of optim contained within LARSWrapper
        for param_group in optimizer.param_groups:
            # print(len(self.lr_schedule))
            param_group["lr"] = self.lr_schedule[self.trainer.global_step]

        # from lightning
        if not isinstance(optimizer, LightningOptimizer):
            # wraps into LightingOptimizer only for running step
            optimizer = LightningOptimizer.to_lightning_optimizer(optimizer, self.trainer)
        optimizer.step(closure=optimizer_closure)

    def nt_xent_loss(self, out_1, out_2, temperature, eps=1e-6):
        """
            assume out_1 and out_2 are normalized
            out_1: [batch_size, dim]
            out_2: [batch_size, dim]
        """

        '''
        out_1.shape - [512, 128]
        dim is 128 because the output from projection head is of 128 dims.
        '''
        
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out_1_dist = SyncFunction.apply(out_1)
            out_2_dist = SyncFunction.apply(out_2)
        else:
            out_1_dist = out_1
            out_2_dist = out_2

        # out: [2 * batch_size, dim] -> [1024, 128]
        # out_dist: [2 * batch_size * world_size, dim] -> [1024, 128]
        out = torch.cat([out_1, out_2], dim=0)
        
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out_dist.t().contiguous())  
        sim = torch.exp(cov / temperature) # denominator part of the loss function, since out_1 and out_2 is a batch that is passed, and not a single image.
        neg = sim.sum(dim=-1) # refer RM notes for understanding dimensionality


        # from each row, subtract e^1 to remove similarity measure for x1.x1
        row_sub = torch.Tensor(neg.shape).fill_(math.e).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size] -> numerator part of the loss function
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature) # only mat_mul requires the column and rows to be similar. This is element by element multiplication
        pos = torch.cat([pos, pos], dim=0) 

        loss = -torch.log(pos / (neg + eps)).mean() # num/den

        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        
        # temporal params
        #parser.add_argument("--temporal_mode", default="2images", type=str, help="images to push in a temporal window")
        # model arch params
        parser.add_argument("--arch", default="resnet18", type=str, help="convnet architecture") 
        # specify flags to store false
        parser.add_argument("--first_conv", action='store_false')
        parser.add_argument("--maxpool1", action='store_false')
        parser.add_argument("--hidden_mlp", default=512, type=int, help="hidden layer dimension in projection head")
        parser.add_argument("--hidden_depth", default=1, type=int, help="number of hidden layers in projection head")
        parser.add_argument("--feat_dim", default=128, type=int, help="feature dimension")

        # transform params
        #parser.add_argument("--gaussian_blur", action="store_true", help="add gaussian blur")
        #parser.add_argument("--jitter_strength", type=float, default=0.5, help="jitter strength")
        parser.add_argument("--data_dir", type=str, default=".", help="directory containing dataset")

        # training params
        parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")
        parser.add_argument("--optimizer", default="adam", type=str, help="choose between adam/sgd")
        parser.add_argument("--lars_wrapper", action='store_true', help="apple lars wrapper over optimizer used")
        parser.add_argument('--exclude_bn_bias', action='store_true', help="exclude bn/bias from weight decay")
        parser.add_argument("--warmup_epochs", default=5, type=int, help="number of warmup epochs")
        parser.add_argument("--batch_size", default=512, type=int, help="batch size per gpu")

        parser.add_argument("--temperature", default=0.5, type=float, help="temperature parameter in training loss")
        parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
        parser.add_argument("--learning_rate", default=1.0, type=float, help="base learning rate")
        parser.add_argument("--start_lr", default=0, type=float, help="initial warmup learning rate")
        parser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate")

        return parser
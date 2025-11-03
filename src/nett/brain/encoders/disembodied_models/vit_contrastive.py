"""
Vision Transformer (ViT) with contrastive learning for self-supervised representation learning.

This module implements Vision Transformer architectures with SimCLR-style contrastive learning.
It provides two ViT variants (standard and simple) along with a projection head for contrastive
learning and a Lightning module for training with the NT-Xent loss.

The implementation supports:
- Standard Vision Transformer (ViT) with dropout
- Simplified Vision Transformer (SimpleViT) without dropout
- SimCLR-style projection head with configurable depth
- NT-Xent (Normalized Temperature-scaled Cross Entropy) contrastive loss
- Temporal contrastive learning with multiple image windows
- PyTorch Lightning integration for easy training

Classes:
    VisionTransformer: Standard ViT wrapper with Xavier initialization.
    SimpleVisionTransformer: Simplified ViT wrapper with Xavier initialization.
    Projection: MLP projection head for contrastive learning (SimCLR-style).
    LitClassifier: Lightning module for training ViT with contrastive learning.
"""

# LIBRARIES

# Pytorch modules
import torch
from torch import nn
from torch.nn import functional as F

# Pytorch-Lightning
from lightning import LightningModule
from vit_pytorch import ViT, SimpleViT

import math


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) wrapper with Xavier weight initialization.

    This class wraps the standard ViT architecture from vit_pytorch and provides
    custom weight initialization using Xavier uniform initialization for better
    convergence during contrastive learning.

    Args:
        config: Dictionary containing ViT configuration parameters including:
            - image_size (int): Input image size (e.g., 224)
            - patch_size (int): Size of image patches (e.g., 16)
            - num_classes (int): Number of output classes/features
            - dim (int): Dimension of token embeddings
            - depth (int): Number of transformer layers
            - heads (int): Number of attention heads
            - mlp_dim (int): Dimension of MLP layers
            - channels (int): Number of input channels (e.g., 3 for RGB)
            - dropout (float, optional): Dropout probability
            - emb_dropout (float, optional): Embedding dropout probability

    Attributes:
        model: The underlying ViT model instance.
    """

    def __init__(self, config):
        """
        Initialize the Vision Transformer wrapper.

        Args:
            config: Dictionary with ViT configuration parameters.
        """
        super().__init__()
        self.model = ViT(**config)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize model weights using Xavier uniform initialization.

        Applies Xavier uniform initialization to Linear and Conv2d layers,
        and initializes biases with small normal noise. The final fully
        connected layer weights and biases are set to zero.

        Note:
            This method is decorated with @torch.no_grad() to avoid tracking
            gradients during initialization.
        """
        def _init(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

        self.apply(_init)
        nn.init.constant_(self.model.fc.weight, 0)
        nn.init.constant_(self.model.fc.bias, 0)

    def forward(self, x):
        """
        Forward pass through the Vision Transformer.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Output features of shape (batch_size, num_classes).
        """
        return self.model(x)


class SimpleVisionTransformer(nn.Module):
    """
    Simplified Vision Transformer wrapper with Xavier weight initialization.

    This class wraps the SimpleViT architecture from vit_pytorch, which is a
    streamlined version of ViT without dropout regularization. It provides the
    same custom weight initialization as the standard VisionTransformer.

    Args:
        config: Dictionary containing SimpleViT configuration parameters including:
            - image_size (int): Input image size (e.g., 224)
            - patch_size (int): Size of image patches (e.g., 16)
            - num_classes (int): Number of output classes/features
            - dim (int): Dimension of token embeddings
            - depth (int): Number of transformer layers
            - heads (int): Number of attention heads
            - mlp_dim (int): Dimension of MLP layers
            - channels (int): Number of input channels (e.g., 3 for RGB)

    Attributes:
        model: The underlying SimpleViT model instance.

    Note:
        Use this class when dropout is not needed in the transformer architecture.
    """

    def __init__(self, config):
        """
        Initialize the Simple Vision Transformer wrapper.

        Args:
            config: Dictionary with SimpleViT configuration parameters.
        """
        super().__init__()
        self.model = SimpleViT(**config)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize model weights using Xavier uniform initialization.

        Applies Xavier uniform initialization to Linear and Conv2d layers,
        and initializes biases with small normal noise. The final fully
        connected layer weights and biases are set to zero.

        Note:
            This method is decorated with @torch.no_grad() to avoid tracking
            gradients during initialization.
        """
        def _init(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

        self.apply(_init)
        nn.init.constant_(self.model.fc.weight, 0)
        nn.init.constant_(self.model.fc.bias, 0)

    def forward(self, x):
        """
        Forward pass through the Simple Vision Transformer.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Output features of shape (batch_size, num_classes).
        """
        return self.model(x)


class Projection(nn.Module):
    """
    MLP projection head for SimCLR-style contrastive learning.

    This module implements a multi-layer perceptron (MLP) that projects the
    backbone features into a lower-dimensional space where contrastive learning
    is performed. Each hidden layer consists of a Linear layer, BatchNorm, and ReLU
    activation. The final output is L2-normalized.

    The projection head architecture follows the SimCLR paper design, which showed
    that using a projection head improves the quality of learned representations.

    Args:
        input_dim: Dimension of input features from the backbone. Defaults to 2048.
        hidden_dim: Dimension of hidden layers in the MLP. Defaults to 2048.
        output_dim: Dimension of the output projection space. Defaults to 128.
        depth: Number of hidden layers (excluding output layer). Defaults to 1.
            If depth=0, a single linear layer maps input to output.

    Attributes:
        output_dim: Dimension of the output projection.
        input_dim: Dimension of input features.
        hidden_dim: Dimension of hidden layers (adjusted if depth=0).
        depth: Number of hidden layers.
        model: Sequential model containing all layers.

    Example:
        >>> proj = Projection(input_dim=512, hidden_dim=512, output_dim=128, depth=2)
        >>> features = torch.randn(32, 512)
        >>> projected = proj(features)
        >>> projected.shape
        torch.Size([32, 128])
    """

    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128, depth=1):
        """
        Initialize the Projection head.

        Args:
            input_dim: Dimension of input features. Defaults to 2048.
            hidden_dim: Dimension of hidden layers. Defaults to 2048.
            output_dim: Dimension of output projection. Defaults to 128.
            depth: Number of hidden layers. Defaults to 1.
        """
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
        """
        Forward pass through the projection head.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            L2-normalized projected features of shape (batch_size, output_dim).
        """
        x = self.model(x)
        return F.normalize(x, dim=1)


class LitClassifier(LightningModule):
    """
    PyTorch Lightning module for training Vision Transformer with contrastive learning.

    This module implements a complete training pipeline for self-supervised learning
    using Vision Transformer backbones and SimCLR-style contrastive loss. It supports
    both standard temporal contrastive learning (2 images) and extended temporal windows
    (multiple images).

    The architecture consists of:
    - A ViT backbone encoder (VisionTransformer or SimpleVisionTransformer)
    - A projection head (MLP) that maps backbone features to contrastive space
    - NT-Xent loss for contrastive learning

    Args:
        backbone_config: Dictionary with ViT configuration parameters.
            Must include: image_size, patch_size, num_classes, dim, depth, heads,
            mlp_dim, channels. Optional: dropout, emb_dropout.
        temporal_mode: Type of temporal window for contrastive learning.
            - "2images": Standard contrastive learning with 2 augmented views
            - None or other: Extended temporal window with multiple images
            Defaults to None.
        learning_rate: Learning rate for Adam optimizer. Defaults to 1e-3.
        hidden_mlp: Dimension of hidden layers in projection head. Defaults to 512.
        feat_dim: Dimension of projected features for contrastive learning. Defaults to 128.
        hidden_depth: Number of hidden layers in projection head. Defaults to 1.

    Attributes:
        backbone: Vision Transformer encoder (ViT or SimpleViT).
        projection: MLP projection head for contrastive learning.
        temperature: Temperature parameter for NT-Xent loss (fixed at 0.5).
        temporal_mode: Type of temporal window being used.
        hidden_mlp: Hidden dimension of projection head.
        feat_dim: Output dimension of projection head.
        hidden_depth: Depth of projection head.
        learning_rate: Learning rate for optimization.

    Example:
        >>> config = {
        ...     'image_size': 84,
        ...     'patch_size': 8,
        ...     'num_classes': 512,
        ...     'dim': 64,
        ...     'depth': 3,
        ...     'heads': 3,
        ...     'mlp_dim': 128,
        ...     'channels': 3
        ... }
        >>> model = LitClassifier(
        ...     backbone_config=config,
        ...     temporal_mode="2images",
        ...     learning_rate=1e-3
        ... )
    """

    def __init__(
        self,
        backbone_config: dict,
        temporal_mode: str = None,
        learning_rate: float = 1e-3,
        hidden_mlp=512,
        feat_dim=128,
        hidden_depth=1,
    ):
        """
        Initialize the LitClassifier module.

        Args:
            backbone_config: Dictionary with ViT configuration.
            temporal_mode: Temporal window type ("2images" or None). Defaults to None.
            learning_rate: Adam optimizer learning rate. Defaults to 1e-3.
            hidden_mlp: Projection head hidden dimension. Defaults to 512.
            feat_dim: Projection head output dimension. Defaults to 128.
            hidden_depth: Projection head depth. Defaults to 1.
        """
        super().__init__()
        self.save_hyperparameters()
        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim
        self.hidden_depth = hidden_depth
        self.learning_rate = learning_rate

        self.backbone = (
            VisionTransformer(backbone_config)
            if "dropout" in backbone_config
            else SimpleVisionTransformer(backbone_config)
        )  # ViT encoder
        self.temporal_mode = temporal_mode  # type of temporal window

        self.projection = Projection(  # SimCLR projection head
            input_dim=self.hidden_mlp,  # 512
            hidden_dim=self.hidden_mlp,  # 512
            output_dim=self.feat_dim,  # 128
            depth=self.hidden_depth,  # 1
        )

        self.temperature = 0.5  # default from SimCLR contrastive model

    def forward(self, x):
        """
        Forward pass to extract features from input images.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Feature embeddings from the ViT backbone of shape (batch_size, hidden_mlp).

        Note:
            This method returns backbone features directly (h), not projected features (z).
            For contrastive learning, use shared_step() which applies projection.
        """
        # SimCLR proposes to take the embeddings from the learned encoder
        feats = self.backbone(x)
        return feats

    def shared_step(self, batch):
        """
        Compute contrastive loss for a batch of data.

        This method handles both standard 2-image contrastive learning and extended
        temporal window learning with multiple images. The loss is computed using
        the NT-Xent contrastive loss between projected representations.

        Args:
            batch: Batch data with different formats depending on temporal_mode:
                - temporal_mode="2images": (img1, img2, index) or ((img1, img2, index), eval_img)
                - Other modes: (img1, img2, ..., imgN, index) for N images

        Returns:
            NT-Xent contrastive loss as a scalar tensor.

        Note:
            For temporal_mode="2images", loss is computed between z1 and z2.
            For other modes, loss is computed between z_list[0] and all other z_list[i].
        """
        # push two images together in a temporal window -

        if self.temporal_mode == "2images":
            # len(batch) = 3 for temporal model, not confirmed for non-temporal
            if len(batch) == 3:
                img1, img2, _ = batch  # (img1, img2, index)
            else:
                # final image in tuple is for online eval
                (img1, img2, _), _ = batch

            # get h representations, bolts resnet returns a list
            h1 = self.backbone(img1)
            h2 = self.backbone(img2)

            # get z representations
            z1 = self.projection(h1)
            z2 = self.projection(h2)

            loss = self.nt_xent_loss(z1, z2, self.temperature)

        # push 2+ images in a temporal window -
        else:
            z_list = []  # list of z representations
            # len = 4 for window_size = 3 because 3 images, and 1 index value
            for i in range(len(batch) - 1):  # last element is index
                # get h representation, bolts resnet returns a list
                h = self.backbone(batch[i])
                # get z representation
                z = self.projection(h)
                z_list.append(z)

            # get loss
            loss = 0
            # loss between z1 and other neighboring samples
            for i in range(1, len(z_list)):
                loss += self.nt_xent_loss(z_list[0], z_list[i], self.temperature)
        return loss

    def nt_xent_loss(self, out_1, out_2, temperature, eps=1e-6):
        """
        Compute NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

        This is the contrastive loss function used in SimCLR. It maximizes agreement
        between differently augmented views of the same image while minimizing
        agreement with other images in the batch.

        The loss encourages:
        - High similarity between positive pairs (out_1[i] and out_2[i])
        - Low similarity between negative pairs (out_1[i] and out_2[j] where i != j)

        Args:
            out_1: First set of normalized embeddings of shape (batch_size, dim).
            out_2: Second set of normalized embeddings of shape (batch_size, dim).
            temperature: Temperature parameter for scaling similarities. Lower values
                make the model more confident in its predictions.
            eps: Small epsilon for numerical stability. Defaults to 1e-6.

        Returns:
            Scalar loss value averaged over the batch.

        Note:
            - Assumes out_1 and out_2 are already L2-normalized
            - Uses temperature scaling to control the concentration of the distribution
            - The loss is symmetric: computed for both (out_1, out_2) and (out_2, out_1)

        Reference:
            Chen et al., "A Simple Framework for Contrastive Learning of Visual
            Representations" (SimCLR), ICML 2020.
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]

        out_1_dist = out_1
        out_2_dist = out_2

        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^1 to remove similarity measure for x1.x1
        row_sub = torch.Tensor(neg.shape).fill_(math.e).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss

    def training_step(self, batch): #, batch_idx):
        """
        Perform a single training step.

        Args:
            batch: Batch of training data (format depends on temporal_mode).

        Returns:
            Training loss for this batch.

        Note:
            Logs 'train_loss' to TensorBoard/console with on_step=True and on_epoch=True.
        """
        # loss = self.step(batch, batch_idx)
        loss = self.shared_step(batch)
        self.log("train_loss", loss, on_epoch=True, on_step=True)  # training_loss
        return loss

    def validation_step(self, batch, batch_idx): # pylint: disable=unused-argument
        """
        Perform a single validation step.

        Args:
            batch: Batch of validation data (format depends on temporal_mode).
            batch_idx: Index of the current batch.

        Returns:
            Validation loss for this batch.

        Note:
            Logs 'val_loss' to TensorBoard/console with on_step=False and on_epoch=True.
        """
        # loss = self.step(batch, batch_idx)
        loss = self.shared_step(batch)

        # TODO: log val_acc
        self.log("val_loss", loss, on_step=False, on_epoch=True)  # for val_loss
        return loss

    def configure_optimizers(self):
        """
        Configure optimizer for training.

        Returns:
            Adam optimizer with the configured learning rate.

        Note:
            Uses self.learning_rate which is saved as a hyperparameter via
            self.save_hyperparameters() in __init__.
        """
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     parser = ArgumentParser(parents=[parent_parser], add_help=False)
    #     parser.add_argument('--learning_rate', type=float, default=0.0001)
    #     # transform params
    #     parser.add_argument("--gaussian_blur", action="store_true", help="add gaussian blur")
    #     parser.add_argument("--jitter_strength", type=float, default=0.5, help="jitter strength")
    #     parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
    #     parser.add_argument("--start_lr", default=0, type=float, help="initial warmup learning rate")
    #     parser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate")
    #     parser.add_argument("--temperature", default=0.5, type=float, help="temperature parameter in training loss")
    #     parser.add_argument("--lars_wrapper", action='store_true', help="apple lars wrapper over optimizer used")
    #     parser.add_argument('--exclude_bn_bias', action='store_true', help="exclude bn/bias from weight decay")
    #     parser.add_argument("--warmup_epochs", default=5, type=int, help="number of warmup epochs")

    #     return parser

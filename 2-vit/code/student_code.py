import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.functional import fold, unfold
from torchvision.utils import make_grid
import math

from utils import resize_image
import custom_transforms as transforms
from custom_blocks import (PatchEmbed, window_partition, window_unpartition,
                           DropPath, MLP, trunc_normal_)


#################################################################################
# You will need to fill in the missing code in this file
#################################################################################


#################################################################################
# Part I.1: Understanding Convolutions
#################################################################################
class CustomConv2DFunction(Function):
    @staticmethod
    def forward(ctx, input_feats, weight, bias, stride=2, padding=0):
        """
        Forward propagation of convolution operation.
        We only consider square filters with equal stride/padding in width and height!

        Args:
          input_feats: input feature map of size N * C_i * H * W
          weight: filter weight of size C_o * C_i * K * K
          bias: (optional) filter bias of size C_o
          stride: (int, optional) stride for the convolution. Default: 1
          padding: (int, optional) Zero-padding added to both sides of the input. Default: 0

        Outputs:
          output: responses of the convolution  w*x+b

        """
        # sanity check
        assert weight.size(2) == weight.size(3)
        assert input_feats.size(1) == weight.size(1)
        assert isinstance(stride, int) and (stride > 0)
        assert isinstance(padding, int) and (padding >= 0)

        # save the conv params
        kernel_size = weight.size(2)
        ctx.stride = stride
        ctx.padding = padding
        ctx.input_height = input_feats.size(2)
        ctx.input_width = input_feats.size(3)

        # make sure this is a valid convolution
        assert kernel_size <= (input_feats.size(2) + 2 * padding)
        assert kernel_size <= (input_feats.size(3) + 2 * padding)

        #################################################################################
        # Fill in the code here
        #################################################################################

        # save for backward (you need to save the unfolded tensor into ctx)
        # ctx.save_for_backward(your_vars, weight, bias)
        
        # Save Weights and Bias
        ctx.weight = weight
        ctx.bias = bias
        
        # Declare output tensor
        out_height = (input_feats.size(2) + 2 * padding - kernel_size) // stride + 1
        out_width = (input_feats.size(3) + 2 * padding - kernel_size) // stride + 1
        output = torch.zeros(input_feats.size(0), weight.size(0), out_height, out_width)
        
        
        # Padding
        if ctx.padding > 0:
            input_feats = nn.functional.pad(input_feats, (padding, padding, padding, padding))
            
        # Unfold the input tensor
        input_unf = unfold(input_feats, kernel_size= (kernel_size, kernel_size), stride = stride)
        
        # Convolution
        output = torch.matmul(weight.view(weight.size(0), -1), input_unf)
        output = output.view(input_feats.size(0), weight.size(0), out_height, out_width)
            
        # Add Bias
        output += bias.view(1, -1, 1, 1)
        
        # Save unfolded tensor
        ctx.save_for_backward(input_unf)
        
        return output.clone()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward propagation of convolution operation

        Args:
          grad_output: gradients of the outputs

        Outputs:
          grad_input: gradients of the input features
          grad_weight: gradients of the convolution weight
          grad_bias: gradients of the bias term

        """
        # unpack tensors and initialize the grads
        #your_vars, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        weight = ctx.weight

        # recover the conv params
        kernel_size = weight.size(2)
        stride = ctx.stride
        padding = ctx.padding
        input_height = ctx.input_height
        input_width = ctx.input_width
        bias = ctx.bias

        #################################################################################
        # Fill in the code here
        #################################################################################
        # compute the gradients w.r.t. input and params
        
        # Unfold the input tensor
        input_unf = ctx.saved_tensors[0]
        output_unf = grad_output.view(grad_output.size(0), grad_output.size(1), -1)
        
        
        # Matrix Multiplication
        #grad_weight = torch.matmul(output_unf.permute(1, 0, 2), input_unf.permute(0, 2, 1))
        grad_weight = torch.matmul(output_unf.permute(1, 0, 2).reshape(output_unf.shape[1],-1) , input_unf.permute(0, 2, 1).reshape(-1, input_unf.shape[1]))
        grad_input = torch.matmul(ctx.weight.view(ctx.weight.size(0), -1).permute(1, 0), output_unf)
        
        
        # Fold the gradients
        grad_input = fold(input = grad_input, 
                          output_size = (input_height + 2*padding, input_width + 2*padding), 
                          kernel_size = (kernel_size, kernel_size), 
                          padding = 0, 
                          stride = stride)
        
        if padding > 0:
            grad_input = grad_input[:,:,padding:-padding, padding:-padding] # Remove Padding
        
        #grad_weight = grad_weight.mean(dim=0)
        grad_weight = grad_weight.view(ctx.weight.size())
        
        # compute the gradients w.r.t. bias (if any)
        if bias is not None and ctx.needs_input_grad[2]:
            # compute the gradients w.r.t. bias (if any)
            grad_bias = grad_output.sum((0, 2, 3))
        
        return grad_input, grad_weight, grad_bias, None, None

custom_conv2d = CustomConv2DFunction.apply


class CustomConv2d(Module):
    """
    The same interface as torch.nn.Conv2D
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(CustomConv2d, self).__init__()
        assert isinstance(kernel_size, int), "We only support squared filters"
        assert isinstance(stride, int), "We only support equal stride"
        assert isinstance(padding, int), "We only support equal padding"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # not used (for compatibility)
        self.dilation = dilation
        self.groups = groups

        # register weight and bias as parameters
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        # initialization using Kaiming uniform
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # call our custom conv2d op
        return custom_conv2d(input, self.weight, self.bias, self.stride, self.padding)

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}, padding={padding}"
        )
        if self.bias is None:
            s += ", bias=False"
        return s.format(**self.__dict__)


#################################################################################
# Part I.2: Design and train a convolutional network
#################################################################################
class SimpleNet(nn.Module):
    # a simple CNN for image classifcation
    def __init__(self, conv_op=nn.Conv2d, num_classes=100):
        super(SimpleNet, self).__init__()
        # you can start from here and create a better model
        self.features = nn.Sequential(
            # conv1 block: conv 7x7
            conv_op(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # conv2 block: simple bottleneck
            conv_op(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            conv_op(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            conv_op(64, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # conv3 block: simple bottleneck
            conv_op(256, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            conv_op(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            conv_op(128, 512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        # global avg pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def reset_parameters(self):
        # init all params
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.consintat_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # you can implement adversarial training here
        # if self.training:
        #   # generate adversarial sample based on x
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CustomNet(nn.Module):
    """
    Added Batch Normalization after each Convolutional Layer in SimpleNet
    """
    def __init__(self, conv_op=nn.Conv2d, num_classes=100):
        super(CustomNet, self).__init__()
        # you can start from here and create a better model
        self.features = nn.Sequential(
            # conv1 block: conv 7x7
            conv_op(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # conv2 block: simple bottleneck
            conv_op(64, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv_op(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv_op(64, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # conv3 block: simple bottleneck
            conv_op(256, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            conv_op(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            conv_op(128, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        # global avg pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def reset_parameters(self):
        # init all params
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.consintat_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # you can implement adversarial training here
        # if self.training:
        #   # generate adversarial sample based on x
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CustomNet2(nn.Module):
    """
    Deeper network than CustomNet
    """
    def __init__(self, conv_op=nn.Conv2d, num_classes=100):
        super(CustomNet2, self).__init__()
        # you can start from here and create a better model
        self.features = nn.Sequential(
            # conv1 block: conv 7x7
            conv_op(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # conv2 block:
            conv_op(64, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv_op(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv_op(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            
            # conv3 block: simple bottleneck
            conv_op(64, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            conv_op(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            conv_op(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            
            # conv4 block: simple bottleneck
            conv_op(128, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            conv_op(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            conv_op(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            
            # conv5 block: simple bottleneck
            conv_op(256, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            conv_op(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            conv_op(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),

        )
        # global avg pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def reset_parameters(self):
        # init all params
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.consintat_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # you can implement adversarial training here
        # if self.training:
        #   # generate adversarial sample based on x
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# change this to your model!
default_cnn_model = SimpleNet

#################################################################################
# Part II.1: Understanding self-attention
#################################################################################
class Attention(nn.Module):
    """Multi-head Self-Attention."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
    ):
        """
        Args:
            dim (int): Embedding dimension. Number of input channels. 
                We assume Q, K, V will be of same dimension as the input.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        # linear projection for query, key, value
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        # linear projection at the end
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor with shape 
                (batch_size, num_patch_along_height, num_patch_along_width, embed_dim)
                Or (B, H, W, C).
        """
        # input size (B, H, W, C). C = self.head_dim * self.num_heads
        B, H, W, C = x.shape
        
        x_qkv = self.qkv(x) # (B, H, W, 3 * C)
        x_qkv_reshaped = x_qkv.reshape(
                B, H * W, 3, self.num_heads, -1
            ) # (B, H * W, 3, num_heads, head_dim)

        qkv = x_qkv_reshaped.permute(2, 0, 3, 1, 4) # (3, B, nHead, H * W, head_dim)
        
        # q, k, v with shape (B * nHead, H * W, head_dim)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        #################################################################################
        # Fill in the code here
        #################################################################################
        
        x_scores = torch.matmul(q, k.transpose(-2,-1))
        x_attn_weights = self.softmax(x_scores / self.scale) # (B * nHead, H * W, H * W)
        
        x = torch.matmul(x_attn_weights, v) # (B * nHead, H * W, head_dim)   
        x_reshaped = (
            x.reshape(B, self.num_heads, H * W, self.head_dim)
            .permute(0, 2, 1, 3) # (B, H * W, nHead, head_dim)
            .reshape(B, H * W, -1) # (B, H * W, C)
            .reshape(B, H, W, C)
        ) 
        x = self.proj(x_reshaped) # (B, H, W, C)
        return x

class TransformerBlock(nn.Module):
    """Transformer blocks with support of local window self-attention"""
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        window_size=0,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            window_size (int): Window size for window attention blocks.
                If it equals 0, then global attention is used.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer
        )

        self.window_size = window_size

    def forward(self, x):
        """
        Args: 
            x (torch.Tensor): Input tensor. Shape 
                (batch_size, H, W, embed_dim).
        """
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


#################################################################################
# Part II.2: Design and train a vision Transformer
#################################################################################
class SimpleViT(nn.Module):
    """
    This module implements Vision Transformer (ViT) backbone in
    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(
        self,
        img_size=128,
        num_classes=100,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=6,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_abs_pos=True,
        window_size=4,
        window_block_indexes=(0, 2),
    ):
        """
        Args:
            img_size (int): Input image size.
            num_classes (int): Number of object categories
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            window_size (int): Window size for local attention blocks.
            window_block_indexes (list): Indexes for blocks using local attention.
                Local window attention allows more efficient computation, and can be
                coupled with standard global attention. E.g., [0, 2] indicates the
                first and the third blocks will use local window attention, while
                other block use standard attention.

        Feel free to modify the default parameters here.
        """
        super(SimpleViT, self).__init__()

        if use_abs_pos:
            # Initialize absolute positional embedding with image size
            # The embedding is learned from data
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1, img_size // patch_size, img_size // patch_size, embed_dim
                )
            ) # (1, H, W, C)
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # patch embedding layer
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        ########################################################################
        # Fill in the code here
        ########################################################################
        # The implementation shall define some Transformer blocks

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
            
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                window_size=window_size if i in window_block_indexes else 0,
            )
            for i in range(depth)
        ])
        
        interm_dim = embed_dim // 2
        self.fc = nn.Linear(embed_dim, interm_dim)
        self.fc2 = nn.Linear(interm_dim, num_classes)

        self.apply(self._init_weights)
        # add any necessary weight initialization here

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of an image. Shape (batch_size, num_channel, img_height, img_width).
        
        Returns:
            (torch.Tensor): Shape (batch_size, )
        """
        ########################################################################
        # Fill in the code here
        ########################################################################
        
        x_emb = self.patch_embed(x) # (B, H, W, C)
        
        if self.pos_embed is not None:
            x_emb = x_emb + self.pos_embed
            
            
        for block in self.blocks:
            x_emb = block(x_emb) # (B, H, W, C)
        
        x_emb = x_emb.reshape(x_emb.shape[0], -1, x_emb.shape[-1]) # (B, H * W, C)
        x_emb_avg = x_emb.mean(dim=1) # (B, C)
        x_interm = self.fc(x_emb_avg)
        x_cls = self.fc2(x_interm)
        return x_cls

# change this to your model!
default_vit_model = SimpleViT

# define data augmentation used for training, you can tweak things if you want
def get_train_transforms():
    train_transforms = []
    train_transforms.append(transforms.Scale(144))
    train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.append(transforms.RandomColor(0.15))
    train_transforms.append(transforms.RandomRotate(15))
    train_transforms.append(transforms.RandomSizedCrop(128))
    train_transforms.append(transforms.ToTensor())
    # mean / std from imagenet
    train_transforms.append(transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ))
    train_transforms = transforms.Compose(train_transforms)
    return train_transforms

# define data augmentation used for validation, you can tweak things if you want
def get_val_transforms():
    val_transforms = []
    val_transforms.append(transforms.Scale(144))
    val_transforms.append(transforms.CenterCrop(128))
    val_transforms.append(transforms.ToTensor())
    # mean / std from imagenet
    val_transforms.append(transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ))
    val_transforms = transforms.Compose(val_transforms)
    return val_transforms


#################################################################################
# Part III: Adversarial samples
#################################################################################
class PGDAttack(object):
    def __init__(self, loss_fn, num_steps=10, step_size=0.01, epsilon=0.1):
        """
        Attack a network by Project Gradient Descent. The attacker performs
        k steps of gradient descent of step size a, while always staying
        within the range of epsilon (under l infinity norm) from the input image.

        Args:
          loss_fn: loss function used for the attack
          num_steps: (int) number of steps for PGD
          step_size: (float) step size of PGD (i.e., alpha in our lecture)
          epsilon: (float) the range of acceptable samples
                   for our normalization, 0.1 ~ 6 pixel levels
                   
        num_steps 10 originally
        """
        self.loss_fn = loss_fn
        self.num_steps = num_steps
        self.step_size = step_size
        self.epsilon = epsilon

    def perturb(self, model, input):
        """
        Given input image X (torch tensor), return an adversarial sample
        (torch tensor) using PGD of the least confident label.

        See https://openreview.net/pdf?id=rJzIBfZAb

        Args:
          model: (nn.module) network to attack
          input: (torch tensor) input image of size N * C * H * W

        Outputs:
          output: (torch tensor) an adversarial sample of the given network
        """
        # clone the input tensor and disable the gradients
        output = input.clone()
        input.requires_grad = False
        model.eval()

        # loop over the number of steps
        # for _ in range(self.num_steps):
        #################################################################################
        # Fill in the code here
        #################################################################################
        for _ in range(self.num_steps):
            # Forward Pass
            output.requires_grad = True
            y = model(output)
            gt = y.detach().argmin(dim=1)
     
            # Loss
            loss = self.loss_fn(y, gt)
            loss.backward()
            
            with torch.no_grad():
                # Gradient
                grad = output.grad.data
            
                # Update
                # grad.sign() is +1/-1 
                output = output + self.step_size * grad.sign()
            
                # Clip
                diff = output - input
                diff = torch.clamp(diff, -self.epsilon, self.epsilon)
                output = input + diff

        model.train()
        output.requires_grad = False
        return output
    
    def change_state(self, num_steps, epsilon):
        self.num_steps = num_steps
        self.epsilon = epsilon

default_attack = PGDAttack


def vis_grid(input, n_rows=10):
    """
    Given a batch of image X (torch tensor), compose a mosaic for visualziation.

    Args:
      input: (torch tensor) input image of size N * C * H * W
      n_rows: (int) number of images per row

    Outputs:
      output: (torch tensor) visualizations of size 3 * HH * WW
    """
    # concat all images into a big picture
    output_imgs = make_grid(input.cpu(), nrow=n_rows, normalize=True)
    return output_imgs

default_visfunction = vis_grid


if __name__ == "__main__":
    # You can test your functions here
    conv = CustomConv2DFunction()
    input_feats = torch.randn(1, 3, 32, 32, requires_grad=True, device="cpu")
    weight = torch.randn(64, 3, 3, 3, requires_grad=True, device="cpu")
    bias = torch.randn(64, requires_grad=True, device="cpu")
    
    
    
    # # Torch Conv
    # custom_layer = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=0)
    # custom_layer.weight = nn.Parameter(weight)
    # custom_layer.bias = nn.Parameter(bias)
    
    # input_custom = torch.clone(input_feats).detach().requires_grad_(True)
    # y_custom = custom_layer(input_custom)
    # y_custom.retain_grad()
    # z_custom = y_custom.sum()
    # z_custom.backward()
    
    
    # # Check backward Prop
    # layer = CustomConv2d(in_channels = 3,out_channels = 64,kernel_size = 5, stride=1,padding=0,dilation=1,groups=1,bias=True)
    # layer.weight = nn.Parameter(weight)
    # layer.bias = nn.Parameter(bias)
    
    # y = layer(input_feats)
    # y.retain_grad()
    # z = y.sum()
    # z.backward()
    
    
    # print(input_custom.grad - input_feats.grad)
    
    # Torch Conv
    torch_layer = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=10)
    torch_layer.weight = nn.Parameter(weight)
    torch_layer.bias = nn.Parameter(bias)
    torch_output = torch_layer(input_feats)
    
    # Cuswtom Layer
    custom_layer = CustomConv2d(in_channels = 3,out_channels = 64,kernel_size = 3, stride=2,padding=10,bias=True)
    custom_layer.weight = nn.Parameter(weight)
    custom_layer.bias = nn.Parameter(bias)
    custom_output = custom_layer(input_feats)
    
    
    # Error
    print(torch_output - custom_output)
    
    print(custom_output.shape)
    
    print(torch_output.shape)
    
    
"""Decoder module for producing denoised images"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from early_vision.models.divisive_normalization.divisive_norm import DivNorm
from early_vision.utils.model_utils import crop_tensor


class Upconv(nn.Module):
    """Decoder with learned upsampling for denoising images."""

    def __init__(self,
                 in_channels,
                 hidden_dim,
                 num_layers,
                 out_dim=3,
                 fsize=3,
                 nl=nn.ReLU,
                 final_tanh=True,
                 ):
        super(Upconv, self).__init__()
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.final_tanh = final_tanh
        self.nl = nl
        self.fsize = fsize
        self.conv_layers, self.norm_layers = {}, {}

        self.init_conv = nn.Conv2d(in_channels,
                                   hidden_dim,
                                   1)
        self.init_norm = nn.BatchNorm2d(self.hidden_dim, affine=True)
        self.out_conv = nn.Conv2d(hidden_dim,
                                  out_dim,
                                  1)

        self.conv1 = nn.ConvTranspose2d(self.hidden_dim,
                                        self.hidden_dim,
                                        self.fsize, stride=2)
        self.norm1 = nn.BatchNorm2d(self.hidden_dim, affine=True)
        self.conv2 = nn.ConvTranspose2d(self.hidden_dim,
                                        self.hidden_dim,
                                        self.fsize, stride=2)
        self.norm2 = nn.BatchNorm2d(self.hidden_dim, affine=True)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.init_norm(x)
        x = F.relu(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.nl()(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.nl()(x)

        x = self.out_conv(x)
        if self.final_tanh:
            x = torch.tanh(x)
            x = (x + 1) / 2.
        return x

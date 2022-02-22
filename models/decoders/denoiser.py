# Decoder module for producing denoised images
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from early_vision.models.divisive_normalization.divisive_norm import DivNorm
from early_vision.utils.model_utils import crop_tensor

class Denoiser(nn.Module):
    """Decoder with learned upsampling for denoising images."""
    def __init__(self,
                 in_channels,
                 hidden_dim,
                 num_layers,
                 out_dim=3,
                 fsize=3,
                 nl=nn.ReLU,
                 ):
        super(Denoiser, self).__init__()
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.nl = nl
        self.fsize = fsize
        self.conv_layers, self.norm_layers = {}, {}

        self.init_conv = nn.Conv2d(in_channels,
                                   hidden_dim,
                                   1)
        self.init_norm = nn.BatchNorm2d(self.hidden_dim, affine=False)
        self.out_conv = nn.Conv2d(hidden_dim,
                                  out_dim,
                                  1)

        for ii in range(self.num_layers):
            conv = nn.ConvTranspose2d(self.hidden_dim,
                                      self.hidden_dim,
                                      self.fsize, stride=2)
            self.conv_layers[ii] = conv
        
        for ii in range(self.num_layers):
            norm = nn.BatchNorm2d(self.hidden_dim, affine=False)
            self.norm_layers[ii] = norm
      
    def forward(self, x):
        x = self.init_conv(x)
        x = self.init_norm(x)
        x = F.relu(x)

        for ii in range(self.num_layers):
            x = self.conv_layers[ii](x)
            x = self.norm_layers[ii](x)
            x = self.nl()(x)

        x = self.out_conv(x)
        x = nn.Tanh()(x)
        return x
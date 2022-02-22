"""Encoder decoder model for denoising images."""
import torch.nn as nn

from early_vision.models.divisive_normalization.divisive_norm import DivNorm
from early_vision.models.divisive_normalization.divisive_norm_exc_inh import DivNormExcInh
from early_vision.models.decoders.upconv import Upconv


class Denoiser(nn.Module):
    """Decoder with learned upsampling for denoising images."""

    def __init__(self, in_channels,
                 # Encoder params
                 encoder_class,
                 l_sz, l_theta, l_sfs,
                 l_phase, divnorm_fsize,
                 # Decoder params
                 num_layers, out_dim,
                 fsize, nl, final_tanh):
        super(Denoiser, self).__init__()
        self.encoder = encoder_class(in_channels,
                               l_sz, l_theta,
                               l_sfs, l_phase,
                               divnorm_fsize)
        self.encoder_dim = len(l_sz) * len(l_theta) * \
            len(l_sfs) * len(l_phase)
        self.decoder = Upconv(self.encoder_dim,
                              self.encoder_dim,
                              num_layers,
                              out_dim, fsize,
                              nl, final_tanh)

    def forward(self, x):
        """Forward pass for building encoder and decoder"""
        x = self.encoder(x)
        x = x['out']
        x = self.decoder(x)
        return x

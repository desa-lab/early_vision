"""Schwartz and Simoncelli 2001, in pytorch."""
import torch  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import torch.nn.functional as F  # pylint: disable=import-error
from early_vision.models.divisive_normalization.gabor_filter_bank import GaborFilterBank


def nonnegative_weights_init(m):
    """Non-negative initialization of weights."""
    if isinstance(m, nn.Conv2d):
        m.weight.data.uniform_(0, 1)
        m.weight.data.clamp_(0)
        if m.bias is not None:
            raise ValueError("Convolution should not contain bias")
    else:
        m.data.zero_()


class DivNormExcInh(nn.Module):
    """
    Implements Schwartz and Simoncelli 2001 style divisive normalization.
    params:
      input_dim: Number of channels in input
      hidden_dim: Number of hidden channels
      kernel_size: Size of kernel in convolutions
    Example:
      x = torch.zeros(1, 1, 100, 100)
      net = SS_2001(1, 16, 15)
      out = net(x)
    """

    def __init__(self,
                 in_channels,
                 l_filter_size,
                 l_theta, l_sfs,
                 l_phase, divnorm_fsize=1,
                 ):
        super(DivNorm, self).__init__()
        self.gfb = GaborFilterBank(in_channels, l_filter_size,
                                   l_theta, l_sfs, l_phase,
                                   contrast=0.1)
        self.hidden_dim = self.gfb.out_dim
        self.gain_control = nn.Conv2d(
            self.hidden_dim,
            self.hidden_dim,
            divnorm_fsize,
            bias=False)
        self.sigma = nn.Parameter(torch.ones([1, self.hidden_dim, 1, 1]))
        self.gamma = nn.Parameter(torch.ones(self.hidden_dim))
        nonnegative_weights_init(self.gain_control)

    def forward(self, x):
        """
        params:
          x: Input grayscale image tensor
        Returns:
          output: Output post divisive normalization
        """
        # Gabor filter bank
        simple_cells = self.gfb(x)
        # Divisive normalization, Schwartz and Simoncelli 2001
        norm = self.gain_control(simple_cells**2) + self.sigma**2
        x = self.gamma * (simple_cells ** 2) / norm
        return x, norm

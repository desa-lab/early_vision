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


class DivNorm(nn.Module):
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
                 l_phase,
                 divnorm_fsize=5,
                 stride=4,
                 padding_mode='reflect',
                 groups=1,
                 device='cuda',
                 ):
        super(DivNorm, self).__init__()
        self.gfb = GaborFilterBank(in_channels, l_filter_size,
                                   l_theta, l_sfs, l_phase, stride=stride,
                                   padding_mode=padding_mode,
                                   contrast=1.).to(device)
        self.hidden_dim = self.gfb.out_dim
        self.div = nn.Conv2d(
            self.hidden_dim,
            self.hidden_dim,
            divnorm_fsize,
            padding=(divnorm_fsize - 1) // 2,
            bias=False,
            padding_mode=padding_mode,
            groups=groups,
        )
        self.sigma = nn.Parameter(torch.ones([1, self.hidden_dim, 1, 1]))
        nonnegative_weights_init(self.div)


    def forward(self, x, residual=False, square_act=True):
        """
        params:
          x: Input grayscale image tensor
        Returns:
          output: Output post divisive normalization
        """
        identity = x
        simple_cells = self.gfb(x)

        if square_act:
            simple_cells = simple_cells ** 2
            norm = self.div(simple_cells) + self.sigma ** 2 + 1e-5
            simple_cells = simple_cells / norm
        else:
            norm = 1 + F.relu(self.div(simple_cells))
            #norm = F.relu(self.div(simple_cells)) + self.sigma ** 2 + 1e-5
            simple_cells = simple_cells / norm
        output = simple_cells
        
        output = self.output_bn(output)
        if residual:
            output += identity
        output = self.output_relu(output)
        output = {'out': output}
        return output
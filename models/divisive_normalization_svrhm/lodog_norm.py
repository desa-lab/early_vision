"""Schwartz and Simoncelli 2001, in pytorch."""
import torch  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
from early_vision.utils.model_utils import generate_gabor_filter_weights


def nonnegative_weights_init(m):
    """Non-negative initialization of weights."""
    if isinstance(m, nn.Conv2d):
        m.weight.data.uniform_(0, 1)
        m.weight.data.clamp_(0)
        if m.bias is not None:
            raise ValueError("Convolution should not contain bias")
    else:
        m.data.zero_()


class LODOG(nn.Module):
    """Implements linear filtering using a Gabor Filter Bank."""

    def __init__(self,
                 in_channels,
                 l_filter_size,
                 l_theta,
                 l_sfs,
                 l_phase,
                 contrast=1.
                 ):
        super(LODOG, self).__init__()
        self.l_filter_size = [int(i) for i in l_filter_size]
        self.l_theta = l_theta
        self.l_sfs = l_sfs
        self.l_phase = l_phase
        self.contrast = contrast
        self.gabor_convs = []
        self.out_dim = len(l_filter_size) * len(l_theta) * \
            len(l_sfs) * len(l_phase)

        for _, sz in enumerate(self.l_filter_size):
            filter_weights = generate_gabor_filter_weights(sz, self.l_theta,
                                                           self.l_sfs,
                                                           self.l_phase,
                                                           contrast=self.contrast)
            curr_conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=filter_weights.shape[0],
                                  kernel_size=sz, stride=1,
                                  padding=(sz - 1) // 2,
                                  bias=False)
            curr_conv.weight.data = filter_weights
            curr_conv.weight.requires_grad = False
            self.gabor_convs.append(curr_conv)

    def forward(self, x):
        outputs = []
        for conv in self.gabor_convs:
            outputs.append(conv(x))
        outputs = torch.cat(outputs, 1)
        # outputs = F.tanh(outputs)
        return outputs

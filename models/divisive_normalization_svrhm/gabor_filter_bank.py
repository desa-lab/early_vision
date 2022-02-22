"""Implements Gabor filtering."""
import torch
import torch.nn as nn
from early_vision.utils.model_utils import generate_gabor_filter_weights


class GaborFilterBank(nn.Module):
    """Implements linear filtering using a Gabor Filter Bank."""

    def __init__(self,
                 in_channels,
                 l_filter_size,
                 l_theta,
                 l_sfs,
                 l_phase,
                 padding_mode='zeros',
                 contrast=1.,
                 stride=1,
                 device='cuda',
                 ):
        super(GaborFilterBank, self).__init__()
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
            print(filter_weights.shape)
            curr_conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=filter_weights.shape[0],
                                  kernel_size=sz, stride=stride,
                                  padding=(sz - 1) // 2,
                                  padding_mode=padding_mode,
                                  bias=False).to(device)
            with torch.no_grad():
                curr_conv.weight.copy_(
                    torch.from_numpy(filter_weights).float())
                curr_conv.weight.requires_grad = False
            self.gabor_convs.append(curr_conv)

    def forward(self, x):
        outputs = []
        for conv in self.gabor_convs:
            outputs.append(conv(x))
        outputs = torch.cat(outputs, 1)
        return outputs

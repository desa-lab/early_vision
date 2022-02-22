'''VGG11/13/16/19 in Pytorch.'''
import numpy as np
import torch  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import torchvision.models as models  # pylint: disable=import-error


class VGG(nn.Module):
    """VGG16 deep network."""

    def __init__(self, output="conv1"):
        super(VGG, self).__init__()
        self.rgb_mean = np.array((0.485, 0.456, 0.406))
        self.rgb_std = np.array((0.229, 0.224, 0.225))
        # Convert to n, c, h, w
        self.rgb_mean = self.rgb_mean.reshape((1, 3, 1, 1))
        self.rgb_mean = torch.Tensor(self.rgb_mean).float().cuda()
        self.rgb_std = self.rgb_std.reshape((1, 3, 1, 1))
        self.rgb_std = torch.Tensor(self.rgb_std).float().cuda()
        self.model = models.vgg16(pretrained=True).cuda()
        self.output = output
        self.convs = {i: self.extract_layer(self.model, "vgg16", i)
                      for i in range(1, 6)}

    def forward(self, inputs):
        """Forward pass for VGG."""
        x = inputs
        output_idx = int(self.output.split("conv")[-1])
        for idx in range(1, output_idx+1):
            x = self.convs[idx](x)
        return x

    def extract_layer(self, model, backbone_mode, ind):
        if backbone_mode == 'vgg16':
            index_dict = {
                1: (0, 4),
                2: (4, 9),
                3: (9, 16),
                4: (16, 23),
                5: (23, 30)}
        elif backbone_mode == 'vgg16_bn':
            index_dict = {
                1: (0, 6),
                2: (6, 13),
                3: (13, 23),
                4: (23, 33),
                5: (33, 43)}

        start, end = index_dict[ind]
        modified_model = nn.Sequential(*list(
            model.features.children()
        )[start:end])
        return modified_model

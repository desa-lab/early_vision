"""VGG16-HED early visual processing implementation."""
import numpy as np
import torch  # pylint: disable=import-error
import torch.nn as nn # pylint: disable=import-error
import torchvision.transforms as transforms  # pylint: disable=import-error
import torchvision.models as models  # pylint: disable=import-error


class VGG_HED(nn.Module):
  def __init__(self, config):
    super(VGG_HED, self).__init__()
    self.model_name = config.model_name
    self.num_classes = config.num_classes
    self.rgb_mean = np.array((0.485, 0.456, 0.406)) * 255.
    self.rgb_std = np.array((0.229, 0.224, 0.225)) * 255.
    # Convert to n, c, h, w
    self.rgb_mean = self.rgb_mean.reshape((1, 3, 1, 1))
    self.rgb_mean = torch.Tensor(self.rgb_mean).float().cuda()
    self.rgb_std = self.rgb_std.reshape((1, 3, 1, 1))
    self.rgb_std = torch.Tensor(self.rgb_std).float().cuda()
    if self.model_name.startswith("vgg16_bn"):
      model = models.vgg16_bn(pretrained=True).cuda()
    elif self.model_name.startswith("vgg16"):
      model = models.vgg16(pretrained=True).cuda()
    # Pad input before VGG
    self.first_padding = nn.ZeroPad2d(35)

    self.conv_1 = self.extract_layer(model, 
                                     self.model_name, 
                                     1)
"""Common model utilities."""
import glob
import math

import numpy as np
import torch  # pylint: disable=import-error
from PIL import Image


def load_imgs_fromdir(pattern="data/*tiff", size=512):
    """Load standard compression test images."""
    images = glob.glob(pattern)
    np_imgs = [np.array(Image.open(img).convert("L").resize((size, size)))
               for img in images]
    np_imgs = np.array(np_imgs)
    return np_imgs


def crop_tensor(net, out_h, out_w):
    """Crop net to input height and width."""
    _, _, in_h, in_w = net.shape
    assert in_h >= out_h and in_w >= out_w
    x_offset = (in_w - out_w) // 2
    y_offset = (in_h - out_h) // 2
    if x_offset or y_offset:
        cropped_net = net[:, :, y_offset:y_offset +
                          out_h, x_offset:x_offset+out_w]
    return cropped_net


def get_upsampling_weight(in_channels=1, out_channels=1, kernel_size=4):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
        (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels,
                       kernel_size, kernel_size),
                      dtype=np.float32)
    weight[range(in_channels), range(out_channels), :, :] = filt
    weight = torch.from_numpy(weight).float()
    weight = weight.cuda()
    return weight


def tile_tensor(net, timesteps):
    """Tile tensor timesteps times for temporally varying input."""
    n, c, h, w = net.shape
    net_tiled = net.repeat(timesteps, 1, 1, 1, 1).view(timesteps, n, c, h, w)
    net_tiled = torch.transpose(net_tiled, 1, 0)
    return net_tiled


def deprecated_genFilterBank(nTheta=8, kernel_size=15, phase=("on", "off")):
    """Generates a bank of gabor filters."""
    def norm(x):
        """Normalize input to [-1, 1]."""
        x = (x - x.min())/(x.max() - x.min())
        return 2*x-1

    def genGabor(sz, omega, theta, func=np.cos, K=np.pi):
        """Generate a single gabor filter."""
        radius = (int(sz[0]/2.0), int(sz[1]/2.0))
        [x, y] = np.meshgrid(range(-radius[0], radius[0]+1),
                             range(-radius[1], radius[1]+1))

        x1 = x * np.cos(theta) + y * np.sin(theta)
        y1 = -x * np.sin(theta) + y * np.cos(theta)

        gauss = omega**2 / (4*np.pi * K**2) * \
            np.exp(- omega**2 / (8*K**2) * (4 * x1**2 + y1**2))
        sinusoid = func(omega * x1) * np.exp(K**2 / 2)
        gabor = gauss * sinusoid
        return gabor

    theta = np.arange(0, np.pi, np.pi/nTheta)  # range of theta
    omega = np.arange(1., 1.01, 0.1)  # range of omega
    params = [(t, o) for o in omega for t in theta]
    sinFilterBank = []
    cosFilterBank = []
    gaborParams = []

    for (t, o) in params:
        gaborParam = {'omega': o, 'theta': t, 'sz': (kernel_size, kernel_size)}
        cosGabor = norm(genGabor(func=np.cos, **gaborParam))
        if "on" in phase:
            cosFilterBank.append(cosGabor)
        if "off" in phase:
            cosFilterBank.append(-cosGabor)
    cosFilterBank = np.array(cosFilterBank)
    cosFilterBank = np.expand_dims(cosFilterBank, axis=1)
    return cosFilterBank


def genGabor(sz, theta, gamma, sigma, sf, phi=0, contrast=2):
    """Generate gabor filter based on argument parameters."""
    location = (sz[0] // 2, sz[1] // 2)
    [x, y] = np.meshgrid(np.arange(sz[0])-location[0],
                         np.arange(sz[1])-location[1])

    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    envelope = .5 * contrast * \
        np.exp(-(x_theta**2 + (y_theta * gamma)**2)/(2 * sigma**2))
    gabor = envelope * np.cos(2 * math.pi * x_theta * sf + phi)
    return gabor


def generate_gabor_filter_weights(sz, l_theta, l_sfs,
                                  l_phase, gamma=1,
                                  contrast=1, return_dict=False):
    """Generate a bank of gabor filter weights.
    Args:
      sz: (filter height, filter width), +-2 SD of gaussian envelope
      l_theta: List of gabor orientations
      l_sfs: List of spatial frequencies, cycles per SD of envelope
      l_phase: List of gabor phase
    Returns:
      gabor filter weights with parameters sz X l_theta X l_sfs X l_phase
    """
    gabor_bank = []
    theta2filter = {}
    for theta in l_theta:
        curr_filters = []
        for sf in l_sfs:
            for phase in l_phase:
                g = genGabor(sz=(sz, sz), theta=theta,
                             gamma=gamma, sigma=sz/4,
                             sf=sf/sz, phi=phase,
                             contrast=contrast)
                gabor_bank.append(g)
                curr_filters.append(g)
        theta2filter[theta] = torch.from_numpy(
            np.array(curr_filters, dtype=np.float32))
    theta2filter = {t: torch.unsqueeze(g_b, 1)
                    for t, g_b in theta2filter.items()}
    gabor_bank = np.array(gabor_bank, dtype=np.float32)
    gabor_bank = np.expand_dims(gabor_bank, 1)
    if return_dict:
        return gabor_bank, theta2filter
    return gabor_bank


def spatial_softmax(x):
    """Compute spatial softmax on x (softmax over height and width)."""
    x_min = x.min(2, keepdim=True)[0].min(3, keepdim=True)[0]
    x_max = x.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
    x = (x - x_min)/(x_max - x_min)
    exp_x = torch.exp(x)
    denom = torch.sum(exp_x, dim=(2, 3), keepdim=True)
    return exp_x/(denom + 1e-8)


def get_objective_fn(objective):
    """Get reference to objective function."""
    def ss2001(x):
        norm, out = x['norm'], x['out']
        objective = torch.log(norm) + out
        return objective

    def wainwright(x):
        out = x['out']
        loss = torch.log(out)**2
        loss[out == 0.] = 0.
        return loss

    objectives = {'ss_2001': ss2001,
                  'wainwright': wainwright}
    return objectives[objective]

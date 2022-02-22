"""Train normalization models to maximize independence."""
import random
import string

import numpy as np
import torch
import torch.backends.cudnn as cudnn  # pylint: disable=import-error
import torch.optim as optim
from absl import app, flags  # pylint: disable=import-error
from early_vision.utils.model_utils import load_imgs_fromdir, get_objective_fn
from early_vision.models.divisive_normalization.divisive_norm import DivNorm

FLAGS = flags.FLAGS

flags.DEFINE_integer("image_size", 256, "Spatial resolution of input images")
flags.DEFINE_integer("num_steps", 15, "Number of training steps")
flags.DEFINE_integer("groups", 1, "Number of convolution groups")
flags.DEFINE_float("learning_rate", 5e-4, "Optimizer learning rate")
flags.DEFINE_list("l_theta", [0, 45, 90, 135], "Number of orientations")
flags.DEFINE_list("l_sfs", [2, 3], "List of spatial frequencies")
flags.DEFINE_list("l_phase", [0, 90, 180], "List of phase")
flags.DEFINE_list("l_sz", [9, 15, 21], "List of kernel sizes")
flags.DEFINE_string("model_name", "ss_2001", "Name of model to build")
flags.DEFINE_string("padding_mode", "zeros", "Kind of padding for convolutions")
flags.DEFINE_string("objective", "wainwright", "Objective function to use")

def get_random_string(n):
    """Generate random string of n characters."""
    # printing lowercase
    letters = string.ascii_lowercase
    rand_str = ''.join(random.choice(letters) for i in range(n))
    return rand_str


def get_model_by_name():
    """Get instance of model class by model name argument."""
    if FLAGS.model_name.startswith("ss_2001"):
        model_provider = DivNorm
    elif FLAGS.model_name.startswith("v1net"):
        model_provider = None  # Not implemented
    if model_provider is None:
        raise ValueError("Model not yet implemented")
    return model_provider


def main(argv):
    del argv  # unused here
    FLAGS.l_theta = [np.pi * t/180 for t in FLAGS.l_theta]
    FLAGS.l_phase = [np.pi * p/180 for p in FLAGS.l_phase]

    model_provider = get_model_by_name()
    net = model_provider(in_channels=1,
                         l_theta=FLAGS.l_theta,
                         l_sfs=FLAGS.l_sfs,
                         l_phase=FLAGS.l_phase,
                         l_filter_size=FLAGS.l_sz,
                         padding_mode=FLAGS.padding_mode,
                         groups=FLAGS.groups,
                         )
    optim_params = [
        p for n, p in net.named_parameters()]
    optimizer = optim.SGD(optim_params, lr=FLAGS.learning_rate, momentum=0.9)
    net, l_loss, best_params = train(net, optimizer)
    state_dict = {"params": best_params, "l_loss": l_loss}
    rand_str = get_random_string(6)
    torch.save(state_dict, open("runs/best_params_%s.pth" % rand_str, "wb"))
    return


def train(net, optimizer):
    """Single epoch of training normalization parameters."""
    imgs = load_imgs_fromdir(size=FLAGS.image_size)
    gain_control_params = []
    for n, p in net.named_parameters():
        print(n, p.shape)
        gain_control_params.append(p)
    l_loss = []
    min_loss = np.inf
    best_params = None
    criterion = get_objective_fn(FLAGS.objective)
    for step in range(FLAGS.num_steps):
        imgs = np.random.permutation(imgs)
        for img_ii, img in enumerate(imgs):
            img = torch.Tensor(img/255.)
            img = img.unsqueeze_(0).unsqueeze_(1)

            optimizer.zero_grad()
            output = net(img)
            objective = criterion(output)
            loss_ind = torch.mean(objective)/2
            loss_l2 = sum([torch.norm(p, p=2)
                            for n, p in net.named_parameters()])
            loss = loss_ind + 1e-4*loss_l2
            import ipdb; ipdb.set_trace()
            loss.backward()
            
            print(output['out'].max(), output['norm'].max(), net.div.weight.max())
            optimizer.step()
            for param in gain_control_params:
                param.data.clamp_(0.)
            if loss < min_loss:
                min_loss = loss
                best_params = net.div.weight.detach().cpu().numpy()
            print("Iter-%s, Image-%s, Loss: %s, Min loss: %s" %
                  (step, img_ii, loss, min_loss))
            l_loss.append(loss)
            if min_loss not in l_loss[-25:]:
                print("Learning saturated, decaying lr")
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 2.
    return net, l_loss, best_params


if __name__ == "__main__":
    app.run(main)

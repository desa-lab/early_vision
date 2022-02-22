"""Train normalization models to do image denoising."""
import argparse
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn  # pylint: disable=import-error
import torch.nn as nn
from torch.utils.tensorboard import \
    SummaryWriter  # pylint: disable=import-error
import torchvision  # pylint: disable=import-error

from early_vision.data_provider import BSDSDataProvider
from early_vision.models.denoiser import Denoiser
from early_vision.models.divisive_normalization.divisive_norm import DivNorm
from early_vision.models.divisive_normalization.divisive_norm_exc_inh import \
    DivNormExcInh
from early_vision.models.recurrence.dale_rnn import DaleRNN
from early_vision.utils.model_utils import crop_tensor
from pytorch_msssim import ssim  # pylint: disable=import-error
parser = argparse.ArgumentParser(description='Image denoising')

parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training (default: 16)')
parser.add_argument('--crop_size', type=int, default=48,
                    help='Image crop size for training (default: 48)')
parser.add_argument('--downsample', type=int, default=4,
                    help='Image downsample factor (default: 4)')
parser.add_argument('--noise_sigma', type=int, default=8,
                    help='Standard deviation of additive noise (default: 8)')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--eval_every', type=int, default=10,
                    help='number of epochs per evaluation (default: 1)')
parser.add_argument('--update_iters', type=int, default=10,
                    help='number of steps to update params (default: 10)')
parser.add_argument('--decay_steps', type=int, default=10,
                    help='number of steps to decay params (default: 1000)')
parser.add_argument('--save_epoch', type=int, default=1,
                    help='number of epochs per checkpoint')
parser.add_argument('--divnorm_fsize', type=int, default=7,
                    help='Divisive normalization filter size (default: 7)')
parser.add_argument('--num_layers', type=int, default=2,
                    help='Number of layers in decoder (default: 2)')
parser.add_argument('--l_sz', nargs='+', default=[15, 21],
                    help='Initial gabor filter size list (default: [15, 21])')
parser.add_argument('--l_theta', nargs='+', default=[0, 45, 90, 135],
                    help='Initial gabor filter orientations list')
parser.add_argument('--l_phase', nargs='+', default=[0, 90, 180, 270],
                    help='Initial gabor filter phase list')
parser.add_argument('--l_sfs', nargs='+', default=[2, 3],
                    help='Initial gabor filter spatial frequency list')
parser.add_argument('--model_name', type=str, default="",
                    help='Model name for encoder class')
parser.add_argument('--base_dir', type=str, default="",
                    help='Base directory for experiments')
parser.add_argument('--data_dir', type=str, default="",
                    help='Base directory with dataset')
parser.add_argument('--expt_name', type=str, default="",
                    help='Name identifier for experiment')
parser.add_argument('--optimizer', type=str, default="adam",
                    help='Optimizer for training')
parser.add_argument('--transform', type=str, default="gaussian",
                    help='Input image transform to use for training')
parser.add_argument('--checkpoint', type=str, default="",
                    help='Filename for checkpoint restore')
parser.add_argument('--learning_rate', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=2e-4,
                    help='Weight decay multiplier')
parser.add_argument('--clip', type=float, default=5.,
                    help='Gradient clipping norm')
parser.add_argument('--final_tanh', type=bool, default=True,
                    help='whether to add tanh to final output of decoder')

args = parser.parse_args()
writer = SummaryWriter("runs/%s" % args.expt_name)


def train_epoch(model, train_dataloader, optimizer, global_step):
    """Train one epoch."""
    model.train()
    start_time = time.time()
    iter_loss = 0
    optimizer.zero_grad()
    criterion = nn.MSELoss(reduction='sum')
    print("Dataloader length: %s" % (len(train_dataloader)))

    for idx, data in enumerate(train_dataloader):
        # Load images and labels
        imgs, lbls = data
        imgs, lbls = imgs.float().cuda(), lbls.float().cuda()
        out = model(imgs)
        out = crop_tensor(out, lbls.shape[2], lbls.shape[3])
        loss = criterion(out, lbls) / args.update_iters
        iter_loss += loss
        loss.backward()

        if idx > 0 and (idx % args.update_iters == 0):
            # Update parameters
            p_epoch_idx = (
                global_step * args.update_iters) // len(train_dataloader)
            l2_norm = torch.sqrt(torch.mean(torch.square(imgs-lbls)))*255.
            print("Epoch(%s) - Iter (%s) - Loss: %.4f - SSIM: %.4f" % (
                p_epoch_idx,
                idx/args.update_iters,
                iter_loss.item(),
                ssim(imgs, out,
                     data_range=1.)))
            global_step = global_step + 1
            # nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            for param in model.encoder.parameters():
                param.data.clamp_(0.)
            curr_lr = optimizer.param_groups[0]['lr']
            optimizer.zero_grad()
            add_summary(global_step,
                        iter_loss, l2_norm, curr_lr,
                        ssim(imgs, out, data_range=1.),
                        imgs, lbls, out)
            iter_loss = 0
    print("Finished training epoch in %s" % int(time.time() - start_time))
    return global_step


def evaluate(model, val_dataloader, global_step):
    """Perform evaluation on the validation set."""
    mse = nn.MSELoss(reduction='sum')
    print("Starting evaluation..")
    l_mse, l_ssim = [], []
    model.eval()
    with torch.no_grad():
        for _, data in enumerate(val_dataloader):
            imgs, lbls = data
            imgs, lbls = imgs.float().cuda(), lbls.float().cuda()
            out = model(imgs)
            out = crop_tensor(out, lbls.shape[2], lbls.shape[3])
            loss = mse(out, lbls)
            l_mse.append(loss)
            d_ssim = ssim(imgs, out,
                          data_range=1.)
            l_ssim.append(d_ssim)
        print("Completed evaluation..")
        l_mse, l_ssim = torch.Tensor(l_mse), torch.Tensor(l_ssim)
        d_mse, d_ssim = torch.mean(l_mse), torch.mean(l_ssim)
        writer.add_scalar("val/mse", d_mse, global_step)
        writer.add_scalar("val/ssim", d_ssim, global_step)
    return d_mse, d_ssim


def add_summary(idx, loss, l2_norm, lr, ssim_tb,
                images, labels, outputs):
    """Write tensorboard summaries."""
    images = torch.clip(images, 0., 1.)
    labels = torch.clip(labels, 0., 1.)
    outputs = torch.clip(outputs, 0., 1.)
    img_grid = torchvision.utils.make_grid(images)
    labels_grid = torchvision.utils.make_grid(labels)
    outputs_grid = torchvision.utils.make_grid(outputs)
    writer.add_scalar("train/loss", loss, idx)
    writer.add_scalar("train/ssim", ssim_tb, idx)
    writer.add_scalar("train/norm", l2_norm, idx)
    writer.add_scalar("train/lr", lr, idx)
    writer.add_image("Images/noisy_in", img_grid, idx)
    writer.add_image("Images/clean_gt", labels_grid, idx)
    writer.add_image("Images/denoise_pred", outputs_grid, idx)
    return


def get_encoder_class(model_name):
    """Get reference to Encoder class."""
    if model_name.startswith("div_norm_exc_inh"):
        encoder_class = DivNormExcInh
    elif model_name.startswith("div_norm"):
        encoder_class = DivNorm
    elif model_name.startswith("dalernn"):
        encoder_class = DaleRNN
    else:
        raise ValueError("Not implemented model type")
    return encoder_class


def main():  # pylint: disable = too-many-locals
    """Train models on image denoising.
    Gaussian noise added to clean BSDS images"""
    cudnn.benchmark = True
    expt_dir = os.path.join(args.base_dir,
                            args.expt_name)
    if not os.path.exists(expt_dir):
        os.mkdir(expt_dir)
    with open(os.path.join(expt_dir, "args.txt"), "w") as f:
        f.write(str(args))

    args.l_theta = [t*np.pi/180 for t in args.l_theta]
    args.l_phase = [p*np.pi/180 for p in args.l_phase]
    full_start = time.time()
    train_data = BSDSDataProvider(crop_size=args.crop_size,
                                  is_training=True,
                                  data_dir=args.data_dir,
                                  noise_sigma=args.noise_sigma,
                                  transform=args.transform,
                                  downsample=args.downsample,
                                  )
    val_data = BSDSDataProvider(crop_size=args.crop_size,
                                is_training=False,
                                data_dir=args.data_dir,
                                noise_sigma=args.noise_sigma,
                                transform=args.transform,
                                downsample=args.downsample,
                                )
    import ipdb; ipdb.set_trace()
    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=min(
                                                       16, args.batch_size),
                                                   )
    val_dataloader = torch.utils.data.DataLoader(val_data,
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 num_workers=min(
                                                     16, args.batch_size),
                                                 )
    encoder_class = get_encoder_class(args.model_name)
    model = Denoiser(1, encoder_class, args.l_sz, args.l_theta,
                     args.l_sfs, args.l_phase,
                     args.divnorm_fsize, args.num_layers,
                     out_dim=1, fsize=3, nl=nn.ReLU,
                     final_tanh=args.final_tanh)
    model = model.to('cuda')
    print("Model created..")

    if args.checkpoint:
        print("Restoring from %s" % args.checkpoint)
        state_dict = torch.load(args.checkpoint)
        model.load_state_dict(state_dict)

    base_lr = args.learning_rate
    weight_decay = args.weight_decay
    if args.optimizer.startswith("adam"):
        if weight_decay:
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=base_lr,
                                         weight_decay=weight_decay)
        else:
            print("Weight decay set to", weight_decay)
            optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9,
                                    nesterov=True, weight_decay=weight_decay,
                                    lr=base_lr)
    optimizer.zero_grad()
    global_step = 0
    args.epochs = int(args.epochs)
    best_ssim = -10
    best_epoch = 0
    for epoch_idx in range(args.epochs):
        global_step = train_epoch(
            model, train_dataloader, optimizer, global_step)
        if not epoch_idx % args.save_epoch:
            ckpt_dir = os.path.join(args.base_dir,
                                    args.expt_name)
            if not os.path.exists(ckpt_dir):
                os.mkdir(ckpt_dir)
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir,
                                    "saved-model-epoch-%s.pth" % (epoch_idx)))
        if epoch_idx % args.eval_every == 0:
            _, d_ssim = evaluate(model, val_dataloader, global_step)
            d_ssim = d_ssim.detach().cpu().numpy()
            if np.isnan(d_ssim):
                print("SSIM is nan, debugging needed.")
                return
            if best_ssim < d_ssim:
                torch.save(model.state_dict(),
                           os.path.join(ckpt_dir,
                                        "best-model.pth"))
                best_ssim = d_ssim
                best_epoch = epoch_idx
                print("Best validation SSIM:%.3f, best epoch:%s" % (
                    best_ssim, best_epoch))
            elif (epoch_idx - best_epoch) > 25:
                print("Early stopping..")
                return
        if (epoch_idx) and (epoch_idx % args.decay_steps) == 0:
            for param_group in optimizer.param_groups:
                    param_group['lr'] /= 10

    full_duration = time.time() - full_start
    print("Training finished until"
          "%s epochs in %s" % (args.epochs,
                               full_duration))


if __name__ == "__main__":
    main()

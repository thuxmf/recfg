import os

import pickle
import tqdm
import click
import torch
import numpy as np
import torch.distributed

import dnnlib
from training.dataset import PartImageFolderDataset
from torch_utils import misc
from torch_utils import distributed as dist


def all_reduce(x):
    x = x.clone()
    torch.distributed.all_reduce(x)
    return x


@torch.no_grad()
def edm2_sampler(net, gnet, latents, labels, generator, num_steps, device=torch.device('cuda'), dtype=torch.float32, rho=7, sigma_min=0.002, sigma_max=80):
    step_indices = torch.arange(num_steps, dtype=dtype, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    latents = latents.to(dtype)
    input_labels = torch.cat([labels] * 2).to(device)
    noises = torch.randn(latents.shape, generator=generator, device=device, dtype=dtype)

    noises_pred_cond, noises_pred_uncond = [], []
    for t in t_steps:
        noisy_latents = latents + noises * t
        input_noisy_latents = torch.cat([noisy_latents] * 2)
        noise_pred_cond = (input_noisy_latents - net(input_noisy_latents, t, input_labels)) / t
        noise_pred_uncond = (input_noisy_latents - gnet(input_noisy_latents, t, input_labels)) / t
        noises_pred_cond.append(noise_pred_cond)
        noises_pred_uncond.append(noise_pred_uncond)
    return torch.stack(noises_pred_cond), torch.stack(noises_pred_uncond)


@click.group()
def cmdline():
    """"""


@cmdline.command()
@click.option('--net',                           help='Network pickle filename', metavar='PATH|URL',             type=str, required=True)
@click.option('--gnet',                          help='Network pickle filename', metavar='PATH|URL',             type=str)
@click.option('--data',                          help='Path to the dataset', metavar='ZIP|DIR',                  type=str, required=True)
@click.option('--nfe', 'num_steps',              help='Dataset reference statistics ', metavar='PKL|NPZ|URL',    type=int, default=32)
@click.option('--images', 'image_path',          help='Path to the images', metavar='PATH',                      type=str)
@click.option('--num', 'num_images_per_label',   help='Number of images to traverse per label', metavar='INT',   type=click.IntRange(min=2), default=500, show_default=True)
@click.option('--label', 'num_labels',           help='Number of labels to traverse', metavar='INT',             type=click.IntRange(min=1), default=100, show_default=True)
@click.option('--seed',                          help='Random seed for the first image', metavar='INT',          type=int, default=0, show_default=True)
@click.option('--batch', 'max_batch_size',       help='Maximum batch size', metavar='INT',                       type=click.IntRange(min=1), default=32, show_default=True)

def edm2(net, gnet, data, num_steps, image_path, num_images_per_label, num_labels, seed, max_batch_size):
    # Rank 0 goes first.
    dist.init()
    os.makedirs(image_path, exist_ok=True)
    device = torch.device('cuda')
    dtype = torch.float32
    generator = torch.Generator(device)
    generator.manual_seed(seed + dist.get_rank())
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load main network.
    if isinstance(net, str):
        dist.print0(f'Loading network from {net} ...')
        with dnnlib.util.open_url(net, verbose=(dist.get_rank() == 0)) as f:
            pkl = pickle.load(f)
        net = pkl['ema'].to(device)
        encoder = pkl.get('encoder', None)
        if encoder is None:
            encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardRGBEncoder')
    assert net is not None

    # Load guidance network.
    if isinstance(gnet, str):
        dist.print0(f'Loading guidance network from {gnet} ...')
        with dnnlib.util.open_url(gnet, verbose=(dist.get_rank() == 0)) as f:
            gnet = pickle.load(f)['ema'].to(device)
    if gnet is None:
        gnet = net

    # Initialize encoder.
    assert encoder is not None
    dist.print0(f'Setting up {type(encoder).__name__}...')
    encoder.init(device)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide seeds into batches.
    num_images = num_images_per_label * num_labels
    num_batches = max((num_images - 1) // (max_batch_size * dist.get_world_size()) + 1, 1)
    dist.print0(f'Traversing {num_images} images...')

    # Dataset.
    dataset = PartImageFolderDataset(path=data, max_num_labels=num_labels, use_labels=True)
    dataset_sampler = misc.InfiniteSampler(dataset=dataset, rank=dist.get_rank(), shuffle=False, num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(dnnlib.util.construct_class_by_name(dataset=dataset, sampler=dataset_sampler, batch_size=max_batch_size, class_name='torch.utils.data.DataLoader', pin_memory=True))

    torch.distributed.barrier()
    sum1s = torch.zeros(num_labels, num_steps, net.img_channels, net.img_resolution, net.img_resolution, device=device)
    sum2s = torch.zeros(num_labels, num_steps, net.img_channels, net.img_resolution, net.img_resolution, device=device)
    progress_bar = tqdm.tqdm(range(num_batches), total=num_batches) if dist.get_rank() == 0 else range(num_batches)
    for _ in progress_bar:
        images, labels = next(dataset_iterator)
        images = encoder.encode_latents(images.to(device))
        cond, uncond = edm2_sampler(net, gnet, images, labels, generator, num_steps, device, dtype)
        label_indices = torch.argmax(labels, dim=1)
        sum1s[label_indices] += (cond * uncond).mean(dim=1)
        sum2s[label_indices] += (uncond * uncond).mean(dim=1)

    sum1s = all_reduce(sum1s) / num_batches
    sum2s = all_reduce(sum2s) / num_batches
    if dist.get_rank() == 0:
        coeffs = -sum1s / torch.where(sum2s == 0., torch.ones_like(sum2s), sum2s)
        coeffs = torch.cat([coeffs, coeffs.mean(dim=0, keepdim=True)], dim=0)
        with open(os.path.join(image_path, f'coeffs_{num_labels}labels_{num_images_per_label}imgs_per_label.pkl'), 'wb') as f:
            pickle.dump(coeffs, f)
    dist.print0('done')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------

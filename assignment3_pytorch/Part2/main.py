import os
import imageio
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from training.networks import Generator
from helper_functions import *
from your_code_here import *

def run_mini_draggan(
    network_pkl: str = './weights/G_cifar10.pkl',
    outdir: str = './outputs',
    seed: int = 2,
    device: str = 'cpu',
    threshold: float = 2,
    feat_layer: int = 3,
    num_convs_opt: int = 3, 
    r1: int = 1,
    r2: int = 2,
    class_label: int = 5,
    max_iter: int = 500,
    lambda_mask: float = 5.,
    lr: float = 2e-3,
    coords: list = None,
    mask_points: list = None,
    save_experiment_data: bool = False
):
    """Simplified reimplementation of DragGAN on small networks.

    # Parameters:
        @network_pkl: (str) Path to the network pickle file.
        @outdir: (int) Where to save the output images.
        @seed: (int) Random seed.
        @device: (str) cpu|cuda
        @threshold: (float) Stopping thresholding between starting and end point.
        @feat_layer: (int) Index of feature blocks used for optimisation.
        @num_convs_opt: (int) Number of convolutional layers used for optimisation, the rest is fixed.
        @r1: (int) Feature point radius (l1)
        @r2: (int) Search radius (l1)
        @class_label: (int) Cifar10 class label, 0-9
        @max_iter: (int) Maximum number of iterations.
        @lambda_mask: (float) Weight of mask loss.
        @lr: (float) Learning rate.
    """
    
    ## Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(outdir, exist_ok=True)
    
    ## Initialize generator and load weights
    device = torch.device(device)
    G = Generator(**torch.load(f"{network_pkl[:-4]}_kwargs.pkl"))
    G.load_state_dict(torch.load(network_pkl), strict=True)
    G.requires_grad_(False).to(device)
    G = G.float()

    ## Define all inputs to the network: noise, class vector, and latent vector
    # Sample max_num_samples random gaussian vectors (z) and class vectors (c) 
    num_samples = 20
    z = torch.from_numpy(np.random.RandomState(seed).randn(num_samples, G.z_dim)).to(device)
    c = torch.nn.functional.one_hot(torch.tensor([class_label]), 10).expand(num_samples, -1).to(device)
    # Lift z into the 'learned' latent space W. 
    # Further, take the mean over all samples to get a single latent vector with a higher likelihood of being in the data manifold.
    w = G.mapping(z, c).mean(dim=0, keepdim=True)

    # Define which weights are optimised and which are fixed
    w_opt = w[:, :num_convs_opt, :].clone()
    w_fixed = w[:, num_convs_opt:, :].clone()
    w_opt.requires_grad = True
    w_fixed.requires_grad = False

    # Init noise but do not optimise it
    noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = False 

    ## Define parameters and optimizer
    params = [w_opt]
    optimizer = torch.optim.Adam(params, betas=(0.9, 0.999), lr=lr)

    ## Get initial image and intermediate feature volumes
    ws = torch.concat([w_opt, w_fixed], dim=1)
    image, features = G.synthesis(ws, noise_mode='const', max_res=G.img_resolution, force_fp32=True)

    ## Define handle and target point based on current image
    # NOTE: This function returns the values in (X, Y) format
    coords, mask_points = select_points(image, coords, mask_points)
    if len(coords) != 2:
        return
    mask_used = len(mask_points) > 2

    p = torch.tensor(coords[0], dtype=torch.float32, device=device)
    t = torch.tensor(coords[1], dtype=torch.float32, device=device)

    identifier = f"{class_label}_{seed}_{p[0].item():2.1f}_{p[1].item():2.2f}_{t[0].item():2.1f}_{t[1].item():2.2f}"

    ## Calculate segmentation mask from points and save
    if mask_used:
        mask = points_to_mask(image[0].detach().cpu().numpy(), np.array(mask_points))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(mask)
        mask_path = f'{outdir}/mask_{identifier}.png'
        fig.savefig(mask_path)
        plt.close()
        mask = torch.tensor(mask, dtype=torch.float32, device=device)

    ## Save experiment setup
    if save_experiment_data:
        data = {
            'seed': seed, 'threshold': threshold, 'feat_layer': feat_layer,
            'num_convs_opt': num_convs_opt, 'r1': r1, 'r2': r2,
            'class_label': class_label, 'max_iter': max_iter, 'lambda_mask': lambda_mask,
            'lr': lr, 'coords': coords, 'mask_points': mask_points,
        }

        save_pkl(f'./data/data_{identifier}.pkl', data)

    ## Sample initial (fixed) feature vector of handle point p
    resize = torchvision.transforms.Resize((G.img_resolution, G.img_resolution), antialias=True)
    F_0 = features[feat_layer]['feat']
    F_0 = resize(F_0)  # [1, 512, G.img_resolution, G.img_resolution]

    # TODO: 1. Sample features from neighbourhood
    # Sample features from feature map F at points q_N
    f_p = sample_p_from_feature_map(p[None], F_0).detach() # [1, C], detach() because this feature point is fixed

    # Initialize variables for loop
    F = F_0.clone()
    F_0 = F_0.detach()
    dist = (p - t).pow(2).sum().add(1e-6).sqrt()

    # Set up video writer
    video_path = f'{outdir}/video_{identifier}.mp4'
    video = imageio.get_writer(video_path, mode='I', fps=60, codec='libx264', bitrate='32M')
    for i in range(max_iter):

        ## Motion Supervision
        d_i = (t - p) / dist

        # TODO: 2. Get Neighbourhood
        # Compute neighbourhood of points around p based on given radius
        q_N = get_neighbourhood(p, r1)
        # TODO: 1. Sample features from neighbourhood
        # Sample features from feature map F at points q_N
        F_q_i = sample_p_from_feature_map(q_N, F)

        # Compute neighbourhood of points around the shifted point and the given radius
        q_N_d_i = get_neighbourhood(p + d_i, r1)
        # Sample features from feature map F around the neighbourhood of the shifted point
        F_q_d_i = sample_p_from_feature_map(q_N_d_i, F)

        loss = (F_q_i.detach() - F_q_d_i).abs().mean()
        # Compute loss and do gradient step
        # TODO: 4. Calculate mask loss
        if mask_used:
            loss = loss + lambda_mask * get_mask_loss(F_0, F, mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        ## Point Tracking
        # Synthesize image from current w_opt and extract wanted feature volume
        ws = torch.concat([w_opt, w_fixed], dim=1)
        synth_image, features = G.synthesis(ws, noise_mode='const', max_res=G.img_resolution, force_fp32=True)
        F = features[feat_layer]['feat']
        F = resize(F)
        
        q_N = get_neighbourhood(p, r2)
        F_q_N = sample_p_from_feature_map(q_N, F)

        # TODO: 3. Neighbourhood search
        # Find nearest neighbour of feature vector of handle point p in new feature volume F
        p = nearest_neighbour_search(f_p, F_q_N, q_N)
        
        # Compute new distance and print progress
        dist = (p - t).pow(2).sum().add(1e-6).sqrt()
        print(f'step {i:>4d}: distance {float(dist.item()):<5.6f}, loss {float(loss):<5.6f}')

        ## Save images
        video_frame = make_video_frame(p, t, synth_image)
        video.append_data(video_frame)

        # Check whether handle point has reached target point
        if dist.mean() <= threshold:
            break
    
    # Generate final image and save video
    ws = torch.concat([w_opt, w_fixed], dim=1)
    video_frame, features = G.synthesis(ws, noise_mode='const', max_res=G.img_resolution, force_fp32=True)
    video_frame = make_video_frame(p, t, video_frame)

    video.append_data(video_frame)
    video.close()

    if mask_used:
        print(f"Mask saved at: {mask_path}")
    print(f"Video saved at: {video_path}")

if __name__ == "__main__":
    experiment = 'car' # 'car', 'horse', 'manual'
    if experiment == 'car':
        data = read_pkl('./data/car_no_mask.pkl')
    elif experiment == 'horse':
        data = read_pkl('./data/horse_mask.pkl')
    else:
        data = {}
        # NOTE: Classes 1, 5, and 7 work best.

    run_mini_draggan(**data, save_experiment_data=False)
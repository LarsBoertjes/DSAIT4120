import torch
import torch.nn.functional as Func
from helper_functions import *
# WARNING: Do not import any other libraries or files

def get_neighbourhood(p, radius):
    """ Returns the neighbourhood grid of points around p.

    NOTE: The neighbourhood should contain (radius * 2 + 1)**2 points, including p.
    For example, for p = [3.5, 3.5] and radius = 1, this function should return the following values:
        (2.5, 2.5) (2.5, 3.5) (2.5, 4.5) 
        (3.5, 2.5) (3.5, 3.5) (3.5, 4.5) 
        (4.5, 2.5) (4.5, 3.5) (4.5, 4.5)
    The order of the points does not matter but has to be consistent inbetween calls.

    # Parameters:
        @p: torch.tensor size [2], the current handle point p
        @radius: int, the radius of the grid neighbourhood to return

    # Returns: torch.tensor size [(radius * 2 + 1)**2, 2], the neighbourhood of points around p, including p
    """
    offsets = torch.arange(-radius, radius + 1, device=p.device)

    x_offsets, y_offsets = torch.meshgrid(offsets, offsets)

    grid_points = torch.stack([x_offsets.flatten(), y_offsets.flatten()], dim=1)

    # Shift all points with p
    neighbourhood = p + grid_points
    #print(neighbourhood)

    return neighbourhood

def sample_p_from_feature_map(q_N, F):
    """ Samples the feature map F at the points q_N.

    NOTE: As the points in q_N are floats, we can not directly sample the points from the feature map via indexing.
    Bilinear interpolation is needed, PyTorch has a function for this: F.grid_sample.
    If you struggle with this function, try to set up a minimal working example, where you manually control q_N and F.

    # Parameters:
        @q_N: torch.tensor size [N, 2], the points to sample from the feature map
        @F: torch.tensor size [1, C, H, W], the feature map of the current image

    # Returns: torch.tensor size [N, C], the sampled features at q_N
    """
    assert F.shape[-1] == F.shape[-2]

    # Get the feature map's dimensions
    _, C, H, W = F.shape

    # Normalize the sampling coordinates for Pytorch's grid_sample
    q_N_normalized = q_N.clone()
    # Normalize x-coordinates relative to width W
    q_N_normalized[:, 0] = 2.0 * q_N[:, 0] / (W - 1) - 1.0
    # Normalize y-coordinates relative to height H
    q_N_normalized[:, 1] = 2.0 * q_N[:, 1] / (H - 1) - 1.0

    # Reshape coordinates for input format grid_sample (add batch dimension and spatial dimension)
    grid = q_N_normalized.unsqueeze(0).unsqueeze(2)


    sampled_features = Func.grid_sample(F, grid, mode='bilinear', align_corners=True)

    sampled_features = sampled_features.squeeze(2).squeeze(0).T

    return sampled_features


def nearest_neighbour_search(f_p, F_q_N, q_N):
    """ Does a nearest neighbourhood search in feature space to find the new handle point position.

    NOTE: The nearest neighbour should be determined by the L1 distance.

    # Parameters:
        @f_p: torch.tensor size [1, C], the feature vector of the handle point p
        @F_q_N: torch.tensor size [N, C], feature vectors at points q_N
        @q_N: torch.tensor size [N, 2], corresponding points to F_q_N in the image space

    # Returns: torch.tensor size [2], the new handle point p
    """
    F_q_N = F_q_N.squeeze(0) # Now F_q_N has shape [25, 512]
    print(f"F_q_N shape: {F_q_N.shape}") # check

    # Calculate L1 distances between f_p and all of F_q_N
    l1_distances = torch.sum(torch.abs(F_q_N - f_p), dim=1)  # Now l1_distances has shape [25]

    print(f"F_q_N shape: {F_q_N.shape}, q_N shape: {q_N.shape}, distances shape: {l1_distances.shape}")

    # Find the index of the minimum L1 distance and set nearest neighbor
    min_index = torch.argmin(l1_distances)  # min_index should be in range [0, 24]

    print(f"min_index: {min_index}")

    # Return the corresponding point in q_N
    new_handle_point = q_N[min_index]

    return new_handle_point

def get_mask_loss(F_1, F_2, mask):
    """ Returns the mask loss.

    # Parameters:
        @F_1: torch.tensor size [1, C, H, W], the feature map of the first image
        @F_2: torch.tensor size [1, C, H, W], the feature map of the second image
        @mask: torch.tensor size [H, W], the segmentation mask. 
            NOTE: 1 encodes what areas should move and 0 what areas should stay fixed.
        @lambda_mask: float, the weight of the mask loss
    
    # Returns: torch.tensor of size [1], the mask loss
    """

    # make mask broadcastable to F_1 and F_2
    mask = mask.unsqueeze(0).unsqueeze(0)

    # compute difference between feature maps
    diff = torch.abs(F_1 - F_2)

    # compute loses for moving and fixed regions
    moving_loss = torch.mean(diff * mask)
    fixed_loss = torch.mean(diff * ( 1 - mask))

    mask_loss = moving_loss + fixed_loss

    return mask_loss
